import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.config.settings import GROQ_API_KEY, EMBEDDING_DIMENSION


class VectorStore:
    """FAISS Vector Store for storing and retrieving knowledge embeddings."""
    def __init__(self, index_path: str = "data/vector_index"):
        """Initialize the vector store.
        Args:
            index_path: Path to save and load the FAISS index
        """
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=GROQ_API_KEY)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        if os.path.exists(f"{index_path}.faiss"):
            self.index = faiss.read_index(f"{index_path}.faiss")
            with open(f"{index_path}.pkl", "rb") as f:
                self.documents = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            self.documents = []
    
    def add_text(self, text: str, metadata: Dict[str, Any]) -> None:
        """Add text to the vector store with associated metadata.
        
        Args:
            text: The text to add
            metadata: Associated metadata for the text
        """
        embedding = self.embeddings.embed_query(text)
        embedding_np = np.array([embedding], dtype=np.float32)
        
        self.index.add(embedding_np)
        
        doc = Document(page_content=text, metadata=metadata)
        self.documents.append(doc)
        
        self._save_index()
    
    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Search for similar documents.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if len(self.documents) == 0:
            return []
        
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding_np, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                doc = self.documents[idx]
                distance = distances[0][i]
                similarity = 1.0 / (1.0 + distance)
                results.append((doc, similarity))
        
        return results
    
    def _save_index(self) -> None:
        """Save the index and documents to disk."""
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump(self.documents, f)
    
    def delete_by_metadata(self, metadata_key: str, metadata_value: Any) -> None:
        """Delete documents with matching metadata.
        
        Args:
            metadata_key: The metadata key to match
            metadata_value: The metadata value to match
        """
        new_documents = []
        for doc in self.documents:
            if doc.metadata.get(metadata_key) != metadata_value:
                new_documents.append(doc)
        
        if len(new_documents) != len(self.documents):
            self.documents = new_documents
            self.index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            if new_documents:
                embeddings = [self.embeddings.embed_query(doc.page_content)
                              for doc in new_documents]
                embeddings_np = np.array(embeddings, dtype=np.float32)
                self.index.add(embeddings_np)
            
            self._save_index()
