from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from src.database.vector_store import VectorStore
from src.config.settings import SIMILARITY_THRESHOLD

JSON_CODE_BLOCK = "```json"


class KnowledgeBase:
    """Knowledge base that stores and retrieves information."""
    def __init__(self):
        """Initialize the knowledge base."""
        self.vector_store = VectorStore()
        self.user_preferences = {}
    
    def add_fact(self, fact: str, validated: bool = False, source: str = "user") -> bool:
        """Add a validated fact to the knowledge base.
        
        Args:
            fact: The fact to add
            validated: Whether the fact has been validated
            source: Source of the fact (user, system, etc.)
            
        Returns:
            bool: Whether the fact was added
        """
        if not validated:
            return False
        
        metadata = {
            "type": "fact",
            "validated": validated,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.vector_store.add_text(fact, metadata)
        return True
    
    def add_preference(self, preference_key: str, preference_value: Any) -> None:
        """Add or update a user preference.
        
        Args:
            preference_key: The preference key (e.g., "tone", "verbosity")
            preference_value: The preference value
        """
        self.user_preferences[preference_key] = preference_value
        
        preference_text = f"User preference: {preference_key} = {preference_value}"
        metadata = {
            "type": "preference",
            "key": preference_key,
            "value": preference_value,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.vector_store.add_text(preference_text, metadata)
    
    def get_preferences(self) -> Dict[str, Any]:
        """Get all user preferences.
        
        Returns:
            Dict of user preferences
        """
        return self.user_preferences
    
    def get_relevant_facts(self, query: str, k: int = 3) -> List[str]:
        """Get facts relevant to the query.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of relevant facts
        """
        results = self.vector_store.search(query, k)
        
        facts = []
        for doc, score in results:
            if (
                doc.metadata.get("type") == "fact" and
                doc.metadata.get("validated", False) and
                score > SIMILARITY_THRESHOLD
            ):
                facts.append(doc.page_content)
        
        return facts
    
    def validate_fact(self, fact: str, llm) -> Tuple[bool, str]:
        """Validate if a statement is factual using the LLM.
        
        Args:
            fact: The fact to validate
            llm: Language model for validation
            
        Returns:
            Tuple of (is_valid, reason)
        """
        prompt = f"""
        Determine if the following statement is factually accurate based on general knowledge.
        If it's a subjective preference, opinion, or a personal experience, indicate that it's not a factual statement.
        
        Statement: "{fact}"
        
        Provide your assessment as JSON with these fields:
        - is_factual: true/false (is this a factual claim rather than an opinion or preference)
        - is_accurate: true/false (if factual, is it generally accurate)
        - reason: brief explanation
        
        Response (JSON):
        """
        response = llm.invoke(prompt)
        try:
            import json
            response_text = response.content
            if JSON_CODE_BLOCK in response_text:
                json_text = response_text.split(JSON_CODE_BLOCK)[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text
            result = json.loads(json_text)
            is_valid = result.get("is_factual", False) and result.get("is_accurate", False)
            reason = result.get("reason", "No reason provided")

            return is_valid, reason
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def identify_preference(self, message: str, llm) -> Optional[Dict[str, Any]]:
        """Identify if a message contains a user preference.
        Args:
            message: User message
            llm: Language model for preference identification
        Returns:
            Dict with preference information or None
        """
        prompt = f"""
        Determine if the following message contains a user preference about how they want the chatbot to behave.
        Examples of preferences include:
        - Preferred tone (formal, casual, friendly, professional)
        - Verbosity (concise, detailed)
        - Style of responses (technical, simple)
        - Specific topics of interest or topics to avoid
        Message: "{message}"
        Provide your assessment as JSON with these fields:
        - contains_preference: true/false
        - preference_type: the type of preference (if any)
        - preference_value: the specific value for this preference
        - confidence: 0-1 value of how confident you are in this assessment
        Response (JSON):
        """
        response = llm.invoke(prompt)

        try:
            import json
            response_text = response.content
            if JSON_CODE_BLOCK in response_text:
                json_text = response_text.split(JSON_CODE_BLOCK)[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text
                    
            result = json.loads(json_text)
            
            if result.get("contains_preference", False) and result.get("confidence", 0) > 0.7:
                return {
                    "type": result.get("preference_type"),
                    "value": result.get("preference_value")
                }
            return None
        except Exception:
            return None
