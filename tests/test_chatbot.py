import pytest
import asyncio
from unittest.mock import MagicMock, patch
from langchain.schema import HumanMessage

from src.agents.chatbot import ChatbotAgent
from src.memory.knowledge_base import KnowledgeBase


@pytest.fixture
def mock_llm():
    """Mock LLM responses."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="This is a test response")
    return mock


@pytest.fixture
def chatbot_with_mock_llm(mock_llm):
    """Create a chatbot with a mock LLM."""
    chatbot = ChatbotAgent()
    chatbot.llm = mock_llm
    return chatbot


class TestChatbotAgent:
    """Tests for the ChatbotAgent class."""

    @pytest.mark.asyncio
    async def test_chat_generates_response(self, chatbot_with_mock_llm):
        """Test that the chat method generates a response."""
        # Arrange
        message = "Hello, I'm a test message"
        history = [HumanMessage(content="Previous message")]
        
        # Act
        with patch.object(chatbot_with_mock_llm.knowledge_base, 'get_relevant_facts', return_value=[]):
            with patch.object(chatbot_with_mock_llm.knowledge_base, 'identify_preference', return_value=None):
                response = await chatbot_with_mock_llm.chat(message, history)
        
        # Assert
        assert isinstance(response, str)
        assert len(response) > 0


class TestKnowledgeBase:
    """Tests for the KnowledgeBase class."""

    def test_add_fact_validated(self):
        """Test adding a validated fact."""
        # Arrange
        kb = KnowledgeBase()
        fact = "The Earth orbits the Sun"
        
        # Mock the vector store
        kb.vector_store = MagicMock()
        
        # Act
        result = kb.add_fact(fact, validated=True)
        
        # Assert
        assert result is True
        kb.vector_store.add_text.assert_called_once()

    def test_add_fact_not_validated(self):
        """Test adding a non-validated fact."""
        # Arrange
        kb = KnowledgeBase()
        fact = "The Earth is flat"
        
        # Mock the vector store
        kb.vector_store = MagicMock()
        
        # Act
        result = kb.add_fact(fact, validated=False)
        
        # Assert
        assert result is False
        kb.vector_store.add_text.assert_not_called()

    def test_add_preference(self):
        """Test adding a user preference."""
        # Arrange
        kb = KnowledgeBase()
        
        # Mock the vector store
        kb.vector_store = MagicMock()
        
        # Act
        kb.add_preference("tone", "formal")
        
        # Assert
        assert kb.user_preferences.get("tone") == "formal"
        kb.vector_store.add_text.assert_called_once()
