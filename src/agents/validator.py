from typing import Dict, Any, List
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from src.config.settings import GROQ_API_KEY, LLM_MODEL
from src.memory.knowledge_base import KnowledgeBase


class ValidationAgent:
    """Agent responsible for validating user inputs and identifying factual statements."""

    def __init__(self, knowledge_base: KnowledgeBase):
        """Initialize the validation agent.

        Args:
            knowledge_base: Knowledge base for storing validated facts
        """
        self.knowledge_base = knowledge_base
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.2,
        )

    async def process(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """Process a user message to extract facts and preferences.
        Args:
            message: User message
            chat_history: Chat history
        Returns:
            Dict with processing results
        """
        # Check for preferences
        preference = self.knowledge_base.identify_preference(message, self.llm)

        # Extract potential factual statements
        facts = self._extract_potential_facts(message)
        validated_facts = []

        # Validate each potential fact
        for fact in facts:
            is_valid, reason = self.knowledge_base.validate_fact(fact, self.llm)
            if is_valid:
                self.knowledge_base.add_fact(fact, validated=True)
                validated_facts.append(fact)

        return {
            "message": message,
            "preference": preference,
            "validated_facts": validated_facts,
        }

    def _extract_potential_facts(self, message: str) -> List[str]:
        """Extract potential factual statements from a message.
        Args:
            message: User message
        Returns:
            List of potential factual statements
        """
        prompt = f"""
        Extract any factual claims from the following message. A factual claim is a statement
        that asserts something about the world that can be verified as true or false.        
        Message: "{message}"
        Extract each distinct factual claim, ignoring opinions, questions, and subjective statements.
        Return the results as a JSON list of strings, with each string being a single factual claim.
        If there are no factual claims, return an empty list.
        Example output: ["The Earth orbits the Sun", "Water freezes at 0 degrees Celsius"]
        Response (JSON list):
        """
        response = self.llm.invoke(prompt)
        try:
            import json
            response_text = response.content
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    return []
            facts_list = json.loads(json_text)
            return facts_list if isinstance(facts_list, list) else []
        except Exception as e:
            return e
