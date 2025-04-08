from typing import Dict, Any, List, Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from groq import Groq
from ..config.settings import (
    GROQ_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS
)
from src.config.settings import LLM_MODEL
from src.memory.knowledge_base import KnowledgeBase
from src.agents.validator import ValidationAgent


class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    validated_facts: Annotated[List[str], "Validated facts from the conversation"]
    preferences: Annotated[Dict[str, Any], "User preferences"]
    current_input: Annotated[str, "The current user input"]
    knowledge_base: Annotated[Any, "The knowledge base instance"]
    response: Annotated[str, "The response to return to the user"]


class ChatbotAgent:
    """Adaptive chatbot agent built with LangGraph."""

    def __init__(self):
        """Initialize the chatbot agent."""
        self.knowledge_base = KnowledgeBase()
        self.validator = ValidationAgent(self.knowledge_base)
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.7,
        )

        self.workflow = self._build_graph()

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = DEFAULT_MODEL
        self.temperature = DEFAULT_TEMPERATURE
        self.setup_graph()

    def _build_graph(self) -> StateGraph:
        """Build the state graph for the agent.
        Returns:
            StateGraph instance
        """
        builder = StateGraph(ChatState)
        builder.add_node("validate_input", self._validate_input)
        builder.add_node("process_preferences", self._process_preferences)
        builder.add_node("retrieve_context", self._retrieve_context)
        builder.add_node("generate_response", self._generate_response)
        builder.add_edge("validate_input", "process_preferences")
        builder.add_edge("process_preferences", "retrieve_context")
        builder.add_edge("retrieve_context", "generate_response")
        builder.add_edge("generate_response", END)
        builder.set_entry_point("validate_input")

        return builder.compile()

    async def _validate_input(self, state: ChatState) -> ChatState:
        """Validate the user input and extract facts and preferences.
        Args:
            state: Current state
        Returns:
            Updated state
        """
        current_input = state["current_input"]
        messages = state["messages"]
        result = await self.validator.process(current_input, messages)
        validated_facts = state.get("validated_facts", []) + result.get("validated_facts", [])
        return {
            **state,
            "validated_facts": validated_facts,
        }

    async def _process_preferences(self, state: ChatState) -> ChatState:
        """Process and store user preferences.
        Args:
            state: Current state
        Returns:
            Updated state
        """
        current_input = state["current_input"]
        knowledge_base = state["knowledge_base"]
        preference = knowledge_base.identify_preference(current_input, self.llm)
        if preference:
            pref_type = preference.get("type")
            pref_value = preference.get("value")
            if pref_type and pref_value:
                knowledge_base.add_preference(pref_type, pref_value)
        preferences = knowledge_base.get_preferences()
        return {
            **state,
            "preferences": preferences,
        }

    async def _retrieve_context(self, state: ChatState) -> ChatState:
        """Retrieve relevant context for the response.
        Args:
            state: Current state
        Returns:
            Updated state with context
        """
        current_input = state["current_input"]
        knowledge_base = state["knowledge_base"]
        relevant_facts = knowledge_base.get_relevant_facts(current_input)
        return {
            **state,
            "relevant_facts": relevant_facts,
        }

    async def _generate_response(self, state: ChatState) -> ChatState:
        """Generate a response based on the context and preferences.
        Args:
            state: Current state
        Returns:
            Updated state with response
        """
        current_input = state["current_input"]
        messages = state["messages"]
        preferences = state.get("preferences", {})
        relevant_facts = state.get("relevant_facts", [])
        
        preference_instructions = ""
        for pref_type, pref_value in preferences.items():
            preference_instructions += f"- {pref_type}: {pref_value}\n"

        fact_context = ""
        if relevant_facts:
            fact_context = "Based on these facts I've learned:\n"
            for fact in relevant_facts:
                fact_context += f"- {fact}\n"

        system_prompt = f"""
        You are an adaptive and helpful chatbot assistant. Respond to the user's message thoughtfully.

        {fact_context if fact_context else ""}

        User preferences:
        {preference_instructions if preference_instructions else "No specific preferences set yet."}

        Always be accurate, helpful, and adapt to the user's preferences.
        """

        history = [
            {"role": "system", "content": system_prompt}
        ]

        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        history.append({"role": "user", "content": current_input})
        response = self.llm.invoke(history)

        return {
            **state,
            "response": response.content,
        }

    async def chat(self, message: str, history: List[BaseMessage]) -> str:
        """Process a user message and generate a response.

        Args:
            message: User message
            history: Chat history

        Returns:
            Generated response
        """
        state = {
            "messages": history,
            "validated_facts": [],
            "preferences": self.knowledge_base.get_preferences(),
            "current_input": message,
            "knowledge_base": self.knowledge_base,
            "response": ""
        }
        final_state = await self.workflow.ainvoke(state)
        return final_state.get("response", "I'm not sure how to respond to that.")

    def _convert_messages_to_groq_format(
        self,
        messages: List[HumanMessage | AIMessage]
    ) -> List[Dict]:
        """Converts LangChain messages to Groq format."""
        groq_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                groq_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                groq_messages.append(
                    {"role": "assistant", "content": msg.content}
                )
        return groq_messages

    def _get_groq_response(
        self,
        messages: List[HumanMessage | AIMessage],
        temperature: float | None = None,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """Get response from Groq API."""
        groq_messages = self._convert_messages_to_groq_format(messages)
        temp = temperature if temperature is not None else self.temperature

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=groq_messages,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=1,
            stream=False
        )

        return completion.choices[0].message.content

    def setup_graph(self):
        """Configure the message processing graph."""
        def route_message(state: Dict) -> str:
            """Route the message to the appropriate node."""
            message = state["messages"][-1].content.lower()

            if "corrigir" in message or "correção" in message:
                return "correction"
            elif "preferência" in message or "preferir" in message:
                return "preference"
            return "chat"

        def process_chat(state: Dict) -> Dict:
            """Process normal chat message."""
            messages = state["messages"]
            response = self._get_groq_response(messages)
            return {"messages": messages + [AIMessage(content=response)]}

        def process_correction(state: Dict) -> Dict:
            """Process user corrections."""
            messages = state["messages"]
            response = "Thank you for the correction! I will learn from it."
            return {"messages": messages + [AIMessage(content=response)]}

        def process_preference(state: Dict) -> Dict:
            """Process user preferences."""
            messages = state["messages"]
            response = "I understand your preference! I will adapt to it."
            return {"messages": messages + [AIMessage(content=response)]}

        workflow = StateGraph(ChatState)
        workflow.add_node("route", route_message)
        workflow.add_node("chat", process_chat)
        workflow.add_node("correction", process_correction)
        workflow.add_node("preference", process_preference)

        workflow.add_edge("route", "chat")
        workflow.add_edge("route", "correction")
        workflow.add_edge("route", "preference")
        workflow.add_edge("chat", "route")
        workflow.add_edge("correction", "route")
        workflow.add_edge("preference", "route")
        
        workflow.set_entry_point("route")
        self.graph = workflow.compile()

    def process_message(self, message: str) -> str:
        """Process a user message and return the response."""
        state = {"messages": [HumanMessage(content=message)]}
        result = self.graph.invoke(state)
        return result["messages"][-1].content

    def update_temperature(self, temperature: float):
        """Update the model temperature."""
        self.temperature = temperature
