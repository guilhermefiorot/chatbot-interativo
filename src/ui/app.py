import streamlit as st
import asyncio
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from src.agents.chatbot import ChatbotAgent

load_dotenv()


def main():
    """Main function for the Streamlit UI."""
    st.set_page_config(
        page_title="Chatbot Interativo",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 Chatbot Interativo")
    st.markdown("""
    Este chatbot aprende e se adapta com base em suas interações. 
    Ele pode:
    - Armazenar informações relevantes
    - Aprender com suas correções
    - Adaptar-se ao seu estilo de comunicação
    """)
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatbotAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.sidebar:
        st.header("Sobre")
        st.markdown("""
        Este chatbot utiliza:
        - LangChain para processamento
        - LangGraph para fluxo de trabalho
        - Groq para LLM
        - FAISS para armazenamento vetorial
        """)
        st.header("Configurações")
        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Controla a criatividade das respostas"
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Digite sua mensagem..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if "chatbot" not in st.session_state:
            st.session_state.chatbot = ChatbotAgent()
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        history.append(AIMessage(content=msg["content"]))

                try:
                    loop = asyncio.new_event_loop()
                    response = loop.run_until_complete(
                        st.session_state.chatbot.chat(prompt, history)
                    )
                    loop.close()
                except Exception as e:
                    response = f"Ocorreu um erro: {str(e)}"
                    st.error("Erro ao processar sua mensagem. Veja os logs para mais detalhes.")
                    print(f"Erro no chatbot: {str(e)}")

                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    with st.sidebar:
        st.header("Learning Status")

        if "chatbot" in st.session_state:
            preferences = st.session_state.chatbot.knowledge_base.get_preferences()

            st.subheader("User Preferences")
            if preferences:
                for pref_type, pref_value in preferences.items():
                    st.write(f"**{pref_type}:** {pref_value}")
            else:
                st.write("No preferences learned yet.")
            if st.button("View Learned Facts"):
                st.info("Functionality to display learned facts will be shown here.")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()


if __name__ == "__main__":
    main()
