import streamlit as st
import asyncio
import sys
import os
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import tempfile


# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.agents.chatbot import ChatbotAgent
from src.utils.pdf_extract import extract_cnh_fields_ocr

load_dotenv()

def main():
    """Main function for the Streamlit UI."""
    st.set_page_config(
        page_title="Chatbot para Validar N Registro da CNH",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 Chatbot para Validar N Registro da CNH")
    st.markdown(""" 
    Ele pode:
    - Armazenar informações relevantes
    - Validar CNH
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

    # Add PDF uploader at the top of the main function, before chat input
    uploaded_file = st.file_uploader("Envie sua CNH em PDF", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name
        st.session_state["uploaded_pdf_path"] = tmp_pdf_path
        st.success("PDF recebido! Verificando se é uma CNH...")
        # Extract CNH fields immediately
        extracted = extract_cnh_fields_ocr(tmp_pdf_path)
        if extracted["registro"]:
            st.info(f"N registro extraído: {extracted['registro']}")
            # Change button to send validation message to chatbot
            if st.button("Enviar informações extraídas para o chatbot"):
                pdf_message = (
                    f"Considere a seguinte regra para validar o número de registro da CNH brasileira:\n"
                    f'O número de registro da CNH possui 11 dígitos. Os dois últimos dígitos são calculados a partir dos nove primeiros, usando o seguinte algoritmo:\n'
                    f'1. Multiplique cada um dos 9 primeiros dígitos, da esquerda para a direita, pelos pesos de 9 a 1 e some os resultados.\n'
                    f'2. O resto da divisão dessa soma por 11 é o primeiro dígito verificador (se for 10, use 0).\n'
                    f'3. Para o segundo dígito verificador, multiplique os 9 primeiros dígitos pelos pesos de 1 a 9, some, divida por 11 e o resto é o segundo dígito verificador (se for 10, use 0).\n'
                    f'4. O número completo é: 9 dígitos + 2 dígitos verificadores.\n\n'
                    f'Agora, explique se o número de registro abaixo é válido, mostrando o cálculo passo a passo:\n'
                    f'{extracted["registro"]}'
                )
                # Não adiciona a mensagem do usuário ao chat visível
                hidden_prompt = pdf_message

                # Build conversation history (sem o prompt técnico)
                history = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        history.append(AIMessage(content=msg["content"]))

                # Adiciona o prompt técnico só para o LLM
                history.append(HumanMessage(content=hidden_prompt))
                prompt = hidden_prompt

                # Call the chatbot and append the response
                with st.spinner("Pensando..."):
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

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    st.experimental_rerun()
        else:
            st.error("Não foi possível extrair as informações principais da CNH. Certifique-se de que o PDF é uma CNH válida.")

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
