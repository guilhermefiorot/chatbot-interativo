# Chatbot de Aprendizado Adaptativo

Um chatbot interativo que aprende com as interações do usuário e se adapta às preferências. O chatbot armazena informações sobre um tópico específico, valida fatos e ajusta-se às preferências do usuário.

## Recursos

- Interface de usuário interativa construída com Streamlit
- Aprendizado contínuo a partir das interações do usuário
- Validação de fatos antes de adicionar à base de conhecimento
- Adaptação às preferências do usuário
- Banco de dados vetorial para recuperação eficiente de conhecimento
- Containerização com Docker para fácil implantação

## Arquitetura

O projeto é construído usando as seguintes tecnologias:

- **Python**: Linguagem de programação principal
- **Streamlit**: Interface do usuário
- **LangChain**: Framework para trabalhar com LLMs
- **LangGraph**: Orquestração de agentes e fluxo de trabalho
- **Groq**: Provedor de LLM
- **FAISS**: Banco de dados vetorial para armazenamento de conhecimento
- **Docker**: Containerização

## Estrutura do Projeto
```
project/
├── .env.example          # Modelo de variáveis de ambiente
├── .gitignore           
├── README.md             # Documentação do projeto
├── docker-compose.yml    # Configuração Docker para a aplicação e banco de dados
├── Dockerfile            # Configuração Docker para a aplicação
├── src/
│   ├── __init__.py
│   ├── main.py           # Ponto de entrada da aplicação
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── chatbot.py    # Implementação do agente LangGraph
│   │   └── validator.py  # Agente de validação de entrada
│   ├── database/
│   │   ├── __init__.py
│   │   └── vector_store.py  # Integração com banco de dados vetorial FAISS
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py   # Configurações
│   ├── memory/
│   │   ├── __init__.py
│   │   └── knowledge_base.py  # Armazenamento e recuperação de conhecimento
│   └── ui/
│       ├── __init__.py
│       └── app.py        # Implementação da interface Streamlit
└── tests/
    ├── __init__.py
    └── test_chatbot.py   # Testes básicos
```

## Instruções de Configuração

### Pré-requisitos

- Python 3.10+
- Poetry
- Docker e Docker Compose
- Chave de API Groq

### Desenvolvimento Local

1. Clone o repositório:
   ```bash
   git clone https://github.com/yourusername/adaptive-learning-chatbot.git
   cd adaptive-learning-chatbot
   ```

2. Instale as dependências:
   ```bash
   poetry install
   ```

3. Crie um arquivo `.env` a partir do modelo:
   ```bash
   cp .env.example .env
   ```

4. Edite o arquivo `.env` para adicionar sua chave de API Groq:
   ```
   GROQ_API_KEY=sua_chave_api_groq_aqui
   ```

5. Execute a aplicação:
   ```bash
   streamlit run src/ui/app.py
   ```

### Implantação com Docker

1. Construa e execute com Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Acesse a aplicação em `http://localhost:8501`

## Guia de Uso

1. **Iniciando uma Conversa**: Comece fazendo perguntas ou compartilhando informações com o chatbot.

2. **Ensinando o Chatbot**: O chatbot aprenderá informações que você compartilhar após validá-las.

3. **Definindo Preferências**: O chatbot se adaptará às suas preferências:
   - Para comunicação formal: "Eu prefiro um tom formal"
   - Para respostas concisas: "Eu gosto de respostas curtas e diretas"
   - Para respostas detalhadas: "Por favor, forneça explicações detalhadas"

4. **Visualizando Informações Aprendidas**: Use a barra lateral para ver o que o chatbot aprendeu.

## Testes

Execute os testes com:
```bash
pytest
```

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.
