import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = ""
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store")
DEFAULT_MODEL = os.getenv(
    "DEFAULT_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct"
)
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "6379"))
MAX_HISTORY_LENGTH = 10
EMBEDDING_DIMENSION = 1536
SIMILARITY_THRESHOLD = 0.75
