import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR="./data/dummy" # change into 'dummy' or 'original'
POSTGRES_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "rag_chatbot_v3"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "12345678"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this environment variable
GROQ_MODEL = "llama-3.1-8b-instant"

# Hybrid Search Configuration
DEFAULT_SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.5"))  # Alpha
DEFAULT_LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.5"))   # Beta
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# Response Generation Configuration  
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))