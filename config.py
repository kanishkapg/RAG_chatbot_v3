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