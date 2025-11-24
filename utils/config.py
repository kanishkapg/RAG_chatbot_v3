import os
from dotenv import load_dotenv

load_dotenv(override=True)

DATA_DIR="./data/dummy" # change into 'dummy' or 'original'
TXT_DIR="./data/txt_files"
POSTGRES_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "rag_chatbot_v3"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "12345678"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this environment variable
GROQ_MODEL = "llama-3.1-8b-instant"

# Hybrid Search Configuration for Multiplicative Gating
MULTIPLICATIVE_WEIGHTS = {
    "semantic": float(os.getenv("MULT_SEMANTIC_WEIGHT", "0.7")),
    "lexical": float(os.getenv("MULT_LEXICAL_WEIGHT", "0.3")),
    "recency": float(os.getenv("MULT_RECENCY_WEIGHT", "0.7"))
}

# Hybrid Search Configuration for Penalty Scored Fusion
PENALTY_WEIGHTS = {
    "semantic": float(os.getenv("PEN_SEMANTIC_WEIGHT", "0.7")),
    "lexical": float(os.getenv("PEN_LEXICAL_WEIGHT", "0.3")),
    "recency": float(os.getenv("PEN_RECENCY_WEIGHT", "1.0"))
}

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))

# Temporal Relevance Configuration
DEFAULT_DECAY_BASE = float(os.getenv("DECAY_BASE", "0.65"))
DEFAULT_HALF_LIFE_DAYS = float(os.getenv("HALF_LIFE_DAYS", "300"))

# Response Generation Configuration  
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Penalty Scored Fusion Configuration
DEFAULT_MARGIN = float(os.getenv("MARGIN", "0.15"))
DEFAULT_P_COEFF = float(os.getenv("P_COEFF", "2.0"))