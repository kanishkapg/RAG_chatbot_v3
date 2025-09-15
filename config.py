import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR="./data/dummy" # change into 'dummy' or 'original'
POSTGRES_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT")
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model_type": "word2vec",  # Options: 'word2vec', 'sentence_transformers', 'openai'
    "model_name": "word2vec-google-news-300",  # For Word2Vec
    "vector_dimension": 300,
    "chunk_size": 512,  # For splitting documents into chunks
    "chunk_overlap": 50
}

# LLM Configuration
LLM_CONFIG = {
    "model_name": "llama-3.1-8b-instant",
    "api_base": os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
    "api_key": os.getenv("GROQ_API_KEY"),
    "max_tokens": 1000,
    "temperature": 0.1,
    "context_window": 4096
}

# Search Configuration
SEARCH_CONFIG = {
    "max_results": 5,
    "similarity_threshold": 0.7,
    "hybrid_search_weights": {
        "semantic": 0.7,
        "keyword": 0.3
    }
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")