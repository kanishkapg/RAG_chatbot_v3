import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR="./data/dummy" # change into 'dummy' or 'original'
POSTGRES_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "Circulars_Chatbot_v3"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "12345678"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}