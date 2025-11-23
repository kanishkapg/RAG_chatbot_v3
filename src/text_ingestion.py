import os
import sys
import hashlib
import logging
import psycopg2

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import TXT_DIR, POSTGRES_CONFIG
from src.database_setup import get_db_connection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_file_hash(file_path: str) -> str:
    """Computes the MD5 hash of a file."""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                hash_md5.update(byte_block)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {e}")
        raise

def ingest_text_file(file_path: str):
    """Reads a text file and stores its content in PostgreSQL if it's new or updated."""
    file_hash = compute_file_hash(file_path)
    filename = os.path.splitext(os.path.basename(file_path))[0]

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_hash FROM pdf_files WHERE filename = %s",
                (filename,)
            )
            result = cur.fetchone()
            if result and result[0] == file_hash:
                logger.info(f"Skipping '{filename}' - already processed with the same hash.")
                return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pdf_files (filename, file_hash, extracted_text, extraction_date)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (filename) DO UPDATE
                SET file_hash = EXCLUDED.file_hash,
                    extracted_text = EXCLUDED.extracted_text,
                    extraction_date = CURRENT_TIMESTAMP;
                """,
                (filename, file_hash, content)
            )
        conn.commit()
        logger.info(f"Successfully ingested text from '{filename}'.")

    except Exception as e:
        logger.error(f"Failed to ingest text from '{filename}': {e}")
        conn.rollback()
    finally:
        conn.close()

def process_all_text_files(directory: str = TXT_DIR):
    """Processes all .txt files in the specified directory."""
    if not os.path.exists(directory):
        logger.error(f"Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(directory, filename)
            logger.info(f"Processing '{file_path}'...")
            ingest_text_file(file_path)

if __name__ == "__main__":
    process_all_text_files()
