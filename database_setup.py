import psycopg2
from config import POSTGRES_CONFIG
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def create_pdf_files_table():
    """Creates the pdf_files table in the database if it does not already exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_files (
                    filename TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    extracted_text TEXT,
                    extraction_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
        logger.info("Table 'pdf_files' created successfully or already exists.")
    except Exception as e:
        logger.error(f"Failed to create 'pdf_files' table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_pdf_files_table()
