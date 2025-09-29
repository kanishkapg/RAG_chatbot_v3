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
                    extraction_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(filename, file_hash)
                );
            """)
        conn.commit()
        logger.info("Table 'pdf_files' created successfully or already exists.")
    except Exception as e:
        logger.error(f"Failed to create 'pdf_files' table: {e}")
    finally:
        conn.close()

def create_document_metadata_table():
    """Creates the document_metadata table in the database if it does not already exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    metadata JSONB,
                    CONSTRAINT fk_document_metadata_pdf_files 
                        FOREIGN KEY (filename, file_hash) 
                        REFERENCES pdf_files(filename, file_hash)
                        ON DELETE CASCADE,
                    UNIQUE(filename, file_hash)
                );
            """)
            # Create index on filename and file_hash for better join performance
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_metadata_filename_hash 
                ON document_metadata(filename, file_hash);
            """)
        conn.commit()
        logger.info("Table 'document_metadata' created successfully or already exists.")
    except Exception as e:
        logger.error(f"Failed to create 'document_metadata' table: {e}")
    finally:
        conn.close()

def create_document_chunks_table():
    """
    Creates the document_chunks table for storing text chunks and their embeddings.
    The 'embedding' column uses JSONB to be flexible, but can be switched to a
    dedicated vector type if the pgvector extension is used.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Optionally, enable the pgvector extension if you plan to use it.
            # cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding JSONB, -- Can be changed to 'vector(dimension)' with pgvector
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_document_chunks_pdf_files 
                        FOREIGN KEY (filename, file_hash) 
                        REFERENCES pdf_files(filename, file_hash)
                        ON DELETE CASCADE,
                    UNIQUE(filename, file_hash, chunk_index)
                );
            """)
            # Create indexes for better performance
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_filename_hash 
                ON document_chunks(filename, file_hash);
            """)
        conn.commit()
        logger.info("Table 'document_chunks' created successfully or already exists.")
    except Exception as e:
        logger.error(f"Failed to create 'document_chunks' table: {e}")
    finally:
        conn.close()

def create_all_tables():
    """Creates all required tables for the RAG chatbot."""
    logger.info("Creating all database tables...")
    create_pdf_files_table()
    create_document_metadata_table()
    create_document_chunks_table()
    logger.info("All tables created successfully.")

if __name__ == "__main__":
    create_all_tables()
