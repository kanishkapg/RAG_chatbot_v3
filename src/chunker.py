import re
import logging
from typing import List, Dict, Tuple
from database_setup import get_db_connection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_documents_from_db() -> List[Tuple[str, str, str]]:
    """Retrieve all documents with extracted text from the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.filename, p.file_hash, p.extracted_text 
                FROM pdf_files p
                LEFT JOIN document_chunks dc ON p.filename = dc.filename AND p.file_hash = dc.file_hash
                WHERE p.extracted_text IS NOT NULL 
                AND p.extracted_text != ''
                AND dc.filename IS NULL
            """)
            documents = cur.fetchall()
            logger.info(f"Retrieved {len(documents)} documents pending chunking.")
            return documents
    except Exception as e:
        logger.error(f"Error retrieving documents from database: {e}")
        return []
    finally:
        conn.close()

def chunk_text_by_page(text: str) -> List[Dict]:
    """
    Splits text into chunks by page boundaries only.
    
    Args:
        text (str): The full text extracted from a document.

    Returns:
        List[Dict]: A list of chunks, each with page number and text.
    """
    chunks = []
    # Split the document into pages based on "Page X" markers
    pages = re.split(r'Page \d+\n', text)
    
    for i, page_content in enumerate(pages):
        page_content = page_content.strip()
        if not page_content:
            continue

        page_number = i + 1
        # Each page becomes a single chunk
        chunks.append({'page_number': page_number, 'chunk_text': page_content})
            
    return chunks

def store_chunks_in_db(filename: str, file_hash: str, chunks: List[Dict]):
    """Deletes old chunks and inserts new ones for a given document."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Delete existing chunks for the document to avoid duplicates
            cur.execute(
                "DELETE FROM document_chunks WHERE filename = %s AND file_hash = %s",
                (filename, file_hash)
            )
            
            # Insert new chunks
            for i, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO document_chunks (filename, file_hash, chunk_index, chunk_text)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (filename, file_hash, i, chunk['chunk_text'])
                )
        conn.commit()
        logger.info(f"Stored {len(chunks)} chunks for {filename}.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to store chunks for {filename}: {e}")
    finally:
        conn.close()

def process_and_chunk_all_documents():
    """Main function to fetch, chunk, and store all documents."""
    logger.info("Starting document chunking process...")
    documents = get_documents_from_db()
    
    if not documents:
        logger.info("No documents to process.")
        return

    for filename, file_hash, extracted_text in documents:
        logger.info(f"Chunking document: {filename}")
        chunks = chunk_text_by_page(extracted_text)
        if chunks:
            store_chunks_in_db(filename, file_hash, chunks)
            
    logger.info("Completed document chunking process.")

if __name__ == "__main__":
    process_and_chunk_all_documents()
