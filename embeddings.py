import logging
import json
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from database_setup import get_db_connection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_chunks_without_embeddings() -> List[Tuple[int, str]]:
    """Retrieve chunks that have not yet been embedded."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, chunk_text 
                FROM document_chunks 
                WHERE embedding IS NULL
            """)
            chunks = cur.fetchall()
            logger.info(f"Found {len(chunks)} chunks without embeddings.")
            return chunks
    except Exception as e:
        logger.error(f"Error retrieving chunks from database: {e}")
        return []
    finally:
        conn.close()

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of text chunks."""
    logger.info("Loading BAAI/bge-m3 model...")
    model = SentenceTransformer('BAAI/bge-m3')
    logger.info("Model loaded. Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    logger.info("Embeddings generated successfully.")
    return embeddings.tolist()

def store_embeddings_in_db(chunk_ids: List[int], embeddings: List[List[float]]):
    """Store the generated embeddings in the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                # Convert embedding to JSON string
                embedding_json = json.dumps(embedding)
                cur.execute(
                    "UPDATE document_chunks SET embedding = %s WHERE id = %s",
                    (embedding_json, chunk_id)
                )
        conn.commit()
        logger.info(f"Stored {len(embeddings)} embeddings in the database.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to store embeddings: {e}")
    finally:
        conn.close()

def process_and_embed_chunks():
    """Main function to generate and store embeddings for all chunks."""
    logger.info("Starting embedding generation process...")
    
    chunks_to_process = get_chunks_without_embeddings()
    
    if not chunks_to_process:
        logger.info("No chunks to embed.")
        return

    chunk_ids, chunk_texts = zip(*chunks_to_process)
    
    embeddings = generate_embeddings(list(chunk_texts))
    
    if embeddings:
        store_embeddings_in_db(list(chunk_ids), embeddings)
            
    logger.info("Completed embedding generation process.")

if __name__ == "__main__":
    process_and_embed_chunks()
