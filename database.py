import psycopg2
from config import POSTGRES_CONFIG
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_database():
    """Connect to PostgreSQL and set up the necessary table."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_texts (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) UNIQUE NOT NULL,
                    file_hash VARCHAR(32) NOT NULL,
                    text_content TEXT,
                    issued_date DATE,
                    effective_date DATE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Ensure columns exist for older deployments where the table was created without them
            cur.execute("""
                ALTER TABLE pdf_texts
                ADD COLUMN IF NOT EXISTS issued_date DATE;
            """)
            cur.execute("""
                ALTER TABLE pdf_texts
                ADD COLUMN IF NOT EXISTS effective_date DATE;
            """)
        conn.commit()
        conn.close()
        logger.info("Database setup complete. 'pdf_texts' table is ready.")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

def get_all_texts():
    """Fetch all records from the pdf_texts table.

    Returns list of tuples: (id, text_content)
    """
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, text_content FROM pdf_texts;")
            records = cur.fetchall()
            return records
    except Exception as e:
        logger.error(f"Failed to fetch records: {e}")
        return []
    finally:
        conn.close()

def get_texts_missing_metadata():
    """Fetch records that are missing metadata (issued_date or effective_date is NULL).

    Returns list of tuples: (id, text_content)
    """
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, text_content
                FROM pdf_texts
                WHERE (issued_date IS NULL OR effective_date IS NULL)
                  AND text_content IS NOT NULL
                ;
                """
            )
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to fetch records missing metadata: {e}")
        return []
    finally:
        conn.close()

def update_metadata(record_id, issued_date, effective_date) -> bool:
    """Update the metadata for a specific record.

    Returns True if the update affected a row, False otherwise.
    """
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pdf_texts
                SET issued_date = COALESCE(%s, issued_date),
                    effective_date = COALESCE(%s, effective_date)
                WHERE id = %s
                """,
                (issued_date, effective_date, record_id)
            )
            updated = cur.rowcount > 0
            conn.commit()
            if not updated:
                logger.warning(f"No rows updated for record {record_id}.")
            return updated
    except Exception as e:
        logger.error(f"Failed to update metadata for record {record_id}: {e}")
        return False
    finally:
        conn.close()

if __name__ == '__main__':
    setup_database()
