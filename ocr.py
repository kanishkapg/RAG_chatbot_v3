import pytesseract
from pdf2image import convert_from_path
import os
import hashlib
import psycopg2
from config import DATA_DIR, POSTGRES_CONFIG
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_db_connection():
    """Establish PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise

def ocr_and_store(pdf_path: str):
    """
    Performs OCR on a PDF file and stores the extracted text in a PostgreSQL database.
    Skips processing if the file has not changed.
    """
    pdf_filename = os.path.basename(pdf_path)
    file_hash = compute_file_hash(pdf_path)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_hash FROM pdf_texts WHERE filename = %s",
                (pdf_filename,)
            )
            result = cur.fetchone()
            if result and result[0] == file_hash:
                logger.info(f"Skipping '{pdf_filename}' - already processed with the same content.")
                return

            logger.info(f"Processing '{pdf_filename}'...")
            images = convert_from_path(pdf_path, dpi=300)
            full_text = ""
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang='eng')
                full_text += f"----- Page {i+1} -----\n{text}\n"

            if result:
                cur.execute(
                    """
                    UPDATE pdf_texts
                    SET text_content = %s, file_hash = %s, created_at = CURRENT_TIMESTAMP
                    WHERE filename = %s
                    """,
                    (full_text, file_hash, pdf_filename)
                )
                logger.info(f"Updated text for '{pdf_filename}' in the database.")
            else:
                cur.execute(
                    """
                    INSERT INTO pdf_texts (filename, file_hash, text_content)
                    VALUES (%s, %s, %s)
                    """,
                    (pdf_filename, file_hash, full_text)
                )
                logger.info(f"Stored new text for '{pdf_filename}' in the database.")
            
            conn.commit()

    except Exception as e:
        logger.error(f"Failed to process {pdf_filename}: {e}")
    finally:
        conn.close()

def process_all_pdfs(data_dir: str = DATA_DIR):
    """Process all PDFs in the data directory."""
    if not os.path.exists(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        return

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        ocr_and_store(pdf_path)

if __name__ == '__main__':
    process_all_pdfs()
    