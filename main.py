from database import setup_database
from ocr import process_all_pdfs
from metadata_extraction import process_and_store_metadata
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to set up the database, process all PDFs, and extract metadata.
    """
    logger.info("Starting the application...")
    
    # Step 1: Set up the database
    logger.info("Setting up the database...")
    setup_database()
    
    # Step 2: Process all PDF files
    logger.info("Starting PDF processing...")
    process_all_pdfs()

    # Step 3: Extract and store metadata
    logger.info("Starting metadata extraction...")
    process_and_store_metadata()
    
    logger.info("Application finished.")

if __name__ == '__main__':
    main()
