#!/usr/bin/env python3
"""
Complete RAG Chatbot Pipeline
=============================

This script executes the complete pipeline for the RAG chatbot system:
1. Database setup (create tables)
2. OCR processing (extract text from PDFs and store in database)
3. Metadata extraction (extract metadata from stored texts and store in database)

Usage:
    python full_pipeline.py [--data-dir DATA_DIR] [--force-reprocess]
    
    --data-dir: Directory containing PDF files (default: from config.py)
    --force-reprocess: Force reprocessing of all files even if already processed
"""

import sys
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_setup import create_all_tables, get_db_connection
from ocr import process_all_pdfs
from metadata_extractor import MetadataExtractor
from config import DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete pipeline for RAG chatbot data processing."""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.metadata_extractor = MetadataExtractor()
        
    def validate_prerequisites(self) -> bool:
        """Validate that all prerequisites are met."""
        logger.info("Validating prerequisites...")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} does not exist")
            return False
            
        # Check if there are PDF files
        pdf_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logger.error(f"No PDF files found in {self.data_dir}")
            return False
            
        logger.info(f"Found {len(pdf_files)} PDF files in {self.data_dir}")
        
        # Test database connection
        try:
            conn = get_db_connection()
            conn.close()
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
            
        logger.info("All prerequisites validated successfully")
        return True
        
    def step1_setup_database(self) -> bool:
        """Step 1: Create all required database tables."""
        logger.info("="*50)
        logger.info("STEP 1: Setting up database tables")
        logger.info("="*50)
        
        try:
            create_all_tables()
            logger.info("✅ Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create database tables: {e}")
            return False
            
    def step2_ocr_processing(self, force_reprocess: bool = False) -> bool:
        """Step 2: Process PDFs and extract text using OCR."""
        logger.info("="*50)
        logger.info("STEP 2: OCR Processing and Text Extraction")
        logger.info("="*50)
        
        try:
            if force_reprocess:
                logger.info("Force reprocessing enabled - will reprocess all files")
                # Clear existing data if force reprocess
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM document_metadata")
                    cur.execute("DELETE FROM pdf_files")
                conn.commit()
                conn.close()
                logger.info("Cleared existing data from database")
            
            extracted_texts = process_all_pdfs(self.data_dir)
            
            if extracted_texts:
                logger.info(f"✅ Successfully processed {len(extracted_texts)} PDF files")
                return True
            else:
                logger.warning("⚠️  No new files were processed (may already exist)")
                return True
        except Exception as e:
            logger.error(f"❌ Failed during OCR processing: {e}")
            return False
            
    def step3_metadata_extraction(self) -> bool:
        """Step 3: Extract metadata from stored texts using LLM."""
        logger.info("="*50)
        logger.info("STEP 3: Metadata Extraction")
        logger.info("="*50)
        
        try:
            results = self.metadata_extractor.process_all_documents()
            
            if results:
                logger.info(f"✅ Successfully extracted metadata for {len(results)} documents")
                
                # Log summary of extracted metadata
                logger.info("\nMetadata extraction summary:")
                for filename, metadata in results.items():
                    circular_no = metadata.get('circular_number', 'N/A')
                    title = metadata.get('title', 'N/A')
                    logger.info(f"  📄 {filename}: {circular_no} - {title}")
                    
                return True
            else:
                logger.warning("⚠️  No metadata was extracted")
                return False
        except Exception as e:
            logger.error(f"❌ Failed during metadata extraction: {e}")
            return False
            
    def step4_verify_results(self) -> bool:
        """Step 4: Verify the complete pipeline results."""
        logger.info("="*50)
        logger.info("STEP 4: Verifying Results")
        logger.info("="*50)
        
        try:
            conn = get_db_connection()
            
            # Check pdf_files table
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM pdf_files WHERE extracted_text IS NOT NULL")
                pdf_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM document_metadata")
                metadata_count = cur.fetchone()[0]
                
                # Check for proper joins
                cur.execute("""
                    SELECT COUNT(*) FROM pdf_files p 
                    INNER JOIN document_metadata dm 
                    ON p.filename = dm.filename AND p.file_hash = dm.file_hash
                """)
                joined_count = cur.fetchone()[0]
                
            conn.close()
            
            logger.info(f"📊 Pipeline Results:")
            logger.info(f"   - PDF files with extracted text: {pdf_count}")
            logger.info(f"   - Documents with metadata: {metadata_count}")
            logger.info(f"   - Successfully joined records: {joined_count}")
            
            if pdf_count > 0 and metadata_count > 0 and joined_count > 0:
                logger.info("✅ Pipeline completed successfully!")
                return True
            else:
                logger.warning("⚠️  Pipeline completed but some data may be missing")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed during verification: {e}")
            return False
            
    def run_complete_pipeline(self, force_reprocess: bool = False) -> bool:
        """Execute the complete RAG pipeline."""
        start_time = datetime.now()
        logger.info("🚀 Starting complete RAG chatbot pipeline")
        logger.info(f"📂 Data directory: {self.data_dir}")
        logger.info(f"🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            logger.error("❌ Prerequisites validation failed")
            return False
            
        steps = [
            ("Database Setup", self.step1_setup_database),
            ("OCR Processing", lambda: self.step2_ocr_processing(force_reprocess)),
            ("Metadata Extraction", self.step3_metadata_extraction),
            ("Results Verification", self.step4_verify_results)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n🔄 Executing: {step_name}")
            
            step_start = datetime.now()
            success = step_func()
            step_duration = datetime.now() - step_start
            
            if success:
                logger.info(f"✅ {step_name} completed in {step_duration}")
            else:
                logger.error(f"❌ {step_name} failed after {step_duration}")
                return False
                
        total_duration = datetime.now() - start_time
        logger.info("="*50)
        logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️  Total execution time: {total_duration}")
        logger.info("="*50)
        
        return True


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete RAG Chatbot Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--data-dir',
        default=DATA_DIR,
        help=f'Directory containing PDF files (default: {DATA_DIR})'
    )
    
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing of all files even if already processed'
    )
    
    parser.add_argument(
        '--step',
        choices=['db', 'ocr', 'metadata', 'verify', 'all'],
        default='all',
        help='Run specific step only (default: all)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(args.data_dir)
    
    success = False
    
    if args.step == 'all':
        success = pipeline.run_complete_pipeline(args.force_reprocess)
    elif args.step == 'db':
        success = pipeline.step1_setup_database()
    elif args.step == 'ocr':
        success = pipeline.step2_ocr_processing(args.force_reprocess)
    elif args.step == 'metadata':
        success = pipeline.step3_metadata_extraction()
    elif args.step == 'verify':
        success = pipeline.step4_verify_results()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
