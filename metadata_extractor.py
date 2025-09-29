import json
import logging
from typing import Dict, List, Tuple
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from database_setup import get_db_connection
from psycopg2.extras import Json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetadataExtractor:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.model_name = GROQ_MODEL

    def get_documents_from_db(self) -> List[Tuple[str, str, str]]:
        """Retrieve all documents with extracted text from the database."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT filename, file_hash, extracted_text 
                    FROM pdf_files 
                    WHERE extracted_text IS NOT NULL AND extracted_text != ''
                """)
                documents = cur.fetchall()
                logger.info(f"Retrieved {len(documents)} documents from database")
                return documents
        except Exception as e:
            logger.error(f"Error retrieving documents from database: {e}")
            return []
        finally:
            conn.close()

    def extract_metadata_from_text(self, text: str, filename: str) -> Dict:
        """Extract metadata from document text using LLM."""
        prompt = f"""
        You are an expert at extracting metadata from official documents and circulars. 
        
        Analyze the following text from {filename} and extract metadata in VALID JSON format. Look carefully for:
        1. Document/circular numbers (often after "Circular No:", "Reference:", "No:", etc.)
        2. Titles or subjects (often after "Subject:", "Re:", "Title:", etc.)  
        3. Dates (issued date, effective date, etc. in YYYY-MM-DD format)
                   
        Return ONLY a valid JSON object with these exact fields:
        {{
            "circular_number": "string or null",
            "title": "string or null", 
            "issued_date": "string or null",
            "effective_date": "string or null"
        }}
        
        Document text:
        {text}
        
        JSON Response:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise metadata extractor. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                max_tokens=800,
                temperature=0.1
            )
            
            metadata_str = response.choices[0].message.content.strip()
            logger.info(f"LLM Response for {filename}: {metadata_str}")
            
            # Extract JSON from response
            json_start = metadata_str.find('{')
            json_end = metadata_str.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = metadata_str[json_start:json_end]
                metadata = json.loads(json_str)
                
                # Clean up metadata values
                for key, value in metadata.items():
                    if isinstance(value, str) and (value.strip() == "" or value.lower() in ["null", "none", "n/a"]):
                        metadata[key] = None
                
                logger.info(f"Successfully extracted metadata for {filename}: {metadata}")
                return metadata
            else:
                raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {filename}: {e}")
            return {
                "circular_number": None,
                "title": None,
                "issued_date": None,
                "effective_date": None
            }

    def store_metadata(self, filename: str, file_hash: str, metadata: Dict) -> None:
        """Store metadata in PostgreSQL."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_metadata (filename, file_hash, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (filename, file_hash) DO UPDATE
                    SET metadata = %s
                """, (filename, file_hash, Json(metadata), Json(metadata)))
            conn.commit()
            logger.info(f"Stored metadata for {filename} in PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to store metadata for {filename}: {e}")
        finally:
            conn.close()

    def process_all_documents(self) -> None:
        """Process all documents in the database to extract and store metadata."""
        documents = self.get_documents_from_db()
        
        if not documents:
            logger.info("No documents found in database")
            return
        
        for filename, file_hash, extracted_text in documents:
            logger.info(f"Processing document: {filename}")
            
            # Extract metadata from text
            metadata = self.extract_metadata_from_text(extracted_text, filename)
            
            # Store metadata in database
            self.store_metadata(filename, file_hash, metadata)
        
        logger.info(f"Completed processing {len(documents)} documents")

    def process_single_document(self, filename: str) -> Dict:
        """Process a single document by filename."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT filename, file_hash, extracted_text 
                    FROM pdf_files 
                    WHERE filename = %s AND extracted_text IS NOT NULL
                """, (filename,))
                result = cur.fetchone()
                
                if not result:
                    logger.error(f"Document {filename} not found or has no extracted text")
                    return {}
                
                filename, file_hash, extracted_text = result
                
                # Extract metadata
                metadata = self.extract_metadata_from_text(extracted_text, filename)
                
                # Store metadata
                self.store_metadata(filename, file_hash, metadata)
                
                return metadata
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            return {}
        finally:
            conn.close()


if __name__ == "__main__":
    # Example usage
    extractor = MetadataExtractor()
    extractor.process_all_documents()