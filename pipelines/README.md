# RAG Chatbot Pipelines

This directory contains pipeline scripts for the RAG chatbot system that orchestrate the complete data processing workflow.

## Files

### `full_pipeline.py`
**Complete data processing pipeline that executes all steps:**

1. **Database Setup**: Creates required PostgreSQL tables
2. **OCR Processing**: Extracts text from PDF files and stores in database
3. **Metadata Extraction**: Uses LLM to extract structured metadata from texts
4. **Results Verification**: Validates the complete process

#### Usage:
```bash
# Run complete pipeline
python full_pipeline.py

# Use custom data directory
python full_pipeline.py --data-dir /path/to/pdf/files

# Force reprocess all files (clears existing data)
python full_pipeline.py --force-reprocess

# Run specific steps only
python full_pipeline.py --step db        # Database setup only
python full_pipeline.py --step ocr       # OCR processing only
python full_pipeline.py --step metadata  # Metadata extraction only
python full_pipeline.py --step verify    # Verification only
```

### `query_utils.py`
**Database query utilities for exploring processed data:**

#### Usage:
```bash
# List all processed files
python query_utils.py list-files

# List files with extracted metadata
python query_utils.py list-metadata

# Search by circular number
python query_utils.py search-circular "15-2020"

# Search by title keywords
python query_utils.py search-title "financial authority"

# Get complete details for specific file
python query_utils.py file-details "CEO's Circular No. 15-2020"

# Export all data to CSV
python query_utils.py export-csv output.csv
```

## Database Schema

### `pdf_files` table:
- `filename` (TEXT, PRIMARY KEY): PDF filename without extension
- `file_hash` (TEXT): MD5 hash of the file content
- `extracted_text` (TEXT): Full extracted text via OCR
- `extraction_date` (TIMESTAMP): When OCR was performed

### `document_metadata` table:
- `id` (SERIAL, PRIMARY KEY): Auto-increment ID
- `filename` (TEXT): References pdf_files.filename
- `file_hash` (TEXT): References pdf_files.file_hash
- `metadata` (JSONB): Extracted metadata in JSON format
- `extraction_date` (TIMESTAMP): When metadata extraction was performed

**Foreign Key Relationship:**
```sql
CONSTRAINT fk_document_metadata_pdf_files 
    FOREIGN KEY (filename, file_hash) 
    REFERENCES pdf_files(filename, file_hash)
```

## Metadata Structure

The extracted metadata follows this JSON structure:
```json
{
    "filename": "document_name",
    "file_hash": "abc123...",
    "circular_number": "CEO's Circular No. 15-2020",
    "title": "Delegation of Financial Authority",
    "issued_date": "2020-03-15",
    "effective_date": "2020-04-01"
}
```

## Prerequisites

1. **PostgreSQL Database**: Ensure PostgreSQL is running with correct credentials in `config.py`
2. **Environment Variables**: Set required API keys in `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   POSTGRES_DB=rag_chatbot_v3
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   ```
3. **Python Dependencies**: Install required packages:
   ```bash
   pip install psycopg2-binary pytesseract pdf2image groq python-dotenv
   ```

## Separate Component Usage

Each component can also be run separately:

### Database Setup
```bash
python ../database_setup.py
```

### OCR Processing Only
```bash
python ../ocr.py
```

### Metadata Extraction Only
```bash
python ../metadata_extractor.py                    # Process all documents
python ../metadata_extractor.py "document_name"    # Process specific document
```

## Joining Data

To query joined data from both tables:
```sql
SELECT 
    p.filename,
    p.file_hash,
    p.extracted_text,
    dm.metadata->>'circular_number' as circular_number,
    dm.metadata->>'title' as title,
    dm.metadata->>'issued_date' as issued_date
FROM pdf_files p
INNER JOIN document_metadata dm 
    ON p.filename = dm.filename 
    AND p.file_hash = dm.file_hash
WHERE dm.metadata->>'circular_number' IS NOT NULL;
```

## Logging

All pipeline activities are logged to:
- **Console output**: Real-time progress
- **Log file**: `pipeline_YYYYMMDD_HHMMSS.log` in the current directory

## Error Handling

The pipeline includes comprehensive error handling:
- **File validation**: Checks for PDF file existence
- **Database connectivity**: Validates connection before processing
- **OCR failures**: Logs errors and continues with next file
- **Metadata extraction failures**: Falls back to default metadata structure
- **Step-by-step validation**: Each step validates its success before proceeding

## Performance Considerations

- **Incremental processing**: Only processes new/changed files (based on file hash)
- **Database indexing**: Optimized indexes for join operations
- **Chunked processing**: Large texts are processed in manageable chunks
- **Memory management**: Connections are properly closed after use

## Troubleshooting

1. **Database connection errors**: Check PostgreSQL service and credentials
2. **OCR failures**: Ensure Tesseract is installed and PDF files are readable
3. **LLM API errors**: Verify GROQ API key and rate limits
4. **Missing files**: Ensure PDF files exist in the specified data directory
