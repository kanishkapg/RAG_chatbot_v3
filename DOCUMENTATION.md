# RAG Chatbot v3 - Comprehensive Documentation

## Table of Contents
1. [Project Overview and Introduction](#project-overview-and-introduction)
2. [Getting Started (Setup and Installation)](#getting-started-setup-and-installation)
3. [Architecture and Design](#architecture-and-design)
4. [Usage Guide](#usage-guide)
5. [Code Explanation and Structure](#code-explanation-and-structure)
6. [Development and Maintenance](#development-and-maintenance)

---

## Project Overview and Introduction

### Title
**RAG Chatbot v3 with Hybrid Search and Intelligent Document Processing**

### Abstract/Summary
This project implements a sophisticated Retrieval-Augmented Generation (RAG) chatbot system that combines hybrid search capabilities with intelligent document processing. The system processes organizational circulars and policy documents through OCR, extracts structured metadata using Large Language Models (LLMs), and provides accurate responses to user queries through a combination of semantic and lexical search techniques.

The chatbot employs a multi-stage pipeline that includes document ingestion, text extraction, intelligent chunking, embedding generation, metadata extraction, and a hybrid search mechanism that balances semantic understanding with keyword-based retrieval. The system is designed specifically for organizational knowledge management, focusing on policy documents, circulars, and regulatory materials.

### Key Features

#### 🔍 **Hybrid Search Engine**
- **Dual-Mode Retrieval**: Combines TF-IDF lexical search with BGE-M3 semantic embeddings
- **Configurable Weights**: Adjustable α (semantic) and β (lexical) parameters for search optimization
- **Intelligent Scoring**: Normalized scoring system with relevance-based ranking

#### 📄 **Document Processing Pipeline**
- **OCR Integration**: Automatic text extraction from PDF documents using Tesseract
- **Smart Chunking**: Page-aware text segmentation with configurable overlap
- **Metadata Extraction**: LLM-powered extraction of document metadata (circular numbers, titles, dates)
- **Incremental Processing**: File hash-based deduplication and incremental updates

#### 🧠 **AI-Powered Response Generation**
- **Groq LLM Integration**: Utilizes Llama-3.1-8b-instant for response generation
- **Date-Based Reranking**: Prioritizes more recent policy documents
- **Source Attribution**: Comprehensive source citation with document metadata
- **Context-Aware Responses**: Maintains awareness of document hierarchy and supersession

#### 🗄️ **Robust Data Management**
- **PostgreSQL Backend**: Structured data storage with JSONB support for metadata
- **Foreign Key Relationships**: Maintains data integrity across document processing stages
- **Embedding Storage**: Efficient vector storage for semantic search capabilities

#### 🖥️ **Multi-Interface Support**
- **Streamlit Web Interface**: Interactive chat interface with search result visualization
- **Command-Line Interface**: Batch processing and administrative operations
- **API Integration**: Extensible architecture for future API development

#### ⚡ **Performance Optimizations**
- **Caching Mechanisms**: Model loading optimization and embedding caching
- **Batch Processing**: Efficient handling of multiple documents
- **Memory Management**: Optimized resource usage for large document collections

### Technology Stack

#### **Core Technologies**
- **Python 3.8+**: Primary development language
- **PostgreSQL**: Relational database with JSONB support for metadata
- **Streamlit**: Web-based user interface framework

#### **Machine Learning & NLP**
- **Sentence Transformers**: BGE-M3 model for semantic embeddings
- **Scikit-learn**: TF-IDF vectorization and cosine similarity calculations
- **Groq API**: Llama-3.1-8b-instant model for response generation
- **NumPy**: Numerical computations and array operations

#### **Document Processing**
- **Tesseract OCR**: Optical Character Recognition for PDF text extraction
- **PDF2Image**: PDF to image conversion for OCR processing
- **Pytesseract**: Python wrapper for Tesseract OCR engine

#### **Data Management**
- **psycopg2**: PostgreSQL adapter for Python
- **python-dotenv**: Environment variable management
- **Pandas**: Data manipulation and analysis (supporting libraries)

#### **Development Tools**
- **Logging**: Comprehensive logging framework for debugging and monitoring
- **JSON**: Structured data handling for metadata and embeddings
- **Hashlib**: File integrity verification through MD5 hashing

---

## Getting Started (Setup and Installation)

### Prerequisites

#### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.8 or higher
- **PostgreSQL**: Version 12 or higher
- **Tesseract OCR**: Latest stable version
- **Memory**: Minimum 8GB RAM (16GB recommended for large document collections)
- **Storage**: At least 5GB free space for models and data

#### External Dependencies
```bash
# Ubuntu/Debian systems
sudo apt-get update
sudo apt-get install tesseract-ocr postgresql postgresql-contrib

# macOS using Homebrew
brew install tesseract postgresql

# Windows
# Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Install PostgreSQL from: https://www.postgresql.org/download/windows/
```

### Installation Steps

#### 1. Clone and Setup Project
```bash
# Clone the repository
git clone <your-repository-url>
cd RAG_chatbot_v3

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

#### 2. Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import sentence_transformers, streamlit, psycopg2; print('All packages installed successfully')"
```

#### 3. Database Setup
```bash
# Start PostgreSQL service
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# Create database
sudo -u postgres createdb rag_chatbot_v3

# Create database user (optional but recommended)
sudo -u postgres createuser --interactive your_username
```

#### 4. Environment Configuration
```bash
# Create environment file
cp .env.example .env  # If example exists, otherwise create new file

# Edit .env file with your configuration
nano .env
```

**Required Environment Variables:**
```env
# Database Configuration
POSTGRES_DB=rag_chatbot_v3
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# API Keys
GROQ_API_KEY=your_groq_api_key_here

# Search Configuration (Optional)
SEMANTIC_WEIGHT=0.6
LEXICAL_WEIGHT=0.4
DEFAULT_TOP_K=10
MAX_TOKENS=500
TEMPERATURE=0.1
```

#### 5. Initialize Database Schema
```bash
# Create all required tables
python src/database_setup.py
```

### Configuration

#### Database Configuration
The system uses PostgreSQL with the following schema:
- **pdf_files**: Stores original PDF metadata and extracted text
- **document_chunks**: Contains text chunks with embeddings
- **document_metadata**: Holds extracted metadata in JSONB format

#### Search Parameters
Configure hybrid search behavior through environment variables:
- **SEMANTIC_WEIGHT**: Weight for semantic similarity (0.0-1.0)
- **LEXICAL_WEIGHT**: Weight for keyword matching (0.0-1.0)
- **DEFAULT_TOP_K**: Number of results to retrieve (5-20 recommended)

#### Model Configuration
The system automatically downloads required models on first run:
- **BGE-M3**: Semantic embedding model (~2.2GB)
- **Tesseract Language Data**: OCR language models (~500MB)

#### Data Directory Setup
```bash
# Create data directories
mkdir -p data/original  # For actual documents
mkdir -p data/dummy     # For sample/test documents

# Configure data directory in utils/config.py
DATA_DIR = "./data/dummy"  # Change to 'original' for production
```

---

## Architecture and Design

### RAG Pipeline Explanation

The RAG Chatbot v3 implements a sophisticated multi-stage pipeline that transforms raw PDF documents into an intelligent question-answering system. The architecture follows a modular design pattern with clear separation of concerns across different processing stages.

#### **Stage 1: Document Ingestion and OCR Processing**
The pipeline begins with PDF documents stored in the configured data directory. Each document undergoes OCR processing using Tesseract to extract raw text. The system implements intelligent file management through MD5 hashing to detect changes and avoid reprocessing unchanged documents.

```
PDF Documents → OCR Processing → Text Extraction → Database Storage
     ↓              ↓                ↓                ↓
File Hash     Page Detection    Raw Text      pdf_files table
Calculation   & Conversion      Extraction    with metadata
```

#### **Stage 2: Text Chunking and Segmentation**
Extracted text undergoes intelligent chunking that respects document structure. The chunking algorithm identifies page boundaries and creates overlapping text segments to maintain context continuity while ensuring optimal chunk sizes for embedding generation.

#### **Stage 3: Metadata Extraction and Enrichment**
A specialized LLM-powered component analyzes document content to extract structured metadata including circular numbers, titles, effective dates, and classification information. This metadata enables sophisticated document filtering and prioritization.

#### **Stage 4: Embedding Generation and Indexing**
The system generates high-dimensional embeddings using the BGE-M3 model, which provides multilingual support and strong performance on technical documents. Simultaneously, TF-IDF vectors are computed for lexical search capabilities.

#### **Stage 5: Hybrid Search and Response Generation**
User queries trigger a dual-path search mechanism that combines semantic similarity search with keyword-based retrieval. Results undergo intelligent reranking based on document recency and relevance before being processed by the response generation system.

### Block Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Files     │    │  OCR Processing │    │  Text Storage   │
│  (Input Data)   │───▶│   (Tesseract)   │───▶│ (PostgreSQL)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Metadata Extractor│   │  Text Chunking  │    │  Raw Text Data  │
│   (Groq LLM)    │◀───│   (Page-aware)  │◀───│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│ Document        │    │ Text Chunks     │
│ Metadata        │    │ Storage         │
│ (JSONB)         │    │                 │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                      ┌─────────────────┐
                      │ Embedding       │
                      │ Generation      │
                      │ (BGE-M3)        │
                      └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │    │  Hybrid Search  │    │ Search Results  │
│                 │───▶│ Engine          │───▶│ (Top-K Chunks)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │ │                       │
                              │ │                       ▼
                              │ └─────────────┐ ┌─────────────────┐
                              │               ▼ │ Date-based      │
                              │    ┌─────────────────┐ Reranking   │
                              │    │ TF-IDF Search   │────────────┐│
                              │    │ (Lexical)       │            ││
                              │    └─────────────────┘            ││
                              │                                   ││
                              │    ┌─────────────────┐            ││
                              └───▶│ Semantic Search │            ││
                                   │ (BGE-M3)        │            ││
                                   └─────────────────┘            ││
                                                                  ▼│
                                   ┌─────────────────┐    ┌─────────────────┐
                                   │ Response        │    │ Ranked Context  │
                                   │ Generation      │◀───│ Chunks          │
                                   │ (Groq LLM)      │    │                 │
                                   └─────────────────┘    └─────────────────┘
                                           │
                                           ▼
                                   ┌─────────────────┐
                                   │ Final Response  │
                                   │ with Sources    │
                                   └─────────────────┘
```

### Core Components Architecture

#### **1. Hybrid Search Engine (`hybrid_search.py`)**
**Purpose**: Implements the dual-mode search mechanism combining semantic and lexical retrieval.

**Key Classes**:
- `HybridSearchEngine`: Main search orchestrator
- **Methods**:
  - `semantic_search()`: BGE-M3 embedding-based similarity search
  - `lexical_search()`: TF-IDF vectorization and cosine similarity
  - `hybrid_search()`: Score fusion and result ranking
  - `_normalize_scores_dict()`: Score normalization for fair combination

**Algorithm**:
```python
final_score = α × semantic_score + β × lexical_score
where α + β = 1.0 (default: α=0.6, β=0.4)
```

#### **2. Response Generator (`response_generator.py`)**
**Purpose**: Manages LLM interactions and implements intelligent response generation with source attribution.

**Key Classes**:
- `ResponseGenerator`: LLM interface and response orchestrator
- **Methods**:
  - `rerank_by_date()`: Temporal relevance sorting
  - `generate_response()`: Context-aware response generation
  - `process_query()`: Complete query processing pipeline
  - `_get_chunk_metadata()`: Metadata retrieval and enrichment

#### **3. Document Processing Pipeline**
**Components**:
- **OCR Module (`ocr.py`)**: PDF text extraction with incremental processing
- **Chunker (`chunker.py`)**: Page-aware text segmentation with overlap handling
- **Metadata Extractor (`metadata_extractor.py`)**: LLM-powered structured metadata extraction
- **Embeddings Generator (`embeddings.py`)**: BGE-M3 embedding computation and storage

#### **4. Database Layer (`database_setup.py`)**
**Schema Design**:
- **Referential Integrity**: Foreign key constraints maintain data consistency
- **JSONB Metadata**: Flexible structured metadata storage
- **Optimized Indexing**: Performance-optimized indexes for join operations
- **Incremental Updates**: Support for partial data updates and refreshes

### Data Flow Architecture

#### **Ingestion Flow**
```
PDF Input → Hash Calculation → OCR Processing → Text Storage → 
Chunking → Embedding Generation → Index Update → Ready for Search
```

#### **Query Processing Flow**  
```
User Query → Query Analysis → Parallel Search Execution →
[Semantic Path] + [Lexical Path] → Score Fusion → 
Temporal Reranking → Context Assembly → LLM Processing → 
Response with Sources
```

#### **Search Score Fusion**
The hybrid search implements a sophisticated score fusion mechanism:

1. **Independent Scoring**: Semantic and lexical searches execute in parallel
2. **Normalization**: Scores normalized to [0,1] range using min-max scaling
3. **Weighted Combination**: Linear combination using configurable weights
4. **Ranking**: Results sorted by combined score in descending order

### Performance Considerations

#### **Memory Management**
- **Model Caching**: Embedding models loaded once and reused across requests
- **Connection Pooling**: Database connections managed efficiently
- **Batch Processing**: Documents processed in configurable batch sizes

#### **Scalability Features**
- **Incremental Processing**: Only processes new or modified documents
- **Configurable Parameters**: Search weights and result counts adjustable per query
- **Modular Architecture**: Components can be scaled independently

#### **Optimization Strategies**
- **Embedding Caching**: Pre-computed embeddings stored in database
- **Index Optimization**: Database indexes optimized for common query patterns
- **Lazy Loading**: Models and resources loaded on-demand

---

## Usage Guide

### Initial Setup and Document Processing

Before using the chatbot, you need to process your documents through the complete pipeline:

#### **Step 1: Prepare Your Documents**
```bash
# Place your PDF files in the data directory
cp your_documents/*.pdf data/dummy/  # or data/original/

# Verify files are accessible
ls -la data/dummy/
```

#### **Step 2: Run the Complete Processing Pipeline**
```bash
# Process all documents in sequence
python src/ocr.py                    # Extract text from PDFs
python src/chunker.py               # Create text chunks
python src/metadata_extractor.py    # Extract metadata using LLM
python src/embeddings.py           # Generate embeddings
```

**Alternative: Use the automated pipeline (if available):**
```bash
python pipelines/full_pipeline.py
```

### Running the Chatbot

#### **Streamlit Web Interface (Recommended)**
```bash
# Start the web application
streamlit run src/app.py

# Access the interface at: http://localhost:8501
```

**Web Interface Features:**
- 💬 Interactive chat interface with history
- ⚙️ Real-time search parameter adjustment
- 📊 Search results visualization and analysis
- 📄 Document metadata display
- 🔍 Dual search result views (hybrid and date-reranked)

#### **Command-Line Interface**
For batch processing and testing:

```bash
# Direct search testing
python src/hybrid_search.py

# Response generation testing  
python src/response_generator.py
```

### Examples

#### **Example 1: Basic Query Processing**
```bash
# Using the Streamlit interface
# 1. Open http://localhost:8501
# 2. Enter query: "What is the maternity leave policy?"
# 3. Click "Search" to get results with sources
```

**Expected Output:**
- Generated response with policy details
- Source documents with metadata
- Relevance scores and document dates
- Search result analysis in separate tabs

#### **Example 2: Configuring Search Parameters**
```bash
# In the Streamlit sidebar:
# - Semantic Weight: 0.7 (for more contextual search)
# - Lexical Weight: 0.3 (for less keyword matching)
# - Top K Results: 8 (retrieve 8 best chunks)
```

#### **Example 3: Administrative Queries**
```python
from src.hybrid_search import HybridSearchEngine
from src.response_generator import ResponseGenerator

# Initialize components
search_engine = HybridSearchEngine()
response_gen = ResponseGenerator()

# Process a query
query = "Show me financial delegation limits for managers"
results = search_engine.hybrid_search(query, top_k=5)
response = response_gen.process_query(query, results)

print(f"Response: {response['response']}")
print(f"Sources: {len(response['sources'])} documents")
```

### Customization

#### **Adjusting Search Behavior**
Modify `utils/config.py` to change default parameters:

```python
# Semantic vs Lexical Balance
DEFAULT_SEMANTIC_WEIGHT = 0.7  # More semantic understanding
DEFAULT_LEXICAL_WEIGHT = 0.3   # Less keyword matching

# Response Generation
DEFAULT_MAX_TOKENS = 750       # Longer responses
DEFAULT_TEMPERATURE = 0.05     # More deterministic responses
```

#### **Custom Chunking Strategy**
Modify `src/chunker.py` for different document types:

```python
def chunk_text_by_page(text: str, chunk_size: int = 1500, overlap: int = 200):
    # Larger chunks for technical documents
    # More overlap for better context preservation
```

#### **Database Query Customization**
Add custom queries in `src/database_setup.py`:

```python
def get_documents_by_date_range(start_date, end_date):
    """Custom query for date-filtered documents."""
    conn = get_db_connection()
    # Implementation here
```

#### **LLM Prompt Customization**
Modify prompts in `src/response_generator.py`:

```python
prompt = f"""You are an expert assistant specializing in [YOUR_DOMAIN].
Focus on [SPECIFIC_REQUIREMENTS].
Context Documents: {context}
User Question: {query}
Answer:"""
```

### Advanced Usage

#### **Batch Document Processing**
```python
from src.ocr import process_all_pdfs
from src.metadata_extractor import MetadataExtractor

# Process large document collections
extractor = MetadataExtractor()
results = process_all_pdfs("./data/original")
extractor.process_all_documents()
```

#### **Performance Monitoring**
```python
import logging
logging.basicConfig(level=logging.INFO)

# Monitor search performance
search_engine = HybridSearchEngine()
results = search_engine.hybrid_search(query, top_k=10)
# Check logs for timing information
```

#### **Index Management**
```python
# Refresh search index after adding new documents
search_engine.refresh_index()

# Clear and rebuild embeddings
from src.embeddings import process_and_embed_chunks
process_and_embed_chunks()
```

---

## Code Explanation and Structure

### High-Level Structure

The RAG Chatbot v3 follows a modular architecture with clear separation between data processing, search functionality, and user interfaces. The codebase is organized into logical components that can be developed, tested, and maintained independently.

```
RAG_chatbot_v3/
├── src/                    # Core application modules
│   ├── app.py             # Streamlit web interface
│   ├── hybrid_search.py   # Hybrid search engine
│   ├── response_generator.py # LLM response generation
│   ├── database_setup.py  # Database schema and connections
│   ├── ocr.py            # PDF text extraction
│   ├── chunker.py        # Text segmentation
│   ├── metadata_extractor.py # LLM-powered metadata extraction
│   └── embeddings.py     # Embedding generation
├── utils/                  # Configuration and utilities
│   └── config.py         # Environment and system configuration
├── pipelines/             # Automated processing workflows
│   └── full_pipeline.py  # Complete document processing pipeline
├── data/                  # Document storage
│   ├── dummy/            # Sample documents
│   └── original/         # Production documents
└── requirements.txt       # Python dependencies
```

### Core Functions and Components

#### **1. Hybrid Search Engine (`src/hybrid_search.py`)**

**Primary Class: `HybridSearchEngine`**
```python
class HybridSearchEngine:
    def __init__(self):
        """Initialize search engine with BGE-M3 model and TF-IDF vectorizer."""
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        self.vectorizer = TfidfVectorizer(...)
        self._load_documents()  # Load chunks from database
        self._build_tfidf_index()  # Build keyword search index
```

**Key Methods:**

**`semantic_search(query: str) -> Dict[int, float]`**
- Generates query embedding using BGE-M3
- Computes cosine similarity against all document embeddings  
- Returns mapping of chunk_id to similarity score

```python
def semantic_search(self, query: str) -> Dict[int, float]:
    query_embedding = self.embedding_model.encode([query])[0]
    semantic_scores = {}
    for chunk in self.chunk_data:
        if chunk['embedding']:
            similarity = cosine_similarity([query_embedding], [chunk['embedding']])[0][0]
            semantic_scores[chunk['id']] = similarity
    return semantic_scores
```

**`lexical_search(query: str) -> Dict[int, float]`**
- Transforms query using pre-built TF-IDF vectorizer
- Computes cosine similarity against document TF-IDF matrix
- Returns keyword-based relevance scores

**`hybrid_search(query, semantic_weight, lexical_weight, top_k) -> List[Dict]`**
- Combines semantic and lexical scores using weighted fusion
- Normalizes scores to ensure fair combination
- Returns ranked list of most relevant chunks

#### **2. Response Generator (`src/response_generator.py`)**

**Primary Class: `ResponseGenerator`**
```python
class ResponseGenerator:
    def __init__(self):
        """Initialize with Groq LLM client."""
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.model_name = "llama-3.1-8b-instant"
```

**Key Methods:**

**`rerank_by_date(chunks: List[Dict]) -> List[Dict]`**
- Enriches search results with document metadata
- Sorts by effective_date or issued_date in descending order
- Prioritizes recent policy documents over older versions

```python
def rerank_by_date(self, chunks: List[Dict]) -> List[Dict]:
    enriched_chunks = []
    for chunk in chunks:
        metadata = self._get_chunk_metadata(chunk['chunk_id'])
        ranking_date = metadata.get('effective_date') or metadata.get('issued_date')
        enriched_chunk = chunk.copy()
        enriched_chunk['metadata'] = metadata
        enriched_chunk['ranking_date'] = ranking_date
        enriched_chunks.append(enriched_chunk)
    
    # Sort by date (most recent first)
    return sorted(enriched_chunks, 
                 key=lambda x: x['ranking_date'] if x['ranking_date'] else '0000-00-00', 
                 reverse=True)
```

**`generate_response(query: str, top_chunks: List[Dict]) -> Dict`**
- Constructs context from top-ranked chunks and their metadata
- Generates LLM prompt with document hierarchy awareness
- Returns response with source attribution

#### **3. Document Processing Pipeline**

**OCR Module (`src/ocr.py`)**
```python
def extract_text_from_pdf(pdf_path: str):
    """Extract text using Tesseract OCR with incremental processing."""
    # Compute file hash for change detection
    file_hash = compute_file_hash(pdf_path)
    
    # Skip if already processed with same hash
    if already_processed(pdf_filename, file_hash):
        return [], file_hash
    
    # Convert PDF to images and extract text
    images = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    for i, image in enumerate(images):
        text = f"Page {i + 1}\n{pytesseract.image_to_string(image, lang='eng')}"
        full_text += text + "\n\n"
    
    # Store in database
    store_in_database(pdf_filename, file_hash, full_text)
    return full_text, file_hash
```

**Text Chunker (`src/chunker.py`)**
```python
def chunk_text_by_page(text: str, chunk_size: int = 1000, overlap: int = 150):
    """Page-aware chunking with overlap for context preservation."""
    chunks = []
    pages = re.split(r'Page \d+\n', text)  # Respect page boundaries
    
    for i, page_content in enumerate(pages):
        if len(page_content) <= chunk_size:
            chunks.append({'page_number': i + 1, 'chunk_text': page_content})
        else:
            # Split large pages with overlap
            start = 0
            while start < len(page_content):
                end = start + chunk_size
                chunk_text = page_content[start:end]
                chunks.append({'page_number': i + 1, 'chunk_text': chunk_text})
                start += chunk_size - overlap
    return chunks
```

**Metadata Extractor (`src/metadata_extractor.py`)**
```python
def extract_metadata_from_text(self, text: str, filename: str) -> Dict:
    """LLM-powered metadata extraction with structured output."""
    prompt = f"""
    Analyze this document and extract metadata in JSON format:
    {{
        "circular_number": "string or null",
        "title": "string or null", 
        "issued_date": "YYYY-MM-DD or null",
        "effective_date": "YYYY-MM-DD or null"
    }}
    Document: {text}
    """
    
    response = self.groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=self.model_name,
        temperature=0.1  # Low temperature for consistent extraction
    )
    
    # Parse JSON response and validate
    metadata = json.loads(response.choices[0].message.content)
    return self._clean_metadata(metadata)
```

#### **4. Database Layer (`src/database_setup.py`)**

**Database Schema Design:**
```python
def create_pdf_files_table():
    """Primary table for document storage."""
    schema = """
    CREATE TABLE IF NOT EXISTS pdf_files (
        filename TEXT PRIMARY KEY,
        file_hash TEXT NOT NULL,
        extracted_text TEXT,
        extraction_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

def create_document_chunks_table():
    """Chunked text with embeddings."""
    schema = """
    CREATE TABLE IF NOT EXISTS document_chunks (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding JSONB,  -- Vector storage for semantic search
        FOREIGN KEY (filename, file_hash) REFERENCES pdf_files(filename, file_hash)
    );
    """

def create_document_metadata_table():
    """Extracted metadata in JSONB format."""
    schema = """
    CREATE TABLE IF NOT EXISTS document_metadata (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        metadata JSONB,  -- Flexible metadata storage
        FOREIGN KEY (filename, file_hash) REFERENCES pdf_files(filename, file_hash)
    );
    """
```

### Focus on Core Functions

#### **Search Score Fusion Algorithm**
```python
def hybrid_search(self, query, semantic_weight=0.6, lexical_weight=0.4, top_k=10):
    # Get individual search results
    semantic_scores = self.semantic_search(query)
    lexical_scores = self.lexical_search(query)
    
    # Normalize scores to [0,1] range
    norm_semantic = self._normalize_scores_dict(semantic_scores)
    norm_lexical = self._normalize_scores_dict(lexical_scores)
    
    # Weighted combination
    combined_scores = {}
    for chunk_id in semantic_scores:
        combined_scores[chunk_id] = (
            semantic_weight * norm_semantic[chunk_id] + 
            lexical_weight * norm_lexical[chunk_id]
        )
    
    # Return top-k results with metadata
    return self._format_results(combined_scores, top_k)
```

#### **Intelligent Context Assembly**
```python
def generate_response(self, query: str, top_chunks: List[Dict]) -> Dict:
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        metadata = chunk.get('metadata', {})
        context_parts.append(f"""
        Document {i}:
        Source: {chunk['filename']}
        Circular: {metadata.get('circular_number', 'Unknown')}
        Date: {metadata.get('effective_date', 'Not specified')}
        Content: {chunk['text']}
        """)
    
    # Construct hierarchical prompt
    prompt = f"""
    You are an expert assistant. PRIORITIZE recent documents.
    Context: {'\n'.join(context_parts)}
    Question: {query}
    Answer with source citations:
    """
    
    return self._generate_llm_response(prompt)
```

#### **Streamlit Interface Integration**
The web interface (`src/app.py`) demonstrates advanced Streamlit patterns:

```python
# Session state management for search engine persistence
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = HybridSearchEngine()
    st.session_state.response_generator = ResponseGenerator()

# Real-time parameter adjustment
semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.6, 0.1)
lexical_weight = st.slider("Lexical Weight", 0.0, 1.0, 0.4, 0.1)

# Dual-tab result visualization
tab1, tab2 = st.tabs(["Hybrid Results", "Date-Reranked Results"])
with tab1:
    display_search_results(hybrid_results, "hybrid")
with tab2:
    display_search_results(reranked_results, "reranked")
```

### Integration Patterns

#### **Error Handling and Logging**
```python
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_database_operation():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Database operations
            cur.execute("SELECT ...")
        conn.commit()
        logger.info("Operation completed successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Database operation failed: {e}")
        raise
    finally:
        conn.close()
```

#### **Configuration Management**
```python
# Centralized configuration in utils/config.py
from dotenv import load_dotenv
load_dotenv()

POSTGRES_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "rag_chatbot_v3"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    # ... other config
}

# Environment-based feature flags
DEFAULT_SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
```

### Performance Optimization Techniques

#### **Model Loading and Caching**
```python
class HybridSearchEngine:
    def __init__(self):
        # Load model once and reuse
        logger.info("Loading BGE-M3 model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        
        # Build index on initialization
        self._build_tfidf_index()
        logger.info("Search engine ready")

    def refresh_index(self):
        """Incremental index updates without full reload."""
        self.chunk_data = []
        self._load_documents()  # Only reload data
        self._build_tfidf_index()  # Rebuild search index
```

#### **Database Connection Management**
```python
def get_db_connection():
    """Connection factory with error handling."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

# Connection context manager pattern
with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT ...")
    # Auto-commit and cleanup
```

---

## Development and Maintenance

### Known Issues

#### **Current Limitations**

**1. OCR Quality Dependencies**
- **Issue**: OCR accuracy varies significantly with document quality, scan resolution, and formatting
- **Impact**: Poor-quality PDFs may result in incomplete or inaccurate text extraction
- **Workaround**: Pre-process documents at 300+ DPI, ensure good contrast and minimal skew
- **Status**: Monitoring - investigating advanced OCR solutions like AWS Textract or Azure Document Intelligence

**2. Memory Usage with Large Document Collections**
- **Issue**: BGE-M3 model and embeddings consume significant memory (2-4GB+ for large datasets)
- **Impact**: Performance degradation with 1000+ documents or limited RAM systems
- **Mitigation**: Implement batch processing, consider embedding dimensionality reduction
- **Priority**: Medium - affects scalability for enterprise deployments

**3. LLM Response Latency**
- **Issue**: Groq API calls introduce 2-5 second latency for response generation
- **Impact**: User experience affected during peak usage or network issues
- **Temporary Solution**: Implement response caching for common queries
- **Future Plan**: Investigate local LLM deployment or streaming responses

#### **Technical Debt**

**Database Performance Optimization**
- **Current**: Sequential database queries for metadata enrichment
- **Impact**: Slower response times with large result sets
- **Solution**: Implement JOIN-based queries and connection pooling
- **Estimated Effort**: 2-3 days development

**Configuration Management**
- **Current**: Mixed environment variable and hardcoded configuration
- **Impact**: Difficult deployment across different environments
- **Solution**: Centralized configuration management with validation
- **Priority**: Low - functional but not optimal

**Error Recovery Mechanisms**
- **Current**: Basic exception handling without retry logic
- **Impact**: Pipeline failures require manual intervention
- **Solution**: Implement exponential backoff and circuit breaker patterns
- **Status**: Planned for next release

### Future Enhancements

#### **Short-term Roadmap (3-6 months)**

**1. Advanced Search Capabilities**
```python
# Planned: Query expansion and intent detection
class QueryProcessor:
    def expand_query(self, query: str) -> List[str]:
        """Generate semantically related query variations."""
        # Implementation: Use LLM to generate query paraphrases
        # Expected improvement: 15-20% better recall
        
    def detect_intent(self, query: str) -> str:
        """Classify query type for specialized handling."""
        # Categories: factual, policy, procedural, comparative
        # Enable intent-specific response templates
```

**2. Multi-Modal Document Support**
- **Scope**: Support for Word documents, Excel files, PowerPoint presentations
- **Implementation**: Add docx, xlsx parsing with format preservation
- **Timeline**: Q2 2026
- **Dependencies**: python-docx, openpyxl integration

**3. Advanced Metadata Extraction**
- **Current**: Basic circular information extraction
- **Enhancement**: Entity recognition, relationship mapping, policy hierarchy detection
- **Benefit**: Better document understanding and cross-referencing
- **Technical Approach**: Fine-tune named entity recognition models

#### **Medium-term Roadmap (6-12 months)**

**1. Real-time Document Processing**
```python
# Planned: File system monitoring and auto-processing
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentWatcher(FileSystemEventHandler):
    def on_created(self, event):
        """Auto-process new documents as they're added."""
        if event.src_path.endswith('.pdf'):
            self.process_document_async(event.src_path)
```

**2. Advanced Analytics and Monitoring**
- **Search Analytics**: Query patterns, result relevance tracking, user satisfaction metrics
- **Performance Monitoring**: Response times, error rates, resource utilization
- **Business Intelligence**: Document usage patterns, policy gap analysis

**3. Multi-tenant Architecture**
```python
# Planned: Organization-based data isolation
class TenantManager:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.db_schema = f"tenant_{tenant_id}"
        
    def get_tenant_documents(self):
        """Isolate documents by organization."""
        # Implementation: Schema-based or table-prefix isolation
```

#### **Long-term Vision (12+ months)**

**1. Conversational Memory and Context**
- **Feature**: Multi-turn conversations with context preservation
- **Implementation**: Conversation history storage and context injection
- **User Experience**: Follow-up questions, clarifications, progressive refinement

**2. Advanced AI Capabilities**
```python
# Vision: Intelligent document analysis and insights
class DocumentIntelligence:
    def detect_policy_conflicts(self, documents: List[str]) -> List[Dict]:
        """Identify contradictory policies across documents."""
        
    def suggest_policy_updates(self, context: str) -> List[str]:
        """Recommend policy changes based on trends and gaps."""
        
    def generate_compliance_reports(self) -> Dict:
        """Automated compliance gap analysis and reporting."""
```

**3. Integration Ecosystem**
- **API Gateway**: RESTful API for external system integration
- **Webhook Support**: Real-time notifications for document updates
- **SSO Integration**: SAML, OAuth2 support for enterprise authentication
- **Audit Trail**: Comprehensive logging for compliance and security

### Development Guidelines

#### **Code Quality Standards**

**Testing Strategy**
```python
# Required: Comprehensive test coverage
class TestHybridSearch(unittest.TestCase):
    def test_semantic_search_accuracy(self):
        """Verify semantic search returns relevant results."""
        
    def test_score_normalization(self):
        """Ensure score fusion produces valid ranges."""
        
    def test_database_integration(self):
        """Validate database operations and transactions."""

# Target: 85%+ code coverage for core modules
# Tools: pytest, coverage.py, mock for external services
```

**Documentation Requirements**
- **Docstrings**: Google-style docstrings for all public methods
- **Type Hints**: Comprehensive type annotations for better IDE support
- **Architecture Decisions**: ADR documents for major technical choices
- **API Documentation**: Auto-generated docs using Sphinx or similar tools

**Performance Benchmarks**
```python
# Performance targets and monitoring
PERFORMANCE_TARGETS = {
    "search_response_time": "< 500ms",  # 95th percentile
    "document_processing": "< 30s per PDF",  # OCR + embedding
    "memory_usage": "< 4GB for 1000 documents",
    "concurrent_users": "50+ simultaneous queries"
}
```

#### **Deployment and DevOps**

**Container Strategy**
```dockerfile
# Multi-stage Docker build for production deployment
FROM python:3.9-slim as base
RUN apt-get update && apt-get install -y tesseract-ocr postgresql-client

FROM base as development
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

FROM base as production
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
```

**Monitoring and Observability**
```python
# Planned: Comprehensive monitoring stack
import prometheus_client
from opentelemetry import trace, metrics

class RAGMetrics:
    def __init__(self):
        self.query_counter = prometheus_client.Counter('rag_queries_total')
        self.response_time = prometheus_client.Histogram('rag_response_seconds')
        self.embedding_cache_hits = prometheus_client.Counter('embedding_cache_hits')
```

**CI/CD Pipeline**
```yaml
# Planned: GitHub Actions workflow
name: RAG Chatbot CI/CD
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
      - name: Run tests
        run: pytest --cov=src/
      - name: Check code quality
        run: flake8 src/ && black --check src/
```

### Maintenance Procedures

#### **Regular Maintenance Tasks**

**Weekly Tasks**
```bash
#!/bin/bash
# maintenance_weekly.sh

# 1. Database maintenance
psql -c "VACUUM ANALYZE document_chunks, pdf_files, document_metadata;"

# 2. Log rotation and cleanup
find logs/ -name "*.log" -mtime +7 -delete

# 3. Embedding index optimization
python scripts/optimize_embeddings.py

# 4. Performance metrics collection
python scripts/collect_metrics.py --period week
```

**Monthly Tasks**
- **Model Updates**: Check for new versions of BGE-M3 or alternative embedding models
- **Dependency Updates**: Update Python packages, security patches
- **Data Backup**: Full database backup with integrity verification
- **Performance Analysis**: Review query patterns, identify optimization opportunities

**Quarterly Tasks**
- **Security Audit**: Dependency vulnerability scanning, access control review
- **Capacity Planning**: Resource usage analysis, scaling requirements assessment
- **User Feedback Integration**: Feature request prioritization, usability improvements
- **Documentation Updates**: Keep technical documentation synchronized with code changes

#### **Troubleshooting Guide**

**Common Issues and Solutions**

**Search Returns No Results**
```python
# Diagnostic steps
def diagnose_search_issue(query: str):
    # 1. Check if documents are loaded
    search_engine = HybridSearchEngine()
    print(f"Loaded documents: {len(search_engine.chunk_data)}")
    
    # 2. Test individual search components
    semantic_scores = search_engine.semantic_search(query)
    lexical_scores = search_engine.lexical_search(query)
    
    # 3. Verify database connectivity
    # 4. Check embedding generation
```

**Database Connection Errors**
```bash
# Connection troubleshooting
# 1. Check PostgreSQL service status
sudo systemctl status postgresql

# 2. Verify database existence
psql -l | grep rag_chatbot

# 3. Test connection with environment variables
python -c "from src.database_setup import get_db_connection; get_db_connection().close(); print('Connection successful')"
```

**High Memory Usage**
```python
# Memory optimization strategies
import gc
from functools import lru_cache

class OptimizedSearchEngine:
    @lru_cache(maxsize=100)
    def get_embeddings_cached(self, text_hash: str):
        """Cache frequently accessed embeddings."""
        
    def cleanup_resources(self):
        """Force garbage collection and clear caches."""
        gc.collect()
        self.get_embeddings_cached.cache_clear()
```

#### **Backup and Recovery**

**Database Backup Strategy**
```bash
#!/bin/bash
# backup_database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/rag_chatbot"
DB_NAME="rag_chatbot_v3"

# Full database backup
pg_dump -h localhost -U postgres -d $DB_NAME | gzip > "$BACKUP_DIR/full_backup_$DATE.sql.gz"

# Schema-only backup for development
pg_dump -h localhost -U postgres -d $DB_NAME --schema-only > "$BACKUP_DIR/schema_$DATE.sql"

# Verify backup integrity
gunzip -t "$BACKUP_DIR/full_backup_$DATE.sql.gz" && echo "Backup verified successfully"
```

**Disaster Recovery Procedures**
```bash
# Recovery from backup
#!/bin/bash
# restore_database.sh

BACKUP_FILE="/backups/rag_chatbot/full_backup_20251013_120000.sql.gz"
DB_NAME="rag_chatbot_v3"

# 1. Create fresh database
dropdb $DB_NAME
createdb $DB_NAME

# 2. Restore from backup
gunzip -c $BACKUP_FILE | psql -h localhost -U postgres -d $DB_NAME

# 3. Verify restoration
python -c "from src.database_setup import get_db_connection; print('Recovery successful')"
```

### Contributing Guidelines

#### **Development Workflow**

**Branch Strategy**
```bash
# Feature development workflow
git checkout -b feature/new-search-algorithm
# Development work
git commit -am "Implement improved semantic search"
git push origin feature/new-search-algorithm
# Create pull request for code review
```

**Code Review Checklist**
- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have appropriate docstrings and type hints
- [ ] Unit tests cover new functionality (minimum 80% coverage)
- [ ] Performance impact assessed for search-critical components
- [ ] Database migrations are backward compatible
- [ ] Security implications reviewed (especially for LLM interactions)

**Pull Request Template**
```markdown
## Description
Brief description of changes and motivation

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Performance Impact
- Search latency: [No change/Improved by X%/Degraded by X%]
- Memory usage: [No change/Details]
- Database queries: [New queries added/Optimized existing]
```

#### **Release Management**

**Version Strategy**
- **Major Version (X.0.0)**: Breaking API changes, major architectural updates
- **Minor Version (X.Y.0)**: New features, backward-compatible changes
- **Patch Version (X.Y.Z)**: Bug fixes, security updates

**Release Process**
1. **Feature Freeze**: Code complete, final testing phase
2. **Release Candidate**: Deploy to staging environment, user acceptance testing
3. **Production Release**: Tag version, update documentation, deploy to production
4. **Post-Release**: Monitor metrics, address any critical issues

This comprehensive Development and Maintenance section provides a roadmap for the project's evolution while ensuring sustainable development practices and robust operational procedures.
````
```
