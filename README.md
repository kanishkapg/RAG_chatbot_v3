# RAG Chatbot with Hybrid Search

A simple RAG (Retrieval-Augmented Generation) chatbot that uses hybrid search combining lexical (TF-IDF) and semantic (embeddings) search to find relevant documents and generate responses using Groq's Llama model.

## Features

- **Hybrid Search**: Combines TF-IDF lexical search and semantic embedding search
- **Adjustable Weights**: Configurable weights for semantic vs lexical search (α and β parameters)
- **Simple Architecture**: Clean, minimal code structure with only necessary components
- **Groq Integration**: Uses Llama-3.1-8b-instant for response generation
- **Multiple Interfaces**: Command-line interactive mode and single-query mode

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   POSTGRES_DB=rag_chatbot_v3
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   
   # Optional: Search configuration
   SEMANTIC_WEIGHT=0.6
   LEXICAL_WEIGHT=0.4
   DEFAULT_TOP_K=5
   MAX_TOKENS=500
   TEMPERATURE=0.1
   ```

3. **Database Setup**:
   Make sure your PostgreSQL database is running and execute:
   ```bash
   python database_setup.py
   ```

4. **Process Documents**:
   Make sure you have processed your documents and generated embeddings:
   ```bash
   python embeddings.py
   ```

## Usage

### Interactive Mode (Default)
```bash
python chatbot.py
```

### Single Query Mode
```bash
python chatbot.py --mode single --query "What is the delegation of financial authority?" --sources
```

### Test Search Functionality
```bash
python chatbot.py --mode test
```

### Custom Search Weights
```bash
python chatbot.py --alpha 0.7 --beta 0.3
```

## Architecture

### Core Components

1. **`similarity_search.py`**: 
   - `HybridSearcher`: Main class combining TF-IDF and semantic search
   - Configurable weights (α for semantic, β for lexical)
   - Returns ranked chunks with relevance scores

2. **`response_generator.py`**:
   - `ResponseGenerator`: Handles Groq API integration
   - `ChatBot`: Simple interface combining search and generation
   - Supports responses with or without source citations

3. **`chatbot.py`**:
   - Main CLI interface
   - Interactive and single-query modes
   - Command parsing and error handling

### Search Algorithm

The hybrid search combines two approaches:

1. **Lexical Search (TF-IDF)**: 
   - Good for exact keyword matches
   - Handles specific terms and phrases well

2. **Semantic Search (Embeddings)**:
   - Understands context and meaning
   - Finds conceptually similar content

**Combined Score**: `final_score = α × semantic_score + β × lexical_score`

Where α + β = 1.0 (default: α=0.6, β=0.4)

## API Reference

### HybridSearcher Class
```python
searcher = HybridSearcher(alpha=0.6, beta=0.4)
results = searcher.hybrid_search(query="your question", top_k=5)
```

### ResponseGenerator Class
```python
generator = ResponseGenerator()
response = generator.generate_response(query="your question")
# OR with sources
result = generator.generate_response_with_sources(query="your question")
```

### Simple Search Function
```python
from similarity_search import search_documents
chunks = search_documents(query="your question", top_k=5, alpha=0.6, beta=0.4)
```

## Configuration

All configuration is handled in `config.py`. Key parameters:

- `DEFAULT_SEMANTIC_WEIGHT`: Weight for semantic search (default: 0.6)
- `DEFAULT_LEXICAL_WEIGHT`: Weight for lexical search (default: 0.4)  
- `DEFAULT_TOP_K`: Number of chunks to retrieve (default: 5)
- `DEFAULT_MAX_TOKENS`: Max tokens in LLM response (default: 500)
- `DEFAULT_TEMPERATURE`: LLM temperature (default: 0.1)

## Example Usage

```python
from response_generator import ChatBot

# Initialize chatbot
bot = ChatBot()

# Ask a question
response = bot.ask("What are the financial delegation limits?")
print(response)

# Ask with sources
response_with_sources = bot.ask(
    "What are the financial delegation limits?", 
    include_sources=True
)
print(response_with_sources)
```

## Troubleshooting

1. **No results found**: Ensure documents are processed and embeddings are generated
2. **Groq API errors**: Check your API key and internet connection
3. **Database connection issues**: Verify PostgreSQL is running and credentials are correct
4. **Low relevance scores**: Adjust search weights or check document quality

## Performance Notes

- Initial model loading takes a few seconds
- Embedding generation is done once during setup
- Search is fast (typically < 1 second for small document sets)
- Response generation time depends on Groq API response time

## Extending the System

To add new features:

1. **New search algorithms**: Extend `HybridSearcher` class
2. **Different LLM providers**: Modify `ResponseGenerator` class
3. **Advanced ranking**: Add more sophisticated scoring in hybrid search
4. **Caching**: Add response caching for frequently asked questions
