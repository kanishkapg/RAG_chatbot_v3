import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database_setup import get_db_connection
from config import DEFAULT_SEMANTIC_WEIGHT, DEFAULT_LEXICAL_WEIGHT, DEFAULT_TOP_K

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSearchEngine:
    def __init__(self):
        """Initialize the hybrid search engine with embedding model and BM25 vectorizer."""
        logger.info("Initializing Hybrid Search Engine...")
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        self.vectorizer = None
        self.document_vectors = None
        self.chunk_data = []
        self._load_documents()
        self._build_bm25_index()
        logger.info("Hybrid Search Engine initialized successfully.")

    def _load_documents(self):
        """Load all document chunks and their embeddings from the database."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, filename, chunk_text, embedding 
                    FROM document_chunks 
                    WHERE embedding IS NOT NULL
                    ORDER BY filename, chunk_index
                """)
                results = cur.fetchall()
                
                for chunk_id, filename, chunk_text, embedding_data in results:
                    # Handle both JSON string and list formats for embeddings
                    if embedding_data:
                        if isinstance(embedding_data, str):
                            # If it's a JSON string, parse it
                            embedding = json.loads(embedding_data)
                        elif isinstance(embedding_data, list):
                            # If it's already a list, use it directly
                            embedding = embedding_data
                        else:
                            # Handle other formats (like dict from JSONB)
                            embedding = list(embedding_data) if embedding_data else None
                    else:
                        embedding = None
                    
                    self.chunk_data.append({
                        'id': chunk_id,
                        'filename': filename,
                        'text': chunk_text,
                        'embedding': embedding
                    })
                
                logger.info(f"Loaded {len(self.chunk_data)} chunks from database.")
        except Exception as e:
            logger.error(f"Error loading documents from database: {e}")
            self.chunk_data = []
        finally:
            conn.close()

    def _build_bm25_index(self):
        """Build TF-IDF index for BM25-like lexical search."""
        if not self.chunk_data:
            logger.warning("No documents to index for BM25.")
            return
        
        logger.info("Building BM25 index...")
        documents = [chunk['text'] for chunk in self.chunk_data]
        
        # Use TF-IDF as a proxy for BM25
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000,
            lowercase=True
        )
        
        self.document_vectors = self.vectorizer.fit_transform(documents)
        logger.info("BM25 index built successfully.")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if not scores or len(scores) == 0:
            return scores
        
        scores = np.array(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()

    def _semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform semantic search using embeddings and cosine similarity."""
        if not self.chunk_data:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Calculate cosine similarities
            similarities = []
            for i, chunk in enumerate(self.chunk_data):
                if chunk['embedding']:
                    chunk_embedding = np.array(chunk['embedding'])
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [chunk_embedding]
                    )[0][0]
                    similarities.append((i, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _lexical_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform lexical search using BM25-like TF-IDF scoring."""
        if not self.vectorizer or self.document_vectors is None:
            return []
        
        try:
            # Transform query using the fitted vectorizer
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarities (TF-IDF approximation of BM25)
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Get indices and scores
            scored_docs = [(i, similarities[i]) for i in range(len(similarities))]
            
            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in lexical search: {e}")
            return []

    def hybrid_search(
        self, 
        query: str, 
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and lexical search results.
        
        Args:
            query: Search query string
            semantic_weight: Weight for semantic search (alpha)
            lexical_weight: Weight for lexical search (beta)
            top_k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        if not self.chunk_data:
            logger.warning("No documents available for search.")
            return []
        
        # Normalize weights
        total_weight = semantic_weight + lexical_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            lexical_weight /= total_weight
        
        logger.info(f"Performing hybrid search for query: '{query[:50]}...'")
        logger.info(f"Weights - Semantic: {semantic_weight:.2f}, Lexical: {lexical_weight:.2f}")
        
        # Perform both types of searches
        semantic_results = self._semantic_search(query, top_k * 2)
        lexical_results = self._lexical_search(query, top_k * 2)
        
        # Normalize scores
        semantic_scores = [score for _, score in semantic_results]
        lexical_scores = [score for _, score in lexical_results]
        
        normalized_semantic = self._normalize_scores(semantic_scores)
        normalized_lexical = self._normalize_scores(lexical_scores)
        
        # Create score dictionaries for fusion
        semantic_dict = {
            semantic_results[i][0]: normalized_semantic[i] 
            for i in range(len(semantic_results))
        }
        lexical_dict = {
            lexical_results[i][0]: normalized_lexical[i] 
            for i in range(len(lexical_results))
        }
        
        # Combine scores using weighted fusion
        combined_scores = {}
        all_doc_indices = set(semantic_dict.keys()) | set(lexical_dict.keys())
        
        for doc_idx in all_doc_indices:
            semantic_score = semantic_dict.get(doc_idx, 0.0)
            lexical_score = lexical_dict.get(doc_idx, 0.0)
            
            combined_score = (
                semantic_weight * semantic_score + 
                lexical_weight * lexical_score
            )
            combined_scores[doc_idx] = combined_score
        
        # Sort by combined score and get top_k results
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Format results with metadata
        results = []
        for doc_idx, final_score in sorted_results:
            chunk = self.chunk_data[doc_idx]
            result = {
                'chunk_id': chunk['id'],
                'filename': chunk['filename'],
                'text': chunk['text'],
                'final_score': final_score,
                'semantic_score': semantic_dict.get(doc_idx, 0.0),
                'lexical_score': lexical_dict.get(doc_idx, 0.0)
            }
            results.append(result)
        
        logger.info(f"Hybrid search completed. Returned {len(results)} results.")
        return results

    def search_by_filename(
        self, 
        query: str, 
        filename: str, 
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        Perform hybrid search within a specific document.
        
        Args:
            query: Search query string
            filename: Name of the document to search within
            semantic_weight: Weight for semantic search
            lexical_weight: Weight for lexical search
            top_k: Number of results to return
            
        Returns:
            List of search results from the specified document
        """
        # Filter results by filename
        all_results = self.hybrid_search(query, semantic_weight, lexical_weight, top_k * 3)
        filtered_results = [
            result for result in all_results 
            if result['filename'] == filename
        ]
        
        return filtered_results[:top_k]

    def get_document_list(self) -> List[str]:
        """Get list of available document filenames."""
        filenames = list(set(chunk['filename'] for chunk in self.chunk_data))
        return sorted(filenames)

    def refresh_index(self):
        """Refresh the search index by reloading documents from database."""
        logger.info("Refreshing search index...")
        self.chunk_data = []
        self._load_documents()
        self._build_bm25_index()
        logger.info("Search index refreshed successfully.")


# Example usage and testing
if __name__ == "__main__":
    # Initialize search engine
    search_engine = HybridSearchEngine()
    
    # Example search
    if search_engine.chunk_data:
        query = "what is the latest maternity leave policy"
        results = search_engine.hybrid_search(query, top_k=5)
        
        print(f"\nSearch Results for: '{query}'")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document: {result['filename']}")
            print(f"   Final Score: {result['final_score']:.4f}")
            print(f"   Semantic: {result['semantic_score']:.4f}, Lexical: {result['lexical_score']:.4f}")
            print(f"   Text Preview: {result['text'][:200]}...")
    else:
        print("No documents available for search. Please ensure documents are embedded first.")