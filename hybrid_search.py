import json
import logging
import numpy as np
from typing import List, Dict, Tuple
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
        """Initialize the hybrid search engine with embedding model and TF-IDF vectorizer."""
        logger.info("Initializing Hybrid Search Engine...")
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        self.vectorizer = None
        self.document_vectors = None
        self.chunk_data = []
        self._load_documents()
        self._build_tfidf_index()
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

    def _build_tfidf_index(self):
        """Build TF-IDF index for keyword-based search."""
        if not self.chunk_data:
            logger.warning("No documents to index.")
            return
        
        logger.info("Building TF-IDF index...")
        documents = [chunk['text'] for chunk in self.chunk_data]
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000,
            lowercase=True
        )
        
        self.document_vectors = self.vectorizer.fit_transform(documents)
        logger.info("TF-IDF index built successfully.")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        scores = np.array(scores)
        min_score, max_score = np.min(scores), np.max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return ((scores - min_score) / (max_score - min_score)).tolist()

    def _semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform semantic search using embeddings."""
        if not self.chunk_data:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            
            similarities = []
            for i, chunk in enumerate(self.chunk_data):
                if chunk['embedding']:
                    chunk_embedding = np.array(chunk['embedding'])
                    similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                    similarities.append((i, similarity))
            
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def _lexical_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform keyword-based search using TF-IDF."""
        if not self.vectorizer or self.document_vectors is None:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            scored_docs = [(i, similarities[i]) for i in range(len(similarities))]
            return sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Lexical search error: {e}")
            return []

    def hybrid_search(
        self, 
        query: str, 
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        HYBRID SEARCH CORE ALGORITHM:
        
        1. Perform SEMANTIC search (meaning-based) using embeddings
        2. Perform LEXICAL search (keyword-based) using TF-IDF  
        3. Normalize both sets of scores to 0-1 range
        4. Combine scores: final_score = (semantic_weight × semantic_score) + (lexical_weight × lexical_score)
        5. Rank ALL candidates by combined score and return top_k
        
        Args:
            query: Search query string
            semantic_weight: Weight for semantic search (default: 0.7)
            lexical_weight: Weight for lexical search (default: 0.3)
            top_k: Number of final results to return (default: 5)
        """
        if not self.chunk_data:
            return []
        
        # Normalize weights to sum to 1
        total_weight = semantic_weight + lexical_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            lexical_weight /= total_weight
        
        logger.info(f"Hybrid search: '{query[:50]}...' (semantic:{semantic_weight:.1f}, lexical:{lexical_weight:.1f})")
        
        # STEP 1 & 2: Get results from both search methods (top_k*2 to have more candidates)
        semantic_results = self._semantic_search(query, top_k * 2)
        lexical_results = self._lexical_search(query, top_k * 2)
        
        # STEP 3: Normalize scores to 0-1 range
        semantic_scores = self._normalize_scores([score for _, score in semantic_results])
        lexical_scores = self._normalize_scores([score for _, score in lexical_results])
        
        # Convert to dictionaries for easy lookup
        semantic_dict = {semantic_results[i][0]: semantic_scores[i] for i in range(len(semantic_results))}
        lexical_dict = {lexical_results[i][0]: lexical_scores[i] for i in range(len(lexical_results))}
        
        # STEP 4: Combine scores for ALL documents that appeared in either search
        combined_scores = {}
        all_candidates = set(semantic_dict.keys()) | set(lexical_dict.keys())
        
        for doc_idx in all_candidates:
            semantic_score = semantic_dict.get(doc_idx, 0.0)  # 0 if not found in semantic search
            lexical_score = lexical_dict.get(doc_idx, 0.0)    # 0 if not found in lexical search
            
            combined_scores[doc_idx] = (semantic_weight * semantic_score + lexical_weight * lexical_score)
        
        # STEP 5: Sort by combined score and return top_k
        top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format final results
        results = []
        for doc_idx, final_score in top_results:
            chunk = self.chunk_data[doc_idx]
            results.append({
                'chunk_id': chunk['id'],
                'filename': chunk['filename'],
                'text': chunk['text'],
                'final_score': final_score,
                'semantic_score': semantic_dict.get(doc_idx, 0.0),
                'lexical_score': lexical_dict.get(doc_idx, 0.0)
            })
        
        logger.info(f"Returned {len(results)} results from {len(all_candidates)} candidates.")
        return results

    def get_document_list(self) -> List[str]:
        """Get list of available document filenames."""
        filenames = list(set(chunk['filename'] for chunk in self.chunk_data))
        return sorted(filenames)

    def refresh_index(self):
        """Refresh the search index by reloading documents from database."""
        logger.info("Refreshing search index...")
        self.chunk_data = []
        self._load_documents()
        self._build_tfidf_index()
        logger.info("Search index refreshed successfully.")


# Example usage
if __name__ == "__main__":
    search_engine = HybridSearchEngine()
    
    if search_engine.chunk_data:
        query = "what is the latest maternity leave policy"
        results = search_engine.hybrid_search(query, top_k=5)
        
        print(f"\nHybrid Search Results for: '{query}'")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']}")
            print(f"   Combined Score: {result['final_score']:.3f}")
            print(f"   (Semantic: {result['semantic_score']:.3f}, Lexical: {result['lexical_score']:.3f})")
            print(f"   Preview: {result['text'][:150]}...")
    else:
        print("No documents found. Please run embeddings first.")