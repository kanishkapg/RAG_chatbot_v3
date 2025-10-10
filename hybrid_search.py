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

    def _normalize_scores_dict(self, score_dict: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores in a dictionary to 0-1 range."""
        if not score_dict:
            return score_dict
        scores = np.array(list(score_dict.values()))
        min_score, max_score = np.min(scores), np.max(scores)
        if max_score == min_score:
            norm_scores = {k: 0.5 for k in score_dict}
        else:
            norm_scores = {k: (v - min_score) / (max_score - min_score) for k, v in score_dict.items()}
        return norm_scores

    def hybrid_search(
        self, 
        query: str, 
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        HYBRID SEARCH CORE ALGORITHM:
        
        1. Perform SEMANTIC search (meaning-based) using embeddings to get scores for all chunks
        2. Perform LEXICAL search (keyword-based) using TF-IDF to get scores for all chunks
        3. Normalize both sets of scores to 0-1 range
        4. Combine scores for every chunk: final_score = (semantic_weight × semantic_score) + (lexical_weight × lexical_score)
        5. Rank ALL chunks by combined score and return top_k
        
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
        
        # In hybrid_search method:

        # STEP 1: Get semantic scores as dict {chunk_id: score}
        semantic_scores = {}
        query_embedding = self.embedding_model.encode([query])[0]
        for chunk in self.chunk_data:
            if chunk['embedding']:
                chunk_embedding = np.array(chunk['embedding'])
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                semantic_scores[chunk['id']] = similarity
            else:
                semantic_scores[chunk['id']] = 0.0
        print(list(semantic_scores.items())[0:5])  # Debug print to check semantic scores

        # STEP 2: Get lexical scores as dict {chunk_id: score}
        query_vector = self.vectorizer.transform([query])
        lexical_similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        lexical_scores = {}
        for idx, chunk in enumerate(self.chunk_data):
            lexical_scores[chunk['id']] = lexical_similarities[idx]
        print(list(lexical_scores.items())[0:5])

        # STEP 3: Normalize scores (adapt _normalize_scores to handle dict values)
        norm_semantic = self._normalize_scores_dict(semantic_scores)
        norm_lexical = self._normalize_scores_dict(lexical_scores)

        # STEP 4: Combine scores as dict {chunk_id: combined_score}
        combined_scores = {}
        for chunk_id in semantic_scores:  # Assuming same keys
            semantic_score = norm_semantic[chunk_id]
            lexical_score = norm_lexical[chunk_id]
            combined_scores[chunk_id] = (semantic_weight * semantic_score + lexical_weight * lexical_score)

        # STEP 5: Sort by combined score and return top_k
        top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Format final results (update to use chunk_id directly)
        results = []
        chunk_id_to_data = {chunk['id']: chunk for chunk in self.chunk_data}  # One-time map for lookup
        for chunk_id, final_score in top_results:
            chunk = chunk_id_to_data[chunk_id]
            results.append({
                'chunk_id': chunk_id,
                'filename': chunk['filename'],
                'text': chunk['text'],
                'final_score': final_score,
                'semantic_score': norm_semantic[chunk_id],
                'lexical_score': norm_lexical[chunk_id]
            })
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
        query = "How many days are allowed for a mother after giving a child birth"
        results = search_engine.hybrid_search(query, top_k=7)
        
        print(f"\nHybrid Search Results for: '{query}'")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']}")
            print(f"   Combined Score: {result['final_score']:.3f}")
            print(f"   (Semantic: {result['semantic_score']:.3f}, Lexical: {result['lexical_score']:.3f})")
            print(f"   Preview: {result['text'][:150]}...")
    else:
        print("No documents found. Please run embeddings first.")