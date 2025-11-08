import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database_setup import get_db_connection
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import (
    DEFAULT_SEMANTIC_WEIGHT, DEFAULT_LEXICAL_WEIGHT, DEFAULT_RECENCY_WEIGHT, 
    DEFAULT_TOP_K, DEFAULT_DECAY_BASE, DEFAULT_HALF_LIFE_DAYS
)

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
                    SELECT dc.id, dc.filename, dc.chunk_text, dc.embedding, dm.metadata
                    FROM document_chunks dc
                    LEFT JOIN document_metadata dm ON dc.filename = dm.filename AND dc.file_hash = dm.file_hash
                    WHERE dc.embedding IS NOT NULL
                    ORDER BY dc.filename, dc.chunk_index
                """)
                results = cur.fetchall()
                
                for chunk_id, filename, chunk_text, embedding_data, metadata in results:
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
                    
                    # Extract effective_date from metadata
                    effective_date = None
                    if metadata and isinstance(metadata, dict):
                        effective_date_str = metadata.get('effective_date')
                        if effective_date_str:
                            try:
                                from datetime import datetime
                                effective_date = datetime.strptime(effective_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                            except ValueError:
                                logger.warning(f"Invalid effective_date format for {filename}: {effective_date_str}")
                    
                    self.chunk_data.append({
                        'id': chunk_id,
                        'filename': filename,
                        'text': chunk_text,
                        'embedding': embedding,
                        'effective_date': effective_date
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

    def semantic_search(self, query: str) -> Dict[int, float]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary mapping chunk_id to similarity score
        """
        semantic_scores = {}
        query_embedding = self.embedding_model.encode([query])[0]
        
        for chunk in self.chunk_data:
            if chunk['embedding']:
                chunk_embedding = np.array(chunk['embedding'])
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                semantic_scores[chunk['id']] = similarity
            else:
                semantic_scores[chunk['id']] = 0.0

        return semantic_scores

    def lexical_search(self, query: str) -> Dict[int, float]:
        """
        Perform lexical search using TF-IDF.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary mapping chunk_id to similarity score
        """
        query_vector = self.vectorizer.transform([query])
        lexical_similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        lexical_scores = {}
        for idx, chunk in enumerate(self.chunk_data):
            lexical_scores[chunk['id']] = lexical_similarities[idx]

        return lexical_scores

    def recency_search(
        self, 
        decay_base: float = DEFAULT_DECAY_BASE, 
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    ) -> Dict[int, float]:
        """
        Calculate recency scores based on document effective date.
        
        Args:
            decay_base: Base for exponential decay (default: 0.5)
            half_life_days: Half-life in days (default: 200)
            
        Returns:
            Dictionary mapping chunk_id to recency score
        """
        current_time = datetime.now(timezone.utc)
        recency_scores = {}
        
        for chunk in self.chunk_data:
            if chunk['effective_date']:
                age_days = (current_time - chunk['effective_date']).total_seconds() / (24 * 3600)
                recency_score = decay_base ** (age_days / half_life_days)
                recency_scores[chunk['id']] = recency_score
            else:
                recency_scores[chunk['id']] = 0.0
        
        return recency_scores

    def hybrid_search(
        self, 
        query: str, 
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        recency_weight: float = DEFAULT_RECENCY_WEIGHT,
        decay_base: float = DEFAULT_DECAY_BASE,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        Combine semantic, lexical, and recency search results.
        
        Args:
            query: Search query string
            semantic_weight: Weight for semantic search (default: 0.6)
            lexical_weight: Weight for lexical search (default: 0.3)
            recency_weight: Weight for recency score (default: 0.1)
            decay_base: Base for exponential decay (default: 0.5)
            half_life_days: Half-life in days for recency decay (default: 200)
            top_k: Number of final results to return (default: 10)
        """
        if not self.chunk_data:
            return []
        
        # Normalize weights to sum to 1
        total_weight = semantic_weight + lexical_weight + recency_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            lexical_weight /= total_weight
            recency_weight /= total_weight
        
        logger.info(f"Hybrid search: '{query[:50]}...' (semantic:{semantic_weight:.2f}, lexical:{lexical_weight:.2f}, recency:{recency_weight:.2f})")
        
        # Get scores from separate search methods
        semantic_scores = self.semantic_search(query)
        lexical_scores = self.lexical_search(query)
        recency_scores = self.recency_search(decay_base, half_life_days)
        
        # Normalize scores
        norm_semantic = self._normalize_scores_dict(semantic_scores)
        norm_lexical = self._normalize_scores_dict(lexical_scores)
        norm_recency = self._normalize_scores_dict(recency_scores)
        
        # Combine scores
        combined_scores = {}
        for chunk_id in semantic_scores:
            semantic_score = norm_semantic[chunk_id]
            lexical_score = norm_lexical[chunk_id]
            recency_score = norm_recency[chunk_id]
            combined_scores[chunk_id] = (
                semantic_weight * semantic_score + 
                lexical_weight * lexical_score + 
                recency_weight * recency_score
            )
        
        # Sort and get top results
        top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        chunk_id_to_data = {chunk['id']: chunk for chunk in self.chunk_data}
        for chunk_id, final_score in top_results:
            chunk = chunk_id_to_data[chunk_id]
            results.append({
                'chunk_id': chunk_id,
                'filename': chunk['filename'],
                'text': chunk['text'],
                'final_score': final_score,
                'semantic_score': norm_semantic[chunk_id],
                'lexical_score': norm_lexical[chunk_id],
                'recency_score': norm_recency[chunk_id],
                'effective_date': chunk.get('effective_date')
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
        query = "How many paid leaves are allowed for the maternity leave policy?"
        results = search_engine.hybrid_search(query, top_k=10)
        
        print(f"\nHybrid Search Results for: '{query}'")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']}")
            print(f"   Combined Score: {result['final_score']:.3f}")
            print(f"   (Semantic: {result['semantic_score']:.3f}, Lexical: {result['lexical_score']:.3f}, Recency: {result['recency_score']:.3f})")
            if result.get('effective_date'):
                print(f"   Effective Date: {result['effective_date'].strftime('%Y-%m-%d')}")
            print(f"   Preview: {result['text'][:150]}...")
    else:
        print("No documents found. Please run embeddings first.")