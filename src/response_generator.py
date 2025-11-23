import logging
from typing import List, Dict, Optional
from groq import Groq
from database_setup import get_db_connection
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import GROQ_API_KEY, GROQ_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self):
        """Initialize the response generator with Groq client."""
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.model_name = GROQ_MODEL

    def _get_chunk_metadata(self, chunk_id: int) -> Optional[Dict]:
        """Get metadata for a specific chunk from the database."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT dc.filename, dc.file_hash, dm.metadata
                    FROM document_chunks dc
                    LEFT JOIN document_metadata dm ON dc.filename = dm.filename 
                                                   AND dc.file_hash = dm.file_hash
                    WHERE dc.id = %s
                """, (chunk_id,))
                
                result = cur.fetchone()
                if result:
                    filename, file_hash, metadata = result
                    return metadata if metadata else {}
                return {}
        except Exception as e:
            logger.error(f"Error getting metadata for chunk {chunk_id}: {e}")
            return {}
        finally:
            conn.close()

    def enrich_chunks_with_metadata(self, chunks: List[Dict]) -> List[Dict]:
        """
        Enrich chunks with metadata information.
        
        Args:
            chunks: List of chunks from hybrid search
            
        Returns:
            List of chunks enriched with metadata
        """
        logger.info(f"Enriching {len(chunks)} chunks with metadata...")
        
        enriched_chunks = []
        for chunk in chunks:
            metadata = self._get_chunk_metadata(chunk['chunk_id'])
            
            enriched_chunk = chunk.copy()
            enriched_chunk['metadata'] = metadata
            enriched_chunks.append(enriched_chunk)
        
        logger.info("Chunks enriched with metadata successfully.")
        return enriched_chunks

    def generate_response(self, query: str, top_chunks: List[Dict]) -> Dict:
        """
        Generate response using LLM based on the top chunks.
        
        Args:
            query: User's question
            top_chunks: Top 5 chunks from hybrid search
            
        Returns:
            Dictionary with response and metadata
        """
        if not top_chunks:
            return {
                'response': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'chunk_count': 0
            }
        
        # Create context from chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(top_chunks, 1):
            metadata = chunk.get('metadata', {})
            
            context_parts.append(
                f"Document {i}:\n"
                f"Source: {chunk['filename']}\n"
                f"Circular: {metadata.get('circular_number', 'Unknown')}\n"
                f"Title: {metadata.get('title', 'Unknown Title')}\n"
                f"Date: {metadata.get('effective_date') or metadata.get('issued_date') or 'Not specified'}\n"
                f"Content: {chunk['text']}\n"
            )
            
            sources.append({
                'filename': chunk['filename'],
                'circular_number': metadata.get('circular_number', 'Unknown'),
                'title': metadata.get('title', 'Unknown Title'),
                'date': metadata.get('effective_date') or metadata.get('issued_date'),
                'score': chunk.get('final_score', 0),
                'relevance_score': chunk.get('relevance_score', 0),
                'recency_boost': chunk.get('recency_boost', 1.0),
                'semantic_score': chunk.get('semantic_score', 0),
                'lexical_score': chunk.get('lexical_score', 0),
                'recency_score': chunk.get('recency_score', 0)
            })
        
        context = "\n" + "="*50 + "\n".join(context_parts)
        
        # Create prompt for LLM
        prompt = f"""You are an expert assistant specializing in organizational policies and circulars. 
        Your task is to provide accurate, concise answers based on the provided context documents.

        IMPORTANT INSTRUCTIONS:
        1. When providing an answer always prioritize most recent document's information
        2. Be CONCISE - provide direct answers without unnecessary elaboration
        3. ALWAYS cite the specific document(s) you're referencing in your answer
        4. In the query, if it is asked for a historical information, mentioning a specific year, prioritize the relevant documents accordingly rather than the recency

        Context Documents (ordered by hybrid search with multiplicative gating):
        {context}
     
        User Question: {query}
        
        Answer:"""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that answers questions based on official documents. Always cite the source documents."
                    },
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated response for query: '{query[:50]}...'")
            
            return {
                'response': answer,
                'sources': sources,
                'chunk_count': len(top_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"Sorry, I encountered an error: {str(e)}",
                'sources': sources,
                'chunk_count': len(top_chunks)
            }

    def process_query(self, query: str, hybrid_search_results: List[Dict]) -> Dict:
        """
        Complete pipeline: enrich with metadata and generate response using top 5 results.
        
        Args:
            query: User's question
            hybrid_search_results: Results from hybrid search (already ranked by relevance)
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing query: '{query[:50]}...'")

        # Take top 3 results from hybrid search (already optimally ranked)
        top_chunks = hybrid_search_results[:3]
        
        # Enrich with metadata for context
        enriched_chunks = self.enrich_chunks_with_metadata(top_chunks)
        
        # Generate response
        result = self.generate_response(query, enriched_chunks)
        return result


# Example usage and testing
if __name__ == "__main__":
    from hybrid_search import HybridSearchEngine
    
    search_engine = HybridSearchEngine()
    response_generator = ResponseGenerator()
    
    if search_engine.chunk_data:
        query = "Is it true that the Sales team has different in-office requirements than the Engineering team?"
        hybrid_results = search_engine.hybrid_search(query, top_k=5)
        
        if hybrid_results:
            result = response_generator.process_query(query, hybrid_results)
            
            print(f"\nQuery: {query}")
            print("=" * 80)
            print(f"\nResponse:\n{result['response']}")
            
            print(f"\n\nSources ({result['chunk_count']} documents used for response):")
            print("-" * 50)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['filename']}")
                print(f"   Circular: {source['circular_number']}")
                print(f"   Title: {source['title']}")
                print(f"   Date: {source['date'] or 'Not specified'}")
                print(f"   Final Score: {source['score']:.4f}")
                print(f"   Relevance: {source['relevance_score']:.4f} × Boost: {source['recency_boost']:.4f}")
                print(f"   (Sem: {source['semantic_score']:.3f}, Lex: {source['lexical_score']:.3f}, Rec: {source['recency_score']:.3f})")
                print()
        else:
            print("No search results found.")
    else:
        print("No documents available.")