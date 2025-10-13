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

    def rerank_by_date(self, chunks: List[Dict]) -> List[Dict]:
        """
        Rerank chunks by effective_date (or issued_date as fallback) in descending order.
        
        Args:
            chunks: List of chunks from hybrid search with their metadata
            
        Returns:
            List of chunks sorted by date (most recent first)
        """
        logger.info(f"Reranking {len(chunks)} chunks by date...")
        
        # Add metadata to each chunk
        enriched_chunks = []
        for chunk in chunks:
            metadata = self._get_chunk_metadata(chunk['chunk_id'])
            
            # Get ranking date (effective_date or issued_date as fallback)
            ranking_date = metadata.get('effective_date') or metadata.get('issued_date')
            
            enriched_chunk = chunk.copy()
            enriched_chunk['metadata'] = metadata
            enriched_chunk['ranking_date'] = ranking_date
            enriched_chunks.append(enriched_chunk)
        
        # Sort by ranking_date (most recent first)
        # Chunks without dates go to the end
        reranked_chunks = sorted(
            enriched_chunks, 
            key=lambda x: x['ranking_date'] if x['ranking_date'] else '0000-00-00', 
            reverse=True
        )
        
        logger.info("Chunks reranked by date successfully.")
        return reranked_chunks

    def generate_response(self, query: str, top_chunks: List[Dict]) -> Dict:
        """
        Generate response using LLM based on the top chunks.
        
        Args:
            query: User's question
            top_chunks: Top 3 chunks after reranking
            
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
                'score': chunk.get('final_score', 0)
            })
        
        context = "\n" + "="*50 + "\n".join(context_parts)
        
        # Create prompt for LLM
        prompt = f"""You are an expert assistant specializing in organizational policies and circulars. 
        Your task is to provide accurate, concise answers based on the provided context documents.

        IMPORTANT INSTRUCTIONS:
        1. PRIORITIZE information from documents with more recent dates (effective_date or issued_date)
        2. IGNORE or clearly note if information appears to be from superseded/repealed documents
        3. Be CONCISE - provide direct answers without unnecessary elaboration
        4. ALWAYS cite the specific document(s) you're referencing in your answer
        5. If information conflicts between documents, prioritize the most recent and explain the discrepancy

        Context Documents (ordered by relevance and date):
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
        Complete pipeline: rerank by date and generate response.
        
        Args:
            query: User's question
            hybrid_search_results: Top 10 chunks from hybrid search
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing query: '{query[:50]}...'")
        
        # Rerank by date and select top 3
        reranked_chunks = self.rerank_by_date(hybrid_search_results)
        ranked_top_chunks = reranked_chunks[:10]
        
        # Generate response
        result = self.generate_response(query, ranked_top_chunks)
        return result


# Example usage and testing
if __name__ == "__main__":
    from hybrid_search import HybridSearchEngine
    
    search_engine = HybridSearchEngine()
    response_generator = ResponseGenerator()
    
    if search_engine.chunk_data:
        query = "What is the maternity leave policy in 2022?"
        hybrid_results = search_engine.hybrid_search(query, top_k=10)
        
        if hybrid_results:
            result = response_generator.process_query(query, hybrid_results)
            
            print(f"\nQuery: {query}")
            print("=" * 80)
            print(f"\nResponse:\n{result['response']}")
            
            print(f"\n\nSources ({result['chunk_count']} documents):")
            print("-" * 50)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['filename']}")
                print(f"   Circular: {source['circular_number']}")
                print(f"   Title: {source['title']}")
                print(f"   Date: {source['date'] or 'Not specified'}")
                print(f"   Score: {source['score']:.4f}")
                print()
        else:
            print("No search results found.")
    else:
        print("No documents available.")