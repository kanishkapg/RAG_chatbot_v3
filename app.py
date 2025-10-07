import streamlit as st
import pandas as pd
from typing import List, Dict
from hybrid_search import HybridSearchEngine
from response_generator import ResponseGenerator
import json

# Page config
st.set_page_config(
    page_title="RAG Chatbot v3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'search_engine' not in st.session_state:
    with st.spinner("Initializing search engine..."):
        st.session_state.search_engine = HybridSearchEngine()

if 'response_generator' not in st.session_state:
    with st.spinner("Initializing response generator..."):
        st.session_state.response_generator = ResponseGenerator()

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'semantic_results' not in st.session_state:
    st.session_state.semantic_results = None

if 'lexical_results' not in st.session_state:
    st.session_state.lexical_results = None

def format_chunk_preview(text: str, max_length: int = 200) -> str:
    """Format chunk text for preview."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def display_chunk_details(chunks: List, title: str, search_type: str):
    """Display detailed chunk information in an expandable section."""
    with st.expander(f"📊 {title} ({len(chunks)} chunks)", expanded=False):
        if chunks:
            for i, chunk in enumerate(chunks, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Handle different chunk formats
                        if search_type == "hybrid":
                            # Hybrid results are dictionaries
                            filename = chunk.get('filename', 'Unknown File')
                            chunk_id = chunk.get('chunk_id', i)
                            text = chunk.get('text', '')
                        else:
                            # Semantic and lexical results are tuples (index, score)
                            chunk_idx, score = chunk
                            chunk_data = st.session_state.search_engine.chunk_data[chunk_idx]
                            filename = chunk_data['filename']
                            chunk_id = chunk_data['id']
                            text = chunk_data['text']
                        
                        st.markdown(f"**{i}. {filename}**")
                        
                        # Show scores based on search type
                        if search_type == "hybrid":
                            st.markdown(f"🎯 **Final Score:** {chunk.get('final_score', 0):.4f}")
                            st.markdown(f"🧠 Semantic: {chunk.get('semantic_score', 0):.4f} | 📝 Lexical: {chunk.get('lexical_score', 0):.4f}")
                        elif search_type == "semantic":
                            st.markdown(f"🧠 **Semantic Score:** {score:.4f}")
                        elif search_type == "lexical":
                            st.markdown(f"📝 **Lexical Score:** {score:.4f}")
                    
                    with col2:
                        # Button to show full text
                        button_key = f"{search_type}_{i}_{chunk_id}"
                        if st.button(f"View Full Text", key=button_key):
                            st.session_state[f"show_full_{search_type}_{i}"] = not st.session_state.get(f"show_full_{search_type}_{i}", False)
                    
                    # Show preview text
                    if st.session_state.get(f"show_full_{search_type}_{i}", False):
                        st.markdown(f"**Full Text:**")
                        st.text_area("", value=text, height=200, key=f"full_text_{search_type}_{i}", disabled=True)
                    else:
                        st.markdown(f"**Preview:** {format_chunk_preview(text)}")
                    
                    st.divider()
        else:
            st.info("No results found.")

def main():
    st.title("🤖 RAG Chatbot v3")
    st.markdown("**Retrieval-Augmented Generation Chatbot with Hybrid Search**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Search Configuration")
        
        # Search weights
        semantic_weight = st.slider(
            "Semantic Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Weight for meaning-based search using embeddings"
        )
        
        lexical_weight = st.slider(
            "Lexical Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.1,
            help="Weight for keyword-based search using TF-IDF"
        )
        
        # Top K results
        top_k = st.selectbox(
            "Number of Results",
            options=[3, 5, 10, 15, 20],
            index=1,
            help="Number of top results to retrieve"
        )
        
        st.divider()
        
        # Document statistics
        if st.session_state.search_engine.chunk_data:
            st.header("📈 Document Statistics")
            total_chunks = len(st.session_state.search_engine.chunk_data)
            unique_docs = len(st.session_state.search_engine.get_document_list())
            
            st.metric("Total Chunks", total_chunks)
            st.metric("Unique Documents", unique_docs)
            
            # Available documents
            with st.expander("📁 Available Documents"):
                docs = st.session_state.search_engine.get_document_list()
                for doc in docs:
                    st.text(doc)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the latest maternity leave policy?",
            help="Ask any question about the documents in the database"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Process search when button is clicked or Enter is pressed
    if (search_button or query) and query.strip():
        with st.spinner("Searching and generating response..."):
            try:
                # Perform hybrid search
                hybrid_results = st.session_state.search_engine.hybrid_search(
                    query=query,
                    semantic_weight=semantic_weight,
                    lexical_weight=lexical_weight,
                    top_k=top_k
                )
                
                # Get individual search results for detailed view
                semantic_results = st.session_state.search_engine._semantic_search(query, top_k=10)
                lexical_results = st.session_state.search_engine._lexical_search(query, top_k=10)
                
                # Store results in session state
                st.session_state.search_results = hybrid_results
                st.session_state.semantic_results = semantic_results
                st.session_state.lexical_results = lexical_results
                
                # Generate response using top 5 hybrid results
                response_data = st.session_state.response_generator.process_query(query, hybrid_results[:5])
                
                # Display main response
                st.header("💬 Response")
                st.markdown(response_data['response'])
                
                # Display top 3 sources (reranked by date)
                if response_data['sources']:
                    st.header("📚 Top 3 Sources (Ranked by Effective Date)")
                    
                    for i, source in enumerate(response_data['sources'], 1):
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{i}. {source['filename']}**")
                                st.markdown(f"📄 **Circular:** {source['circular_number']}")
                                st.markdown(f"📝 **Title:** {source['title']}")
                            
                            with col2:
                                st.markdown(f"📅 **Date:** {source['date'] or 'Not specified'}")
                            
                            with col3:
                                st.markdown(f"⭐ **Score:** {source['score']:.4f}")
                            
                            st.divider()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Display detailed search results if available
    if st.session_state.search_results is not None:
        st.header("🔍 Detailed Search Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Hybrid Search (Top 5)", 
            "🧠 Semantic Search (Top 10)", 
            "📝 Lexical Search (Top 10)",
            "📈 Score Comparison"
        ])
        
        with tab1:
            display_chunk_details(
                st.session_state.search_results[:5], 
                "Hybrid Search Results", 
                "hybrid"
            )
        
        with tab2:
            display_chunk_details(
                st.session_state.semantic_results[:10], 
                "Semantic Search Results", 
                "semantic"
            )
        
        with tab3:
            display_chunk_details(
                st.session_state.lexical_results[:10], 
                "Lexical Search Results", 
                "lexical"
            )
        
        with tab4:
            # Score comparison chart
            if len(st.session_state.search_results) > 0:
                st.subheader("Score Distribution Analysis")
                
                # Prepare data for visualization
                hybrid_data = []
                for i, result in enumerate(st.session_state.search_results[:10]):
                    hybrid_data.append({
                        'Rank': i + 1,
                        'Document': result['filename'][:30] + '...' if len(result['filename']) > 30 else result['filename'],
                        'Final Score': result['final_score'],
                        'Semantic Score': result['semantic_score'],
                        'Lexical Score': result['lexical_score']
                    })
                
                df = pd.DataFrame(hybrid_data)
                
                # Display as chart
                st.line_chart(df.set_index('Rank')[['Final Score', 'Semantic Score', 'Lexical Score']])
                
                # Display as table
                st.subheader("Score Details")
                st.dataframe(df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**RAG Chatbot v3** - Powered by Hybrid Search (Semantic + Lexical) | "
        "Built with Streamlit, sentence-transformers, and Groq LLM"
    )

if __name__ == "__main__":
    main()