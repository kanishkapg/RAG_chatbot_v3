import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import logging
from typing import Dict
from datetime import datetime

# Import your custom modules
from multiplicative import MultiplicativeGatingSearchEngine
from penalty import PenaltyScoredFusionSearchEngine
from response_generator import ResponseGenerator
from utils.config import (
    MULTIPLICATIVE_WEIGHTS, PENALTY_WEIGHTS,
    DEFAULT_TOP_K, DEFAULT_MARGIN, DEFAULT_P_COEFF
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Circular Chatbot - Hybrid Search Scored Fusion Method",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chunk-card {
        background-color: #f1f3f4;
        border-left: 4px solid #4285f4;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
    }
    .score-badge {
        background-color: #e8f0fe;
        color: #1967d2;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .date-badge {
        background-color: #fef7e0;
        color: #f57c00;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .metadata-text {
        font-size: 0.9rem;
        color: #666;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_engine(method):
    with st.spinner(f"🔄 Initializing {method} search engine..."):
        if method == 'Multiplicative Gating':
            st.session_state.search_engine = MultiplicativeGatingSearchEngine()
        else:
            st.session_state.search_engine = PenaltyScoredFusionSearchEngine()
        st.session_state.response_generator = ResponseGenerator(search_method='multiplicative' if method == 'Multiplicative Gating' else 'penalty')
        st.session_state.search_method = method

if 'search_engine' not in st.session_state:
    initialize_engine('Multiplicative Gating') # Default method

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'last_hybrid_results' not in st.session_state:
    st.session_state.last_hybrid_results = []

if 'search_method' not in st.session_state:
    st.session_state.search_method = 'Multiplicative Gating'



# Helper functions
def format_chunk_preview(text: str, max_length: int = 200) -> str:
    """Format chunk text for preview display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_date(date_str: str) -> str:
    """Format date string for display."""
    if not date_str or date_str == 'Not specified':
        return "📅 No date"
    try:
        # Try to parse and format the date
        if len(date_str) == 10 and '-' in date_str:  # YYYY-MM-DD format
            return f"📅 {date_str}"
        return f"📅 {date_str}"
    except:
        return f"📅 {date_str}"

def display_chunk_card(chunk: Dict, index: int, card_type: str = "hybrid"):
    """Display a chunk card with metadata."""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**📄 {chunk['filename']}**")
            
            # Display metadata if available
            metadata = chunk.get('metadata', {})
            if metadata:
                circular_no = metadata.get('circular_number', 'N/A')
                title = metadata.get('title', 'N/A')
                date = metadata.get('effective_date') or metadata.get('issued_date')
                
                st.markdown(f"<span class='metadata-text'>🔖 Circular: {circular_no}</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='metadata-text'>📋 Title: {title}</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='date-badge'>{format_date(date)}</span>", unsafe_allow_html=True)
        
        with col2:
            if 'final_score' in chunk:
                st.markdown(f"<span class='score-badge'>Final: {chunk['final_score']:.3f}</span>", unsafe_allow_html=True)
            
            if card_type == "hybrid":
                if 'relevance_score' in chunk and 'recency_boost' in chunk:
                    st.caption(f"Relevance: {chunk['relevance_score']:.3f}")
                    st.caption(f"Boost: {chunk['recency_boost']:.3f}")
                
                if 'semantic_score' in chunk and 'lexical_score' in chunk:
                    st.caption(f"Sem: {chunk['semantic_score']:.3f} | Lex: {chunk['lexical_score']:.3f}")
                
                if 'recency_score' in chunk:
                    st.caption(f"Recency: {chunk['recency_score']:.3f}")
        
        # Text preview
        st.markdown(f"<div class='chunk-card'>{format_chunk_preview(chunk['text'])}</div>", 
                   unsafe_allow_html=True)
        
        # Expandable full text
        with st.expander(f"View full text - Chunk {index + 1}"):
            st.text_area("Full content:", value=chunk['text'], height=200, disabled=True, key=f"{card_type}_chunk_{index}_{chunk.get('chunk_id', index)}")

# Main UI
st.markdown("""
<div class="main-header">
    <h1>🤖 Circular Chatbot - Hybrid Search Scored Fusion Method</h1>
    <p>Intelligent Document Search & Question Answering</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Search method selection
    search_method = st.radio(
        "Select Search Method",
        ('Multiplicative Gating', 'Penalty Scored Fusion'),
        index=0 if st.session_state.search_method == 'Multiplicative Gating' else 1
    )

    if search_method != st.session_state.search_method:
        initialize_engine(search_method)
    
    # Search parameters
    st.subheader("🔍 Search Parameters")
    if search_method == 'Multiplicative Gating':
        weights = MULTIPLICATIVE_WEIGHTS
    else:
        weights = PENALTY_WEIGHTS

    semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, weights['semantic'], 0.1)
    lexical_weight = st.slider("Lexical Weight", 0.0, 1.0, weights['lexical'], 0.1)
    recency_weight = st.slider("Recency Weight", 0.0, 1.0, weights['recency'], 0.1)
    top_k = st.slider("Top K Results", 5, 20, DEFAULT_TOP_K, 1)

    if search_method == 'Penalty Scored Fusion':
        st.subheader("Penalty Parameters")
        margin = st.slider("Margin", 0.0, 1.0, DEFAULT_MARGIN, 0.05)
        p_coeff = st.slider("Penalty Coefficient", 0.0, 5.0, DEFAULT_P_COEFF, 0.1)
    
    # Show formula info
    if search_method == 'Multiplicative Gating':
        relevance_total = semantic_weight + lexical_weight
        if relevance_total > 0:
            semantic_norm = semantic_weight / relevance_total
            lexical_norm = lexical_weight / relevance_total
        else:
            semantic_norm = 0.7
            lexical_norm = 0.3
        st.info(f"**Multiplicative Gating:**\nRelevance = {semantic_norm:.2f}×Sem + {lexical_norm:.2f}×Lex\nFinal = Relevance × (1 + {recency_weight:.2f}×Rec)")
    else:
        st.info(f"**Penalty Scored Fusion:**\nRelevance = {semantic_weight:.2f}×Sem + {lexical_weight:.2f}×Lex\nPenalty = {p_coeff:.2f} × max(0, Rec - Rel - {margin:.2f})\nFinal = Rel + Rec - Penalty")
    
    # Document statistics
    st.subheader("📊 Document Statistics")
    if st.session_state.search_engine.chunk_data:
        total_chunks = len(st.session_state.search_engine.chunk_data)
        unique_docs = len(st.session_state.search_engine.get_document_list())
        st.metric("Total Chunks", total_chunks)
        st.metric("Unique Documents", unique_docs)
    
    # Refresh button
    if st.button("🔄 Refresh Index"):
        with st.spinner("Refreshing search index..."):
            st.session_state.search_engine.refresh_index()
            st.success("Index refreshed!")

# Main chat interface
st.header("💬 Ask a Question")

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is the maternity leave policy?",
    help="Ask questions about the documents in your database"
)

# Search button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button("🔍 Search", type="primary")
with col2:
    clear_button = st.button("🗑️ Clear History")

if clear_button:
    st.session_state.chat_history = []
    st.session_state.last_hybrid_results = []
    st.rerun()

# Process search
if search_button and query.strip():
    if not st.session_state.search_engine.chunk_data:
        st.error("❌ No documents found in the database. Please ensure documents are processed and embeddings are created.")
    else:
        with st.spinner("🔍 Searching and generating response..."):
            try:
                # Perform hybrid search
                if st.session_state.search_method == 'Penalty Scored Fusion':
                    hybrid_results = st.session_state.search_engine.hybrid_search(
                        query, semantic_weight, lexical_weight, recency_weight, top_k=top_k, margin=margin, p_coeff=p_coeff
                    )
                else:
                    hybrid_results = st.session_state.search_engine.hybrid_search(
                        query, semantic_weight, lexical_weight, recency_weight, top_k=top_k
                    )
                
                st.session_state.last_hybrid_results = hybrid_results
                
                # Generate response
                response_data = st.session_state.response_generator.process_query(query, hybrid_results)
                
                # Store chat history
                st.session_state.chat_history.append({
                    'query': query,
                    'response': response_data['response'],
                    'sources': response_data['sources'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                st.error(f"An error occurred: {e}")

# Display chat history
if st.session_state.chat_history:
    st.header("💭 Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"🕐 {chat['timestamp']} - {chat['query'][:50]}...", expanded=(i == 0)):
            st.markdown("**Question:**")
            st.info(chat['query'])
            
            st.markdown("**Answer:**")
            st.markdown(f"<div class='chat-container'>{chat['response']}</div>", unsafe_allow_html=True)
            
            if chat['sources']:
                st.markdown("**Sources:**")
                for j, source in enumerate(chat['sources'], 1):
                    with st.container():
                        st.markdown(f"<div class='source-card'>", unsafe_allow_html=True)
                        st.markdown(f"**{j}. {source['filename']}**")
                        st.markdown(f"📋 **Circular:** {source['circular_number']}")
                        st.markdown(f"📄 **Title:** {source['title']}")
                        st.markdown(f"{format_date(source['date'])}")
                        st.markdown(f"⭐ **Final Score:** {source['score']:.4f}")
                        if 'relevance_score' in source and 'recency_boost' in source:
                            st.markdown(f"🎯 **Relevance:** {source['relevance_score']:.4f} × **Boost:** {source['recency_boost']:.4f}")
                        st.markdown("</div>", unsafe_allow_html=True)

# Display search results
if st.session_state.last_hybrid_results:
    st.header("🔍 Search Results Analysis")
    
    st.subheader(f"Top {len(st.session_state.last_hybrid_results)} Hybrid Search Results")
    st.caption("Results from multiplicative gating hybrid search (top 3 used for response generation)")
    
    for i, chunk in enumerate(st.session_state.last_hybrid_results):
        # Highlight the top 3 chunks used for response generation
        if i < 3:
            st.markdown(f"### ✨ Result {i + 1} (Used for Response)")
        else:
            st.markdown(f"### Result {i + 1}")
        display_chunk_card(chunk, i, "hybrid")
        st.divider()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 RAG Chatbot v3 - Powered by Multiplicative Gating Hybrid Search & LLM Response Generation</p>
    <p>Built with Streamlit • BGE-M3 Embeddings • TF-IDF • Groq LLM • Multiplicative Gating</p>
</div>
""", unsafe_allow_html=True)
