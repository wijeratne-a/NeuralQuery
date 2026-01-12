"""
Streamlit Frontend for NeuralQuery Semantic Search Engine.

This module provides a clean, modern chat interface for interacting
with the NeuralQuery API.
"""

import streamlit as st
import requests
from typing import Dict, Any, Optional

# API Configuration
API_BASE_URL = "http://localhost:8000"
SEARCH_ENDPOINT = f"{API_BASE_URL}/search"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Page Configuration
st.set_page_config(
    page_title="NeuralQuery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        background-color: #4CAF50;
        color: white;
        font-size: 0.85rem;
    }
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        background-color: #2196F3;
        color: white;
        font-size: 0.85rem;
        margin-left: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """
    Check if the API is running and healthy.
    
    Returns:
        bool: True if API is healthy, False otherwise
    """
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout, requests.RequestException):
        return False


def perform_search(query: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
    """
    Send search request to the API.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        
    Returns:
        Optional[Dict[str, Any]]: API response or None if error
    """
    try:
        payload = {
            "query": query,
            "top_k": top_k
        }
        response = requests.post(
            SEARCH_ENDPOINT,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the API. Is the Docker container running?")
        return None
    except requests.exceptions.Timeout:
        st.error("Request Timeout: The API took too long to respond.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None


def display_results(results: Dict[str, Any]) -> None:
    """
    Display search results in expandable cards.
    
    Args:
        results: API response containing search results
    """
    if not results or not results.get("results"):
        st.warning("No results found.")
        return
    
    st.subheader(f"Found {len(results['results'])} results for: \"{results['query']}\"")
    st.markdown("---")
    
    for idx, result in enumerate(results["results"], 1):
        score = result.get("score", 0.0)
        metadata = result.get("metadata", {})
        category = metadata.get("category", "Unknown")
        doc_id = result.get("id", f"doc_{idx}")
        
        # Create expander for each result
        with st.expander(
            f"Result {idx} | Score: {score:.3f} | Category: {category}",
            expanded=idx == 1  # Expand first result by default
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Document ID:** {doc_id}")
                st.markdown(f"**Similarity Score:** {score:.4f}")
                st.markdown(f"**Category:** {category}")
                
                # Display metadata if available
                if metadata:
                    st.markdown("**Metadata:**")
                    for key, value in metadata.items():
                        if key != "category":  # Already displayed
                            st.text(f"  {key}: {value}")
            
            with col2:
                # Visual score indicator
                st.metric("Score", f"{score:.3f}")
                st.markdown(f"<span class='category-badge'>{category}</span>", unsafe_allow_html=True)
        
        st.markdown("---")


def main():
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">🧠 NeuralQuery: Semantic Search Engine</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Health Check
        st.subheader("API Status")
        if check_api_health():
            st.success("API is healthy")
        else:
            st.error("API is not responding")
            st.info("Make sure the Docker container is running on port 8000")
        
        st.markdown("---")
        
        # Top-k configuration
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=3,
            help="Select how many results to return (1-10)"
        )
        
        st.markdown("---")
        st.markdown("**About NeuralQuery**")
        st.info(
            "NeuralQuery uses semantic search to find relevant technical tips "
            "about Docker, Python, and AWS. It leverages transformer models "
            "to understand the meaning behind your queries."
        )
    
    # Main Content
    st.markdown("Enter your technical question below to search the knowledge base.")
    
    # Search input
    query = st.text_input(
        "Search Query",
        placeholder="e.g., How do I optimize Docker images?",
        label_visibility="collapsed"
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Perform search
    if search_button or (st.session_state.get("auto_search", False)):
        if st.session_state.get("auto_search", False):
            st.session_state.auto_search = False
        
        if not query or len(query.strip()) < 3:
            st.warning("Please enter a search query (at least 3 characters).")
        else:
            # Show loading spinner
            with st.spinner("Searching the knowledge base..."):
                results = perform_search(query.strip(), top_k)
                
                if results:
                    display_results(results)


if __name__ == "__main__":
    main()
