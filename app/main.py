"""
Fault-Tolerant Semantic Search API for NeuralQuery.

This FastAPI application provides semantic search capabilities
using Pinecone and Sentence Transformers.
"""

import os
import contextlib
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from pinecone import Pinecone
from pinecone.exceptions import PineconeException
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "neural-search"
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional model

# Global variables for model and Pinecone client
model: Optional[SentenceTransformer] = None
pc: Optional[Pinecone] = None
index = None


# Pydantic Models
class SearchRequest(BaseModel):
    """Request model for search queries."""
    
    query: str = Field(
        ...,
        min_length=3,
        description="Search query text (minimum 3 characters)"
    )
    top_k: int = Field(
        default=3,
        le=10,
        ge=1,
        description="Number of results to return (1-10)"
    )
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and strip query string."""
        stripped = v.strip()
        if len(stripped) < 3:
            raise ValueError("Query must be at least 3 characters long")
        return stripped


class SearchResult(BaseModel):
    """Individual search result model."""
    
    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for search results."""
    
    results: List[SearchResult]
    query: str
    top_k: int


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads the SentenceTransformer model once at startup.
    """
    global model, pc, index
    
    # Startup
    print("Starting up NeuralQuery API...")
    
    # Validate environment
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY environment variable is required but not set. "
            "Please set it in your .env file or environment."
        )
    
    # Initialize Pinecone client
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=api_key)
    
    # Connect to index
    print(f"Connecting to index '{INDEX_NAME}'...")
    try:
        index = pc.Index(INDEX_NAME)
    except PineconeException as e:
        raise RuntimeError(f"Failed to connect to index '{INDEX_NAME}': {e}") from e
    
    # Load model
    print(f"Loading sentence transformer model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully. API is ready!")
    
    yield
    
    # Shutdown (cleanup if needed)
    print("Shutting down NeuralQuery API...")


# FastAPI app initialization
app = FastAPI(
    title="NeuralQuery Semantic Search API",
    description="Production-grade semantic search API using Pinecone and Sentence Transformers",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint that verifies Pinecone connection is active.
    
    Returns:
        Dict[str, Any]: Health status
    """
    global pc, index
    
    try:
        if pc is None or index is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        # Verify connection by checking index stats
        stats = index.describe_index_stats()
        total_vectors = getattr(stats, "total_vector_count", 0)
        
        return {
            "status": "healthy",
            "service": "NeuralQuery API",
            "index": INDEX_NAME,
            "total_vectors": total_vectors
        }
    except PineconeException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pinecone connection error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Semantic search endpoint.
    
    Args:
        request: SearchRequest with query and top_k
        
    Returns:
        SearchResponse: Search results with metadata
    """
    global model, index
    
    try:
        # Validate model and index are loaded
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        if index is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Index not connected"
            )
        
        # Encode query
        query_vector = model.encode(request.query).tolist()
        
        # Search in Pinecone
        results = index.query(
            vector=query_vector,
            top_k=request.top_k,
            include_metadata=True
        )
        
        # Format response
        search_results = [
            SearchResult(
                id=match.id,
                score=float(match.score),
                metadata=match.metadata or {}
            )
            for match in results.matches
        ]
        
        return SearchResponse(
            results=search_results,
            query=request.query,
            top_k=request.top_k
        )
        
    except PineconeException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pinecone service error: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> Dict[str, str]:
    """
    Root endpoint with API information.
    
    Returns:
        Dict[str, str]: API information
    """
    return {
        "message": "NeuralQuery Semantic Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
