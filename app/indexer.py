"""
Defensive Indexer for NeuralQuery Semantic Search API.

This script handles data ingestion with robust error handling,
environment validation, and index management.
"""

import os
import sys
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "neural-search"
DIMENSION = 384  # Required by sentence-transformers models
BATCH_SIZE = 100
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional model


def validate_environment() -> str:
    """
    Validate that required environment variables are set.
    
    Returns:
        str: The Pinecone API key
        
    Raises:
        ValueError: If PINECONE_API_KEY is missing
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY environment variable is required but not set. "
            "Please set it in your .env file or environment."
        )
    return api_key


def generate_synthetic_data() -> List[Dict[str, Any]]:
    """
    Generate 20 rich, realistic technical tips with metadata.
    
    Returns:
        List[Dict[str, Any]]: List of documents with text and metadata
    """
    data = [
        {
            "text": "Use Docker multi-stage builds to reduce image size by separating build and runtime dependencies.",
            "metadata": {"category": "Docker"}
        },
        {
            "text": "Leverage Python's context managers with 'with' statements to ensure proper resource cleanup and exception handling.",
            "metadata": {"category": "Python"}
        },
        {
            "text": "Configure AWS Lambda with appropriate memory allocation - more memory also increases CPU proportionally.",
            "metadata": {"category": "AWS"}
        },
        {
            "text": "Use Docker Compose for local development to orchestrate multiple containers and manage dependencies easily.",
            "metadata": {"category": "Docker"}
        },
        {
            "text": "Implement async/await in Python for I/O-bound operations to improve concurrency and performance.",
            "metadata": {"category": "Python"}
        },
        {
            "text": "Set up AWS CloudWatch alarms to monitor Lambda function errors, duration, and throttles proactively.",
            "metadata": {"category": "AWS"}
        },
        {
            "text": "Optimize Docker images by using .dockerignore to exclude unnecessary files and reduce build context size.",
            "metadata": {"category": "Docker"}
        },
        {
            "text": "Use Python's dataclasses or Pydantic models for type-safe data validation and serialization in APIs.",
            "metadata": {"category": "Python"}
        },
        {
            "text": "Implement AWS S3 lifecycle policies to automatically transition objects to cheaper storage classes over time.",
            "metadata": {"category": "AWS"}
        },
        {
            "text": "Use Docker health checks to ensure containers are running correctly and enable automatic restart on failure.",
            "metadata": {"category": "Docker"}
        },
        {
            "text": "Leverage Python's type hints with mypy for static type checking to catch errors before runtime.",
            "metadata": {"category": "Python"}
        },
        {
            "text": "Configure AWS VPC endpoints for private connectivity to S3 and other services without internet gateway.",
            "metadata": {"category": "AWS"}
        },
        {
            "text": "Use Docker volumes for persistent data storage that survives container restarts and updates.",
            "metadata": {"category": "Docker"}
        },
        {
            "text": "Implement Python logging with proper levels (DEBUG, INFO, WARNING, ERROR) for better observability.",
            "metadata": {"category": "Python"}
        },
        {
            "text": "Use AWS IAM roles instead of access keys for EC2 instances and Lambda functions for better security.",
            "metadata": {"category": "AWS"}
        },
        {
            "text": "Leverage Docker layer caching by ordering Dockerfile commands from least to most frequently changing.",
            "metadata": {"category": "Docker"}
        },
        {
            "text": "Use Python's pathlib instead of os.path for more readable and cross-platform file path operations.",
            "metadata": {"category": "Python"}
        },
        {
            "text": "Implement AWS CloudFormation or Terraform for Infrastructure as Code to version control your infrastructure.",
            "metadata": {"category": "AWS"}
        },
        {
            "text": "Use Docker secrets management for sensitive data like API keys instead of hardcoding them in images.",
            "metadata": {"category": "Docker"}
        },
        {
            "text": "Leverage Python's functools.lru_cache decorator for memoization to cache expensive function results.",
            "metadata": {"category": "Python"}
        }
    ]
    return data


def validate_or_create_index(pc: Pinecone) -> None:
    """
    Validate that the index exists and has correct dimensions.
    Create or recreate if necessary.
    
    Args:
        pc: Pinecone client instance
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME in existing_indexes:
        # Index exists, verify dimensions
        index_stats = pc.describe_index(INDEX_NAME)
        current_dimension = index_stats.dimension
        
        if current_dimension != DIMENSION:
            print(
                f"Warning: Index '{INDEX_NAME}' exists with dimension {current_dimension}, "
                f"but requires {DIMENSION}. Deleting and recreating..."
            )
            pc.delete_index(INDEX_NAME)
            # Wait for deletion to complete
            import time
            while INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
                time.sleep(1)
            create_index(pc)
        else:
            print(f"Index '{INDEX_NAME}' exists with correct dimension {DIMENSION}.")
    else:
        # Index doesn't exist, create it
        print(f"Index '{INDEX_NAME}' does not exist. Creating...")
        create_index(pc)


def create_index(pc: Pinecone) -> None:
    """
    Create a new Pinecone index with ServerlessSpec.
    
    Args:
        pc: Pinecone client instance
    """
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Successfully created index '{INDEX_NAME}' with dimension {DIMENSION}.")
    except PineconeException as e:
        raise RuntimeError(f"Failed to create index: {e}") from e


def encode_documents(model: SentenceTransformer, documents: List[str]) -> List[List[float]]:
    """
    Encode documents into vectors using the sentence transformer model.
    
    Args:
        model: SentenceTransformer model instance
        documents: List of document texts
        
    Returns:
        List[List[float]]: List of encoded vectors
    """
    return model.encode(documents, show_progress_bar=True).tolist()


def batch_upsert(
    index,
    vectors: List[Tuple[str, List[float], Dict[str, Any]]],
    batch_size: int = BATCH_SIZE
) -> None:
    """
    Upsert vectors in batches to prevent HTTP timeouts.
    
    Args:
        index: Pinecone index instance
        vectors: List of tuples (id, vector, metadata)
        batch_size: Number of vectors per batch
    """
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        try:
            index.upsert(vectors=batch)
            print(f"Upserted batch {batch_num}/{total_batches} ({len(batch)} vectors)")
        except PineconeException as e:
            raise RuntimeError(f"Failed to upsert batch {batch_num}: {e}") from e


def main() -> None:
    """
    Main execution function for the indexer.
    """
    try:
        # Environment validation
        print("Validating environment...")
        api_key = validate_environment()
        print("Environment validation passed.")
        
        # Initialize Pinecone client
        print("Initializing Pinecone client...")
        pc = Pinecone(api_key=api_key)
        
        # Validate or create index
        validate_or_create_index(pc)
        
        # Connect to index
        print(f"Connecting to index '{INDEX_NAME}'...")
        index = pc.Index(INDEX_NAME)
        
        # Load model
        print(f"Loading sentence transformer model '{MODEL_NAME}'...")
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
        
        # Generate synthetic data
        print("Generating synthetic data...")
        data = generate_synthetic_data()
        print(f"Generated {len(data)} documents.")
        
        # Encode documents
        print("Encoding documents...")
        documents = [item["text"] for item in data]
        vectors = encode_documents(model, documents)
        
        # Prepare vectors with IDs and metadata
        print("Preparing vectors for upsert...")
        vectors_to_upsert = [
            (f"doc_{i}", vector, data[i]["metadata"])
            for i, vector in enumerate(vectors)
        ]
        
        # Batch upsert
        print(f"Upserting {len(vectors_to_upsert)} vectors in batches of {BATCH_SIZE}...")
        batch_upsert(index, vectors_to_upsert, BATCH_SIZE)
        
        print("Indexing completed successfully!")
        
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PineconeException as e:
        print(f"Pinecone Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
