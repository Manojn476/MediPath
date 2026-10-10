"""
Health RAG (Retrieval-Augmented Generation) system.

Uses FAISS and sentence embeddings to retrieve relevant health information
from the chunked corpus.
"""
import sys
from pathlib import Path
from typing import List, Dict
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.llm_client import ask_llm


# Global variables for lazy loading
_index = None
_chunks = None
_model = None


def get_embedding_model():
    """Get or initialize the embedding model (lazy loading)."""
    global _model
    if _model is None:
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def build_index(chunks_path: str = "data/processed/health_corpus.pkl",
                index_dir: str = "models/health_faiss") -> None:
    """
    Build FAISS index from health corpus chunks.

    Args:
        chunks_path: Path to the pickled chunks file
        index_dir: Directory to save the FAISS index and metadata
    """
    print("=" * 70)
    print("Building FAISS Index")
    print("=" * 70 + "\n")

    # Load chunks
    print(f"Loading chunks from: {chunks_path}")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks\n")

    # Get embedding model
    model = get_embedding_model()
    print("Model loaded.\n")

    # Extract text content for embedding
    print("Generating embeddings...")
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"Generated embeddings with shape: {embeddings.shape}\n")

    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(embeddings.astype('float32'))
    print(f"Index built with {index.ntotal} vectors\n")

    # Save index and chunks
    index_dir_path = Path(index_dir)
    index_dir_path.mkdir(parents=True, exist_ok=True)

    index_file = index_dir_path / "index.faiss"
    chunks_file = index_dir_path / "chunks.pkl"

    faiss.write_index(index, str(index_file))
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[SAVE] Index saved to: {index_file}")
    print(f"[SAVE] Chunks saved to: {chunks_file}")
    print("\n" + "=" * 70)


def load_index(index_dir: str = "models/health_faiss") -> tuple:
    """
    Load FAISS index and chunks from disk.

    Args:
        index_dir: Directory containing the saved index

    Returns:
        Tuple of (index, chunks)
    """
    global _index, _chunks

    if _index is not None and _chunks is not None:
        return _index, _chunks

    index_dir_path = Path(index_dir)
    index_file = index_dir_path / "index.faiss"
    chunks_file = index_dir_path / "chunks.pkl"

    if not index_file.exists():
        raise FileNotFoundError(
            f"Index not found at {index_file}. Run build_index() first."
        )

    print(f"Loading index from: {index_dir}")
    _index = faiss.read_index(str(index_file))
    with open(chunks_file, "rb") as f:
        _chunks = pickle.load(f)
    print(f"Loaded index with {_index.ntotal} vectors and {len(_chunks)} chunks")

    return _index, _chunks


def retrieve(query: str, k: int = 5, index_dir: str = "models/health_faiss") -> List[Dict]:
    """
    Retrieve top-k most relevant chunks for a query.

    Args:
        query: The search query
        k: Number of top results to return
        index_dir: Directory containing the saved index

    Returns:
        List of chunk dictionaries with added 'score' field
    """
    # Load index and chunks
    index, chunks = load_index(index_dir)

    # Get embedding model and encode query
    model = get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')

    # Search FAISS index
    distances, indices = index.search(query_embedding, k)

    # Prepare results
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = chunks[idx].copy()
        chunk['score'] = float(dist)  # L2 distance (lower is better)
        chunk['rank'] = i + 1
        results.append(chunk)

    return results


def answer_health_question(query: str, k: int = 5) -> str:
    """
    Answer a health question using RAG with grounding.

    Args:
        query: The health question to answer
        k: Number of chunks to retrieve for context

    Returns:
        LLM answer based only on retrieved chunks
    """
    # Retrieve relevant chunks
    chunks = retrieve(query, k=k)

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Chunk {chunk['chunk_id'] + 1}]\n{chunk['content']}"
        )
    context = "\n\n".join(context_parts)

    # Build grounded prompt
    prompt = f"""You are a health information assistant. Answer the following question using ONLY the information provided in the sources below.

IMPORTANT INSTRUCTIONS:
- Use ONLY information from the provided sources
- Cite the source(s) you use (e.g., "According to Source 1...")
- If the sources don't contain enough information to answer the question, respond with: "I don't have reliable information on that in the available health documents."
- Do not add information from your general knowledge
- Be concise and accurate

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

    # Get LLM response
    answer = ask_llm(prompt)
    return answer


def main():
    """Test the RAG system."""
    print("\n" + "=" * 70)
    print("Testing Health RAG System")
    print("=" * 70 + "\n")

    # Build index (skip if already exists)
    index_path = Path("models/health_faiss/index.faiss")
    if not index_path.exists():
        build_index()
    else:
        print("[INFO] Index already exists, skipping build.\n")

    # Test 1: Question that SHOULD be covered by the data
    print("=" * 70)
    print("TEST 1: Question covered by health corpus")
    print("=" * 70)
    question1 = "What are the main causes of mortality globally?"
    print(f"\nQuestion: {question1}\n")
    print("Answer:")
    answer1 = answer_health_question(question1, k=3)
    print(answer1)
    print("\n")

    # Test 2: Question that should NOT be covered by the data
    print("=" * 70)
    print("TEST 2: Question NOT covered by health corpus")
    print("=" * 70)
    question2 = "What is the best recipe for chocolate chip cookies?"
    print(f"\nQuestion: {question2}\n")
    print("Answer:")
    answer2 = answer_health_question(question2, k=3)
    print(answer2)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
