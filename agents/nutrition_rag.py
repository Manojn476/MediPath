"""
Nutrition RAG (Retrieval-Augmented Generation) system.

Uses FAISS and sentence embeddings to retrieve relevant nutrition and
dietary guideline information from the chunked corpus.
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


def build_index(chunks_path: str = "data/processed/nutrition_corpus.pkl",
                index_dir: str = "models/nutrition_faiss") -> None:
    """
    Build FAISS index from nutrition corpus chunks.

    Args:
        chunks_path: Path to the pickled chunks file
        index_dir: Directory to save the FAISS index and metadata
    """
    print("=" * 70)
    print("Building Nutrition FAISS Index")
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


def load_index(index_dir: str = "models/nutrition_faiss") -> tuple:
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

    print(f"Loading nutrition index from: {index_dir}")
    _index = faiss.read_index(str(index_file))
    with open(chunks_file, "rb") as f:
        _chunks = pickle.load(f)
    print(f"Loaded index with {_index.ntotal} vectors and {len(_chunks)} chunks")

    return _index, _chunks


def retrieve(query: str, k: int = 5, index_dir: str = "models/nutrition_faiss") -> List[Dict]:
    """
    Retrieve top-k most relevant chunks for a nutrition query.

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


def answer_nutrition_question(query: str, k: int = 5) -> str:
    """
    Answer a nutrition question using RAG with grounding.

    Args:
        query: The nutrition question to answer
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
    prompt = f"""You are a nutrition information assistant. Answer the following question using ONLY the information provided in the sources below.

IMPORTANT INSTRUCTIONS:
- Use ONLY information from the provided sources
- Cite the source(s) you use (e.g., "According to Source 1...")
- If the sources don't contain enough information to answer the question, respond with: "I don't have reliable information on that in the available nutrition documents."
- Do not add information from your general knowledge
- Provide nutrition and dietary information as general guidance, never tied to specific medical conditions
- Do not diagnose, treat, or provide medical advice
- Be concise and accurate

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

    # Get LLM response
    answer = ask_llm(prompt)
    return answer


def main():
    """Test the nutrition RAG system."""
    print("\n" + "=" * 70)
    print("Testing Nutrition RAG System")
    print("=" * 70 + "\n")

    # Build index (skip if already exists)
    index_path = Path("models/nutrition_faiss/index.faiss")
    if not index_path.exists():
        print("[INFO] Index not found. Creating sample nutrition corpus...\n")

        # Create sample nutrition data for testing
        sample_chunks = [
            {
                "content": "Protein is essential for building and repairing tissues. Good sources include lean meats, poultry, fish, eggs, dairy products, legumes, nuts, and seeds. Adults should aim for 0.8 grams of protein per kilogram of body weight daily.",
                "source": "Dietary Guidelines",
                "chunk_id": 0
            },
            {
                "content": "Vitamin D is crucial for bone health and calcium absorption. It can be obtained from sunlight exposure, fortified dairy products, fatty fish like salmon and mackerel, and egg yolks. Many people may need supplementation, especially in winter months.",
                "source": "Nutrition Handbook",
                "chunk_id": 1
            },
            {
                "content": "Fiber aids in digestion and helps maintain healthy blood sugar levels. Whole grains, fruits, vegetables, legumes, and nuts are excellent sources. Adults should aim for 25-30 grams of fiber per day.",
                "source": "Dietary Guidelines",
                "chunk_id": 2
            },
            {
                "content": "Iron is essential for oxygen transport in the blood. Good sources include red meat, poultry, seafood, beans, dark leafy greens, and fortified cereals. Vitamin C enhances iron absorption, so pair iron-rich foods with citrus fruits or tomatoes.",
                "source": "Nutrition Handbook",
                "chunk_id": 3
            },
            {
                "content": "Omega-3 fatty acids support heart and brain health. They are found in fatty fish (salmon, sardines, mackerel), walnuts, flaxseeds, chia seeds, and algae-based supplements. Aim for 2 servings of fatty fish per week.",
                "source": "Heart Health Guidelines",
                "chunk_id": 4
            },
            {
                "content": "Calcium is vital for strong bones and teeth. Sources include dairy products, fortified plant milks, leafy greens, almonds, and sardines with bones. Adults need 1000-1200mg daily depending on age.",
                "source": "Bone Health Guide",
                "chunk_id": 5
            },
            {
                "content": "A balanced diet includes a variety of fruits and vegetables, whole grains, lean proteins, and healthy fats. Try to fill half your plate with fruits and vegetables, one quarter with whole grains, and one quarter with protein.",
                "source": "MyPlate Guidelines",
                "chunk_id": 6
            },
            {
                "content": "Hydration is essential for all bodily functions. Water is the best choice for staying hydrated. Adults should aim for 8-10 cups of water daily, adjusting for activity level, climate, and individual needs.",
                "source": "Hydration Guidelines",
                "chunk_id": 7
            },
        ]

        # Save sample chunks
        corpus_path = Path("data/processed/nutrition_corpus.pkl")
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, "wb") as f:
            pickle.dump(sample_chunks, f)
        print(f"[SAVE] Created sample nutrition corpus with {len(sample_chunks)} chunks\n")

        # Build index
        build_index()
    else:
        print("[INFO] Index already exists, skipping build.\n")

    # Test 1: Question that SHOULD be covered by the data
    print("=" * 70)
    print("TEST 1: Question covered by nutrition corpus")
    print("=" * 70)
    question1 = "What are good sources of protein?"
    print(f"\nQuestion: {question1}\n")
    print("Answer:")
    answer1 = answer_nutrition_question(question1, k=3)
    print(answer1)
    print("\n")

    # Test 2: Question about vitamin D
    print("=" * 70)
    print("TEST 2: Question about vitamin D")
    print("=" * 70)
    question2 = "How can I get enough vitamin D?"
    print(f"\nQuestion: {question2}\n")
    print("Answer:")
    answer2 = answer_nutrition_question(question2, k=3)
    print(answer2)
    print("\n")

    # Test 3: Question that should NOT be covered by the data
    print("=" * 70)
    print("TEST 3: Question NOT covered by nutrition corpus")
    print("=" * 70)
    question3 = "What is the best recipe for chocolate cake?"
    print(f"\nQuestion: {question3}\n")
    print("Answer:")
    answer3 = answer_nutrition_question(question3, k=3)
    print(answer3)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
