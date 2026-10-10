"""Test script for the LangGraph workflow."""
import sys
from pathlib import Path

# Add parent directory to path to import agents module
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.graph import graph


def main():
    print("=" * 70)
    print("Testing LangGraph with Health RAG")
    print("=" * 70)

    # Test with a health question (text input)
    print("\n[TEST] Health question with RAG-based answer")
    print("-" * 70)

    question = "What were the main findings about maternal mortality in the WHO data?"
    print(f"Question: {question}\n")

    initial_state = {
        "input_text": question,
        "output_text": "",
        "audio_path": None
    }

    print("Invoking graph: START -> transcribe_node -> health_literacy_node -> END\n")
    final_state = graph.invoke(initial_state)

    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Input: {final_state['input_text'][:100]}...")
    print(f"\nOutput:\n{final_state['output_text']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
