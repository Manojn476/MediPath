"""Test script for the LLM client."""
import sys
from pathlib import Path

# Add parent directory to path to import agents module
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_client import ask_llm


def main():
    print("Testing LLM client...")
    print("-" * 50)

    question = "What is a balanced diet in one sentence?"
    print(f"Question: {question}\n")

    response = ask_llm(question)
    print(f"Response: {response}")
    print("-" * 50)


if __name__ == "__main__":
    main()
