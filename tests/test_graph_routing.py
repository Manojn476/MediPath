"""Test script for the LangGraph workflow with intent routing."""
import sys
from pathlib import Path

# Add parent directory to path to import agents module
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.graph import graph


def main():
    print("=" * 70)
    print("Testing LangGraph with Intent-Based Routing")
    print("=" * 70)

    # Test 1: Health-related message (should go to health_literacy_node)
    print("\n" + "=" * 70)
    print("TEST 1: Health Question (should route to health_literacy_node)")
    print("=" * 70)

    health_question = "What are the main causes of maternal mortality?"
    print(f"\nInput: {health_question}\n")

    initial_state_health = {
        "input_text": health_question,
        "output_text": "",
        "audio_path": None,
        "intent": None
    }

    print("Invoking graph...\n")
    final_state_health = graph.invoke(initial_state_health)

    print("FINAL STATE:")
    print(f"  input_text: {final_state_health['input_text']}")
    print(f"  intent: {final_state_health['intent']}")
    print(f"  output_text: {final_state_health['output_text'][:200]}...")

    # Test 2: Facility-related message (should go to not_implemented_node)
    print("\n" + "=" * 70)
    print("TEST 2: Facility Query (should route to not_implemented_node)")
    print("=" * 70)

    facility_question = "Find hospitals near me in Mumbai"
    print(f"\nInput: {facility_question}\n")

    initial_state_facility = {
        "input_text": facility_question,
        "output_text": "",
        "audio_path": None,
        "intent": None
    }

    print("Invoking graph...\n")
    final_state_facility = graph.invoke(initial_state_facility)

    print("FINAL STATE:")
    print(f"  input_text: {final_state_facility['input_text']}")
    print(f"  intent: {final_state_facility['intent']}")
    print(f"  output_text: {final_state_facility['output_text']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
