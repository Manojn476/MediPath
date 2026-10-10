"""Test script for intent classification."""
import sys
from pathlib import Path

# Add parent directory to path to import agents module
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.intake import classify_intent_fast, classify_intent


def main():
    print("\n" + "=" * 70)
    print("Intent Classification Test Suite")
    print("=" * 70 + "\n")

    # Test cases covering each category
    test_cases = [
        {
            "name": "General Health Query",
            "input": "What are the symptoms of diabetes?",
            "state": {"input_text": "What are the symptoms of diabetes?"},
            "expected": "health"
        },
        {
            "name": "Prescription Refill",
            "input": "I need to refill my blood pressure medication",
            "state": {"input_text": "I need to refill my blood pressure medication"},
            "expected": "prescription"
        },
        {
            "name": "Facility Search",
            "input": "Find hospitals near me in Delhi",
            "state": {"input_text": "Find hospitals near me in Delhi"},
            "expected": "facility"
        },
        {
            "name": "Health Insurance Scheme",
            "input": "How do I apply for Ayushman Bharat scheme?",
            "state": {"input_text": "How do I apply for Ayushman Bharat scheme?"},
            "expected": "scheme"
        },
        {
            "name": "Research Paper Query",
            "input": "Show me recent research on covid vaccines",
            "state": {"input_text": "Show me recent research on covid vaccines"},
            "expected": "journal"
        },
        {
            "name": "Nutrition Information",
            "input": "What are good sources of vitamin D?",
            "state": {"input_text": "What are good sources of vitamin D?"},
            "expected": "nutrition"
        },
        {
            "name": "Food Recipe",
            "input": "Can you give me a recipe for chicken curry?",
            "state": {"input_text": "Can you give me a recipe for chicken curry?"},
            "expected": "food"
        },
        {
            "name": "Fitness Exercise",
            "input": "Best cardio exercises for weight loss",
            "state": {"input_text": "Best cardio exercises for weight loss"},
            "expected": "fitness"
        },
        {
            "name": "Maternal Health",
            "input": "What caused the maternal mortality decline globally?",
            "state": {"input_text": "What caused the maternal mortality decline globally?"},
            "expected": "health"
        },
        {
            "name": "Ambiguous/Unclear",
            "input": "Hello, how are you today?",
            "state": {"input_text": "Hello, how are you today?"},
            "expected": "unclear"
        },
        {
            "name": "Ambiguous - Should I exercise?",
            "input": "Should I start doing something active?",
            "state": {"input_text": "Should I start doing something active?"},
            "expected": "fitness"  # LLM should classify as fitness
        },
        {
            "name": "Ambiguous - Feeling tired",
            "input": "I've been feeling really tired lately",
            "state": {"input_text": "I've been feeling really tired lately"},
            "expected": "health"  # LLM should classify as health
        },
    ]

    print("=" * 70)
    print("PART 1: Fast Classifier Only")
    print("=" * 70 + "\n")

    # Run tests with fast classifier
    fast_results = []
    for i, test in enumerate(test_cases, 1):
        result = classify_intent_fast(test["state"])
        is_correct = result == test["expected"]
        fast_results.append(is_correct)

        status = "PASS" if is_correct else "FAIL"
        emoji = "[+]" if is_correct else "[X]"

        print(f"{emoji} Test {i}: {test['name']}")
        print(f"    Input: \"{test['input']}\"")
        print(f"    Expected: {test['expected']}")
        print(f"    Got: {result}")
        if not is_correct:
            print(f"    Status: {status}")
        print()

    # Summary for fast classifier
    correct = sum(fast_results)
    total = len(test_cases)
    accuracy = (correct / total) * 100
    print(f"Fast Classifier Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print()

    print("=" * 70)
    print("PART 2: Combined Classifier (Fast + LLM Fallback)")
    print("=" * 70 + "\n")

    # Run tests with combined classifier
    combined_results = []
    for i, test in enumerate(test_cases, 1):
        result = classify_intent(test["state"])
        is_correct = result == test["expected"]
        combined_results.append(is_correct)

        status = "PASS" if is_correct else "FAIL"
        emoji = "[+]" if is_correct else "[X]"

        print(f"{emoji} Test {i}: {test['name']}")
        print(f"    Input: \"{test['input']}\"")
        print(f"    Expected: {test['expected']}")
        print(f"    Got: {result}")
        if not is_correct:
            print(f"    Status: {status}")
        print()

    # Summary for combined classifier
    correct = sum(combined_results)
    total = len(test_cases)
    accuracy = (correct / total) * 100
    print(f"Combined Classifier Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
