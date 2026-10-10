"""
Intent classification for routing health/wellness queries.

Uses simple keyword and pattern matching for fast intent detection,
with LLM fallback for ambiguous cases.
"""
import re
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.llm_client import ask_llm


def classify_intent_fast(state: Dict[str, Any]) -> str:
    """
    Classify user intent using keyword/pattern matching.

    Args:
        state: Dictionary containing:
            - input_text: The user's query text
            - audio_path: Optional path to audio file
            - image_path: Optional path to image file (for future use)

    Returns:
        One of: "health", "prescription", "facility", "scheme", "journal",
                "nutrition", "food", "fitness", "unclear"
    """
    input_text = state.get("input_text", "").lower()
    image_path = state.get("image_path", "")

    # Check image/file paths for hints
    if image_path:
        if any(word in image_path.lower() for word in ["prescription", "rx", "medicine", "pill"]):
            return "prescription"
        if any(word in image_path.lower() for word in ["food", "meal", "dish", "recipe"]):
            return "food"

    # Prescription-related keywords
    prescription_patterns = [
        r'\b(prescription|rx|medicine|medication|drug|pill|tablet|dosage|refill)\b',
        r'\b(antibiotic|painkiller|insulin|inhaler)\b',
    ]
    for pattern in prescription_patterns:
        if re.search(pattern, input_text):
            return "prescription"

    # Facility-related keywords (hospitals, clinics, pharmacies)
    facility_patterns = [
        r'\b(hospital|clinic|doctor|physician|pharmacy|chemist|dispensary)\b',
        r'\bnear me\b',
        r'\b(find|locate|where|nearest|closest)\b.*(hospital|clinic|doctor|pharmacy)',
        r'\b(hospital|clinic).*(near|around|in)\b',
    ]
    for pattern in facility_patterns:
        if re.search(pattern, input_text):
            return "facility"

    # Health scheme/insurance keywords
    scheme_patterns = [
        r'\b(insurance|scheme|policy|coverage|claim|ayushman|mediclaim)\b',
        r'\b(health card|eligibility|enroll|benefit)\b',
        r'\bhow to (apply|register|enroll)\b.*(insurance|scheme)',
    ]
    for pattern in scheme_patterns:
        if re.search(pattern, input_text):
            return "scheme"

    # Research/journal keywords
    journal_patterns = [
        r'\b(research|study|journal|paper|publication|clinical trial)\b',
        r'\b(peer.?review|meta.?analysis|systematic review)\b',
        r'\b(pubmed|doi|citation|abstract)\b',
    ]
    for pattern in journal_patterns:
        if re.search(pattern, input_text):
            return "journal"

    # Nutrition-related keywords
    nutrition_patterns = [
        r'\b(nutrition|nutrient|vitamin|mineral|supplement|deficiency)\b',
        r'\b(calorie|protein|carb|fat|fiber|micronutrient)\b',
        r'\b(balanced diet|dietary|nutritional value)\b',
    ]
    for pattern in nutrition_patterns:
        if re.search(pattern, input_text):
            return "nutrition"

    # Food-specific keywords (meals, recipes, ingredients)
    food_patterns = [
        r'\b(recipe|meal|dish|cook|cuisine|ingredient|food)\b',
        r'\b(breakfast|lunch|dinner|snack)\b',
        r'\b(vegetarian|vegan|keto|paleo)\b',
        r'\bhow to (make|cook|prepare)\b',
    ]
    for pattern in food_patterns:
        if re.search(pattern, input_text):
            return "food"

    # Fitness/exercise keywords
    fitness_patterns = [
        r'\b(fitness|exercise|workout|gym|training|yoga)\b',
        r'\b(cardio|strength|aerobic|running|walking|jogging)\b',
        r'\b(weight loss|muscle|endurance|flexibility)\b',
    ]
    for pattern in fitness_patterns:
        if re.search(pattern, input_text):
            return "fitness"

    # General health keywords (catch-all for health queries)
    health_patterns = [
        r'\b(health|disease|symptom|condition|treatment|diagnosis|illness)\b',
        r'\b(covid|diabetes|hypertension|cancer|asthma|fever|pain)\b',
        r'\b(mortality|morbidity|epidemic|pandemic)\b',
        r'\b(maternal|child|infant|prenatal|pregnancy)\b',
        r'\b(mental health|depression|anxiety|stress)\b',
    ]
    for pattern in health_patterns:
        if re.search(pattern, input_text):
            return "health"

    # If nothing matches, return unclear
    return "unclear"


def classify_intent_llm(state: Dict[str, Any]) -> str:
    """
    Classify user intent using LLM when fast classifier is uncertain.

    Args:
        state: Dictionary containing input_text

    Returns:
        One of: "health", "prescription", "facility", "scheme", "journal",
                "nutrition", "food", "fitness", "unclear"
    """
    input_text = state.get("input_text", "")

    prompt = f"""You are an intent classifier for a health/wellness assistant.

Classify the following user message into EXACTLY ONE of these categories:
- health: General health questions, diseases, symptoms, conditions, treatments
- prescription: Medication, prescriptions, dosage, refills
- facility: Hospitals, clinics, doctors, pharmacies, location-based queries
- scheme: Health insurance, government schemes, benefits, eligibility
- journal: Research papers, clinical trials, scientific studies
- nutrition: Nutrients, vitamins, minerals, dietary requirements
- food: Recipes, meals, cooking, ingredients
- fitness: Exercise, workouts, physical activity, training
- unclear: Cannot be classified into any of the above

USER MESSAGE: "{input_text}"

Respond with ONLY the category name, nothing else. Choose the best matching category.

CATEGORY:"""

    try:
        response = ask_llm(prompt).strip().lower()

        # Extract just the category if the model added extra text
        valid_categories = [
            "health", "prescription", "facility", "scheme",
            "journal", "nutrition", "food", "fitness", "unclear"
        ]

        for category in valid_categories:
            if category in response:
                return category

        # If no valid category found, default to unclear
        return "unclear"

    except Exception as e:
        print(f"LLM classification error: {e}")
        return "unclear"


def classify_intent(state: Dict[str, Any]) -> str:
    """
    Classify user intent using fast classifier with LLM fallback.

    Tries keyword-based classification first. If result is "unclear",
    falls back to LLM-based classification.

    Args:
        state: Dictionary containing input_text

    Returns:
        One of: "health", "prescription", "facility", "scheme", "journal",
                "nutrition", "food", "fitness", "unclear"
    """
    # Try fast classifier first
    intent = classify_intent_fast(state)

    # If unclear, use LLM fallback
    if intent == "unclear":
        intent = classify_intent_llm(state)

    return intent


def main():
    """Test the intent classifier."""
    print("=" * 70)
    print("Testing Intent Classification")
    print("=" * 70 + "\n")

    # Test cases
    test_cases = [
        {"input_text": "What are the symptoms of diabetes?", "expected": "health"},
        {"input_text": "I need to refill my blood pressure medication", "expected": "prescription"},
        {"input_text": "Find hospitals near me in Delhi", "expected": "facility"},
        {"input_text": "How do I apply for Ayushman Bharat scheme?", "expected": "scheme"},
        {"input_text": "Show me recent research on covid vaccines", "expected": "journal"},
        {"input_text": "What are good sources of vitamin D?", "expected": "nutrition"},
        {"input_text": "Can you give me a recipe for chicken curry?", "expected": "food"},
        {"input_text": "Best cardio exercises for weight loss", "expected": "fitness"},
        {"input_text": "What caused the maternal mortality decline?", "expected": "health"},
        {"input_text": "Hello, how are you today?", "expected": "unclear"},
        {"input_text": "nearest pharmacy that's open now", "expected": "facility"},
    ]

    correct = 0
    for i, test in enumerate(test_cases, 1):
        state = {"input_text": test["input_text"]}
        result = classify_intent_fast(state)
        is_correct = result == test["expected"]
        correct += is_correct

        status = "[PASS]" if is_correct else "[FAIL]"
        print(f"{status} Test {i}")
        print(f"  Input: {test['input_text']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Got: {result}")
        print()

    print("=" * 70)
    print(f"Results: {correct}/{len(test_cases)} correct")
    print("=" * 70)


if __name__ == "__main__":
    main()
