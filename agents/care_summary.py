"""
Care summary aggregator.

Builds a unified summary from multiple agent outputs, preserving factual
data while using LLM to smooth connecting language and translate if needed.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.llm_client import ask_llm


def build_template_summary(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Build template-based summary sections from active branches.

    Args:
        state: Graph state containing active_branches and agent outputs

    Returns:
        Dict mapping branch names to their summary sections
    """
    summaries = {}
    active_branches = state.get("active_branches", [])

    # Health RAG summary
    if "health_literacy_node" in active_branches:
        output = state.get("health_output", state.get("output_text", ""))
        if output:
            summaries["health"] = {
                "title": "Health Information",
                "content": output,
                "type": "verbatim"  # Keep as-is from RAG
            }

    # Nutrition RAG summary
    if "nutrition_node" in active_branches:
        output = state.get("nutrition_output", state.get("output_text", ""))
        if output:
            summaries["nutrition"] = {
                "title": "Nutrition Guidance",
                "content": output,
                "type": "verbatim"  # Keep as-is from RAG
            }

    # Prescription summary
    if "prescription_node" in active_branches:
        prescription_data = state.get("prescription_data", {})
        if prescription_data:
            drug_name = prescription_data.get("drug_name", "Unknown")
            dose = prescription_data.get("dose", "N/A")
            frequency = prescription_data.get("frequency", "N/A")
            duration = prescription_data.get("duration", "N/A")
            confidence = prescription_data.get("confidence_score", 0)
            status = prescription_data.get("status", "pending")

            content = (
                f"Drug: {drug_name}\n"
                f"Dose: {dose}\n"
                f"Frequency: {frequency}\n"
                f"Duration: {duration}\n"
                f"Confidence: {confidence:.2%}\n"
                f"Status: {status}"
            )

            summaries["prescription"] = {
                "title": "Prescription Details",
                "content": content,
                "type": "factual"  # Preserve exact values
            }

    # Reminder/Schedule summary
    if "reminder_node" in active_branches:
        schedule = state.get("medication_schedule", [])
        if schedule:
            schedule_text = "\n".join([
                f"- {s.get('time_of_day', 'N/A')} ({s.get('frequency_label', 'N/A')})"
                for s in schedule
            ])

            summaries["reminder"] = {
                "title": "Medication Schedule",
                "content": schedule_text,
                "type": "factual"
            }

    # Adherence prediction summary
    if "adherence_node" in active_branches:
        risk_score = state.get("adherence_risk", 0.0)
        risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"

        content = f"Miss Risk: {risk_score:.1%} ({risk_level})"

        summaries["adherence"] = {
            "title": "Adherence Prediction",
            "content": content,
            "type": "factual"
        }

    # Journal summary
    if "journal_node" in active_branches:
        journal_output = state.get("journal_output", "")
        if journal_output:
            summaries["journal"] = {
                "title": "Journal Entry",
                "content": journal_output,
                "type": "verbatim"
            }

    # Fitness summary
    if "fitness_node" in active_branches:
        fitness_data = state.get("fitness_data", {})

        # Calories estimation
        calories = fitness_data.get("calories_burned")
        activity = fitness_data.get("activity_type")
        duration = fitness_data.get("duration_minutes")

        # Activity goal
        goal_suggestion = fitness_data.get("goal_suggestion")

        content_parts = []

        if calories and activity:
            content_parts.append(
                f"Activity: {activity}\n"
                f"Duration: {duration} minutes\n"
                f"Calories burned: {calories} kcal"
            )

        if goal_suggestion:
            content_parts.append(f"\n{goal_suggestion}")

        if content_parts:
            summaries["fitness"] = {
                "title": "Fitness Tracking",
                "content": "\n".join(content_parts),
                "type": "mixed"  # Has both factual (calories) and descriptive (goals)
            }

    # Facility finder summary
    if "facility_node" in active_branches:
        facilities = state.get("nearby_facilities", [])
        if facilities:
            facility_list = "\n".join([
                f"- {f.get('name', 'Unknown')} ({f.get('distance', 'N/A')} km, {f.get('type', 'N/A')})"
                for f in facilities[:5]  # Top 5
            ])

            summaries["facility"] = {
                "title": "Nearby Healthcare Facilities",
                "content": facility_list,
                "type": "factual"
            }

    # Scheme/Insurance summary
    if "scheme_node" in active_branches:
        scheme_data = state.get("scheme_eligibility", {})
        if scheme_data:
            scheme_name = scheme_data.get("scheme_name", "Unknown")
            eligible = scheme_data.get("eligible", False)
            coverage = scheme_data.get("coverage_amount", "N/A")

            content = (
                f"Scheme: {scheme_name}\n"
                f"Eligibility: {'Yes' if eligible else 'No'}\n"
                f"Coverage: {coverage}"
            )

            summaries["scheme"] = {
                "title": "Health Insurance",
                "content": content,
                "type": "factual"
            }

    # Food recognition summary
    if "food_node" in active_branches:
        food_data = state.get("food_prediction", {})
        predictions = food_data.get("predictions", [])
        needs_confirmation = food_data.get("needs_confirmation", False)

        if predictions:
            top_prediction = predictions[0]
            content = f"Identified: {top_prediction[0]} (confidence: {top_prediction[1]:.1%})"

            if needs_confirmation:
                content += "\n⚠️ Low confidence - please confirm"

            # Add nutrition lookup if available
            nutrition_info = food_data.get("nutrition_info")
            if nutrition_info:
                content += f"\n{nutrition_info.get('notes', '')}"

            summaries["food"] = {
                "title": "Food Recognition",
                "content": content,
                "type": "mixed"
            }

    return summaries


def smooth_and_translate(
    summaries: Dict[str, Dict[str, str]],
    detected_language: str = "en"
) -> str:
    """
    Use LLM to smooth connecting language and translate if needed.

    Preserves factual fields (doses, numbers, eligibility) while making
    the overall text flow naturally.

    Args:
        summaries: Dict of summary sections from build_template_summary
        detected_language: Target language code (e.g., 'en', 'es', 'hi')

    Returns:
        Smoothed and translated summary string
    """
    if not summaries:
        return "No information to summarize."

    # Build structured summary text
    summary_parts = []

    for key, section in summaries.items():
        summary_parts.append(f"## {section['title']}")
        summary_parts.append(section['content'])
        summary_parts.append("")  # Blank line between sections

    raw_summary = "\n".join(summary_parts)

    # Prepare prompt for LLM
    language_names = {
        "en": "English",
        "es": "Spanish",
        "hi": "Hindi",
        "zh": "Chinese",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
        "ar": "Arabic",
        "ja": "Japanese",
        "ko": "Korean"
    }

    target_language = language_names.get(detected_language, "English")

    prompt = f"""You are helping create a care summary for a patient. Below is structured information from multiple healthcare agents.

IMPORTANT INSTRUCTIONS:
1. Preserve ALL factual information EXACTLY as written:
   - Drug names, doses, frequencies, durations
   - Numbers (calories, distances, percentages, times)
   - Eligibility statuses, scheme names
   - Facility names and locations

2. Improve the connecting language to make it flow naturally:
   - Add smooth transitions between sections
   - Use appropriate medical/health terminology
   - Make it sound cohesive rather than robotic

3. Translate the CONNECTING language (not the factual data) to {target_language}
   - Keep drug names, facility names, technical terms in original language
   - Translate explanatory text, section headers, transitions

4. Keep the summary concise and patient-friendly
5. Maintain the section structure with clear headings

RAW SUMMARY:
{raw_summary}

SMOOTHED SUMMARY in {target_language}:"""

    try:
        smoothed = ask_llm(prompt)
        return smoothed.strip()
    except Exception as e:
        print(f"[WARN] LLM smoothing failed: {e}. Returning raw summary.")
        return raw_summary


def summarize(state: Dict[str, Any]) -> str:
    """
    Generate a unified care summary from multiple agent outputs.

    Main entry point that builds template summaries and smooths with LLM.

    Args:
        state: Graph state containing:
            - active_branches: List of node names that executed
            - Various agent outputs (health_output, prescription_data, etc.)
            - detected_language: Target language for output (default 'en')

    Returns:
        Final smoothed and translated care summary string
    """
    # Build template-based summaries
    summaries = build_template_summary(state)

    # Get target language
    detected_language = state.get("detected_language", "en")

    # Smooth and translate
    final_summary = smooth_and_translate(summaries, detected_language)

    return final_summary


def main():
    """Test care summary with multiple branches."""
    print("=" * 70)
    print("Testing Care Summary System")
    print("=" * 70 + "\n")

    # Test case 1: Single branch (nutrition)
    print("TEST 1: Single branch (Nutrition only)")
    print("-" * 70)

    state1 = {
        "active_branches": ["nutrition_node"],
        "nutrition_output": (
            "According to Source 1 (Dietary Guidelines), good sources of protein include "
            "lean meats, poultry, fish, eggs, dairy products, legumes, nuts, and seeds. "
            "Adults should aim for 0.8 grams of protein per kilogram of body weight daily."
        ),
        "detected_language": "en"
    }

    summary1 = summarize(state1)
    print(summary1)
    print("\n")

    # Test case 2: Multiple branches (nutrition + fitness)
    print("=" * 70)
    print("TEST 2: Multiple branches (Nutrition + Fitness)")
    print("-" * 70)

    state2 = {
        "active_branches": ["nutrition_node", "fitness_node"],
        "nutrition_output": (
            "According to Source 2 (Nutrition Handbook), vitamin D is crucial for bone health "
            "and calcium absorption. It can be obtained from sunlight exposure, fortified dairy "
            "products, fatty fish like salmon and mackerel, and egg yolks."
        ),
        "fitness_data": {
            "activity_type": "running_6mph",
            "duration_minutes": 30,
            "calories_burned": 350.0,
            "goal_suggestion": (
                "You're moderately active (daily average: 8500 steps, 55 active minutes). "
                "Great progress! Goal suggestion: Push toward 10,000 steps and 60+ minutes "
                "of activity. Consider adding strength training 2-3 times per week."
            )
        },
        "detected_language": "en"
    }

    summary2 = summarize(state2)
    print(summary2)
    print("\n")

    # Test case 3: Complex multi-branch (nutrition + prescription + adherence)
    print("=" * 70)
    print("TEST 3: Complex multi-branch (Nutrition + Prescription + Adherence)")
    print("-" * 70)

    state3 = {
        "active_branches": ["nutrition_node", "prescription_node", "adherence_node", "reminder_node"],
        "nutrition_output": (
            "According to Source 3 (Heart Health Guidelines), omega-3 fatty acids support "
            "heart and brain health. They are found in fatty fish (salmon, sardines, mackerel), "
            "walnuts, flaxseeds, chia seeds, and algae-based supplements."
        ),
        "prescription_data": {
            "drug_name": "Amoxicillin",
            "dose": "500 mg",
            "frequency": "Three times daily",
            "duration": "7 days",
            "confidence_score": 0.87,
            "status": "confirmed"
        },
        "medication_schedule": [
            {"time_of_day": "08:00", "frequency_label": "Morning"},
            {"time_of_day": "14:00", "frequency_label": "Afternoon"},
            {"time_of_day": "20:00", "frequency_label": "Night"}
        ],
        "adherence_risk": 0.15,
        "detected_language": "en"
    }

    summary3 = summarize(state3)
    print(summary3)
    print("\n")

    # Test case 4: Facility search + scheme info
    print("=" * 70)
    print("TEST 4: Facility Search + Health Insurance")
    print("-" * 70)

    state4 = {
        "active_branches": ["facility_node", "scheme_node"],
        "nearby_facilities": [
            {"name": "City General Hospital", "distance": 2.3, "type": "Hospital"},
            {"name": "MedPlus Pharmacy", "distance": 0.8, "type": "Pharmacy"},
            {"name": "HealthCare Clinic", "distance": 1.5, "type": "Clinic"}
        ],
        "scheme_eligibility": {
            "scheme_name": "Ayushman Bharat",
            "eligible": True,
            "coverage_amount": "₹5,00,000 per family per year"
        },
        "detected_language": "en"
    }

    summary4 = summarize(state4)
    print(summary4)
    print("\n")

    print("=" * 70)
    print("KEY FEATURES DEMONSTRATED:")
    print("  1. Template-based summaries preserve factual data")
    print("  2. LLM smooths connecting language only")
    print("  3. Multiple branches combine into unified summary")
    print("  4. Empty sections are excluded")
    print("  5. Language detection supported (detected_language)")
    print("=" * 70)


if __name__ == "__main__":
    main()
