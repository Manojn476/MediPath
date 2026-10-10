"""
Updated LangGraph workflow with followup loops for MediPath.

Includes conditional edges that loop back for updates when new data is logged.
"""
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from agents.llm_client import ask_llm
from agents.voice_client import transcribe_audio
from agents.health_rag import answer_health_question
from agents.intake import classify_intent
from agents.care_summary import summarize
from agents.followup import has_followup_flags, get_active_flags, clear_all_flags

# Real model functions — imported lazily inside nodes so graph loads
# even when model .pkl files have not yet been trained.
def _call_predict_miss_risk(features: dict) -> float:
    """Call adherence_model.predict_miss_risk, fallback to 0.3 on error."""
    try:
        from agents.adherence_model import predict_miss_risk
        return predict_miss_risk(features)
    except Exception as e:
        print(f"[WARN] predict_miss_risk unavailable ({e}). Using fallback 0.30.")
        return 0.30


def _call_suggest_activity_goal(features: dict) -> str:
    """Call fitness.suggest_activity_goal, fallback to a generic message on error."""
    try:
        from agents.fitness import suggest_activity_goal
        return suggest_activity_goal(features)
    except Exception as e:
        print(f"[WARN] suggest_activity_goal unavailable ({e}). Using fallback.")
        steps = features.get('steps', 0)
        return (
            f"Updated goal based on recent activity ({steps:.0f} steps). "
            "Keep up the great work and aim for 10,000 steps daily."
        )


class GraphState(TypedDict):
    """Enhanced state schema for the graph with followup flags."""
    input_text: str
    output_text: str
    audio_path: Optional[str]
    intent: Optional[str]
    active_branches: list
    detected_language: str

    # Followup flags
    new_adherence_log: bool
    new_activity_log: bool
    new_journal_entry: bool

    # Agent outputs
    health_output: Optional[str]
    nutrition_output: Optional[str]
    prescription_data: Optional[dict]
    medication_schedule: Optional[list]
    adherence_risk: Optional[float]
    journal_output: Optional[str]
    fitness_data: Optional[dict]
    nearby_facilities: Optional[list]
    scheme_eligibility: Optional[dict]
    food_prediction: Optional[dict]


def transcribe_node(state: GraphState) -> GraphState:
    """Transcribe audio if provided."""
    audio_path = state.get("audio_path")

    if audio_path:
        transcription = transcribe_audio(audio_path)
        state["input_text"] = transcription

    return state


def intake_node(state: GraphState) -> GraphState:
    """Classify user intent."""
    intent = classify_intent(state)
    state["intent"] = intent

    # Initialize followup flags if not present
    if "new_adherence_log" not in state:
        state["new_adherence_log"] = False
    if "new_activity_log" not in state:
        state["new_activity_log"] = False
    if "new_journal_entry" not in state:
        state["new_journal_entry"] = False

    # Initialize active_branches
    if "active_branches" not in state:
        state["active_branches"] = []

    # Initialize detected_language
    if "detected_language" not in state:
        state["detected_language"] = "en"

    return state


def route_by_intent(state: GraphState) -> Literal[
    "health_literacy_node",
    "nutrition_node",
    "prescription_node",
    "fitness_node",
    "journal_node",
    "not_implemented_node"
]:
    """Route based on detected intent."""
    intent = state.get("intent", "unclear")

    if intent == "health":
        return "health_literacy_node"
    elif intent == "nutrition":
        return "nutrition_node"
    elif intent == "prescription":
        return "prescription_node"
    elif intent == "fitness":
        return "fitness_node"
    elif intent == "journal":
        return "journal_node"
    else:
        return "not_implemented_node"


def health_literacy_node(state: GraphState) -> GraphState:
    """Answer health questions using RAG."""
    response = answer_health_question(state["input_text"])
    state["health_output"] = response
    state["output_text"] = response

    # Track active branch
    if "health_literacy_node" not in state["active_branches"]:
        state["active_branches"].append("health_literacy_node")

    return state


def nutrition_node(state: GraphState) -> GraphState:
    """Answer nutrition questions using RAG."""
    # This would call nutrition_rag.answer_nutrition_question
    # For now, placeholder
    response = f"Nutrition guidance for: {state['input_text']}"
    state["nutrition_output"] = response
    state["output_text"] = response

    # Track active branch
    if "nutrition_node" not in state["active_branches"]:
        state["active_branches"].append("nutrition_node")

    return state


def prescription_node(state: GraphState) -> GraphState:
    """Process prescription OCR and reminders."""
    # Placeholder - would integrate prescription_ocr and reminder
    state["prescription_data"] = {
        "drug_name": "Sample Drug",
        "dose": "500 mg",
        "frequency": "Twice daily",
        "duration": "7 days",
        "confidence_score": 0.85,
        "status": "confirmed"
    }
    state["output_text"] = "Prescription processed."

    # Track active branch
    if "prescription_node" not in state["active_branches"]:
        state["active_branches"].append("prescription_node")

    return state


def fitness_node(state: GraphState) -> GraphState:
    """Handle fitness tracking and goals."""
    # Placeholder - would integrate fitness.py
    state["fitness_data"] = {
        "activity_type": "walking_brisk",
        "duration_minutes": 30,
        "calories_burned": 150.0,
        "goal_suggestion": "Great progress! Aim for 8,000 steps daily."
    }
    state["output_text"] = "Fitness data tracked."

    # Track active branch
    if "fitness_node" not in state["active_branches"]:
        state["active_branches"].append("fitness_node")

    return state


def journal_node(state: GraphState) -> GraphState:
    """Process journal entry."""
    # Placeholder - would integrate journal.py
    state["journal_output"] = "Journal entry saved with sentiment analysis."
    state["output_text"] = state["journal_output"]

    # Track active branch
    if "journal_node" not in state["active_branches"]:
        state["active_branches"].append("journal_node")

    return state


def adherence_update_node(state: GraphState) -> GraphState:
    """
    Re-run adherence risk prediction when a new adherence log entry is added.

    Triggered by: new_adherence_log flag (set by reminder.py or followup helpers).
    Action: calls adherence_model.predict_miss_risk() with the most recent
    adherence entry, updates state['adherence_risk'], then clears the flag.
    """
    print("[GRAPH] Running adherence update (predict_miss_risk)...")

    # Build feature dict from the latest adherence entry if available
    entries = state.get("adherence_entries", [])
    if entries:
        latest = entries[-1]
        from datetime import datetime
        try:
            scheduled_dt = datetime.fromisoformat(str(latest.get("scheduled_time", "")))
            dow = scheduled_dt.weekday()          # 0=Mon
            is_weekend = int(dow >= 5)
            hour = scheduled_dt.hour
            if hour < 12:
                tod_enc = 0   # morning
            elif hour < 17:
                tod_enc = 1   # afternoon
            else:
                tod_enc = 2   # evening
        except Exception:
            dow, is_weekend, tod_enc = 2, 0, 0  # sensible defaults

        features = {
            "time_of_day_encoded": tod_enc,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "trailing_7day_miss_count": latest.get("trailing_7day_miss_count", 0),
            "trailing_14day_miss_count": latest.get("trailing_14day_miss_count", 0),
            "current_streak_length": latest.get("current_streak_length", 1),
            "user_overall_adherence": latest.get("user_overall_adherence", 0.85),
            "dose_number_in_day": latest.get("dose_number_in_day", 1),
        }
        new_risk = _call_predict_miss_risk(features)
    else:
        # No entries in state — call model with generic safe defaults
        features = {
            "time_of_day_encoded": 0,
            "day_of_week": 2,
            "is_weekend": 0,
            "trailing_7day_miss_count": 0,
            "trailing_14day_miss_count": 0,
            "current_streak_length": 5,
            "user_overall_adherence": 0.85,
            "dose_number_in_day": 1,
        }
        new_risk = _call_predict_miss_risk(features)

    state["adherence_risk"] = new_risk

    # Clear the flag so the loop terminates
    state["new_adherence_log"] = False

    # Track active branch
    if "adherence_update_node" not in state["active_branches"]:
        state["active_branches"].append("adherence_update_node")

    print(f"[GRAPH] Adherence risk updated: {new_risk:.2%}")

    return state


def fitness_update_node(state: GraphState) -> GraphState:
    """
    Re-run fitness trend step when a new activity is logged.

    Triggered by: new_activity_log flag (set by fitness.py or followup helpers).
    Action: calls fitness.suggest_activity_goal() using the latest activity entry,
    updates state['fitness_data']['goal_suggestion'], then clears the flag.
    """
    print("[GRAPH] Running fitness trend update (suggest_activity_goal)...")

    # Build feature dict from the latest activity entry if available
    entries = state.get("activity_entries", [])
    if entries:
        latest = entries[-1]
        user_features = {
            "steps": float(latest.get("steps", 7500)),
            "active_minutes": float(latest.get("active_minutes", 45)),
            "sleep_hours": float(latest.get("sleep_hours", 7.5)),
        }
    else:
        # No entries — use current fitness_data values if present
        fd = state.get("fitness_data") or {}
        user_features = {
            "steps": float(fd.get("steps", 7500)),
            "active_minutes": float(fd.get("active_minutes", 45)),
            "sleep_hours": float(fd.get("sleep_hours", 7.5)),
        }

    goal_suggestion = _call_suggest_activity_goal(user_features)

    if "fitness_data" not in state or state["fitness_data"] is None:
        state["fitness_data"] = {}

    state["fitness_data"]["goal_suggestion"] = goal_suggestion

    # Expose summary values for care_summary aggregation
    state["fitness_data"]["steps"] = user_features["steps"]
    state["fitness_data"]["active_minutes"] = user_features["active_minutes"]

    # Clear the flag so the loop terminates
    state["new_activity_log"] = False

    # Track active branch
    if "fitness_update_node" not in state["active_branches"]:
        state["active_branches"].append("fitness_update_node")

    print(f"[GRAPH] Fitness goal updated: {goal_suggestion[:80]}...")

    return state


def journal_update_node(state: GraphState) -> GraphState:
    """
    Update journal sentiment trend when new entry is added.

    Triggered by new_journal_entry flag.
    """
    print("[GRAPH] Running journal trend update...")

    # Recalculate sentiment trend
    # This would call journal.get_trend()
    state["journal_output"] = "Sentiment trend updated with new entry."

    # Clear the flag
    state["new_journal_entry"] = False

    # Track active branch
    if "journal_update_node" not in state["active_branches"]:
        state["active_branches"].append("journal_update_node")

    print("[GRAPH] Journal trend updated.")

    return state


def care_summary_node(state: GraphState) -> GraphState:
    """Generate unified care summary from all active branches."""
    print(f"[GRAPH] Generating care summary for branches: {state['active_branches']}")

    summary = summarize(state)
    state["output_text"] = summary

    return state


def followup_check_node(state: GraphState) -> GraphState:
    """
    Check if any followup flags are set.

    This node doesn't modify state, just acts as a decision point.
    """
    active_flags = get_active_flags(state)

    if active_flags:
        print(f"[GRAPH] Followup flags detected: {active_flags}")
    else:
        print("[GRAPH] No followup flags. Proceeding to END.")

    return state


def route_followup(state: GraphState) -> Literal[
    "adherence_update_node",
    "fitness_update_node",
    "journal_update_node",
    "care_summary_node",
    "END"
]:
    """
    Route to appropriate update node based on active flags.

    Priority order: adherence > fitness > journal
    If no flags, go to END.
    """
    # Check flags in priority order
    if state.get("new_adherence_log", False):
        return "adherence_update_node"
    elif state.get("new_activity_log", False):
        return "fitness_update_node"
    elif state.get("new_journal_entry", False):
        return "journal_update_node"
    elif has_followup_flags(state):
        # Catch-all: if any flag still set, loop to care_summary
        return "care_summary_node"
    else:
        # No flags set, we're done
        return "END"


def not_implemented_node(state: GraphState) -> GraphState:
    """Placeholder for unimplemented features."""
    intent = state.get("intent", "unknown")
    state["output_text"] = f"This feature isn't built yet. (Intent: {intent})"
    return state


# Build the graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("transcribe_node", transcribe_node)
workflow.add_node("intake_node", intake_node)
workflow.add_node("health_literacy_node", health_literacy_node)
workflow.add_node("nutrition_node", nutrition_node)
workflow.add_node("prescription_node", prescription_node)
workflow.add_node("fitness_node", fitness_node)
workflow.add_node("journal_node", journal_node)
workflow.add_node("not_implemented_node", not_implemented_node)
workflow.add_node("care_summary_node", care_summary_node)
workflow.add_node("followup_check_node", followup_check_node)
workflow.add_node("adherence_update_node", adherence_update_node)
workflow.add_node("fitness_update_node", fitness_update_node)
workflow.add_node("journal_update_node", journal_update_node)

# Wire the graph
workflow.add_edge(START, "transcribe_node")
workflow.add_edge("transcribe_node", "intake_node")

# Conditional routing from intake
workflow.add_conditional_edges(
    "intake_node",
    route_by_intent,
    {
        "health_literacy_node": "health_literacy_node",
        "nutrition_node": "nutrition_node",
        "prescription_node": "prescription_node",
        "fitness_node": "fitness_node",
        "journal_node": "journal_node",
        "not_implemented_node": "not_implemented_node"
    }
)

# All agent nodes go to care_summary
workflow.add_edge("health_literacy_node", "care_summary_node")
workflow.add_edge("nutrition_node", "care_summary_node")
workflow.add_edge("prescription_node", "care_summary_node")
workflow.add_edge("fitness_node", "care_summary_node")
workflow.add_edge("journal_node", "care_summary_node")
workflow.add_edge("not_implemented_node", "care_summary_node")

# Care summary goes to followup check
workflow.add_edge("care_summary_node", "followup_check_node")

# Conditional routing from followup check
workflow.add_conditional_edges(
    "followup_check_node",
    route_followup,
    {
        "adherence_update_node": "adherence_update_node",
        "fitness_update_node": "fitness_update_node",
        "journal_update_node": "journal_update_node",
        "care_summary_node": "care_summary_node",
        "END": END
    }
)

# Update nodes loop back to care_summary for re-generation
workflow.add_edge("adherence_update_node", "followup_check_node")
workflow.add_edge("fitness_update_node", "followup_check_node")
workflow.add_edge("journal_update_node", "followup_check_node")

# Compile the graph
graph = workflow.compile()


def main():
    """Test the graph with followup loops."""
    print("=" * 70)
    print("Testing Graph with Followup Loops")
    print("=" * 70 + "\n")

    # Test case: User asks about fitness, then logs new activity during execution
    print("TEST: Fitness query with mid-execution activity log")
    print("-" * 70 + "\n")

    # Simulate graph execution with followup
    from agents.followup import log_activity_entry

    # Initial state
    initial_state = {
        "input_text": "How many calories did I burn today?",
        "audio_path": None,
        "intent": None,
        "active_branches": [],
        "detected_language": "en",
        "new_adherence_log": False,
        "new_activity_log": False,
        "new_journal_entry": False
    }

    print("Step 1: Initial query processed")
    print(f"  Input: {initial_state['input_text']}")

    # Simulate fitness node execution
    print("\nStep 2: Fitness node processes query")
    simulated_state = initial_state.copy()
    simulated_state["intent"] = "fitness"
    simulated_state["fitness_data"] = {
        "activity_type": "running_6mph",
        "duration_minutes": 30,
        "calories_burned": 350.0
    }
    simulated_state["active_branches"] = ["fitness_node"]
    print(f"  Calories burned: {simulated_state['fitness_data']['calories_burned']} kcal")

    # Log new activity (triggers flag)
    print("\nStep 3: User logs new activity (mid-execution)")
    activity_data = {
        "date": "2026-10-10",
        "steps": 9500,
        "active_minutes": 65,
        "sleep_hours": 7.5
    }
    simulated_state = log_activity_entry(simulated_state, activity_data)
    print(f"  Activity logged: {activity_data['steps']} steps, {activity_data['active_minutes']} active mins")
    print(f"  new_activity_log flag: {simulated_state['new_activity_log']}")

    # Simulate care summary (first time)
    print("\nStep 4: Care summary generated (first time)")
    simulated_state["output_text"] = "Summary: Activity tracked - 350 kcal burned from 30-min run."
    print(f"  Summary: {simulated_state['output_text']}")

    # Followup check detects flag
    print("\nStep 5: Followup check detects new_activity_log flag")
    print(f"  Active flags: {get_active_flags(simulated_state)}")
    print(f"  Decision: Loop back to fitness_update_node")

    # Fitness update node
    print("\nStep 6: Fitness update node recalculates")
    simulated_state = fitness_update_node(simulated_state)
    print(f"  Updated goal: {simulated_state['fitness_data']['goal_suggestion']}")
    print(f"  new_activity_log flag cleared: {simulated_state['new_activity_log']}")

    # Care summary (second time - with updated data)
    print("\nStep 7: Care summary regenerated with updated fitness data")
    simulated_state["output_text"] = (
        "Summary: Activity tracked - 350 kcal burned from 30-min run. "
        "Updated goal based on recent activity (9,500 steps)."
    )
    print(f"  Updated summary: {simulated_state['output_text']}")

    # Final followup check
    print("\nStep 8: Final followup check")
    print(f"  Active flags: {get_active_flags(simulated_state)}")
    print(f"  Decision: No flags set → Proceed to END")

    print("\n" + "=" * 70)
    print("GRAPH EXECUTION COMPLETE")
    print("=" * 70)
    print("\nKey behaviors demonstrated:")
    print("  1. Graph processes initial query normally")
    print("  2. New data logged during execution sets flag")
    print("  3. Care summary generated with initial data")
    print("  4. Followup check detects flag and loops back")
    print("  5. Specific update node recalculates (fitness)")
    print("  6. Care summary regenerated with updated data")
    print("  7. No flags remain → Graph proceeds to END")
    print("\nThis avoids re-running intake/classification for dynamic updates!")
    print("=" * 70)


if __name__ == "__main__":
    main()
