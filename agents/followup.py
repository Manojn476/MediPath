"""
Followup flag management for dynamic graph loops.

Sets flags when new data is logged (adherence, activity, journal) to trigger
selective re-processing without re-running the entire graph from intake.
"""
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def set_flag(state: Dict[str, Any], flag_name: str) -> Dict[str, Any]:
    """
    Set a followup flag in graph state.

    Flags trigger conditional loops in the graph to update predictions
    without re-running intake/classification.

    Args:
        state: Current graph state
        flag_name: Name of flag to set (e.g., 'new_adherence_log')

    Returns:
        Updated state with flag set to True
    """
    state[flag_name] = True
    return state


def clear_flag(state: Dict[str, Any], flag_name: str) -> Dict[str, Any]:
    """
    Clear a followup flag in graph state.

    Args:
        state: Current graph state
        flag_name: Name of flag to clear

    Returns:
        Updated state with flag set to False
    """
    state[flag_name] = False
    return state


def clear_all_flags(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clear all followup flags.

    Args:
        state: Current graph state

    Returns:
        Updated state with all flags cleared
    """
    state["new_adherence_log"] = False
    state["new_activity_log"] = False
    state["new_journal_entry"] = False
    return state


def has_followup_flags(state: Dict[str, Any]) -> bool:
    """
    Check if any followup flags are set.

    Args:
        state: Current graph state

    Returns:
        True if any flag is set, False otherwise
    """
    return (
        state.get("new_adherence_log", False) or
        state.get("new_activity_log", False) or
        state.get("new_journal_entry", False)
    )


def get_active_flags(state: Dict[str, Any]) -> list:
    """
    Get list of currently active flags.

    Args:
        state: Current graph state

    Returns:
        List of active flag names
    """
    active = []

    if state.get("new_adherence_log", False):
        active.append("new_adherence_log")

    if state.get("new_activity_log", False):
        active.append("new_activity_log")

    if state.get("new_journal_entry", False):
        active.append("new_journal_entry")

    return active


# Integration helpers for other agents

def log_adherence_entry(state: Dict[str, Any], entry_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log a new adherence entry and set the flag.

    Called from reminder.py when a dose is taken/missed.

    Args:
        state: Current graph state
        entry_data: Adherence entry details (taken, scheduled_time, etc.)

    Returns:
        Updated state with entry logged and flag set
    """
    # Store entry in state (in real system, would write to DB)
    if "adherence_entries" not in state:
        state["adherence_entries"] = []

    state["adherence_entries"].append(entry_data)

    # Set flag to trigger adherence prediction update
    state = set_flag(state, "new_adherence_log")

    print(f"[FOLLOWUP] New adherence entry logged. Flag set.")

    return state


def log_activity_entry(state: Dict[str, Any], activity_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log a new fitness activity and set the flag.

    Called from fitness.py when activity is tracked.

    Args:
        state: Current graph state
        activity_data: Activity details (steps, active_minutes, etc.)

    Returns:
        Updated state with entry logged and flag set
    """
    # Store entry in state (in real system, would write to DB)
    if "activity_entries" not in state:
        state["activity_entries"] = []

    state["activity_entries"].append(activity_data)

    # Set flag to trigger fitness trend update
    state = set_flag(state, "new_activity_log")

    print(f"[FOLLOWUP] New activity entry logged. Flag set.")

    return state


def log_journal_entry(state: Dict[str, Any], journal_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log a new journal entry and set the flag.

    Called from journal.py when user writes an entry.

    Args:
        state: Current graph state
        journal_data: Journal entry details (text, sentiment, etc.)

    Returns:
        Updated state with entry logged and flag set
    """
    # Store entry in state (in real system, would write to DB)
    if "journal_entries" not in state:
        state["journal_entries"] = []

    state["journal_entries"].append(journal_data)

    # Set flag to trigger journal trend update
    state = set_flag(state, "new_journal_entry")

    print(f"[FOLLOWUP] New journal entry logged. Flag set.")

    return state


def main():
    """Test followup flag management."""
    print("=" * 70)
    print("Testing Followup Flag Management")
    print("=" * 70 + "\n")

    # Initialize state
    state = {
        "user_id": "test_user_001",
        "new_adherence_log": False,
        "new_activity_log": False,
        "new_journal_entry": False
    }

    print("Initial state:")
    print(f"  new_adherence_log: {state['new_adherence_log']}")
    print(f"  new_activity_log: {state['new_activity_log']}")
    print(f"  new_journal_entry: {state['new_journal_entry']}")
    print(f"  Has followup flags: {has_followup_flags(state)}")
    print()

    # Test 1: Log adherence entry
    print("TEST 1: Log adherence entry")
    print("-" * 70)

    adherence_data = {
        "prescription_id": 1,
        "scheduled_time": "2026-10-10 08:00:00",
        "taken": True,
        "taken_at": "2026-10-10 08:05:00",
        "delay_minutes": 5
    }

    state = log_adherence_entry(state, adherence_data)
    print(f"  new_adherence_log: {state['new_adherence_log']}")
    print(f"  Active flags: {get_active_flags(state)}")
    print(f"  Adherence entries: {len(state.get('adherence_entries', []))}")
    print()

    # Test 2: Log activity entry
    print("TEST 2: Log activity entry")
    print("-" * 70)

    activity_data = {
        "date": "2026-10-10",
        "steps": 8500,
        "active_minutes": 55,
        "sleep_hours": 7.5
    }

    state = log_activity_entry(state, activity_data)
    print(f"  new_activity_log: {state['new_activity_log']}")
    print(f"  Active flags: {get_active_flags(state)}")
    print(f"  Activity entries: {len(state.get('activity_entries', []))}")
    print()

    # Test 3: Check multiple flags
    print("TEST 3: Multiple flags active")
    print("-" * 70)
    print(f"  Has followup flags: {has_followup_flags(state)}")
    print(f"  Active flags: {get_active_flags(state)}")
    print()

    # Test 4: Clear specific flag
    print("TEST 4: Clear adherence flag")
    print("-" * 70)
    state = clear_flag(state, "new_adherence_log")
    print(f"  new_adherence_log: {state['new_adherence_log']}")
    print(f"  Active flags: {get_active_flags(state)}")
    print()

    # Test 5: Log journal entry
    print("TEST 5: Log journal entry")
    print("-" * 70)

    journal_data = {
        "date": "2026-10-10",
        "text": "Feeling better today. Got some exercise and slept well.",
        "sentiment_score": 0.7,
        "sentiment_label": "positive"
    }

    state = log_journal_entry(state, journal_data)
    print(f"  new_journal_entry: {state['new_journal_entry']}")
    print(f"  Active flags: {get_active_flags(state)}")
    print(f"  Journal entries: {len(state.get('journal_entries', []))}")
    print()

    # Test 6: Clear all flags
    print("TEST 6: Clear all flags")
    print("-" * 70)
    state = clear_all_flags(state)
    print(f"  new_adherence_log: {state['new_adherence_log']}")
    print(f"  new_activity_log: {state['new_activity_log']}")
    print(f"  new_journal_entry: {state['new_journal_entry']}")
    print(f"  Has followup flags: {has_followup_flags(state)}")
    print()

    print("=" * 70)
    print("SUMMARY:")
    print("  - Flags track new data logged during graph execution")
    print("  - Each flag triggers specific prediction updates")
    print("  - Allows selective re-processing without full graph restart")
    print("  - Flags are cleared after processing")
    print("=" * 70)


if __name__ == "__main__":
    main()
