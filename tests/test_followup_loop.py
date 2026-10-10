"""
Integration test for the followup loop in graph_with_followup.py.

Scenario
--------
1. Log a new adherence entry via followup.log_adherence_entry() so
   new_adherence_log=True is set in the initial state.
2. Invoke the compiled graph.
3. Confirm the graph detects the flag after care_summary_node, routes to
   adherence_update_node, calls predict_miss_risk (real or fallback), clears
   the flag, then runs care_summary_node a second time and terminates.
4. Assert final state has adherence_risk set and new_adherence_log == False.

Run
---
    python tests/test_followup_loop.py
    pytest tests/test_followup_loop.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.followup import (
    set_flag, clear_flag, clear_all_flags,
    log_adherence_entry, log_activity_entry, log_journal_entry,
    has_followup_flags, get_active_flags,
)
from agents.graph_with_followup import (
    adherence_update_node, fitness_update_node, journal_update_node,
    route_followup, graph,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    base = {
        "input_text": "",
        "output_text": "",
        "audio_path": None,
        "intent": None,
        "active_branches": [],
        "detected_language": "en",
        "new_adherence_log": False,
        "new_activity_log": False,
        "new_journal_entry": False,
        "health_output": None,
        "nutrition_output": None,
        "prescription_data": None,
        "medication_schedule": None,
        "adherence_risk": None,
        "journal_output": None,
        "fitness_data": None,
        "nearby_facilities": None,
        "scheme_eligibility": None,
        "food_prediction": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Unit tests: flag helpers
# ---------------------------------------------------------------------------

def test_set_and_clear_flag():
    state = _base_state()
    assert state["new_adherence_log"] is False

    state = set_flag(state, "new_adherence_log")
    assert state["new_adherence_log"] is True
    assert has_followup_flags(state) is True
    assert "new_adherence_log" in get_active_flags(state)

    state = clear_flag(state, "new_adherence_log")
    assert state["new_adherence_log"] is False
    assert has_followup_flags(state) is False
    print("PASS  test_set_and_clear_flag")


def test_log_helpers_set_flags():
    state = _base_state()

    state = log_adherence_entry(state, {"scheduled_time": "2026-10-10 08:00:00", "taken": True})
    assert state["new_adherence_log"] is True
    assert len(state["adherence_entries"]) == 1

    state = log_activity_entry(state, {"steps": 8000, "active_minutes": 50, "sleep_hours": 7})
    assert state["new_activity_log"] is True
    assert len(state["activity_entries"]) == 1

    state = log_journal_entry(state, {"text": "Feeling good", "sentiment_score": 0.8})
    assert state["new_journal_entry"] is True
    assert len(state["journal_entries"]) == 1

    assert set(get_active_flags(state)) == {
        "new_adherence_log", "new_activity_log", "new_journal_entry"
    }

    state = clear_all_flags(state)
    assert has_followup_flags(state) is False
    print("PASS  test_log_helpers_set_flags")


# ---------------------------------------------------------------------------
# Unit tests: route_followup priority
# ---------------------------------------------------------------------------

def test_route_followup_priority():
    state = _base_state(
        new_adherence_log=True,
        new_activity_log=True,
        new_journal_entry=True,
        active_branches=[],
    )
    assert route_followup(state) == "adherence_update_node"

    state["new_adherence_log"] = False
    assert route_followup(state) == "fitness_update_node"

    state["new_activity_log"] = False
    assert route_followup(state) == "journal_update_node"

    state["new_journal_entry"] = False
    assert route_followup(state) == "END"

    print("PASS  test_route_followup_priority")


# ---------------------------------------------------------------------------
# Unit tests: individual update nodes
# ---------------------------------------------------------------------------

def test_adherence_update_node_clears_flag():
    state = _base_state(new_adherence_log=True, active_branches=[])
    state = log_adherence_entry(state, {
        "scheduled_time": "2026-10-10 08:00:00",
        "taken": True,
        "trailing_7day_miss_count": 0,
        "trailing_14day_miss_count": 1,
        "current_streak_length": 10,
        "user_overall_adherence": 0.90,
        "dose_number_in_day": 1,
    })

    state = adherence_update_node(state)

    assert state["new_adherence_log"] is False, "Flag must be cleared"
    assert state["adherence_risk"] is not None, "adherence_risk must be set"
    assert 0.0 <= state["adherence_risk"] <= 1.0, "Risk must be a valid probability"
    assert "adherence_update_node" in state["active_branches"]
    print(f"PASS  test_adherence_update_node_clears_flag  (risk={state['adherence_risk']:.2%})")


def test_fitness_update_node_clears_flag():
    state = _base_state(new_activity_log=True, active_branches=[])
    state = log_activity_entry(state, {"steps": 9500, "active_minutes": 65, "sleep_hours": 7.5})

    state = fitness_update_node(state)

    assert state["new_activity_log"] is False, "Flag must be cleared"
    assert state["fitness_data"] is not None
    assert "goal_suggestion" in state["fitness_data"]
    assert "fitness_update_node" in state["active_branches"]
    print(f"PASS  test_fitness_update_node_clears_flag")


# ---------------------------------------------------------------------------
# Integration test: full graph invocation
# ---------------------------------------------------------------------------

def test_graph_loops_on_adherence_flag():
    """
    Log a new adherence entry before running the graph.
    Confirm the graph loops back through adherence_update_node and
    re-runs care_summary_node without restarting from intake.
    """
    print("\nINTEGRATION TEST: graph loops back on new_adherence_log")
    print("-" * 70)

    initial = _base_state(
        input_text="I just took my morning medication.",
        new_adherence_log=True,
        adherence_entries=[{
            "scheduled_time": "2026-10-10 08:00:00",
            "taken": True,
            "trailing_7day_miss_count": 0,
            "trailing_14day_miss_count": 1,
            "current_streak_length": 7,
            "user_overall_adherence": 0.88,
            "dose_number_in_day": 1,
        }],
    )

    print(f"  new_adherence_log (before) : {initial['new_adherence_log']}")
    print(f"  adherence_entries count    : {len(initial['adherence_entries'])}")

    final = graph.invoke(initial)

    print(f"\n  active_branches            : {final['active_branches']}")
    print(f"  adherence_risk             : {final.get('adherence_risk')!r}")
    print(f"  new_adherence_log (after)  : {final.get('new_adherence_log')}")
    print(f"  output_text (first 120)    : {str(final.get('output_text',''))[:120]}")

    # --- Assertions ---
    branches = final.get("active_branches", [])
    assert "adherence_update_node" in branches, (
        f"Loop did not execute: 'adherence_update_node' not in {branches}"
    )
    assert final.get("new_adherence_log") is False, (
        "new_adherence_log was not cleared — loop may be infinite"
    )
    risk = final.get("adherence_risk")
    assert risk is not None, "adherence_risk not set by adherence_update_node"
    assert 0.0 <= risk <= 1.0, f"adherence_risk out of range: {risk}"
    assert final.get("output_text"), "output_text (care summary) is empty"

    print("PASS  test_graph_loops_on_adherence_flag")
    return final


def test_graph_no_loop_when_no_flags():
    """Sanity check: no loop executes when all flags are False."""
    print("\nINTEGRATION TEST: no loop when flags are clear")
    print("-" * 70)

    initial = _base_state(
        input_text="What foods support heart health?",
        new_adherence_log=False,
        new_activity_log=False,
        new_journal_entry=False,
    )

    final = graph.invoke(initial)

    branches = final.get("active_branches", [])
    assert "adherence_update_node" not in branches, (
        f"Loop should NOT run without flag; got branches: {branches}"
    )
    assert final.get("new_adherence_log") is False
    print(f"  active_branches : {branches}")
    print("PASS  test_graph_no_loop_when_no_flags")


# ---------------------------------------------------------------------------
# pytest-compatible wrappers
# ---------------------------------------------------------------------------

def test_pytest_set_clear():             test_set_and_clear_flag()
def test_pytest_log_helpers():           test_log_helpers_set_flags()
def test_pytest_route_priority():        test_route_followup_priority()
def test_pytest_adherence_node():        test_adherence_update_node_clears_flag()
def test_pytest_fitness_node():          test_fitness_update_node_clears_flag()
def test_pytest_integration_loop():      test_graph_loops_on_adherence_flag()
def test_pytest_integration_no_loop():   test_graph_no_loop_when_no_flags()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Followup Loop Tests — MediPath")
    print("=" * 70 + "\n")

    # Unit tests (no LLM call)
    test_set_and_clear_flag()
    test_log_helpers_set_flags()
    test_route_followup_priority()
    test_adherence_update_node_clears_flag()
    test_fitness_update_node_clears_flag()

    # Integration tests (invoke graph, LLM called)
    test_graph_loops_on_adherence_flag()
    test_graph_no_loop_when_no_flags()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
