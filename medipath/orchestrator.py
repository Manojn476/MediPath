from __future__ import annotations

from typing import Any, Dict

from .agents import (
    adherence_reminder_agent,
    care_summary_agent,
    diet_nutrition_agent,
    facility_finder_agent,
    fitness_activity_agent,
    follow_up_tracker,
    food_recognition_agent,
    health_literacy_agent,
    intake_agent,
    prescription_digitization_agent,
    scheme_eligibility_agent,
    state_as_dict,
    wellness_journal_agent,
)
from .state import SharedState

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = "END"
    START = "START"
    StateGraph = None


def _run_selected_branches(state: SharedState, payload: Dict[str, Any]) -> SharedState:
    out: Dict[str, Any] = {}
    if "health_literacy" in state.active_branches:
        out["health_literacy"] = health_literacy_agent(state)
    if "prescription_digitization" in state.active_branches:
        out["prescription_digitization"] = prescription_digitization_agent(payload.get("prescription_text", ""))
    if "adherence_reminder" in state.active_branches and payload.get("confirmed_meds"):
        out["adherence_reminder"] = adherence_reminder_agent(
            payload["confirmed_meds"], payload.get("risk_score", 0.5)
        )
    if "facility_finder" in state.active_branches:
        out["facility_finder"] = facility_finder_agent(payload.get("region", "Bengaluru"))
    if "scheme_eligibility" in state.active_branches:
        out["scheme_eligibility"] = scheme_eligibility_agent(
            payload.get("age", 30), payload.get("income", 200000), payload.get("residency", "Karnataka")
        )
    if "wellness_journal" in state.active_branches:
        out["wellness_journal"] = wellness_journal_agent(payload.get("mood", ""), payload.get("notes", ""))
    if "diet_nutrition" in state.active_branches:
        out["diet_nutrition"] = diet_nutrition_agent(payload.get("goal", "general wellness"), payload.get("preferences", ""))
    if "food_recognition" in state.active_branches:
        out["food_recognition"] = food_recognition_agent(payload.get("food_label", "meal"), payload.get("food_confidence", 0.6))
    if "fitness_activity" in state.active_branches:
        out["fitness_activity"] = fitness_activity_agent(
            payload.get("weight_kg", 70.0), payload.get("met", 4.0), payload.get("minutes", 30.0)
        )
    state.branch_outputs = out
    return state


def _build_langgraph() -> Any:
    if StateGraph is None:
        return None

    graph = StateGraph(dict)

    def intake_node(data: dict) -> dict:
        state = SharedState(**data["state"])
        state = intake_agent(state)
        data["state"] = state_as_dict(state)
        return data

    def run_branches_node(data: dict) -> dict:
        state = SharedState(**data["state"])
        state = _run_selected_branches(state, data.get("payload", {}))
        data["state"] = state_as_dict(state)
        return data

    def summary_node(data: dict) -> dict:
        state = SharedState(**data["state"])
        data["care_summary"] = care_summary_agent(state)
        data["follow_up"] = follow_up_tracker(state)
        return data

    graph.add_node("intake", intake_node)
    graph.add_node("run_branches", run_branches_node)
    graph.add_node("summary", summary_node)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "run_branches")
    graph.add_edge("run_branches", "summary")
    graph.add_edge("summary", END)
    return graph.compile()


def run_supervisor(state: SharedState, payload: Dict[str, Any]) -> Dict[str, Any]:
    app = _build_langgraph()
    if app is not None:
        return app.invoke({"state": state_as_dict(state), "payload": payload})

    state = intake_agent(state)
    state = _run_selected_branches(state, payload)
    return {
        "state": state_as_dict(state),
        "care_summary": care_summary_agent(state),
        "follow_up": follow_up_tracker(state),
    }
