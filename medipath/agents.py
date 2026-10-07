from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .state import SharedState

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_json(name: str) -> Any:
    with (DATA_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def intake_agent(state: SharedState) -> SharedState:
    text = (state.raw_input or "").lower()
    mapping = {
        "health": ["health_literacy"],
        "prescription": ["prescription_digitization", "adherence_reminder"],
        "facility": ["facility_finder", "scheme_eligibility"],
        "journal": ["wellness_journal"],
        "nutrition": ["diet_nutrition", "food_recognition", "fitness_activity"],
        "fitness": ["fitness_activity", "diet_nutrition"],
    }
    selected = mapping.get(state.intent, [])
    if not selected:
        if any(k in text for k in ("clinic", "hospital", "scheme")):
            selected = mapping["facility"]
        elif any(k in text for k in ("pill", "medicine", "prescription")):
            selected = mapping["prescription"]
        elif any(k in text for k in ("sad", "mood", "journal")):
            selected = mapping["journal"]
        elif any(k in text for k in ("meal", "food", "calorie", "workout", "activity")):
            selected = mapping["nutrition"]
        else:
            selected = mapping["health"]
    state.active_branches = selected
    return state


def health_literacy_agent(state: SharedState) -> Dict[str, Any]:
    records = _load_json("official_sources.json")
    query = (state.raw_input or "").lower()
    hits = [r for r in records if any(tok in query for tok in r["keywords"])]
    if not hits:
        return {
            "answer": "Official sources in the current knowledge base do not cover this question.",
            "source": "No source match",
            "safety": "General information only; not medical diagnosis or triage.",
        }
    best = hits[0]
    return {
        "answer": best["guidance"],
        "source": f"{best['source_name']} ({best['source_url']})",
        "safety": "General information only; not medical diagnosis or triage.",
    }


def prescription_digitization_agent(raw_text: str) -> Dict[str, Any]:
    medicines = _load_json("drug_reference.json")
    med_pattern = re.compile(r"([A-Za-z][A-Za-z0-9\- ]{2,30})\s+(\d+\s?mg)?\s*(OD|BD|TDS|QID)?\s*(\d+\s?days)?", re.I)
    matches = []
    for m in med_pattern.finditer(raw_text or ""):
        name = (m.group(1) or "").strip()
        if any(name.lower() in known.lower() or known.lower() in name.lower() for known in medicines):
            matches.append(
                {
                    "medication_name": name,
                    "dosage": m.group(2) or "",
                    "frequency": (m.group(3) or "").upper(),
                    "duration": m.group(4) or "",
                    "confidence": {
                        "medication_name": 0.85,
                        "dosage": 0.70 if m.group(2) else 0.4,
                        "frequency": 0.78 if m.group(3) else 0.35,
                        "duration": 0.72 if m.group(4) else 0.3,
                    },
                }
            )
    return {
        "extracted": matches,
        "requires_user_confirmation": True,
        "note": "Please confirm extracted medication details before reminders are activated.",
    }


def adherence_reminder_agent(confirmed_medications: List[Dict[str, Any]], risk_score: float = 0.5) -> Dict[str, Any]:
    adjusted_lead_minutes = 15 if risk_score < 0.5 else 30
    schedule = []
    for med in confirmed_medications:
        schedule.append(
            {
                "medication_name": med.get("medication_name"),
                "frequency": med.get("frequency", ""),
                "reminder_lead_minutes": adjusted_lead_minutes,
            }
        )
    return {
        "risk_score": risk_score,
        "schedule": schedule,
        "safety": "Schedules are based on confirmed prescription text only.",
    }


def facility_finder_agent(region: str) -> Dict[str, Any]:
    facilities = _load_json("facilities.json")
    matches = [f for f in facilities if f["region"].lower() == region.lower()]
    return {
        "results": matches,
        "source": "OpenStreetMap / Overpass (cached sample data)",
    }


def scheme_eligibility_agent(age: int, income: int, residency: str) -> Dict[str, Any]:
    schemes = _load_json("schemes.json")
    outcomes = []
    for s in schemes:
        age_ok = (s["min_age"] is None or age >= s["min_age"]) and (s["max_age"] is None or age <= s["max_age"])
        income_ok = s["income_ceiling"] is None or income <= s["income_ceiling"]
        residency_ok = s["requires_residency"].lower() in residency.lower()
        eligible = age_ok and income_ok and residency_ok
        outcomes.append(
            {
                "scheme_name": s["scheme_name"],
                "eligible": eligible,
                "why": {
                    "age": age_ok,
                    "income": income_ok,
                    "residency": residency_ok,
                },
                "source": s["source"],
            }
        )
    return {"eligibility": outcomes}


def wellness_journal_agent(mood: str, notes: str) -> Dict[str, Any]:
    lowered = f"{mood} {notes}".lower()
    if any(w in lowered for w in ["good", "calm", "happy", "great"]):
        sentiment = "positive"
    elif any(w in lowered for w in ["sad", "anxious", "stressed", "low"]):
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {
        "sentiment": sentiment,
        "trend_note": "Trend tracking is informational only and not diagnostic.",
        "resources_always_visible": True,
    }


def diet_nutrition_agent(goal: str, preferences: str) -> Dict[str, Any]:
    refs = _load_json("nutrition_refs.json")
    return {
        "guidance": f"Based on your goal '{goal}' and preferences '{preferences}', prioritize whole foods and portion balance.",
        "source": refs[0]["source"],
        "constraint": "General nutrition guidance only.",
    }


def food_recognition_agent(food_label: str, confidence: float) -> Dict[str, Any]:
    nutrition = _load_json("food_nutrition_map.json")
    found = nutrition.get(food_label.lower())
    if not found:
        return {
            "recognized_food": food_label,
            "confidence": confidence,
            "requires_user_confirmation": True,
            "message": "Could not match this meal to nutrition data. Please confirm meal details.",
        }
    out = {
        "recognized_food": food_label,
        "confidence": confidence,
        "nutrition": found,
        "benchmark": "Compared to general dietary benchmarks (informational).",
    }
    if confidence < 0.75:
        out["requires_user_confirmation"] = True
        out["message"] = "Low confidence prediction. Please confirm meal identity before logging."
    return out


def fitness_activity_agent(weight_kg: float, met: float, minutes: float) -> Dict[str, Any]:
    calories = met * weight_kg * (minutes / 60.0)
    return {
        "calories_estimate": round(calories, 2),
        "formula": "MET × weight(kg) × duration(hours)",
        "goal": "Aim for steady weekly activity progress based on your own logs.",
    }


def care_summary_agent(state: SharedState) -> Dict[str, Any]:
    return {
        "language": state.language,
        "ran_branches": state.active_branches,
        "merged_outputs": state.branch_outputs,
        "safety": "This summary is informational and does not provide diagnosis or triage.",
    }


def follow_up_tracker(state: SharedState) -> Dict[str, Any]:
    loops = []
    if state.new_data_flags.get("adherence"):
        loops.append("adherence_reminder")
    if state.new_data_flags.get("activity"):
        loops.append("fitness_activity")
    return {
        "loop_targets": loops,
        "new_data_flags": dict(state.new_data_flags),
    }


def state_as_dict(state: SharedState) -> Dict[str, Any]:
    return asdict(state)
