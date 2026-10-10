"""Minimal LangGraph workflow for MediPath."""
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from agents.llm_client import ask_llm
from agents.voice_client import transcribe_audio
from agents.health_rag import answer_health_question
from agents.intake import classify_intent


class GraphState(TypedDict):
    """State schema for the graph."""
    input_text: str
    output_text: str
    audio_path: Optional[str]
    intent: Optional[str]

    # Followup flags: set True by reminder.py, journal.py, or fitness.py
    # to trigger selective re-runs after care_summary_node.
    new_adherence_log: bool
    new_activity_log: bool
    new_journal_entry: bool


def transcribe_node(state: GraphState) -> GraphState:
    """
    Node that transcribes audio if audio_path is provided.

    Args:
        state: Current graph state

    Returns:
        Updated state with transcribed text in input_text
    """
    audio_path = state.get("audio_path")

    if audio_path:
        # Transcribe the audio and put it in input_text
        transcription = transcribe_audio(audio_path)
        return {
            "input_text": transcription,
            "output_text": state.get("output_text", ""),
            "audio_path": audio_path,
            "intent": state.get("intent")
        }
    else:
        # Pass through unchanged
        return state


def intake_node(state: GraphState) -> GraphState:
    """
    Node that classifies user intent.

    Args:
        state: Current graph state

    Returns:
        Updated state with intent classification
    """
    intent = classify_intent(state)
    return {
        "input_text": state["input_text"],
        "output_text": state.get("output_text", ""),
        "audio_path": state.get("audio_path"),
        "intent": intent,
        # Initialize followup flags if not already set by caller
        "new_adherence_log": state.get("new_adherence_log", False),
        "new_activity_log": state.get("new_activity_log", False),
        "new_journal_entry": state.get("new_journal_entry", False),
    }


def route_by_intent(state: GraphState) -> Literal["health_literacy_node", "not_implemented_node"]:
    """
    Route based on detected intent.

    Args:
        state: Current graph state

    Returns:
        Name of next node to execute
    """
    intent = state.get("intent", "unclear")

    # Route health intent to RAG system
    if intent == "health":
        return "health_literacy_node"
    else:
        # All other intents go to placeholder
        return "not_implemented_node"


def health_literacy_node(state: GraphState) -> GraphState:
    """
    Node that answers health questions using RAG.

    Args:
        state: Current graph state

    Returns:
        Updated state with RAG-based answer
    """
    response = answer_health_question(state["input_text"])
    return {
        "input_text": state["input_text"],
        "output_text": response,
        "audio_path": state.get("audio_path"),
        "intent": state.get("intent")
    }


def not_implemented_node(state: GraphState) -> GraphState:
    """
    Placeholder node for features not yet implemented.

    Args:
        state: Current graph state

    Returns:
        Updated state with placeholder message
    """
    intent = state.get("intent", "unknown")
    message = f"This feature isn't built yet. (Detected intent: {intent})"
    return {
        "input_text": state["input_text"],
        "output_text": message,
        "audio_path": state.get("audio_path"),
        "intent": state.get("intent")
    }


def answer_node(state: GraphState) -> GraphState:
    """
    Node that processes input text through the LLM (direct, non-RAG).

    Args:
        state: Current graph state

    Returns:
        Updated state with LLM response
    """
    response = ask_llm(state["input_text"])
    return {
        "input_text": state["input_text"],
        "output_text": response,
        "audio_path": state.get("audio_path")
    }


# Build the graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("transcribe_node", transcribe_node)
workflow.add_node("intake_node", intake_node)
workflow.add_node("health_literacy_node", health_literacy_node)
workflow.add_node("not_implemented_node", not_implemented_node)
workflow.add_node("answer_node", answer_node)

# Wire the graph with conditional routing
workflow.add_edge(START, "transcribe_node")
workflow.add_edge("transcribe_node", "intake_node")
workflow.add_conditional_edges(
    "intake_node",
    route_by_intent,
    {
        "health_literacy_node": "health_literacy_node",
        "not_implemented_node": "not_implemented_node"
    }
)
workflow.add_edge("health_literacy_node", END)
workflow.add_edge("not_implemented_node", END)

# Compile the graph
graph = workflow.compile()
