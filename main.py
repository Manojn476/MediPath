"""Minimal project smoke test entry point."""

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    message: str


def append_ready_message(state: State) -> State:
    return {"message": f"{state['message']} is ready"}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("ready", append_ready_message)
    graph.add_edge(START, "ready")
    graph.add_edge("ready", END)
    return graph.compile()


def main() -> None:
    result = build_graph().invoke({"message": "Project"})
    assert result["message"] == "Project is ready"
    # Importing RunnableLambda also verifies the LangChain installation.
    assert RunnableLambda(lambda value: value).invoke("ok") == "ok"
    print(result["message"])


if __name__ == "__main__":
    main()
