"""
Example showing LangGraph integration with runtime tracking.
"""

try:
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available. Install with: pip install langgraph")

from gradient_agent import entrypoint
from gradient_agent.runtime import get_runtime_manager


class AgentState(TypedDict):
    """State for our simple agent."""

    prompt: str
    response: str
    step_count: int


def process_step(state: AgentState) -> AgentState:
    """Process a single step in the agent."""
    step_count = state.get("step_count", 0) + 1

    # Simulate some processing
    processed = f"Step {step_count}: {state['prompt']}"

    return {**state, "response": processed, "step_count": step_count}


def should_continue(state: AgentState) -> str:
    """Decide whether to continue processing."""
    if state.get("step_count", 0) >= 2:
        return END
    return "process"


@entrypoint
def langgraph_agent(prompt: str) -> str:
    """Agent using LangGraph with runtime tracking."""

    if not LANGGRAPH_AVAILABLE:
        return "LangGraph not available. Please install langgraph to use this agent."

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("process", process_step)

    # Add edges
    workflow.set_entry_point("process")
    workflow.add_conditional_edges(
        "process", should_continue, {"process": "process", END: END}
    )

    # Compile the graph
    app = workflow.compile()

    # Run the graph - runtime tracking happens automatically
    result = app.invoke({"prompt": prompt})

    return result["response"]


if __name__ == "__main__":
    from gradient_agent import run_server

    if LANGGRAPH_AVAILABLE:
        print("Starting LangGraph agent server...")
        print("Runtime tracking is automatically enabled!")
        print(
            "Try: curl -X POST http://localhost:8080/completions -H 'Content-Type: application/json' -d '{\"prompt\": \"Test workflow\"}'"
        )
        run_server(port=8080)
    else:
        print("Please install langgraph to run this example: pip install langgraph")
