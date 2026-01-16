"""
Simple template agent using the gradient agent runtime with Gradient SDK (serverless inference) and LangGraph.
"""

import os
from typing import Dict, TypedDict

from gradient import AsyncGradient
from gradient_adk import entrypoint, RequestContext
from langgraph.graph import StateGraph


class State(TypedDict):
    """The state of our graph."""

    input: str
    output: str


async def llm_call(state: State) -> State:
    """Call the LLM"""

    inference_client = AsyncGradient(
        model_access_key=os.environ.get(
            "GRADIENT_MODEL_ACCESS_KEY"
        )
    )

    # Call the model
    output = await inference_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": state["input"],
            }
        ],
        model="openai-gpt-oss-120b",
    )

    # Set the state
    state["output"] = output.choices[0].message.content
    
    return state


@entrypoint
async def main(input: Dict, context: RequestContext):
    """Entrypoint"""

    # Setup the graph
    initial_state = State(
        input=input.get("prompt"),
        output=None
    )
    graph = StateGraph(State)
    graph.add_node("llm_call", llm_call)
    graph.set_entry_point("llm_call")
    
    # Attach the graph for instrumentation
    app = graph.compile()
    
    # Invoke the app
    result = await app.ainvoke(initial_state)
    return result["output"]
