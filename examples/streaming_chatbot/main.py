from __future__ import annotations

import os
from typing import Any, Dict, Iterator, TypedDict, Annotated, List, Optional

from fastapi.encoders import jsonable_encoder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from gradient import Gradient
from gradient_adk import entrypoint, stream_events
from gradient_adk.langgraph import attach_graph


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    _stream: bool  # Flag to enable streaming
    _stream_generator: Any  # Generator for streaming content


inference = Gradient(model_access_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"))
MODEL_ID = os.getenv("GRADIENT_MODEL_ID", "llama3.3-70b-instruct")

SYSTEM_PROMPT = (
    "You are a helpful assistant specializing in DigitalOcean. "
    "Answer questions about Droplets, Kubernetes, App Platform, Spaces, "
    "Serverless Inference, Functions, networking, and billing. "
    "Be concise and include short code or CLI snippets when useful."
)


def call_model(state: AgentState):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msgs.append({"role": "assistant", "content": m.content or ""})
        else:
            msgs.append(jsonable_encoder(m))

    # Check if streaming is requested via state metadata
    should_stream = state.get("_stream", False)

    if should_stream:
        # Return a generator that will be consumed by the streaming logic
        def stream_generator():
            stream = inference.chat.completions.create(
                model=MODEL_ID,
                messages=msgs,
                stream=True,
            )

            collected_content = ""
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or getattr(
                    chunk, "data", {}
                ).get("choices", [])
                if not choices:
                    continue

                choice0 = choices[0]
                delta = getattr(choice0, "delta", None)
                if delta is None:
                    continue

                if hasattr(delta, "content"):
                    content = delta.content
                elif isinstance(delta, dict):
                    content = delta.get("content")
                else:
                    content = None

                if content:
                    collected_content += content
                    yield content

            # Store final content in state for trace recording
            return collected_content

        # Store the generator in the state for streaming consumption
        return {
            "messages": [AIMessage(content="")],
            "_stream_generator": stream_generator(),
        }
    else:
        # Non-streaming path
        resp = inference.chat.completions.create(model=MODEL_ID, messages=msgs)
        text = resp.choices[0].message.content or ""
        return {"messages": [AIMessage(content=text)]}


graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.set_entry_point("agent")
graph.set_finish_point("agent")
attach_graph(graph)
workflow = graph.compile()


def _stream_chat(query: str):
    """Yield SSE-friendly JSON chunks while running through LangGraph workflow."""
    yield {"type": "start", "model": MODEL_ID}

    try:
        # Use the instrumented LangGraph workflow with streaming enabled
        result = workflow.invoke(
            {"messages": [HumanMessage(content=query)], "_stream": True}
        )

        # Check if we have a stream generator from the call_model node
        stream_generator = result.get("_stream_generator")
        collected_content = ""

        if stream_generator:
            # Stream the tokens
            for content in stream_generator:
                if content:
                    collected_content += content
                    yield {"type": "token", "text": content}

            # Update the AI message with the complete content for trace recording
            if result.get("messages"):
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        msg.content = collected_content
                        break

        # End event
        yield {"type": "end"}
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        yield {"type": "end"}


@entrypoint
def entry(data, context):
    """
    Expected payload:
      { "query": "How do I deploy a FastAPI app on App Platform?" }

    Returns:
      Server-Sent Events stream (NDJSON-ish via your stream_events wrapper)
    """
    query = (data or {}).get("query", "").strip()
    if not query:
        # Fall back to non-stream JSON if no question provided
        out = workflow.invoke({"messages": [HumanMessage(content="")]})
        last_ai = next(
            (m for m in reversed(out["messages"]) if isinstance(m, AIMessage)), None
        )
        return {"answer": last_ai.content if last_ai else ""}

    # Stream tokens back to the client
    return stream_events(_stream_chat(query))
