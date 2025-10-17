from __future__ import annotations

import os
from typing import Any, Dict, Iterator, TypedDict, Annotated, List, Optional

from fastapi.encoders import jsonable_encoder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from gradient import AsyncGradient
from gradient_adk import entrypoint
from gradient_adk.langgraph import attach_graph
from gradient_adk.streaming import stream_events


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    _stream: bool  # request streaming
    _delta: str  # scratch: most recent token from node


inference = AsyncGradient(model_access_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"))
MODEL_ID = os.getenv("GRADIENT_MODEL_ID", "llama3.3-70b-instruct")

SYSTEM_PROMPT = (
    "You are a helpful assistant specializing in DigitalOcean. "
    "Answer questions about Droplets, Kubernetes, App Platform, Spaces, "
    "Serverless Inference, Functions, networking, and billing. "
    "Be concise and include short code or CLI snippets when useful."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    _stream: bool  # request streaming
    _delta: str  # scratch: most recent token from node


async def call_model(state: AgentState):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msgs.append({"role": "assistant", "content": m.content or ""})
        else:
            msgs.append(jsonable_encoder(m))

    should_stream = state.get("_stream", False)

    if should_stream:
        # optional: announce an empty assistant message so downstream renderers have something to append to
        yield {"messages": [AIMessage(content="")]}

        stream = inference.chat.completions.create(
            model=MODEL_ID,
            messages=msgs,
            stream=True,
        )

        collected: List[str] = []
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or getattr(chunk, "data", {}).get(
                "choices", []
            )
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            content = None
            if delta is not None and hasattr(delta, "content"):
                content = delta.content
            elif isinstance(delta, dict):
                content = delta.get("content")

            if content:
                collected.append(content)
                yield {"_delta": content}

        final_text = "".join(collected)
        # Finalize the assistant message for tracing.
        yield {"messages": [AIMessage(content=final_text)], "_delta": ""}  # clear delta
        return  # end node
    else:
        resp = await inference.chat.completions.create(model=MODEL_ID, messages=msgs)
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
        # Stream updates emitted *by the node*.
        for update in workflow.stream(
            {"messages": [HumanMessage(content=query)], "_stream": True},
            stream_mode="updates",  # only the per-step diffs
        ):
            # Forward per-token deltas
            delta = update.get("_delta")
            if delta:
                yield {"type": "token", "text": delta}

        # Done
        yield {"type": "end"}
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        yield {"type": "end"}


@entrypoint
async def entry(data, context):
    query = (data or {}).get("query", "").strip()

    # default empty prompt path
    if not query:
        # Run the graph to completion via streaming and just collect (no SSE)
        buf = []
        last_ai = None
        for update in workflow.stream(
            {
                "messages": [HumanMessage(content="")],
                "_stream": True,
            },  # force node streaming
            stream_mode="updates",
        ):
            if update.get("_delta"):
                buf.append(update["_delta"])
            msgs = update.get("messages")
            if msgs:
                for m in reversed(msgs):
                    if isinstance(m, AIMessage):
                        last_ai = m
                        break
        text = "".join(buf) if buf else (last_ai.content if last_ai else "")
        return {"answer": text}

    return await stream_events(_stream_chat(query))
