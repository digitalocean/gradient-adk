import os
import json
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict, Annotated

from gradient_adk import entrypoint
import asyncio
import httpx

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ============================================================
# Config
# ============================================================

MAX_ITERS = int(os.getenv("DEEP_RESEARCH_MAX_ITERS", "6"))
MAX_TOOL_TEXT_CHARS = int(os.getenv("DEEP_RESEARCH_TOOL_TRUNC_CHARS", "6000"))

# Model used for the final streamed write (Gradient SDK)
FINAL_MODEL = os.getenv("DEEP_RESEARCH_FINAL_MODEL", "openai-gpt-4o")
# If you want the same model as your LangChain calls, set this env var accordingly.

# ============================================================
# Tools
# ============================================================

_duck = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Perform a web search using DuckDuckGo. Input should be a precise query string."""
    return _duck.run(query)


@tool
def get_current_time() -> str:
    """Get the current date and time in ISO format. Useful for time-sensitive research or when the agent needs to know what day/time it is."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# Prompts
# ============================================================

DEEP_RESEARCH_SYSTEM_PROMPT = """You are a DeepResearch agent.

Goal:
Produce an extremely detailed, high-quality answer to the user's original question.

Method:
- Work in iterative research loops.
- In each loop, identify what you still need to know, then call the web_search tool for missing details.
- Prefer primary sources (official docs/specs/papers/standards) and reputable secondary sources.
- Cross-check important claims across multiple sources.
- If sources conflict, explicitly explain the conflict and why one is more credible.
- Do NOT invent details. If you can't verify something, label it as uncertain.

Tool use rules:
- Use the web_search tool proactively.
- In each research iteration, make several web_search calls with different angles (definitions, official docs, comparisons, edge cases, recent changes).
- Stop searching only when you have enough evidence and coverage to write a long, structured, comprehensive answer.

When you are ready to stop researching:
- Reply with a message that starts EXACTLY with: WRITE_FINAL
- Then include a concise outline of the final answer.
- Do not call tools in that message.
"""

NOTES_SUMMARIZER_SYSTEM_PROMPT = """You compress raw search results into durable research notes.

Requirements:
- Extract key facts, numbers, definitions, and caveats.
- Keep source attribution as inline URLs if present; otherwise keep the publication/site name if obvious.
- Organize by topic with bullets.
- Preserve contradictory findings (label them).
- Keep it dense and information-rich; avoid fluff.
"""

FINAL_WRITER_SYSTEM_PROMPT = """You are writing the final DeepResearch answer using research notes.

Requirements:
- Very detailed, long, and structured (clear headings, subheadings, bullet lists, and examples).
- Include concrete, actionable guidance when relevant.
- Include citations as URLs when available in notes; otherwise mention the source name next to the claim.
- Call out uncertainty explicitly.
"""


# ============================================================
# State
# ============================================================

class DeepResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_notes: str
    last_tool_results: List[str]
    iteration: int
    question: str


# ============================================================
# Models (research + summarization)
# ============================================================

base_llm = ChatOpenAI(
    base_url="https://inference.do-ai.run/v1",
    model="openai-gpt-4o",
    api_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
    temperature=0.2,
    max_tokens=8192,
    # reasoning={"effort": "medium"},
)

# Tool-capable LLM
research_llm = base_llm.bind_tools([web_search, get_current_time])

# Notes compressor (no tools)
notes_llm = ChatOpenAI(
    base_url="https://inference.do-ai.run/v1",
    model="openai-gpt-4o",
    api_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
    temperature=0.0,
    max_tokens=8192,
    # reasoning={"effort": "medium"},
)


# ============================================================
# Helpers
# ============================================================

def _safe_json_loads(maybe_json: Any) -> Dict[str, Any]:
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}


def _extract_tool_calls(msg: BaseMessage) -> List[Dict[str, Any]]:
    tool_calls = getattr(msg, "tool_calls", None)
    if isinstance(tool_calls, list):
        return tool_calls

    additional = getattr(msg, "additional_kwargs", {}) or {}
    tc = additional.get("tool_calls")
    if isinstance(tc, list):
        return tc

    return []


def _maybe_get_output_dict(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    LangGraph event payload shapes vary by version; try common places.
    """
    data = ev.get("data") or {}
    out = data.get("output")
    if isinstance(out, dict):
        return out
    if isinstance(data, dict) and isinstance(data.get("state"), dict):
        return data["state"]
    if isinstance(data, dict) and isinstance(data.get("output"), dict):
        return data["output"]
    return None


# ============================================================
# Graph nodes (research only; final writing done in streamed step)
# ============================================================

async def research_node(state: DeepResearchState) -> Dict[str, Any]:
    it = state["iteration"] + 1
    print(f"[RESEARCH NODE] Starting iteration {it}/{MAX_ITERS}")

    msgs: List[BaseMessage] = list(state["messages"])
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=DEEP_RESEARCH_SYSTEM_PROMPT)] + msgs

    dynamic_guidance = SystemMessage(
        content=(
            f"{DEEP_RESEARCH_SYSTEM_PROMPT}\n\n"
            f"Iteration: {it}/{MAX_ITERS}\n\n"
            f"Current research notes (may be incomplete):\n"
            f"{state['research_notes'] or '(none yet)'}"
        )
    )

    print(f"[RESEARCH NODE] Invoking research_llm with {len(msgs) + 1} messages")
    ai = await research_llm.ainvoke(msgs + [dynamic_guidance])
    print(f"[RESEARCH NODE] Got AI response, checking for tool calls...")
    
    tool_calls = _extract_tool_calls(ai)
    print(f"[RESEARCH NODE] Found {len(tool_calls)} tool calls")

    return {"messages": [ai], "iteration": it}


async def tools_node(state: DeepResearchState) -> Dict[str, Any]:
    print("[TOOLS NODE] Starting tool execution")
    last = state["messages"][-1]
    tool_calls = _extract_tool_calls(last)
    print(f"[TOOLS NODE] Executing {len(tool_calls)} tool calls")

    tool_messages: List[ToolMessage] = []
    raw_results: List[str] = []

    for idx, tc in enumerate(tool_calls):
        name = tc.get("name") or tc.get("function", {}).get("name")
        args = tc.get("args") or tc.get("function", {}).get("arguments")
        args_dict = _safe_json_loads(args)
        tool_call_id = tc.get("id") or tc.get("tool_call_id") or ""

        if name == "web_search":
            query = (
                args_dict.get("query")
                or args_dict.get("q")
                or args_dict.get("text")
                or args_dict.get("input")
                or ""
            ).strip()

            if not query:
                print(f"[TOOLS NODE] Tool {idx + 1}: ERROR - No query provided")
                out = "ERROR: web_search called without a query."
            else:
                print(f"[TOOLS NODE] Tool {idx + 1}: Searching for: {query[:100]}...")
                out = web_search.invoke({"query": query})
                print(f"[TOOLS NODE] Tool {idx + 1}: Got {len(out)} chars of results")

            raw_results.append(f"QUERY: {query}\n\n{out}")

            truncated = out[:MAX_TOOL_TEXT_CHARS] + ("…" if len(out) > MAX_TOOL_TEXT_CHARS else "")
            tool_messages.append(
                ToolMessage(
                    content=f"QUERY: {query}\n\n{truncated}",
                    tool_call_id=tool_call_id,
                )
            )
            
        elif name == "get_current_time":
            print(f"[TOOLS NODE] Tool {idx + 1}: Getting current time")
            out = get_current_time.invoke({})
            print(f"[TOOLS NODE] Tool {idx + 1}: Current time: {out}")
            
            raw_results.append(f"CURRENT_TIME: {out}")
            tool_messages.append(
                ToolMessage(
                    content=f"Current time: {out}",
                    tool_call_id=tool_call_id,
                )
            )
            
        else:
            print(f"[TOOLS NODE] Tool {idx + 1}: ERROR - Unknown tool '{name}'")
            tool_messages.append(
                ToolMessage(content=f"ERROR: Unknown tool '{name}'.", tool_call_id=tool_call_id)
            )
            raw_results.append(f"ERROR: Unknown tool '{name}'.")

    print(f"[TOOLS NODE] Completed all tool executions, returning {len(tool_messages)} tool messages")
    return {"messages": tool_messages, "last_tool_results": raw_results}


async def notes_node(state: DeepResearchState) -> Dict[str, Any]:
    print("[NOTES NODE] Starting notes summarization")
    new_material = "\n\n---\n\n".join(state.get("last_tool_results") or [])
    if not new_material.strip():
        print("[NOTES NODE] No new material to summarize")
        return {"last_tool_results": []}

    print(f"[NOTES NODE] Summarizing {len(new_material)} chars of new material")
    summary = await notes_llm.ainvoke(
        [
            SystemMessage(content=NOTES_SUMMARIZER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User question:\n{state['question']}\n\n"
                    f"Existing notes:\n{state['research_notes'] or '(none)'}\n\n"
                    f"New search results:\n{new_material}\n\n"
                    f"Return updated notes only."
                )
            ),
        ]
    )

    print(f"[NOTES NODE] Generated {len(summary.content)} chars of notes")
    return {"research_notes": summary.content, "last_tool_results": []}


def route_after_research(state: DeepResearchState) -> str:
    print("[ROUTER] Determining next step after research")
    # Stop researching after MAX_ITERS
    if state["iteration"] >= MAX_ITERS:
        print(f"[ROUTER] Max iterations ({MAX_ITERS}) reached, ending research")
        return "end"

    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        content = (last.content or "").lstrip()
        if content.startswith("WRITE_FINAL"):
            print("[ROUTER] AI signaled WRITE_FINAL, ending research")
            return "end"

    if _extract_tool_calls(last):
        print("[ROUTER] Tool calls found, routing to tools")
        return "tools"

    print("[ROUTER] No tool calls, ending research")
    return "end"


async def stream_final_answer_tokens(
    *,
    final_messages: list[dict],
    model: str,
) -> AsyncIterator[str]:
    """
    Yields token deltas as strings using direct httpx SSE streaming.
    """
    print(f"[STREAM FINAL] Starting final answer generation with model: {model}")
    
    url = "https://inference.do-ai.run/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GRADIENT_MODEL_ACCESS_KEY')}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model,
        "messages": final_messages,
        "stream": True,
        "max_tokens": 8192,
    }

    print("[STREAM FINAL] Opening httpx stream...")
    token_count = 0
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print(f"[STREAM FINAL] Response status: {response.status_code}")
            
            if response.status_code != 200:
                error_text = await response.aread()
                print(f"[STREAM FINAL] Error response: {error_text}")
                raise Exception(f"API error {response.status_code}: {error_text}")
            
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                
                # Process complete SSE events
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    
                    for line in event.split("\n"):
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                print(f"[STREAM FINAL] Got [DONE], total chunks: {token_count}")
                                return
                            
                            try:
                                parsed = json.loads(data)
                                choices = parsed.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        token_count += 1
                                        print(f"[STREAM FINAL] Chunk {token_count}: {content[:30] if len(content) > 30 else content}")
                                        yield content
                            except json.JSONDecodeError:
                                pass
    
    print(f"[STREAM FINAL] Completed streaming, generated ~{token_count} token chunks")

# ============================================================
# Build graph (research loop only)
# ============================================================

graph = StateGraph(DeepResearchState)
graph.add_node("research", research_node)
graph.add_node("tools", tools_node)
graph.add_node("notes", notes_node)

graph.set_entry_point("research")

graph.add_conditional_edges(
    "research",
    route_after_research,
    {
        "tools": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "notes")
graph.add_edge("notes", "research")

graph = graph.compile()

@entrypoint
async def entry(data, context):
    """Deep research agent that streams progress and final answer."""
    question = data["prompt"]

    initial_state = {
        "question": question,
        "iteration": 0,
        "research_notes": "",
        "last_tool_results": [],
        "messages": [
            SystemMessage(content=DEEP_RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
    }

    yield {"type": "status", "message": "Started."}

    final_notes = None

    async for ev in graph.astream_events(initial_state, version="v1"):
        ev_type = ev.get("event")
        name = ev.get("name")
        ev_data = ev.get("data") or {}

        if ev_type == "on_node_start":
            yield {"type": "status", "message": f"Entering phase: {name}"}

        elif ev_type == "on_tool_start":
            tool_input = ev_data.get("input")
            q = ""
            if isinstance(tool_input, dict):
                q = tool_input.get("query") or tool_input.get("q") or ""
            elif isinstance(tool_input, str):
                q = tool_input
            msg = f"Running tool: {name}" + (f" | query: {q}" if q else "")
            yield {"type": "status", "message": msg}

        elif ev_type in ("on_node_end", "on_chain_end", "on_graph_end"):
            out = _maybe_get_output_dict(ev)
            if isinstance(out, dict):
                rn = out.get("research_notes")
                if isinstance(rn, str) and rn.strip():
                    final_notes = rn

    if final_notes is None:
        final_notes = ""

    yield {"type": "status", "message": "Research complete... Generating final research document."}

    final_messages = [
        {"role": "system", "content": FINAL_WRITER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                f"Research notes:\n{final_notes}\n\n"
                "Write the final answer now."
            ),
        },
    ]

    # Stream final answer directly from inference client
    async for delta in stream_final_answer_tokens(
        final_messages=final_messages,
        model=FINAL_MODEL,
    ):
        yield delta
