import json
import os
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from gradient_adk import entrypoint
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# DuckDuckGo search tool (returns text snippets, often with source links depending on wrapper/version)
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Web search via DuckDuckGo. Input should be a precise query string."""
    return search.run(query)


# -----------------------------
# DeepResearch prompts
# -----------------------------

DEEP_RESEARCH_SYSTEM_PROMPT = """You are a DeepResearch agent.

Goal: Produce an extremely detailed, high-quality answer to the user's original question.

Method:
- Work in iterative research loops.
- In each loop, identify what you still need to know, then call web_search for missing details.
- Prefer primary sources (official docs/specs/papers/standards) and reputable secondary sources.
- Cross-check important claims across multiple sources.
- If sources conflict, explicitly explain the conflict and why one is more credible.
- Do NOT invent details. If you can't verify something, label it as uncertain.

Tool use rules:
- Use the web_search tool proactively.
- Make 2–5 web_search calls per loop with different angles (definitions, official docs, comparisons, edge cases, recent changes).
- Stop searching only when you have enough evidence and coverage to write a long, structured, comprehensive answer.

When you are ready to write the final answer:
- Respond with a message that starts EXACTLY with: WRITE_FINAL
- Then provide a concise outline of the final answer (sections + key points).
- Do not call tools in that message.
"""

NOTES_SUMMARIZER_SYSTEM_PROMPT = """You compress raw search results into durable research notes.

Requirements:
- Extract key facts, numbers, definitions, and caveats.
- Keep source attribution as inline URLs if present in the text; otherwise keep the publication/site name if obvious.
- Organize by topic with bullets.
- Preserve contradictory findings (label them).
- Keep it dense and information-rich; avoid fluff.
"""

FINAL_WRITER_SYSTEM_PROMPT = """Write the final response using the research notes.

Requirements:
- Very detailed, long, and structured (clear headings, subheadings, bullet lists, and examples).
- Include concrete, actionable guidance when relevant.
- Include citations as URLs when available in notes; otherwise mention the source name (e.g., "NIST", "AWS docs") next to the claim.
- Call out uncertainty explicitly.
"""


# -----------------------------
# LangGraph (version-robust) setup
# -----------------------------

MAX_ITERS = int(os.getenv("DEEP_RESEARCH_MAX_ITERS", "6"))
MAX_TOOL_TEXT_CHARS = int(os.getenv("DEEP_RESEARCH_TOOL_TRUNC_CHARS", "3000"))

# LangGraph is imported lazily/defensively to reduce version fragility.
from langgraph.graph import END, StateGraph  # type: ignore

try:
    from langgraph.graph.message import add_messages  # type: ignore
except Exception:  # pragma: no cover
    # Older versions
    from langgraph.prebuilt import add_messages  # type: ignore


class DeepResearchState(TypedDict):
    # All conversational messages (system/human/ai/tool)
    messages: Annotated[List[BaseMessage], add_messages]
    # Compressed notes accumulated across iterations
    research_notes: str
    # Most recent raw tool outputs (not permanently appended to messages)
    last_tool_results: List[str]
    # Loop counter
    iteration: int
    # Original question
    question: str


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
    # LangChain AIMessage commonly has `.tool_calls` list[dict]
    tool_calls = getattr(msg, "tool_calls", None)
    if isinstance(tool_calls, list):
        return tool_calls
    # Some variants store tool call metadata differently
    additional = getattr(msg, "additional_kwargs", {}) or {}
    tc = additional.get("tool_calls")
    if isinstance(tc, list):
        return tc
    return []


base_llm = ChatOpenAI(
    base_url="https://inference.do-ai.run/v1",
    model="openai-gpt-oss-120b",
    api_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
    temperature=0.2,
)

# Tool-capable LLM for research decisions
research_llm = base_llm.bind_tools([web_search])

# Separate LLM instances for summarization / final writing (no tools)
notes_llm = ChatOpenAI(
    base_url="https://inference.do-ai.run/v1",
    model="openai-gpt-oss-120b",
    api_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
    temperature=0.0,
)

writer_llm = ChatOpenAI(
    base_url="https://inference.do-ai.run/v1",
    model="openai-gpt-oss-120b",
    api_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
    temperature=0.2,
)


# -----------------------------
# Graph nodes
# -----------------------------


async def research_node(state: DeepResearchState) -> Dict[str, Any]:
    """
    Decide next actions:
    - Either call web_search via tool calls, OR
    - Emit WRITE_FINAL + outline when ready.
    """
    it = state["iteration"] + 1
    guidance = SystemMessage(
        content=(
            f"{DEEP_RESEARCH_SYSTEM_PROMPT}\n\n"
            f"Iteration: {it}/{MAX_ITERS}\n\n"
            f"Current research notes (may be incomplete):\n{state['research_notes'] or '(none yet)'}"
        )
    )

    # Put the dynamic guidance after the original system message (or at the end if none)
    msgs = list(state["messages"])
    # Ensure there's at least one system prompt anchoring behavior
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=DEEP_RESEARCH_SYSTEM_PROMPT)] + msgs

    # Add dynamic guidance as an additional system message (allowed in most chat stacks)
    msgs = msgs + [guidance]

    ai = await research_llm.ainvoke(msgs)
    return {
        "messages": [ai],
        "iteration": state["iteration"] + 1,
    }


async def tools_node(state: DeepResearchState) -> Dict[str, Any]:
    """
    Execute any tool calls from the last AI message, add ToolMessages (truncated),
    and also store full tool outputs in `last_tool_results` for summarization.
    """
    last = state["messages"][-1]
    tool_calls = _extract_tool_calls(last)

    tool_messages: List[ToolMessage] = []
    raw_results: List[str] = []

    for tc in tool_calls:
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
                out = "ERROR: web_search called without a query."
            else:
                # IMPORTANT: web_search is a StructuredTool; call via .invoke()
                out = web_search.invoke({"query": query})

            raw_results.append(f"QUERY: {query}\n\n{out}")
            truncated = (
                (out[:MAX_TOOL_TEXT_CHARS] + "…")
                if len(out) > MAX_TOOL_TEXT_CHARS
                else out
            )

            tool_messages.append(
                ToolMessage(
                    content=f"QUERY: {query}\n\n{truncated}",
                    tool_call_id=tool_call_id,
                )
            )
        else:
            raw_results.append(f"Unknown tool requested: {name} args={args_dict}")
            tool_messages.append(
                ToolMessage(
                    content=f"ERROR: Unknown tool '{name}'.",
                    tool_call_id=tool_call_id,
                )
            )

    return {
        "messages": tool_messages,
        "last_tool_results": raw_results,
    }


async def notes_node(state: DeepResearchState) -> Dict[str, Any]:
    """
    Compress the newest tool outputs into `research_notes` to avoid runaway context.
    """
    new_material = "\n\n---\n\n".join(state.get("last_tool_results") or [])
    if not new_material.strip():
        return {"last_tool_results": []}

    prompt = [
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

    summary = await notes_llm.ainvoke(prompt)
    return {
        "research_notes": summary.content,
        "last_tool_results": [],
    }


async def write_node(state: DeepResearchState) -> Dict[str, Any]:
    """
    Produce the final detailed response from notes.
    """
    prompt = [
        SystemMessage(content=FINAL_WRITER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User question:\n{state['question']}\n\n"
                f"Research notes:\n{state['research_notes'] or '(none)'}\n\n"
                f"Write the final answer now."
            )
        ),
    ]
    final = await writer_llm.ainvoke(prompt)
    return {"messages": [final]}


def route_after_research(state: DeepResearchState) -> str:
    """
    Decide where to go after the research node.
    """
    # Hard stop to prevent infinite loops
    if state["iteration"] >= MAX_ITERS:
        return "write"

    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        content = (last.content or "").lstrip()
        if content.startswith("WRITE_FINAL"):
            return "write"

    tool_calls = _extract_tool_calls(last)
    if tool_calls:
        return "tools"

    # If the model didn't call tools and didn't explicitly WRITE_FINAL, assume it's ready.
    return "write"


# -----------------------------
# Build graph
# -----------------------------

graph = StateGraph(DeepResearchState)

graph.add_node("research", research_node)
graph.add_node("tools", tools_node)
graph.add_node("notes", notes_node)
graph.add_node("write", write_node)

graph.set_entry_point("research")

graph.add_conditional_edges(
    "research",
    route_after_research,
    {
        "tools": "tools",
        "write": "write",
    },
)

graph.add_edge("tools", "notes")
graph.add_edge("notes", "research")
graph.add_edge("write", END)

graph = graph.compile()


# -----------------------------
# Gradient ADK entrypoint
# -----------------------------


@entrypoint
async def entry(data, context):
    question = data["prompt"]

    initial_state: DeepResearchState = {
        "question": question,
        "iteration": 0,
        "research_notes": "",
        "last_tool_results": [],
        "messages": [
            SystemMessage(content=DEEP_RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
    }

    final_state = await graph.ainvoke(initial_state)
    return final_state["messages"][-1].content
