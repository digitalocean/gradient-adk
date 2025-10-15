from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List
import os, time, json
from gradient import Gradient
from gradient_adk import entrypoint

from gradient_adk.langgraph import attach_graph


@tool
def current_time() -> int:
    """Get current time"""
    return int(time.time())


tools = [current_time]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


inference_client = Gradient(
    model_access_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
)

from fastapi.encoders import jsonable_encoder


def _to_openai_tool_schema(tool_obj):
    params = (
        jsonable_encoder(tool_obj.args_schema.schema())
        if tool_obj.args_schema
        else {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description or "",
            "parameters": params,
        },
    }


tool_schemas = [_to_openai_tool_schema(t) for t in tools]


def call_model(state: AgentState):
    msgs = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            d = {"role": "assistant", "content": m.content or ""}
            if "tool_calls" in m.additional_kwargs:
                d["tool_calls"] = m.additional_kwargs["tool_calls"]
            msgs.append(d)
        elif isinstance(m, ToolMessage):
            msgs.append(
                {"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id}
            )
        elif isinstance(m, dict):
            msgs.append(m)
        else:
            raise ValueError(f"Unsupported message type: {type(m)}")

    resp = inference_client.chat.completions.create(
        model="llama3.3-70b-instruct",
        messages=msgs,
        tools=tool_schemas,
        tool_choice="auto",
    )

    m = resp.choices[0].message
    msg_dict = {
        "role": getattr(m, "role", "assistant"),
        "content": getattr(m, "content", None),
    }
    # normalize tool_calls to plain dicts
    if hasattr(m, "tool_calls") and m.tool_calls:
        norm = []
        for tc in m.tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None)
            args = getattr(fn, "arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args)
            norm.append(
                {
                    "id": getattr(tc, "id", ""),
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            )
        msg_dict["tool_calls"] = norm

    ai = AIMessage(
        content=msg_dict.get("content") or "",
        additional_kwargs={
            k: v for k, v in msg_dict.items() if k not in ("role", "content")
        },
    )
    return {"messages": [ai]}  # reducer will APPEND this to existing history


# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_conditional_edges("agent", tools_condition)  # loops while tool_calls exist
graph.add_edge("tools", "agent")
graph.set_entry_point("agent")
graph.set_finish_point("agent")

attach_graph(graph)
workflow = graph.compile()


@entrypoint
def entry(data, context):
    query = data["query"]
    inputs = {"messages": [HumanMessage(content=query)]}
    result = workflow.invoke(inputs)
    return result
