from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from typing import Dict, Optional
from gradient_adk import entrypoint

load_dotenv()

AGENT_GRAPH: Optional[StateGraph] = None

model = init_chat_model("openai:gpt-4.1")
EXA_API_KEY = os.getenv("EXA_API_KEY")

client = MultiServerMCPClient(
    {
        "search": {
            "url": f"https://mcp.exa.ai/mcp?tools=web_search_exa,company_research,linkedin_search&exaApiKey={EXA_API_KEY}",
            "transport": "streamable_http",
        }
    }
)


async def build_graph():
    print("I be building bruv")
    tools = await client.get_tools()

    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node("tools", ToolNode(tools))  # ← Add explicit name here
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    return graph


@entrypoint
async def main(input: Dict, context: Dict):
    input_request = input.get("prompt")
    global AGENT_GRAPH
    if AGENT_GRAPH is None:
        AGENT_GRAPH = await build_graph()

    # Invoke the app
    result = await AGENT_GRAPH.ainvoke(input_request)
    final_response = result
    return {"response": final_response}


# Send something like this to the agent
# {
#     "prompt" : {
#         "messages" : "who came in second place in the 2025 ICC women's cricket world cup?"
#     }
# }
