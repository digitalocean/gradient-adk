import os
from gradient_adk import entrypoint
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from pydantic import BaseModel

## Define search tool
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Perform a web search using DuckDuckGo."""
    results = search.run(query)
    return results


llm = ChatOpenAI(
    base_url="https://inference.do-ai.run/v1",
    model="openai-gpt-oss-120b",
    api_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
)

agent = create_agent(
    llm, tools=[web_search], system_prompt="You are a helpful assistant."
)


class Message(BaseModel):
    content: str


@entrypoint
async def entry(data, context):
    query = data["prompt"]
    inputs = {"messages": [HumanMessage(content=query)]}
    result = await agent.ainvoke(inputs)
    return result["messages"][-1].content
