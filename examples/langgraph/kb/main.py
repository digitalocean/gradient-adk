import os
from gradient_adk import entrypoint
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from pydantic import BaseModel
from gradient import Gradient

client = Gradient(access_token=os.environ.get("DIGITALOCEAN_API_TOKEN"))


@tool
def query_digitalocean_kb(query: str, num_results: int) -> str:
    """Perform a query against the DigitalOcean Gradient AI knowledge base."""
    response = client.retrieve.documents(
        knowledge_base_id=os.environ.get("DIGITALOCEAN_KB_ID"),
        num_results=num_results,
        query=query,
    )
    print(response)
    if response and response.results:
        return response.results
    return []


llm = ChatOpenAI(
    base_url="https://inference.do-ai.run/v1",
    model="openai-gpt-oss-120b",
    api_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
)

agent = create_agent(
    llm,
    tools=[query_digitalocean_kb],
    system_prompt="You are a helpful assistant that will answer questions about DigitalOcean Gradient AI Platform.",
)


class Message(BaseModel):
    content: str


@entrypoint
async def entry(data, context):
    query = data["prompt"]
    inputs = {"messages": [HumanMessage(content=query)]}
    result = await agent.ainvoke(inputs)
    return result["messages"][-1].content
