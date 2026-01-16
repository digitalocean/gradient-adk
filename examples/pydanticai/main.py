import os
from gradient_adk import entrypoint

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from langchain_community.tools import DuckDuckGoSearchRun


# 1. Set the Base URL and API Key for the OpenAI client using environment variables.
os.environ["OPENAI_BASE_URL"] = "https://inference.do-ai.run/v1"
os.environ["OPENAI_API_KEY"] = os.environ.get("GRADIENT_MODEL_ACCESS_KEY", "")


# 2. Configure the Model
model = OpenAIModel("openai-gpt-4o")

# 3. Define the Agent
agent = Agent(
    name="MyAgent",
    model=model,
    system_prompt="You are a helpful assistant.",
    deps_type=None,
)

# 4. Define the Agent
agenttwo = Agent(
    name="SecondAgent",
    model=model,
    system_prompt="You are a helpful assistant.",
    deps_type=None,
)


# 4. Define the Web Search Tool
@agent.tool
def web_search(ctx: RunContext[None], query: str) -> str:
    """Perform a web search using DuckDuckGo.  This is used to augment any user queries that require current information from the web."""
    search = DuckDuckGoSearchRun()
    return search.run(query)


# 5. Define the Current Time Tool
@agent.tool
def current_time(ctx: RunContext[None]) -> str:
    """Obtain the current time."""
    from datetime import datetime

    return datetime.now().isoformat()


# 6. Define the Entrypoint
@entrypoint
async def entry(data, context):
    query = data["prompt"] or "how are you?"
    result = await agent.run(query)
    print(result)
    result = await agenttwo.run("Refine this result: " + result.output)
    return result.output
