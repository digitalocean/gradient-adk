import os
from typing import Dict, AsyncGenerator, List

from gradient import AsyncGradient
from gradient_adk import (
    entrypoint,
    trace_llm,
    trace_retriever,
    add_agent_span,
    add_tool_span,
    add_llm_span,
)


@trace_retriever("search_database")
async def search_database(query: str) -> list:
    return [
        {"id": 1, "title": "Result 1", "content": "DigitalOcean made $4b in 2022!"},
        {"id": 2, "title": "Result 2", "content": "DigitalOcean made $5b in 2023!"},
    ]


@trace_llm("call_gradient_model")
async def call_model(prompt: str) -> AsyncGenerator[str, None]:
    """
    Stream tokens from Gradient and yield each chunk.
    """
    client = AsyncGradient(
        inference_endpoint="https://inference.do-ai.run",
        model_access_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
    )

    stream = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai-gpt-oss-120b",
        stream=True,
    )

    async for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta


async def prepare_context(user_query: str, search_results: list) -> str:
    context = "Search results:\n"
    for result in search_results:
        context += f"- {result['title']}: {result['content']}\n"

    return f"User query: {user_query}\n\n{context}"


async def process_response(raw_response: str) -> dict:
    return {
        "response": raw_response,
        "confidence": 0.95,
        "sources": ["database"],
    }


@entrypoint
async def main(input: Dict, context: Dict):
    """
    Streaming entrypoint.
    Every `yield` here is flushed to the client.
    """
    user_query = input.get("query", "")

    search_results = await search_database(user_query)
    prepared_context = await prepare_context(user_query, search_results)

    full_response_parts: List[str] = []

    add_llm_span(
        name="external_llm_call",
        input={"messages": [{"role": "user", "content": input["query"]}]},
        output={"response": "response"},
        model="openai-gpt-5",
        num_input_tokens=100,
        num_output_tokens=50,
        temperature=0.7,
    )

    add_tool_span(
        name="data_processor",
        input={"data": "hi"},
        output={"result": "output"},
        tool_call_id="call_abc123",
        metadata={"tool_version": "1.0"},
    )

    add_agent_span(
        name="research_agent",
        input={"task": "hi"},
        output={"result": "result"},
        metadata={"agent_type": "research"},
        tags=["sub-agent", "research"],
    )

    # 🔹 Stream from LLM AND stream outward
    async for chunk in call_model(prepared_context):
        full_response_parts.append(chunk)
        yield chunk
