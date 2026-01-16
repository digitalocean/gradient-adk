"""
Simple template agent using the gradient agent runtime with Gradient SDK (serverless inference) and LangGraph.
"""

import os
import asyncio
from typing import Any, Dict, TypedDict, List, Optional

import httpx
from gradient import AsyncGradient
from gradient_adk import entrypoint
from langgraph.graph import StateGraph
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download("stopwords")
nltk.download("punkt_tab")


async def retrieve_from_knowledge_base(
    query: str, knowledge_base_uuid: str, num_results: int = 10, alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from a knowledge base using the DigitalOcean Knowledge Base API.

    Args:
        query: The search query string
        knowledge_base_uuid: UUID of the knowledge base
        num_results: Number of results to return (0-100)
        alpha: Hybrid search weighting (0-1)

    Returns:
        List of retrieved chunks with metadata and text_content
    """
    api_token = os.environ.get("DIGITALOCEAN_API_TOKEN")
    if not api_token:
        print(
            "Warning: DIGITALOCEAN_API_TOKEN not set, skipping knowledge base retrieval"
        )
        return []

    url = f"https://kbaas.do-ai.run/v1/{knowledge_base_uuid}/retrieve"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}",
    }

    processed_query = extract_keywords(query)
    print(processed_query)

    payload = {"query": processed_query, "num_results": num_results, "alpha": alpha}

    # Retry logic with exponential backoff
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)

                # Check for specific error codes
                if response.status_code == 529:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        print(
                            f"Knowledge base service overloaded (529). Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        print(
                            f"Knowledge base service overloaded (529) after {max_retries} attempts. Proceeding without retrieval."
                        )
                        return []

                response.raise_for_status()
                data = response.json()
                return data.get("results", [])

        except httpx.HTTPStatusError as e:
            if attempt < max_retries - 1 and e.response.status_code in [
                429,
                500,
                502,
                503,
                504,
                529,
            ]:
                delay = base_delay * (2**attempt)
                print(
                    f"HTTP error {e.response.status_code}. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
                continue
            else:
                print(
                    f"Error retrieving from knowledge base: HTTP {e.response.status_code} - {e.response.text[:200]}"
                )
                return []
        except Exception as e:
            print(f"Error retrieving from knowledge base: {type(e).__name__}: {str(e)}")
            return []

    return []


def extract_keywords(text, language=None):
    """
    Extracts keywords from text by removing stopwords based on the given language.
    If no language is provided, attempts to detect it using langdetect.
    Falls back to English if detection fails or if the language isn't supported.
    """

    # Auto-detect language if not provided
    language = "english"

    stop_words = set(stopwords.words(language))
    words = word_tokenize(text)
    # Remove stopwords and punctuation
    keywords = [
        word
        for word in words
        if word.lower() not in stop_words and word not in string.punctuation
    ]

    return " ".join(keywords)


class State(TypedDict, total=False):
    """The state of our graph."""

    input: str
    output: Optional[str]
    retrieved_context: Optional[List[Dict[str, Any]]]


async def retrieve_context(state: State) -> State:
    """Retrieve relevant context from knowledge base"""

    knowledge_base_uuid = os.environ.get("KNOWLEDGE_BASE_UUID")
    if not knowledge_base_uuid:
        print("No KNOWLEDGE_BASE_ID set, skipping retrieval")
        state["retrieved_context"] = []
        return state

    # Retrieve relevant chunks
    results = await retrieve_from_knowledge_base(
        query=state["input"],
        knowledge_base_uuid=knowledge_base_uuid,
        num_results=10,
        alpha=0.5,
    )

    state["retrieved_context"] = results
    return state


async def llm_call(state: State) -> State:
    """Call the LLM with retrieved context"""

    inference_client = AsyncGradient(
        model_access_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
    )

    # Load system prompt
    system_prompt = ""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "config", "prompts", "pdocs-instructions.md"
    )
    try:
        with open(prompt_path, "r") as f:
            system_prompt = f.read()
    except Exception as e:
        print(f"Warning: Could not load system prompt from {prompt_path}: {e}")

    # Build context from retrieved chunks
    context_text = ""
    if state.get("retrieved_context"):
        context_parts = []
        for idx, chunk in enumerate(state["retrieved_context"], 1):
            text = chunk.get("text_content", "")
            metadata = chunk.get("metadata", {})
            source = metadata.get("item_name", "Unknown source")
            context_parts.append(f"[Source {idx}: {source}]\n{text}")

        context_text = "\n\n".join(context_parts)
        context_text = (
            f"Use the following context to answer the question:\n\n{context_text}\n\n"
        )

    print(context_text)
    # Build the final prompt
    user_message = context_text + state["input"]

    # Build messages with system prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    # Call the model
    output = await inference_client.chat.completions.create(
        messages=messages,
        model="openai-gpt-oss-120b",
        temperature=0.1,
        max_tokens=16384,
    )

    # Set the state
    state["output"] = output.choices[0].message.content

    return state


@entrypoint
async def main(input: Dict[str, str], context: Dict[str, str]):
    """Entrypoint"""

    # Setup the graph
    initial_state = State(
        input=input.get("prompt"), output=None, retrieved_context=None
    )
    graph = StateGraph(State)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("llm_call", llm_call)
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "llm_call")
    graph.set_finish_point("llm_call")

    # Attach the graph for instrumentation
    app = graph.compile()

    result = await app.ainvoke(initial_state)
    return result["output"]
