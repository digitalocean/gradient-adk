![Header image for the DigitalOcean Gradient AI Agentic Cloud](https://doimages.nyc3.cdn.digitaloceanspaces.com/do_gradient_ai_agentic_cloud.svg)

# DigitalOcean Gradient™ Agent Development Kit (ADK)

<!-- prettier-ignore -->
[![PyPI version](https://img.shields.io/pypi/v/gradient-adk.svg?label=pypi%20(stable))](https://pypi.org/project/gradient-adk/)
[![Docs](https://img.shields.io/badge/Docs-8A2BE2)](https://docs.digitalocean.com/products/gradient-ai-platform/)

The Gradient™ Agent Development Kit (ADK) is a comprehensive toolkit for building, testing, deploying, and evaluating AI agents on DigitalOcean's Gradient™ AI Platform. It provides both a **CLI** for development workflows and a **runtime environment** for hosting agents with automatic trace capture.

## Features

### 🛠️ CLI (Command Line Interface)
- **Local Development**: Run and test your agents locally with hot-reload support
- **Seamless Deployment**: Deploy agents to DigitalOcean with a single command
- **Evaluation Framework**: Run comprehensive evaluations with custom metrics and datasets
- **Observability**: View traces and runtime logs directly from the CLI

### 🚀 Runtime Environment
- **Framework Agnostic**: Works with any Python framework for building AI agents
- **Automatic LangGraph Integration**: Built-in trace capture for LangGraph nodes and state transitions
- **Custom Decorators**: Capture traces from any framework using `@trace` decorators
- **Streaming Support**: Full support for streaming responses with trace capture
- **Production Ready**: Designed for seamless deployment to DigitalOcean infrastructure

## Installation

```bash
pip install gradient-adk
```

## Quick Start

> **🎥 Watch the [Getting Started Video](https://www.youtube.com/watch?v=23xiqgrGciE)** for a complete walkthrough

### 1. Initialize a New Agent Project

```bash
gradient agent init
```

This creates a new agent project with:
- `main.py` - Agent entrypoint with example code
- `agents/` - Directory for agent implementations
- `tools/` - Directory for custom tools
- `config.yaml` - Agent configuration
- `requirements.txt` - Python dependencies

### 2. Run Locally

```bash
gradient agent run
```

Your agent will be available at `http://localhost:8080` with automatic trace capture enabled.

### 3. Deploy to DigitalOcean

```bash
export DIGITALOCEAN_API_TOKEN=your_token_here
gradient agent deploy
```

### 4. Evaluate Your Agent

```bash
gradient agent evaluate \
  --test-case-name "my-evaluation" \
  --dataset-file evaluation_dataset.csv \
  --categories correctness,context_quality
```

## Usage Examples

### Using LangGraph (Automatic Trace Capture)

LangGraph agents automatically capture traces for all nodes and state transitions:

```python
from gradient_adk import entrypoint
from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    input: str
    output: str

async def llm_call(state: State) -> State:
    # This node execution is automatically traced
    response = await llm.ainvoke(state["input"])
    state["output"] = response
    return state

@entrypoint
async def main(input: dict, context: dict):
    graph = StateGraph(State)
    graph.add_node("llm_call", llm_call)
    graph.set_entry_point("llm_call")
    
    graph = graph.compile()
    result = await graph.ainvoke({"input": input.get("query")})
    return result["output"]
```

### Using Custom Decorators (Any Framework)

For frameworks beyond LangGraph, use trace decorators to capture custom spans:

```python
from gradient_adk import entrypoint, trace_llm, trace_tool, trace_retriever

@trace_retriever("vector_search")
async def search_knowledge_base(query: str):
    # Retriever spans capture search/lookup operations
    results = await vector_db.search(query)
    return results

@trace_llm("generate_response")
async def generate_response(prompt: str):
    # LLM spans capture model calls with token usage
    response = await llm.generate(prompt)
    return response

@trace_tool("calculate")
async def calculate(x: int, y: int):
    # Tool spans capture function execution
    return x + y

@entrypoint
async def main(input: dict, context: dict):
    docs = await search_knowledge_base(input["query"])
    result = await calculate(5, 10)
    response = await generate_response(f"Context: {docs}")
    return response
```

### Streaming Responses

The runtime supports streaming responses with automatic trace capture:

```python
from gradient_adk import entrypoint

@entrypoint
async def main(input: dict, context: dict):
    # Stream text chunks
    async def generate_chunks():
        async for chunk in llm.stream(input["query"]):
            yield chunk
```

## CLI Commands

### Agent Management

```bash
# Initialize new project
gradient agent init

# Configure existing project
gradient agent configure

# Run locally with hot-reload
gradient agent run --dev

# Deploy to DigitalOcean
gradient agent deploy

# View runtime logs
gradient agent logs

# Open traces UI
gradient agent traces
```

### Evaluation

You can evaluate your deployed agent with a number of useful evaluation metrics. See the [DigitalOcean docs](https://docs.digitalocean.com/products/gradient-ai-platform/how-to/create-evaluation-datasets/#evaluation-datasets-for-agents-built-with-agent-development-kit) for details on what belongs in a dataset.

```bash
# Run evaluation (interactive)
gradient agent evaluate

# Run evaluation (non-interactive)
gradient agent evaluate \
  --test-case-name "my-test" \
  --dataset-file data.csv \
  --categories correctness,safety_and_security \
  --star-metric-name "Correctness (general hallucinations)" \
  --success-threshold 80.0
```


## Trace Capture

The ADK runtime automatically captures detailed traces:

### What Gets Traced
- **LangGraph Nodes**: All node executions, state transitions, and edges (including LLM calls, tool calls, and DigitalOcean Knowledge Base calls)
- **LLM Calls**: Function decorated with `@trace_llm`
- **Tool Calls**: Functions decorated with `@trace_tool`
- **Retriever Calls**: Functions decorated with `@trace_retriever`
- **HTTP Requests**: Request/response payloads for LLM API calls
- **Errors**: Full exception details and stack traces
- **Streaming Responses**: Individual chunks and aggregated outputs

### Available Decorators
```python
from gradient_adk import trace_llm, trace_tool, trace_retriever

@trace_llm("model_call")      # For LLM/model invocations
@trace_tool("calculator")      # For tool/function calls
@trace_retriever("db_search")  # For retrieval/search operations
```

These decorators are used to log steps or spans of your agent workflow that are not automatically captured.  These will log things like the input, output, and step duration and make them available in your agent's traces and for use in agent evaluations.

### Viewing Traces
Traces are:
- Automatically sent to DigitalOcean's Gradient Platform
- Available in real-time through the web console
- Accessible via `gradient agent traces` command

## Environment Variables

```bash
# Required for deployment and evaluations
export DIGITALOCEAN_API_TOKEN=your_do_api_token

# Required for Gradient serverless inference (if using)
export GRADIENT_MODEL_ACCESS_KEY=your_gradient_key

# Optional: Enable verbose trace logging
export GRADIENT_VERBOSE=1
```

## Project Structure

```
my-agent/
├── main.py              # Agent entrypoint with @entrypoint decorator
├── .gradient/agent.yml  # Agent configuration (auto-generated)
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not committed)
├── agents/              # Agent implementations
│   └── my_agent.py
└── tools/               # Custom tools
    └── my_tool.py
```

## Framework Compatibility

The Gradient ADK is designed to work with any Python-based AI agent framework:

- ✅ **LangGraph** - Automatic trace capture (zero configuration)
- ✅ **LangChain** - Use trace decorators (`@trace_llm`, `@trace_tool`, `@trace_retriever`) for custom spans
- ✅ **CrewAI** - Use trace decorators for agent and task execution
- ✅ **Custom Frameworks** - Use trace decorators for any function

## Support

- **Templates/Examples**: [https://github.com/digitalocean/gradient-adk-templates](https://github.com/digitalocean/gradient-adk-templates)
- **Gradient Platform**: [https://www.digitalocean.com/products/gradient/platform](https://www.digitalocean.com/products/gradient/platform)
- **Documentation**: [https://docs.digitalocean.com/products/gradient-ai-platform/](https://docs.digitalocean.com/products/gradient-ai-platform/)
- **API Reference**: [https://docs.digitalocean.com/reference/api](https://docs.digitalocean.com/reference/api)
- **Community**: [DigitalOcean Community Forums](https://www.digitalocean.com/community)

## License

Licensed under the Apache License 2.0. See [LICENSE](./LICENSE)
