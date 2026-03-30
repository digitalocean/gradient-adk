![Header image for the DigitalOcean Gradient AI Agentic Cloud](https://doimages.nyc3.cdn.digitaloceanspaces.com/do_gradient_ai_agentic_cloud.svg)

# DigitalOcean Gradient™ Agent Development Kit (ADK)

<!-- prettier-ignore -->
[![PyPI version](https://img.shields.io/pypi/v/gradient-adk.svg?label=pypi%20(stable))](https://pypi.org/project/gradient-adk/)
[![Docs](https://img.shields.io/badge/Docs-8A2BE2)](https://docs.digitalocean.com/products/gradient-ai-platform/how-to/build-agents-using-adk/)

The DigitalOcean Gradient™ Agent Development Kit (ADK) is a Python toolkit designed to help you build, deploy, and operate production-grade AI agents with zero infrastructure overhead.

Building AI agents is challenging enough without worrying about observability, evaluations, and deployment infrastructure. We built the Gradient™ ADK with one simple aim: **bring your agent code, and we handle the rest**—bringing the simplicity you love about DigitalOcean to AI agents.

## Why Use DigitalOcean Gradient™ ADK?

- **Framework Agnostic**: Bring your existing agent code—whether built with LangGraph, LangChain, CrewAI, PydanticAI, or any Python framework. No rewrites, no lock-in.

- **Pay Per Use**: Only pay for what you use with serverless agent hosting. **Currently provided at no compute cost during Public Preview!**

- **Any LLM Provider**: Use OpenAI, Anthropic, Google, or DigitalOcean's own Gradient™ AI serverless inference—your choice, your keys.

- **Built-in Observability**: Get automatic traces, evaluations, and insights out of the box. No OpenTelemetry setup, no third-party integrations required.

- **Production Ready from Day One**: Deploy with a single command to DigitalOcean's managed infrastructure. Focus on building your agent, not managing servers.

- **Seamless DigitalOcean Integration**: Connect effortlessly to the DigitalOcean ecosystem—Knowledge Bases for RAG, Serverless Inference for LLMs, built-in Evaluations, and more.

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

To use local evaluations (powered by DeepEval), install with the eval extra:

```bash
pip install gradient-adk[eval]
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

Run evaluations against a deployed agent on DigitalOcean:

```bash
gradient agent evaluate \
  --test-case-name "my-evaluation" \
  --dataset-file evaluation_dataset.csv \
  --categories correctness,context_quality
```

Or run evaluations locally using DeepEval (no deployment needed):

```bash
gradient agent evaluate --local --preset basic --dataset-file dataset.csv
```

See the [Local Evaluations](#local-evaluations) section for full details.

## Usage Examples

### Using LangGraph (Automatic Trace Capture)

LangGraph agents automatically capture traces for all nodes and state transitions:

```python
from gradient_adk import entrypoint, RequestContext
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
async def main(input: dict, context: RequestContext):
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
from gradient_adk import entrypoint, trace_llm, trace_tool, trace_retriever, RequestContext

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
async def main(input: dict, context: RequestContext):
    docs = await search_knowledge_base(input["query"])
    result = await calculate(5, 10)
    response = await generate_response(f"Context: {docs}")
    return response
```

### Streaming Responses

The runtime supports streaming responses with automatic trace capture:

```python
from gradient_adk import entrypoint, RequestContext

@entrypoint
async def main(input: dict, context: RequestContext):
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

### Remote Evaluation

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

## Local Evaluations

The ADK includes an integrated local evaluation framework powered by [DeepEval](https://github.com/confident-ai/deepeval). Run evaluations locally against your agent code — no deployment required. A judge LLM (via DigitalOcean Serverless Inference or any OpenAI-compatible endpoint) scores your agent's responses across a configurable set of metrics.

### How It Works

Local evaluation runs entirely in-process:

1. `gradient agent evaluate --local` imports your agent module and gets the FastAPI app
2. For each row in the dataset CSV, the runner sends a request to `/run` via ASGI transport (no network, no subprocess)
3. The `@entrypoint` decorator detects eval headers and creates an `EvalRecord` via ContextVar
4. Your agent code calls `eval_record()` to attach retrieval context and tool call data — these are silent no-ops in production
5. After each response, the runner assembles DeepEval `LLMTestCase` objects and evaluates with the judge LLM
6. Metrics that lack required data (e.g. `tool_correctness` without `expected_tools` in the CSV) are auto-skipped with actionable reasons

### Installation

```bash
pip install gradient-adk[eval]
```

### Instrumenting Your Agent

Import `eval_record` from `gradient_adk` and call it during request handling to record retrieval context and tool calls. In production (outside evaluation), these calls are silent no-ops with zero overhead.

```python
from gradient_adk import entrypoint, RequestContext, eval_record

@entrypoint
async def main(input: dict, context: RequestContext):
    prompt = input.get("prompt", "")
    rec = eval_record()

    # Record a tool call and retrieval context
    chunks = retrieve(prompt)
    rec.add_tool_call("retrieve", args={"query": prompt}, output=chunks)
    rec.add_context(chunks)

    # ... call LLM, process response ...

    answer = summarize(raw_answer)
    rec.add_tool_call("summarize", args={"text": raw_answer}, output=answer)

    return answer
```

- `rec.add_context(chunks)` — records retrieval context for faithfulness and contextual metrics
- `rec.add_tool_call(name, args, output)` — records tool invocations for tool correctness metrics

### Creating a Dataset

Create a CSV file with at minimum a `query` column. Additional columns enable more metrics:

| Column | Required | Format | Enables |
|--------|----------|--------|---------|
| `query` | Yes | JSON (e.g. `{"prompt": "What is Python?"}`) | All metrics |
| `expected_output` | No | String | answer_relevancy |
| `expected_context` | No | JSON list of strings | contextual_precision, contextual_recall |
| `expected_tools` | No | JSON list of objects (e.g. `[{"name": "retrieve"}]`) | tool_correctness |

Example `dataset.csv`:

```csv
query,expected_output,expected_context,expected_tools
"{""prompt"": ""What is Python?""}",Python is a high-level interpreted programming language.,"[""Python is a high-level, interpreted programming language.""]","[{""name"": ""retrieve""}, {""name"": ""summarize""}]"
```

### Configuration

Create `.gradient/eval.yml` in your project to configure the judge model, presets, and thresholds:

```yaml
# Judge model for LLM-as-judge evaluation (uses DO serverless inference).
# The "openai/" prefix tells LiteLLM to use the OpenAI-compatible API format.
# All DigitalOcean models are OpenAI API compliant, so the format is:
#   openai/<DO_MODEL_NAME>
judge_model: "openai/openai-gpt-oss-120b"
judge_base_url: "https://inference.do-ai.run/v1"
judge_api_key_env: "GRADIENT_MODEL_ACCESS_KEY"

# Default metric preset: basic | rag | agent | all
preset: "basic"

# Per-metric threshold overrides (default: 0.5)
thresholds:
  answer_relevancy: 0.7
  faithfulness: 0.8

# Per-metric judge model overrides
# metrics:
#   faithfulness:
#     threshold: 0.8
#     judge_model: "openai/meta-llama/Meta-Llama-3.1-405B-Instruct"
```

All settings can be overridden via CLI flags. Precedence: CLI args > YAML > defaults.

> **Note on model names:** The `openai/` prefix is a [LiteLLM routing convention](https://docs.litellm.ai/docs/providers/openai_compatible) that tells the evaluation framework to use the OpenAI-compatible chat completions format. It does not mean the model is hosted by OpenAI. All DigitalOcean Serverless Inference models are OpenAI API compliant, so the format is always `openai/<DO_MODEL_NAME>`.

### Available Metrics

Metrics are organized into three presets:

| Preset | Metrics | Data Required |
|--------|---------|---------------|
| `basic` | answer_relevancy, bias, toxicity | Just input/output — zero config |
| `rag` | faithfulness, contextual_relevancy, contextual_precision, contextual_recall | `eval_record().add_context()` + `expected_context` in CSV |
| `agent` | tool_correctness | `eval_record().add_tool_call()` + `expected_tools` in CSV |
| `all` | All 8 metrics | Auto-skips metrics missing required data |

All scores are normalized so that **1.0 = best** for every metric (including bias and toxicity, where the underlying raw score is inverted).

### Running Evaluations

```bash
# All 8 metrics, verbose per-row breakdown:
gradient agent evaluate --local --preset all --dataset-file dataset.csv --verbose

# Basic metrics only (no retrieval context or tool data needed):
gradient agent evaluate --local --preset basic --dataset-file dataset.csv

# RAG metrics only:
gradient agent evaluate --local --preset rag --dataset-file dataset.csv

# Specific metrics:
gradient agent evaluate --local --metrics answer_relevancy,faithfulness,tool_correctness --dataset-file dataset.csv

# Override judge model and threshold (openai/ prefix = LiteLLM OpenAI-compatible format):
gradient agent evaluate --local --preset basic --dataset-file dataset.csv \
  --judge-model "openai/meta-llama/Meta-Llama-3.1-405B-Instruct" \
  --threshold 0.7
```

#### CLI Options

| Flag | Description |
|------|-------------|
| `--local` | Run evaluation locally with DeepEval (required for local evals) |
| `--dataset-file <path>` | Path to the CSV dataset file |
| `--preset <name>` | Metric preset: `basic`, `rag`, `agent`, or `all` |
| `--metrics <names>` | Comma-separated list of specific metrics to run |
| `--judge-model <model>` | Override the judge model (format: `openai/<DO_MODEL_NAME>`) |
| `--threshold <float>` | Global pass/fail threshold (default: 0.5) |
| `--verbose` | Show per-row score breakdown |

### Example Output

```
Judge model:  openai/openai-gpt-oss-120b
Preset:       all
Dataset:      dataset.csv
Entrypoint:   agent.py

============================================================
  Evaluation Results
============================================================

  Metric                        Score   Threshold  Result
  --------------------------------------------------------
  answer_relevancy              1.00        0.50    PASS
  bias                          1.00        0.50    PASS
  toxicity                      1.00        0.50    PASS
  faithfulness                  1.00        0.50    PASS
  contextual_relevancy          1.00        0.50    PASS
  contextual_precision          1.00        0.50    PASS
  contextual_recall             1.00        0.50    PASS
  tool_correctness              1.00        0.50    PASS

============================================================
  8/8 metrics passed across 3 test case(s)
  Total time: 11.2s
============================================================
```


## Tracing

The ADK provides comprehensive tracing capabilities to capture and analyze your agent's execution. You can use **decorators** for wrapping functions or **programmatic functions** for manual span creation.

### What Gets Traced Automatically

- **LangGraph Nodes**: All node executions, state transitions, and edges (including LLM calls, tool calls, and DigitalOcean Knowledge Base calls)
- **HTTP Requests**: Request/response payloads for LLM API calls
- **Errors**: Full exception details and stack traces
- **Streaming Responses**: Individual chunks and aggregated outputs

### Tracing Decorators

Use decorators to automatically trace function executions:

```python
from gradient_adk import entrypoint, trace_llm, trace_tool, trace_retriever, RequestContext

@trace_llm("model_call")
async def call_model(prompt: str):
    """LLM spans capture model calls with token usage."""
    response = await llm.generate(prompt)
    return response

@trace_tool("calculator")
async def calculate(x: int, y: int):
    """Tool spans capture function/tool execution."""
    return x + y

@trace_retriever("vector_search")
async def search_docs(query: str):
    """Retriever spans capture search/lookup operations."""
    results = await vector_db.search(query)
    return results

@entrypoint
async def main(input: dict, context: RequestContext):
    docs = await search_docs(input["query"])
    result = await calculate(5, 10)
    response = await call_model(f"Context: {docs}")
    return response
```

### Programmatic Span Functions

For more control over span creation, use the programmatic functions. These are useful when you can't use decorators or need to add spans for code you don't control:

```python
from gradient_adk import entrypoint, add_llm_span, add_tool_span, add_agent_span, RequestContext

@entrypoint
async def main(input: dict, context: RequestContext):
    # Add an LLM span with detailed metadata
    response = await external_llm_call(input["query"])
    add_llm_span(
        name="external_llm_call",
        input={"messages": [{"role": "user", "content": input["query"]}]},
        output={"response": response},
        model="gpt-4",
        num_input_tokens=100,
        num_output_tokens=50,
        temperature=0.7,
    )

    # Add a tool span
    tool_result = await run_tool(input["data"])
    add_tool_span(
        name="data_processor",
        input={"data": input["data"]},
        output={"result": tool_result},
        tool_call_id="call_abc123",
        metadata={"tool_version": "1.0"},
    )

    # Add an agent span for sub-agent calls
    agent_result = await call_sub_agent(input["task"])
    add_agent_span(
        name="research_agent",
        input={"task": input["task"]},
        output={"result": agent_result},
        metadata={"agent_type": "research"},
        tags=["sub-agent", "research"],
    )

    return {"response": response, "tool_result": tool_result, "agent_result": agent_result}
```

#### Available Span Functions

| Function           | Description                       | Key Optional Fields                                                                                                |
| ------------------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `add_llm_span()`   | Record LLM/model calls            | `model`, `temperature`, `num_input_tokens`, `num_output_tokens`, `total_tokens`, `tools`, `time_to_first_token_ns` |
| `add_tool_span()`  | Record tool/function executions   | `tool_call_id`                                                                                                     |
| `add_agent_span()` | Record agent/sub-agent executions | —                                                                                                                  |

**Common optional fields for all span functions:** `duration_ns`, `metadata`, `tags`, `status_code`

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

# Optional: A2A protocol — base URL for AgentCard discovery
export A2A_BASE_URL=https://your-app.ondigitalocean.app
```

## Project Structure

```
my-agent/
├── main.py                       # Agent entrypoint with @entrypoint decorator
├── .gradient/
│   ├── agent.yml                 # Agent configuration (auto-generated)
│   ├── eval.yml                  # Local evaluation configuration (optional)
│   └── .gradientignore           # Controls which files are excluded from deployment
├── dataset.csv                   # Evaluation dataset (optional)
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (not committed)
├── agents/                       # Agent implementations
│   └── my_agent.py
└── tools/                        # Custom tools
    └── my_tool.py
```

### Controlling Deployment Contents (`.gradientignore`)

When you deploy with `gradient agent deploy`, the CLI zips your project directory and uploads it. The file `.gradient/.gradientignore` controls which files and directories are excluded from that zip. It is created automatically with sensible defaults when you run `gradient agent init`.

The syntax is one pattern per line:

```
# Comments start with #
dir_name/     # Exclude directories with this name anywhere in the tree
*.ext         # Exclude files matching this extension
exact_name    # Exclude exact file or directory name matches
```

The default `.gradientignore` excludes virtual environments (`env/`, `venv/`, `.venv/`), Python caches (`__pycache__/`, `*.pyc`), version control (`.git/`), build artifacts (`dist/`, `build/`, `*.egg-info`), test caches (`.pytest_cache/`, `.mypy_cache/`), and zip files (`*.zip`).

To customize, edit `.gradient/.gradientignore` directly. For example, to also exclude a local test data directory:

```
# ... existing patterns ...
test_data/
scripts/
```

This is intentionally separate from `.gitignore` because that files you track in git (like setup scripts or test fixtures) may not be needed in your deployed agent.

## Framework Compatibility

The Gradient ADK is designed to work with any Python-based AI agent framework:

- ✅ **LangGraph** - Automatic trace capture (zero configuration)
- ✅ **LangChain** - Use trace decorators (`@trace_llm`, `@trace_tool`, `@trace_retriever`) for custom spans
- ✅ **CrewAI** - Use trace decorators for agent and task execution
- ✅ **Custom Frameworks** - Use trace decorators for any function

## A2A Protocol Support

The Gradient ADK supports the [Agent-to-Agent (A2A) protocol v0.3.0](https://github.com/google/A2A), enabling any `@entrypoint` agent to communicate with A2A-compatible clients. Install with `pip install gradient-adk[a2a]`.

### Wrapping an Agent with A2A

Any `@entrypoint` agent can be exposed as an A2A server with no code changes:

```python
from gradient_adk import entrypoint
from gradient_adk.a2a import create_a2a_server

@entrypoint
async def my_agent(data: dict, context) -> dict:
    return {"output": f"You said: {data.get('prompt', '')}"}

app = create_a2a_server(my_agent)
```

Run with `uvicorn my_module:app --host 0.0.0.0 --port 8000`. The agent is discoverable at `/.well-known/agent-card.json` and accepts JSON-RPC calls (`message/send`, `tasks/get`, `tasks/cancel`).

### How the Protocol Works

A2A uses a discover-then-call pattern over JSON-RPC. Here is the full client-server flow:

1. **Discover** — The client fetches the AgentCard at `GET /.well-known/agent-card.json`. This returns the agent's name, transport URL, supported capabilities, and input/output modes. The client uses this to decide whether it can talk to this agent.

2. **Send** — The client sends a message via `POST /` with JSON-RPC method `message/send`. The server validates the message (text-only in MVP), creates a task, executes the agent, and returns a `Task` object with a `taskId` and current status.

3. **Poll** — The client checks task progress via `tasks/get` with the `taskId`. Once the task reaches a terminal state (`completed`, `failed`, or `canceled`), the response includes the agent's output in the task artifacts. The `historyLength` parameter controls how much conversation history is returned.

4. **Cancel** (optional) — The client can request cancellation via `tasks/cancel`. This is best-effort and idempotent — if the agent already finished, the cancel is a no-op.

```
Client                                 Server
  │                                      │
  ├── GET /.well-known/agent-card.json ──►  AgentCard (capabilities, URL)
  │                                      │
  ├── POST / message/send ──────────────►  Create task → Execute agent
  │◄─────────────────── Task {id, status} │
  │                                      │
  ├── POST / tasks/get ─────────────────►  Return task state + artifacts
  │◄──────────── Task {id, status, result} │
  │                                      │
  └── POST / tasks/cancel ──────────────►  Best-effort cancellation
```

### Deploying to DigitalOcean App Platform

When you deploy to App Platform, the public URL is assigned after deployment. The A2A server needs this URL for the AgentCard so that clients know where to send requests. The workflow is:

1. **Deploy your agent** to App Platform as usual with `gradient agent deploy`
2. **Get your app's public URL** from the App Platform dashboard (e.g., `https://your-agent-abc123.ondigitalocean.app`)
3. **Set the environment variable** in your app's settings:
   ```bash
   A2A_BASE_URL=https://your-agent-abc123.ondigitalocean.app
   ```
4. **Redeploy** — the agent restarts and the AgentCard now advertises the correct public URL

For local development, no configuration is needed — it defaults to `http://localhost:8000`.

### Calling a Remote A2A Agent from Another Agent

Once deployed, any A2A-compatible agent or client can call your agent:

```python
import httpx

# Discover the remote agent
card = httpx.get("https://your-agent.ondigitalocean.app/.well-known/agent-card.json").json()
rpc_url = card["url"]

# Send a message
response = httpx.post(rpc_url, json={
    "jsonrpc": "2.0", "id": "1",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello from another agent!"}],
            "message_id": "msg-1",
            "kind": "message",
        }
    },
})
task = response.json()["result"]

# Poll until done
result = httpx.post(rpc_url, json={
    "jsonrpc": "2.0", "id": "2",
    "method": "tasks/get",
    "params": {"id": task["id"]},
}).json()["result"]
```

See `examples/a2a/client.py` for a complete async client with discovery, send, poll, and cancel.

### Supported Operations

- **`message/send`**: Send a message to the agent, creates or continues a task
- **`tasks/get`**: Poll task state and retrieve results (supports `historyLength`)
- **`tasks/cancel`**: Best-effort task cancellation (idempotent)
- **Agent Discovery**: `GET /.well-known/agent-card.json` for capabilities and transport URL

Text-only input/output (`text/plain`) in the current release. Streaming, push notifications, and authenticated extended cards are explicitly disabled via AgentCard capability flags.

## Support

- **Templates/Examples**: [https://github.com/digitalocean/gradient-adk-templates](https://github.com/digitalocean/gradient-adk-templates)
- **Gradient™ AI Platform**: [https://www.digitalocean.com/products/gradient/platform](https://www.digitalocean.com/products/gradient/platform)
- **Documentation**: [https://docs.digitalocean.com/products/gradient-ai-platform/how-to/build-agents-using-adk/](https://docs.digitalocean.com/products/gradient-ai-platform/how-to/build-agents-using-adk/)
- **API Reference**: [https://docs.digitalocean.com/reference/api](https://docs.digitalocean.com/reference/api)
- **Community**: [DigitalOcean Community Forums](https://www.digitalocean.com/community)

## License

Licensed under the Apache License 2.0. See [LICENSE](./LICENSE)
