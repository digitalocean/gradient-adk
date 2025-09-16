# Gradient Agent

A Python SDK for building and instrumenting AI agents with FastAPI integration and comprehensive runtime tracking.

## Features

- **@entrypoint decorator** - Easy FastAPI server creation from agent functions
- **Runtime instrumentation** - Complete observability for LangGraph-based agents
- **Request tracking** - Monitor agent performance and behavior
- **Context management** - Maintain state across agent executions

## Quick Start

```python
from gradient_sdk import entrypoint

@entrypoint
def my_agent(prompt: str, max_tokens: int = 100) -> str:
    """Simple echo agent"""
    return f"Echo: {prompt} [max_tokens={max_tokens}]"

# The decorator automatically creates a FastAPI server
# Access at http://localhost:8080 when running
```

## Installation

```bash
pip install gradient-agent
```

## Usage

### Basic Agent

```python
from gradient_agent import entrypoint

@entrypoint
def simple_agent(prompt: str) -> str:
    return f"Response to: {prompt}"
```

### LangGraph Integration

```python
from gradient_agent import entrypoint
from langgraph.graph import StateGraph
from gradient_agent.runtime import get_runtime_manager

@entrypoint
def langgraph_agent(prompt: str) -> str:
    # Your LangGraph agent code here
    # Runtime tracking is automatically enabled
    pass
```

### Manual Server Control

```python
from gradient_agent import get_app
import uvicorn

# Import your agent modules first
import my_agent

# Get the FastAPI app
app = get_app()

# Run with custom configuration
uvicorn.run(app, host="0.0.0.0", port=8080)
```

## API Endpoints

When you use the `@entrypoint` decorator, your agent automatically gets:

- `POST /completions` - Main agent endpoint
- `GET /health` - Health check
- `GET /` - Basic info

## Runtime Tracking

The SDK includes comprehensive runtime tracking for LangGraph-based agents:

- Node execution timing
- Request context management  
- Performance metrics
- Error tracking

## Development

This SDK is part of the larger Gradient ecosystem for AI agent development and deployment.

## License

MIT
