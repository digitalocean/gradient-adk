# Gradient Agent CLI - Usage Guide

## Overview

The Gradient Agent CLI provides a complete workflow for building and deploying AI agents with FastAPI servers. The system includes:

1. **SDK with @entrypoint decorator** - Easy way to create agent functions  
2. **Configuration management** - YAML-based agent configuration
3. **Local development server** - Direct FastAPI execution for rapid development

## Quick Start

### 1. Create an Agent Function

```python
# test_agent.py
from gradient.sdk import entrypoint

@entrypoint
def my_agent(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
    """Simple echo agent for testing"""
    return f"Echo: {prompt} [max_tokens={max_tokens}, temperature={temperature}]"
```

### 2. Initialize Your Agent

```bash
# Interactive mode
gradient agent init

# Non-interactive mode
gradient agent init \
  --agent-name my-echo-agent \
  --agent-environment prod \
  --entrypoint-file test_agent.py \
  --no-interactive
```

This creates:
- `.gradient/agent.yml` - Configuration file

### 3. Run Locally

```bash
gradient agent run
```

This starts a FastAPI server directly with your agent accessible at `http://localhost:8080`.

## ✅ **Verified Working Example**

The complete workflow has been tested and verified:

1. **Agent Creation**: `test_agent.py` with `@entrypoint` decorator
2. **Configuration**: Generates `.gradient/agent.yml` configuration file
3. **Direct Server**: Starts FastAPI server using uvicorn for immediate development
4. **API Testing**: Both `/completions` and `/health` endpoints working correctly

```bash
# Test the running agent
curl -X POST http://localhost:8080/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello from the server!", "max_tokens": 50, "temperature": 0.8}'

# Response: {"completion":"Echo: Hello from the server! [max_tokens=50, temperature=0.8]","metadata":null}
```

## API Endpoints

Your agent automatically gets these endpoints:

### POST /completions
```json
{
  "prompt": "Hello world",
  "max_tokens": 50,
  "temperature": 0.8
}
```

Response:
```json
{
  "completion": "Your agent's response",
  "metadata": null
}
```

### GET /health
```json
{
  "status": "healthy"
}
```

## CLI Commands

### Main Commands
- `gradient --help` - Show all available commands
- `gradient auth` - Authentication management
- `gradient agent` - Agent configuration and management
  - `gradient agent init` - Initialize agent configuration
  - `gradient agent run` - Run agents locally

### Agent Configuration
```bash
gradient agent init [OPTIONS]

Options:
  --agent-name TEXT         Name of the agent
  --agent-environment TEXT  Environment (dev/staging/prod)
  --entrypoint-file TEXT    Python file containing @entrypoint functions
  --no-interactive         Skip interactive prompts
```

### Run Options
```bash
gradient agent run

Runs the agent locally using FastAPI server
```

## File Structure

After configuration, your project will have:

```
your-project/
├── test_agent.py           # Your agent code
├── .gradient/
│   └── agent.yml          # Agent configuration  
└── requirements.txt       # Python dependencies (optional)
```

## Generated Configuration

### agent.yml
```yaml
agent_name: my-echo-agent
agent_environment: prod
entrypoint_file: test_agent.py
```

## Testing

You can test your agent locally during development:

```python
from your_agent_file import *  # Import to register @entrypoint
from gradient.sdk.decorator import get_app
from fastapi.testclient import TestClient

app = get_app()
client = TestClient(app)

response = client.post('/completions', json={
    'prompt': 'test prompt',
    'max_tokens': 50
})
print(response.json())
```

## Requirements

- Python 3.8+
- gradient-cli package

## Installation

```bash
pip install gradient-cli
```

## Architecture

The system uses a clean architecture pattern:

- **CLI Layer** (`gradient.cli.cli`) - Command definitions
- **Interface Layer** (`gradient.cli.interfaces`) - Abstract service contracts
- **Service Layer** (`gradient.cli.services`) - Concrete implementations
- **SDK Layer** (`gradient.sdk`) - Developer-facing decorator and tools

This separation allows for easy testing, mocking, and future extensions.
