# Gradient Agent (SDK + CLI)

`gradient-agent` is a unified Python package that provides:

* An SDK with the `@entrypoint` decorator and runtime instrumentation
* The `gradient` CLI for auth, agent init, and local run

## Install

```bash
pip install gradient-agent
```

Verify:
```bash
gradient --help
```

## Quick Start

Create `main.py`:
```python
from gradient_agent import entrypoint

@entrypoint
def my_agent(prompt: str) -> str:
    return f"Echo: {prompt}"
```

Initialize and run:
```bash
gradient agent init --entrypoint-file main.py --agent-name demo --agent-environment dev --no-interactive
gradient agent run
```

Send a request:
```bash
curl -X POST localhost:8080/completions -H 'Content-Type: application/json' -d '{"prompt":"hello"}'
```

## Auth (optional)
```bash
gradient auth init --token YOUR_TOKEN --no-interactive
```

## Versioning
Single package keeps CLI and SDK in sync. `gradient_agent.__version__` shows installed version.

## Migration Notes
Previous separate `gradient-cli` / `gradient-agent` packages are replaced by this unified distribution (>=0.2.0).