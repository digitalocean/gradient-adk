# Publishing Gradient Agent to PyPI

This guide shows you how to publish the Gradient Agent SDK components separately to PyPI.

## Package Structure

The SDK package contains:
- `@entrypoint` decorator for FastAPI server creation
- Runtime instrumentation system for LangGraph agents  
- Request tracking and performance monitoring
- Context management system

## Quick Setup

### Option 1: Development Setup

```bash
cd gradient-sdk-package
./setup-dev.sh
```

### Option 2: Manual Setup

```bash
cd gradient-sdk-package
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Testing

```bash
# Run tests
pytest tests/ -v

# Test examples
python examples/simple_agent.py
# In another terminal:
curl -X POST http://localhost:8080/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello world"}'
```

## Publishing

### 1. Prepare for Publishing

Make sure everything is committed and version is updated in `pyproject.toml`:

```bash
# Check status
git status

# Update version in pyproject.toml if needed
# Then commit changes
git add .
git commit -m "Prepare v0.1.0 release"
git tag v0.1.0
```

### 2. Build and Test

```bash
./publish.sh
```

This will:
- Run tests
- Build the package
- Verify the build
- Show you what files were created

### 3. Publish to Test PyPI (Recommended First)

```bash
# Get a token from https://test.pypi.org/manage/account/token/
python -m twine upload --repository testpypi dist/* --username __token__ --password <your-test-token>
```

Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ gradient-agent
```

### 4. Publish to PyPI

```bash
# Get a token from https://pypi.org/manage/account/token/  
python -m twine upload dist/* --username __token__ --password <your-token>
```

## GitHub Actions (Optional)

The package includes a GitHub workflow for automated publishing:

1. Set up secrets in your GitHub repo:
   - `PYPI_API_TOKEN` - Your PyPI API token
   - `TEST_PYPI_API_TOKEN` - Your Test PyPI API token

2. Create a release on GitHub to trigger publishing to PyPI
3. Use the "Publish Gradient Agent" workflow to manually publish to TestPyPI

## Usage After Publishing

Once published, users can install and use your SDK:

```bash
pip install gradient-agent
```

```python
from gradient_agent import entrypoint

@entrypoint
def my_agent(prompt: str) -> str:
    return f"Response: {prompt}"

# Automatically creates FastAPI server at http://localhost:8080
```

## Package Structure

```
gradient-agent-package/
├── gradient_agent/           # Main package
│   ├── __init__.py        # Package exports
│   ├── decorator.py       # @entrypoint decorator
│   └── runtime/           # Runtime system
│       ├── __init__.py
│       ├── manager.py     # Runtime manager
│       ├── tracker.py     # Execution tracking
│       ├── context.py     # Request context
│       ├── interfaces.py  # Interface definitions
│       └── langgraph_instrumentor.py  # LangGraph integration
├── tests/                 # Test suite
├── examples/              # Usage examples
├── pyproject.toml         # Package configuration
├── README.md              # Package documentation
├── CHANGELOG.md           # Version history
└── publish.sh             # Publishing script
```

## Differences from CLI Package

The SDK package only includes:
- Core `@entrypoint` decorator
- Runtime instrumentation system
- FastAPI integration
- Essential dependencies (FastAPI, uvicorn, pydantic)

It does NOT include:
- CLI commands (`gradient init`, `gradient run`)
- DigitalOcean doctl integration
- Typer CLI framework
- Service layer components
- Heavy CLI-specific dependencies

This keeps the SDK lightweight and focused on the core agent development functionality.
