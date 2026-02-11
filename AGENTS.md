# Agent Guidelines

## Running Tests

**IMPORTANT:** Always run tests after making code changes to verify they work correctly.

### Running Tests

```bash
# Activate the virtual environment first
source /Users/tgillam/code/gradient-adk/env/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/runtime/test_crewai/crewai_instrumentor_test.py -v

# Run a specific test
python -m pytest tests/runtime/test_crewai/crewai_instrumentor_test.py::test_install_sets_installed_flag -v

# Run tests without coverage (faster)
python -m pytest tests/ -v --no-cov
```

### Test Directory Naming

**CRITICAL:** Test directories must NOT have the same name as Python packages being tested.

For example:
- BAD: `tests/runtime/crewai/` - This shadows the real `crewai` package!
- GOOD: `tests/runtime/test_crewai/` - This doesn't conflict with the real package.

When pytest runs, it adds test directories to `sys.path`. If a test directory has the same name as an installed package, Python will import the test directory instead of the real package, causing import errors like "No module named 'package.submodule'".

### Dependencies

Some tests require optional dependencies:
- `crewai` - For CrewAI instrumentor tests
- `pydantic-ai` - For PydanticAI instrumentor tests

Install optional test dependencies:
```bash
pip install crewai pydantic-ai
```

### Environment Variables for Integration Tests

Some integration tests require environment variables:
- `RUN_E2E_TESTS=1` - Enable E2E tests
- `RUN_DEPLOY_TESTS=1` - Enable deployment tests
- `GRADIENT_MODEL_ACCESS_KEY` - For LLM API calls
- `SERPER_API_KEY` - For search tool tests
