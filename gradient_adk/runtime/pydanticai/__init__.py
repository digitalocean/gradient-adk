"""PydanticAI runtime instrumentation for Gradient ADK."""

try:
    from gradient_adk.runtime.pydanticai.pydanticai_instrumentor import (
        PydanticAIInstrumentor,
        WRAPPED_FLAG,
    )

    __all__ = ["PydanticAIInstrumentor", "WRAPPED_FLAG"]
except ImportError:
    # pydantic-ai not installed
    __all__ = []