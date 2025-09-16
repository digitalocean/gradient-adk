"""
Advanced example showing custom response formatting.
"""

from gradient_agent import entrypoint
from typing import Dict, Any


@entrypoint
def advanced_agent(
    prompt: str, temperature: float = 0.7, metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """An agent that returns structured responses with metadata."""

    # Simulate some processing
    processed_prompt = prompt.upper()

    return {
        "completion": f"Processed: {processed_prompt}",
        "metadata": {
            "original_prompt": prompt,
            "temperature_used": temperature,
            "processing_type": "uppercase",
            "input_metadata": metadata or {},
            "agent_version": "1.0.0",
        },
    }


if __name__ == "__main__":
    from gradient_agent import run_server

    print("Starting advanced agent server...")
    print("Try:")
    print("curl -X POST http://localhost:8080/completions \\")
    print('  -H "Content-Type: application/json" \\')
    print(
        '  -d \'{"prompt": "Hello world", "temperature": 0.9, "metadata": {"user_id": "123"}}\''
    )
    run_server(port=8080)
