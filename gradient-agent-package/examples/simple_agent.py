"""
Basic example of using the Gradient Agent SDK.
"""

from gradient_agent import entrypoint


@entrypoint
def simple_agent(prompt: str, max_tokens: int = 100) -> str:
    """A simple echo agent."""
    return f"Echo: {prompt} [max_tokens={max_tokens}]"


if __name__ == "__main__":
    # When running this file directly, start the server
    from gradient_agent import run_server

    print("Starting simple agent server...")
    print("Access at http://localhost:8080")
    print(
        "Try: curl -X POST http://localhost:8080/completions -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello world\"}'"
    )
    run_server(port=8080)
