"""Example A2A server agent using Gradient ADK @entrypoint decorator.

This example demonstrates how to create an A2A-compatible agent that can be
accessed via the A2A protocol using the Gradient ADK.

To run this agent:
1. Ensure you have gradient-adk and a2a-sdk installed:
   pip install gradient-adk[a2a]
2. When deployed, set the A2A base URL (invoke URL without /run; optional for local dev):
   export A2A_BASE_URL=https://agents.do-ai.run/<workspace-uuid>/<deployment-name>
3. Run: gradient agent run
4. The agent will be available with both Gradient and A2A protocols
"""

from gradient_adk import entrypoint


@entrypoint
async def echo_agent(data: dict, context) -> dict:
    """A simple echo agent that repeats the user's input.

    This agent:
    - Receives text input via A2A protocol
    - Echoes it back with a prefix
    - Works with both Gradient /run endpoint and A2A protocol

    Args:
        data: Dictionary containing the user's input.
               For A2A protocol, this will contain {"prompt": "user text"}
        context: Request context (not used in this simple example)

    Returns:
        Dictionary with the agent's response.
        For A2A protocol, the "output" key will be extracted as the response.
    """
    user_input = data.get("prompt", "")

    if not user_input:
        return {"output": "No input provided. Please send a message."}

    response = f"Echo: {user_input}"

    return {"output": response}


@entrypoint
async def greeting_agent(data: dict, context) -> dict:
    """A greeting agent that responds with a personalized message.

    This is an alternative example showing how to create a more
    sophisticated agent that still works with A2A protocol.

    Args:
        data: Dictionary containing the user's input
        context: Request context

    Returns:
        Dictionary with the agent's response
    """
    user_input = data.get("prompt", "").strip()

    if not user_input:
        return {"output": "Hello! What's your name?"}

    if user_input.lower().startswith("hello"):
        return {"output": f"Hello! Nice to meet you. You said: {user_input}"}
    elif user_input.lower().startswith("hi"):
        return {"output": f"Hi there! You said: {user_input}"}
    else:
        return {"output": f"Greetings! You said: {user_input}"}
