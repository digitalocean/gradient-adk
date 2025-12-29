"""
Simple echo agent for integration testing.
Does not make any external API calls - just echoes back the input.
"""

from gradient_adk import entrypoint


@entrypoint
async def main(query, context):
    """Echo the input back to the caller."""
    prompt = query.get("prompt", "no prompt provided")
    return {"echo": prompt, "received": query}