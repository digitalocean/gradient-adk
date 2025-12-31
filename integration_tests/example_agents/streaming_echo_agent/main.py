"""
Streaming echo agent for integration testing.
Does not make any external API calls - just echoes back the input in chunks.
Used to test streaming vs non-streaming behavior with evaluation-id header.
"""

from gradient_adk import entrypoint


@entrypoint
async def main(query, context):
    """Streaming echo agent - yields the response in chunks."""
    prompt = query.get("prompt", "no prompt provided")
    # Stream the response in multiple chunks
    yield "Echo: "
    yield prompt
    yield " [DONE]"