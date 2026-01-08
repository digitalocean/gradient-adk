"""
Agent that tests programmatic span functions.
Uses add_llm_span, add_tool_span, and add_agent_span to manually create spans.
"""

from gradient_adk import entrypoint, RequestContext, add_llm_span, add_tool_span, add_agent_span


@entrypoint
async def main(query, context: RequestContext):
    """Test all programmatic span functions."""
    prompt = query.get("prompt", "no prompt provided")

    # Add an LLM span
    add_llm_span(
        name="test_llm_call",
        input={"messages": [{"role": "user", "content": prompt}]},
        output={"response": f"Mock response to: {prompt}"},
        model="test-model",
        num_input_tokens=10,
        num_output_tokens=20,
        total_tokens=30,
        temperature=0.7,
    )

    # Add a tool span
    add_tool_span(
        name="test_tool_call",
        input={"query": prompt},
        output={"result": "tool result"},
        tool_call_id="test_call_123",
        metadata={"tool_type": "search"},
    )

    # Add an agent span
    add_agent_span(
        name="test_agent_call",
        input={"task": prompt},
        output={"answer": "agent answer"},
        metadata={"agent_version": "1.0"},
        tags=["test", "integration"],
    )

    return {
        "success": True,
        "message": "All programmatic spans created successfully",
        "prompt_received": prompt,
        "session_id": context.session_id if context else None,
    }