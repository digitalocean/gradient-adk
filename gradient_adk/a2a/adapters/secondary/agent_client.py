"""Secondary adapter - calls Gradient agent."""

import inspect
from typing import Any

from gradient_adk.decorator import RequestContext as GradientRequestContext

from gradient_adk.a2a.domain.models import GradientInput
from gradient_adk.a2a.domain.ports import AgentPort


class GradientAgentClient:
    """Executes Gradient agents. Implements AgentPort.

    Only supports non-streaming agent functions. Async generators
    (streaming agents) are rejected at construction time.
    """

    def __init__(self, agent_func: Any, input_key: str = "prompt"):
        """
        Args:
            agent_func: The @entrypoint decorated function (non-streaming)
            input_key: Key to use when passing prompt to agent
        """
        if inspect.isasyncgenfunction(agent_func):
            raise TypeError(
                f"Streaming agent functions are not supported in A2A mode. "
                f"Agent '{getattr(agent_func, '__name__', agent_func)}' is an async generator."
            )
        self.agent_func = agent_func
        self.input_key = input_key

    async def execute(self, agent_input: GradientInput) -> Any:
        """Execute Gradient agent with given input."""
        gradient_data = {self.input_key: agent_input.prompt}

        gradient_context = GradientRequestContext(
            session_id=agent_input.session_id,
            headers={},
        )

        if inspect.iscoroutinefunction(self.agent_func):
            result = await self.agent_func(gradient_data, gradient_context)
        else:
            result = self.agent_func(gradient_data, gradient_context)

        return result
