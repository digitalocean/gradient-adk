from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Tuple, Dict, List

from ..interfaces import NodeExecution
from ..digitalocean_tracker import DigitalOceanTracesTracker
from ..network_interceptor import (
    get_network_interceptor,
    is_inference_url,
    is_kbaas_url,
)


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _mk_exec(name: str, inputs: Any, framework: str = "pydanticai") -> NodeExecution:
    return NodeExecution(
        node_id=str(uuid.uuid4()),
        node_name=name,
        framework=framework,
        start_time=_utc(),
        inputs=inputs,
    )


def _ensure_meta(rec: NodeExecution) -> dict:
    md = getattr(rec, "metadata", None)
    if not isinstance(md, dict):
        md = {}
        try:
            rec.metadata = md
        except Exception:
            pass
    return md


_MAX_DEPTH = 3
_MAX_ITEMS = 100  # keep payloads bounded


def _freeze(obj: Any, depth: int = _MAX_DEPTH) -> Any:
    """Mutation-safe, JSON-ish snapshot for arbitrary Python objects."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # dict-like
    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= _MAX_ITEMS:
                out["<truncated>"] = True
                break
            out[str(k)] = _freeze(v, depth - 1)
        return out

    # sequences
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        out = []
        for i, v in enumerate(seq):
            if i >= _MAX_ITEMS:
                out.append("<truncated>")
                break
            out.append(_freeze(v, depth - 1))
        return out

    # pydantic
    try:
        from pydantic import BaseModel

        if isinstance(obj, BaseModel):
            return _freeze(obj.model_dump(), depth - 1)
    except Exception:
        pass

    # dataclass
    try:
        import dataclasses

        if dataclasses.is_dataclass(obj):
            return _freeze(dataclasses.asdict(obj), depth - 1)
    except Exception:
        pass

    # PydanticAI result types
    try:
        if hasattr(obj, "output"):
            return _freeze(obj.output, depth - 1)
        if hasattr(obj, "data"):
            return _freeze(obj.data, depth - 1)
    except Exception:
        pass

    # fallback
    return repr(obj)


def _snapshot_args_kwargs(a: Tuple[Any, ...], kw: Dict[str, Any]) -> Any:
    """Deepcopy then freeze to avoid mutation surprises."""
    try:
        a_copy = deepcopy(a)
        kw_copy = deepcopy(kw)
    except Exception:
        a_copy, kw_copy = a, kw  # best-effort

    # If there's exactly one arg and no kwargs, return just that arg
    if len(a_copy) == 1 and not kw_copy:
        return _freeze(a_copy[0])

    # If there are kwargs but no args, return just the kwargs
    if not a_copy and kw_copy:
        return _freeze(kw_copy)

    # If there are multiple args or both args and kwargs, return a dict
    if a_copy and kw_copy:
        return {"args": _freeze(a_copy), "kwargs": _freeze(kw_copy)}
    elif len(a_copy) > 1:
        return _freeze(a_copy)

    # Fallback
    return _freeze(a_copy)


def _snap():
    intr = get_network_interceptor()
    try:
        tok = intr.snapshot_token()
    except Exception:
        tok = 0
    return intr, tok


def _had_hits_since(intr, token) -> bool:
    try:
        return intr.hits_since(token) > 0
    except Exception:
        return False


def _get_captured_payloads_with_type(intr, token) -> tuple:
    """Get captured API request/response payloads and classify the call type.

    Returns:
        (request_payload, response_payload, is_llm, is_retriever)
    """
    try:
        captured = intr.get_captured_requests_since(token)
        if captured:
            # Use the first captured request (most common case)
            call = captured[0]
            url = call.url
            is_llm = is_inference_url(url)
            is_retriever = is_kbaas_url(url)
            return call.request_payload, call.response_payload, is_llm, is_retriever
    except Exception:
        pass
    return None, None, False, False


def _transform_kbaas_response(response: Optional[Dict[str, Any]]) -> Optional[list]:
    """Transform KBaaS response to standard retriever format."""
    if not isinstance(response, dict):
        return response

    results = response.get("results", [])
    if not isinstance(results, list):
        return response

    transformed_results = []
    for item in results:
        if isinstance(item, dict):
            new_item = dict(item)

            if "parent_chunk_text" in new_item:
                new_item["page_content"] = new_item.pop("parent_chunk_text")
                if "text_content" in new_item:
                    new_item["embedded_content"] = new_item.pop("text_content")
            elif "text_content" in new_item:
                new_item["page_content"] = new_item.pop("text_content")

            transformed_results.append(new_item)
        else:
            transformed_results.append(item)

    return transformed_results


def _extract_messages_input(messages: List[Any]) -> Any:
    """Extract a clean representation of the messages sent to the LLM."""
    try:
        result = []
        for msg in messages:
            if hasattr(msg, "parts"):
                # ModelRequest or ModelResponse
                msg_data = {"kind": msg.__class__.__name__, "parts": []}
                for part in msg.parts:
                    part_data = _freeze(part)
                    msg_data["parts"].append(part_data)
                if hasattr(msg, "instructions") and msg.instructions:
                    msg_data["instructions"] = msg.instructions
                result.append(msg_data)
            else:
                result.append(_freeze(msg))
        return result
    except Exception:
        return _freeze(messages)


def _extract_model_response_output(response: Any) -> Any:
    """Extract a clean representation of the model response."""
    try:
        if hasattr(response, "parts"):
            result = {"parts": []}
            for part in response.parts:
                part_data = _freeze(part)
                result["parts"].append(part_data)
            if hasattr(response, "usage") and response.usage:
                result["usage"] = _freeze(response.usage)
            if hasattr(response, "model_name") and response.model_name:
                result["model_name"] = response.model_name
            return result
        return _freeze(response)
    except Exception:
        return _freeze(response)


# ---- Workflow Context Management ----


@dataclass
class WorkflowContext:
    """Context for tracking a workflow (Agent.run) and its sub-spans."""

    node: NodeExecution
    sub_spans: List[NodeExecution] = field(default_factory=list)
    agent_name: str = ""


# Context variable to track the current workflow
_current_workflow: ContextVar[Optional[WorkflowContext]] = ContextVar(
    "pydanticai_workflow", default=None
)


def _get_current_workflow() -> Optional[WorkflowContext]:
    """Get the current workflow context, if any."""
    return _current_workflow.get()


def _set_current_workflow(ctx: Optional[WorkflowContext]) -> None:
    """Set the current workflow context."""
    _current_workflow.set(ctx)


class PydanticAIInstrumentor:
    """Wraps PydanticAI agents with tracing using workflow spans."""

    def __init__(self) -> None:
        self._installed = False
        self._tracker: Optional[DigitalOceanTracesTracker] = None
        self._original_call_tool = None
        self._original_model_requests: Dict[type, Any] = {}
        self._original_model_request_streams: Dict[type, Any] = {}
        self._original_agent_run: Any = None
        self._original_agent_run_sync: Any = None
        self._original_agent_run_stream: Any = None

    def install(self, tracker: DigitalOceanTracesTracker) -> None:
        if self._installed:
            return
        self._tracker = tracker

        try:
            from pydantic_ai import Agent
            from pydantic_ai.models import Model
        except ImportError:
            # PydanticAI not installed, skip instrumentation
            return

        t = tracker  # close over

        def _start_sub_span(node_name: str, inputs: Any):
            """Start a sub-span that will be nested inside the current workflow."""
            inputs_snapshot = _freeze(inputs)
            rec = _mk_exec(node_name, inputs_snapshot)
            intr, tok = _snap()

            # Check if we're inside a workflow context
            workflow = _get_current_workflow()
            if workflow is not None:
                # Don't call tracker.on_node_start - we'll batch these with the workflow
                pass
            else:
                # No workflow context - fall back to flat spans
                t.on_node_start(rec)

            return rec, inputs_snapshot, intr, tok

        def _finish_sub_span_ok(
            rec: NodeExecution,
            inputs_snapshot: Any,
            ret: Any,
            intr,
            tok,
            time_to_first_token_ns: Optional[int] = None,
        ):
            """Finish a sub-span successfully."""
            # Check if this node made any tracked API calls
            if _had_hits_since(intr, tok):
                api_request, api_response, is_llm, is_retriever = (
                    _get_captured_payloads_with_type(intr, tok)
                )

                meta = _ensure_meta(rec)
                if is_llm:
                    meta["is_llm_call"] = True
                    # Store raw API payloads for LLM field extraction in tracker
                    if api_request:
                        meta["llm_request_payload"] = api_request
                    if api_response:
                        meta["llm_response_payload"] = api_response
                    # Store time-to-first-token if this was a streaming call
                    if time_to_first_token_ns is not None:
                        meta["time_to_first_token_ns"] = time_to_first_token_ns
                elif is_retriever:
                    meta["is_retriever_call"] = True
                else:
                    meta["is_llm_call"] = True
                    if api_request:
                        meta["llm_request_payload"] = api_request
                    if api_response:
                        meta["llm_response_payload"] = api_response

                if api_request or api_response:
                    if api_request:
                        rec.inputs = _freeze(api_request)

                    if api_response:
                        if is_retriever:
                            api_response = _transform_kbaas_response(api_response)
                        out_payload = _freeze(api_response)
                    else:
                        out_payload = _freeze(ret)
                else:
                    out_payload = _freeze(ret)
            else:
                out_payload = _freeze(ret)

            rec.end_time = _utc()
            rec.outputs = out_payload

            # Check if we're inside a workflow context
            workflow = _get_current_workflow()
            if workflow is not None:
                # Add to workflow's sub-spans
                workflow.sub_spans.append(rec)
            else:
                # No workflow context - call tracker directly
                t.on_node_end(rec, out_payload)

        def _finish_sub_span_err(rec: NodeExecution, intr, tok, e: BaseException):
            """Finish a sub-span with an error."""
            if _had_hits_since(intr, tok):
                api_request, api_response, is_llm, is_retriever = _get_captured_payloads_with_type(
                    intr, tok
                )

                meta = _ensure_meta(rec)
                if is_llm:
                    meta["is_llm_call"] = True
                    # Store raw API payloads for LLM field extraction in tracker
                    if api_request:
                        meta["llm_request_payload"] = api_request
                    if api_response:
                        meta["llm_response_payload"] = api_response
                elif is_retriever:
                    meta["is_retriever_call"] = True
                else:
                    meta["is_llm_call"] = True
                    if api_request:
                        meta["llm_request_payload"] = api_request

                if api_request:
                    rec.inputs = _freeze(api_request)

            rec.end_time = _utc()
            rec.error = str(e)

            # Check if we're inside a workflow context
            workflow = _get_current_workflow()
            if workflow is not None:
                # Add to workflow's sub-spans
                workflow.sub_spans.append(rec)
            else:
                # No workflow context - call tracker directly
                t.on_node_error(rec, e)

        # Import FunctionToolset for tool call instrumentation
        try:
            from pydantic_ai.toolsets.function import FunctionToolset

            self._original_call_tool = FunctionToolset.call_tool
        except ImportError:
            self._original_call_tool = None

        # Get all concrete model classes that need patching
        model_classes: List[type] = []
        try:
            from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

            model_classes.extend([OpenAIChatModel, OpenAIResponsesModel])
        except ImportError:
            pass
        try:
            from pydantic_ai.models.anthropic import AnthropicModel

            model_classes.append(AnthropicModel)
        except ImportError:
            pass
        try:
            from pydantic_ai.models.google import GoogleModel

            model_classes.append(GoogleModel)
        except ImportError:
            pass
        try:
            from pydantic_ai.models.gemini import GeminiModel

            model_classes.append(GeminiModel)
        except ImportError:
            pass
        try:
            from pydantic_ai.models.groq import GroqModel

            model_classes.append(GroqModel)
        except ImportError:
            pass
        try:
            from pydantic_ai.models.mistral import MistralModel

            model_classes.append(MistralModel)
        except ImportError:
            pass
        try:
            from pydantic_ai.models.cohere import CohereModel

            model_classes.append(CohereModel)
        except ImportError:
            pass
        try:
            from pydantic_ai.models.bedrock import BedrockConverseModel

            model_classes.append(BedrockConverseModel)
        except ImportError:
            pass
        try:
            from pydantic_ai.models.huggingface import HuggingFaceModel

            model_classes.append(HuggingFaceModel)
        except ImportError:
            pass
        try:
            from pydantic_ai.models.test import TestModel

            model_classes.append(TestModel)
        except ImportError:
            pass
        try:
            from pydantic_ai.models.function import FunctionModel

            model_classes.append(FunctionModel)
        except ImportError:
            pass

        # Create wrapper factory for each model class
        def make_wrapped_request(original_request):
            async def wrapped_model_request(
                model_self, messages, model_settings, model_request_parameters
            ):
                """Wrapped Model.request that traces each LLM call as a sub-span."""
                model_name = getattr(
                    model_self, "model_name", model_self.__class__.__name__
                )
                node_name = f"llm:{model_name}"

                # Extract input from messages
                inputs = _extract_messages_input(messages)
                rec, snap, intr, tok = _start_sub_span(node_name, inputs)

                try:
                    response = await original_request(
                        model_self, messages, model_settings, model_request_parameters
                    )

                    # Extract output from the response
                    output = _extract_model_response_output(response)
                    _finish_sub_span_ok(rec, snap, output, intr, tok)
                    return response
                except BaseException as e:
                    _finish_sub_span_err(rec, intr, tok, e)
                    raise

            return wrapped_model_request

        def make_wrapped_request_stream(original_request_stream):
            @asynccontextmanager
            async def wrapped_model_request_stream(
                model_self,
                messages,
                model_settings,
                model_request_parameters,
                run_context=None,
            ):
                """Wrapped Model.request_stream that traces each streaming LLM call as a sub-span."""
                model_name = getattr(
                    model_self, "model_name", model_self.__class__.__name__
                )
                node_name = f"llm:{model_name}"

                # Extract input from messages
                inputs = _extract_messages_input(messages)
                rec, snap, intr, tok = _start_sub_span(node_name, inputs)

                try:
                    async with original_request_stream(
                        model_self,
                        messages,
                        model_settings,
                        model_request_parameters,
                        run_context,
                    ) as stream:
                        yield stream

                    # After the stream is consumed, get the response
                    response = stream.get()
                    output = _extract_model_response_output(response)
                    _finish_sub_span_ok(rec, snap, output, intr, tok)
                except BaseException as e:
                    _finish_sub_span_err(rec, intr, tok, e)
                    raise

            return wrapped_model_request_stream

        # Patch all concrete model classes
        for model_cls in model_classes:
            # Store originals
            self._original_model_requests[model_cls] = model_cls.request
            self._original_model_request_streams[model_cls] = model_cls.request_stream

            # Apply patches
            model_cls.request = make_wrapped_request(model_cls.request)
            model_cls.request_stream = make_wrapped_request_stream(
                model_cls.request_stream
            )

        # Wrap FunctionToolset.call_tool to instrument tool calls as sub-spans
        if self._original_call_tool is not None:
            from pydantic_ai.toolsets.function import FunctionToolset

            original_call_tool = self._original_call_tool

            async def wrapped_call_tool(toolset_self, name, tool_args, ctx, tool):
                """Wrapped call_tool that traces tool execution as a sub-span."""
                rec, snap, intr, tok = _start_sub_span(name, _freeze(tool_args))
                try:
                    result = await original_call_tool(
                        toolset_self, name, tool_args, ctx, tool
                    )
                    _finish_sub_span_ok(rec, snap, result, intr, tok)
                    return result
                except BaseException as e:
                    _finish_sub_span_err(rec, intr, tok, e)
                    raise

            FunctionToolset.call_tool = wrapped_call_tool

        # Wrap Agent.run, Agent.run_sync, Agent.run_stream to create workflow spans
        self._original_agent_run = Agent.run
        self._original_agent_run_sync = Agent.run_sync
        self._original_agent_run_stream = Agent.run_stream

        async def wrapped_agent_run(agent_self, user_prompt, **kwargs):
            """Wrapped Agent.run that creates a workflow span containing all sub-spans."""
            agent_name = (
                getattr(agent_self, "name", None) or agent_self.__class__.__name__
            )

            # Create workflow node
            inputs_snapshot = _freeze(user_prompt)
            workflow_node = _mk_exec(agent_name, inputs_snapshot)
            meta = _ensure_meta(workflow_node)
            meta["is_workflow"] = True
            meta["agent_name"] = agent_name

            # Create workflow context
            workflow_ctx = WorkflowContext(
                node=workflow_node,
                agent_name=agent_name,
            )

            # Set the workflow context
            prev_workflow = _get_current_workflow()
            _set_current_workflow(workflow_ctx)

            try:
                # Call original run method
                result = await self._original_agent_run(
                    agent_self, user_prompt, **kwargs
                )

                # Finish workflow node
                workflow_node.end_time = _utc()
                # Extract just the output from the result, not the full state
                if hasattr(result, "output"):
                    workflow_node.outputs = {"output": _freeze(result.output)}
                elif hasattr(result, "data"):
                    workflow_node.outputs = {"output": _freeze(result.data)}
                else:
                    workflow_node.outputs = _freeze(result)

                # Store sub-spans in metadata for the tracker to handle
                meta["sub_spans"] = workflow_ctx.sub_spans

                # Report the workflow span to the tracker
                t.on_node_start(workflow_node)
                t.on_node_end(workflow_node, workflow_node.outputs)

                return result
            except BaseException as e:
                workflow_node.end_time = _utc()
                workflow_node.error = str(e)
                meta["sub_spans"] = workflow_ctx.sub_spans

                t.on_node_start(workflow_node)
                t.on_node_error(workflow_node, e)
                raise
            finally:
                # Restore previous workflow context
                _set_current_workflow(prev_workflow)

        def wrapped_agent_run_sync(agent_self, user_prompt, **kwargs):
            """Wrapped Agent.run_sync that creates a workflow span containing all sub-spans."""
            agent_name = (
                getattr(agent_self, "name", None) or agent_self.__class__.__name__
            )

            # Create workflow node
            inputs_snapshot = _freeze(user_prompt)
            workflow_node = _mk_exec(agent_name, inputs_snapshot)
            meta = _ensure_meta(workflow_node)
            meta["is_workflow"] = True
            meta["agent_name"] = agent_name

            # Create workflow context
            workflow_ctx = WorkflowContext(
                node=workflow_node,
                agent_name=agent_name,
            )

            # Set the workflow context
            prev_workflow = _get_current_workflow()
            _set_current_workflow(workflow_ctx)

            try:
                # Call original run_sync method
                result = self._original_agent_run_sync(
                    agent_self, user_prompt, **kwargs
                )

                # Finish workflow node
                workflow_node.end_time = _utc()
                # Extract just the output from the result, not the full state
                if hasattr(result, "output"):
                    workflow_node.outputs = {"output": _freeze(result.output)}
                elif hasattr(result, "data"):
                    workflow_node.outputs = {"output": _freeze(result.data)}
                else:
                    workflow_node.outputs = _freeze(result)

                # Store sub-spans in metadata for the tracker to handle
                meta["sub_spans"] = workflow_ctx.sub_spans

                # Report the workflow span to the tracker
                t.on_node_start(workflow_node)
                t.on_node_end(workflow_node, workflow_node.outputs)

                return result
            except BaseException as e:
                workflow_node.end_time = _utc()
                workflow_node.error = str(e)
                meta["sub_spans"] = workflow_ctx.sub_spans

                t.on_node_start(workflow_node)
                t.on_node_error(workflow_node, e)
                raise
            finally:
                # Restore previous workflow context
                _set_current_workflow(prev_workflow)

        @asynccontextmanager
        async def wrapped_agent_run_stream(agent_self, user_prompt, **kwargs):
            """Wrapped Agent.run_stream that creates a workflow span containing all sub-spans."""
            agent_name = (
                getattr(agent_self, "name", None) or agent_self.__class__.__name__
            )

            # Create workflow node
            inputs_snapshot = _freeze(user_prompt)
            workflow_node = _mk_exec(agent_name, inputs_snapshot)
            meta = _ensure_meta(workflow_node)
            meta["is_workflow"] = True
            meta["agent_name"] = agent_name

            # Create workflow context
            workflow_ctx = WorkflowContext(
                node=workflow_node,
                agent_name=agent_name,
            )

            # Set the workflow context
            prev_workflow = _get_current_workflow()
            _set_current_workflow(workflow_ctx)

            try:
                async with self._original_agent_run_stream(
                    agent_self, user_prompt, **kwargs
                ) as stream:
                    yield stream

                # Finish workflow node - get the result from the stream
                try:
                    result = stream.result
                    # Extract just the output from the result, not the full state
                    if hasattr(result, "output"):
                        workflow_node.outputs = {"output": _freeze(result.output)}
                    elif hasattr(result, "data"):
                        workflow_node.outputs = {"output": _freeze(result.data)}
                    else:
                        workflow_node.outputs = _freeze(result)
                except Exception:
                    workflow_node.outputs = {"streaming": True}

                workflow_node.end_time = _utc()

                # Store sub-spans in metadata for the tracker to handle
                meta["sub_spans"] = workflow_ctx.sub_spans

                # Report the workflow span to the tracker
                t.on_node_start(workflow_node)
                t.on_node_end(workflow_node, workflow_node.outputs)
            except BaseException as e:
                workflow_node.end_time = _utc()
                workflow_node.error = str(e)
                meta["sub_spans"] = workflow_ctx.sub_spans

                t.on_node_start(workflow_node)
                t.on_node_error(workflow_node, e)
                raise
            finally:
                # Restore previous workflow context
                _set_current_workflow(prev_workflow)

        Agent.run = wrapped_agent_run
        Agent.run_sync = wrapped_agent_run_sync
        Agent.run_stream = wrapped_agent_run_stream

        self._installed = True

    def uninstall(self) -> None:
        """Remove instrumentation hooks."""
        if not self._installed:
            return

        # Restore all patched model classes
        for model_cls, original_request in self._original_model_requests.items():
            model_cls.request = original_request
        for model_cls, original_stream in self._original_model_request_streams.items():
            model_cls.request_stream = original_stream

        # Clear the stored originals
        self._original_model_requests.clear()
        self._original_model_request_streams.clear()

        # Restore FunctionToolset.call_tool
        if self._original_call_tool is not None:
            try:
                from pydantic_ai.toolsets.function import FunctionToolset

                FunctionToolset.call_tool = self._original_call_tool
            except ImportError:
                pass

        # Restore Agent methods
        try:
            from pydantic_ai import Agent

            if self._original_agent_run is not None:
                Agent.run = self._original_agent_run
            if self._original_agent_run_sync is not None:
                Agent.run_sync = self._original_agent_run_sync
            if self._original_agent_run_stream is not None:
                Agent.run_stream = self._original_agent_run_stream
        except ImportError:
            pass

        self._installed = False

    def is_installed(self) -> bool:
        """Check if instrumentation is currently installed."""
        return self._installed