import pytest
from unittest.mock import MagicMock, patch
import os

# Skip all tests if pydantic-ai is not installed
pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from gradient_adk.runtime.pydanticai.pydanticai_instrumentor import (
    PydanticAIInstrumentor,
    _freeze,
    _snapshot_args_kwargs,
    _get_captured_payloads_with_type,
    _transform_kbaas_response,
    _extract_messages_input,
    _extract_model_response_output,
)


@pytest.fixture
def tracker():
    t = MagicMock()
    t.on_node_start = MagicMock()
    t.on_node_end = MagicMock()
    t.on_node_error = MagicMock()
    return t


@pytest.fixture
def interceptor():
    intr = MagicMock()
    intr.snapshot_token.return_value = 42
    intr.hits_since.return_value = 0
    return intr


@pytest.fixture(autouse=True)
def patch_interceptor(interceptor):
    with patch(
        "gradient_adk.runtime.pydanticai.pydanticai_instrumentor.get_network_interceptor",
        return_value=interceptor,
    ):
        yield interceptor


@pytest.fixture
def instrumentor(tracker):
    inst = PydanticAIInstrumentor()
    inst.install(tracker)
    yield inst
    inst.uninstall()


@pytest.fixture
def test_model():
    return TestModel()


def test_install_monkeypatches_model(tracker):
    inst = PydanticAIInstrumentor()
    old_request = TestModel.request
    old_request_stream = TestModel.request_stream

    inst.install(tracker)

    assert TestModel.request is not old_request
    assert TestModel.request_stream is not old_request_stream
    assert inst._installed

    inst.uninstall()

    assert TestModel.request is old_request
    assert TestModel.request_stream is old_request_stream
    assert not inst._installed


def test_install_is_idempotent(tracker):
    inst = PydanticAIInstrumentor()

    inst.install(tracker)
    first_request = TestModel.request

    inst.install(tracker)
    assert TestModel.request is first_request

    inst.uninstall()


def test_is_installed_property(tracker):
    inst = PydanticAIInstrumentor()

    assert not inst.is_installed()

    inst.install(tracker)
    assert inst.is_installed()

    inst.uninstall()
    assert not inst.is_installed()


@pytest.mark.asyncio
async def test_async_run_creates_workflow_span(tracker, instrumentor, test_model):
    """Test that agent.run() creates a workflow span containing LLM sub-spans."""
    agent = Agent(test_model, system_prompt="Test agent")

    result = await agent.run("Hello, test!")

    # Should have one workflow span reported
    assert tracker.on_node_start.call_count >= 1
    assert tracker.on_node_end.call_count >= 1
    tracker.on_node_error.assert_not_called()

    # Check that we got a workflow span
    workflow_span_found = False
    llm_sub_span_found = False
    for call in tracker.on_node_start.call_args_list:
        node_exec = call[0][0]
        if "workflow:" in node_exec.node_name:
            workflow_span_found = True
            assert node_exec.framework == "pydanticai"
            # Check metadata indicates it's a workflow
            assert node_exec.metadata.get("is_workflow") is True
            # Check for sub_spans containing LLM calls
            sub_spans = node_exec.metadata.get("sub_spans", [])
            for sub in sub_spans:
                if "llm:" in sub.node_name:
                    llm_sub_span_found = True
                    break
            break
    assert workflow_span_found, "No workflow span was created"
    assert llm_sub_span_found, "No LLM sub-span was found in the workflow"


def test_sync_run_creates_workflow_span(tracker, instrumentor, test_model):
    """Test that agent.run_sync() creates a workflow span containing LLM sub-spans."""
    agent = Agent(test_model, system_prompt="Test agent")

    result = agent.run_sync("Hello, sync test!")

    # Should have one workflow span reported
    assert tracker.on_node_start.call_count >= 1
    assert tracker.on_node_end.call_count >= 1
    tracker.on_node_error.assert_not_called()

    # Check that we got a workflow span
    workflow_span_found = False
    llm_sub_span_found = False
    for call in tracker.on_node_start.call_args_list:
        node_exec = call[0][0]
        if "workflow:" in node_exec.node_name:
            workflow_span_found = True
            assert node_exec.framework == "pydanticai"
            # Check metadata indicates it's a workflow
            assert node_exec.metadata.get("is_workflow") is True
            # Check for sub_spans containing LLM calls
            sub_spans = node_exec.metadata.get("sub_spans", [])
            for sub in sub_spans:
                if "llm:" in sub.node_name:
                    llm_sub_span_found = True
                    break
            break
    assert workflow_span_found, "No workflow span was created"
    assert llm_sub_span_found, "No LLM sub-span was found in the workflow"


def test_freeze_handles_primitives():
    assert _freeze(None) is None
    assert _freeze("string") == "string"
    assert _freeze(42) == 42
    assert _freeze(3.14) == 3.14
    assert _freeze(True) is True


def test_freeze_handles_dict():
    result = _freeze({"key": "value", "nested": {"a": 1}})
    assert result == {"key": "value", "nested": {"a": 1}}


def test_freeze_handles_list():
    result = _freeze([1, 2, 3, {"x": "y"}])
    assert result == [1, 2, 3, {"x": "y"}]


def test_get_captured_payloads_with_type_inference_url():
    mock_intr = MagicMock()
    mock_captured = MagicMock()
    mock_captured.url = "https://inference.do-ai.run/v1/chat"
    mock_captured.request_payload = {"messages": []}
    mock_captured.response_payload = {"choices": []}

    mock_intr.get_captured_requests_since.return_value = [mock_captured]

    req, resp, is_llm, is_retriever = _get_captured_payloads_with_type(mock_intr, 0)

    assert req == {"messages": []}
    assert resp == {"choices": []}
    assert is_llm is True
    assert is_retriever is False


def test_transform_kbaas_response_converts_text_content():
    response = {
        "results": [
            {"metadata": {"source": "doc1.pdf"}, "text_content": "Document content."}
        ],
        "total_results": 1,
    }

    transformed = _transform_kbaas_response(response)

    assert isinstance(transformed, list)
    assert len(transformed) == 1
    assert transformed[0]["page_content"] == "Document content."
    assert "text_content" not in transformed[0]


def test_extract_messages_input_with_parts():
    class MockPart:
        pass

    class MockMessage:
        def __init__(self, parts, instructions=None):
            self.parts = parts
            self.instructions = instructions

    mock_part = MockPart()
    msg = MockMessage([mock_part], instructions="Test instructions")

    result = _extract_messages_input([msg])

    assert len(result) == 1
    assert result[0]["kind"] == "MockMessage"
    assert "parts" in result[0]
    assert result[0]["instructions"] == "Test instructions"


def test_extract_model_response_output_with_parts():
    class MockPart:
        pass

    class MockResponse:
        def __init__(self, parts, usage=None, model_name=None):
            self.parts = parts
            self.usage = usage
            self.model_name = model_name

    mock_part = MockPart()
    response = MockResponse([mock_part], model_name="test-model")

    result = _extract_model_response_output(response)

    assert "parts" in result
    assert result["model_name"] == "test-model"


def test_uninstall_restores_original_methods(tracker):
    original_request = TestModel.request
    original_request_stream = TestModel.request_stream

    inst = PydanticAIInstrumentor()
    inst.install(tracker)

    assert TestModel.request is not original_request

    inst.uninstall()

    assert TestModel.request is original_request
    assert TestModel.request_stream is original_request_stream


def test_uninstall_without_install_is_safe():
    inst = PydanticAIInstrumentor()
    inst.uninstall()
    assert not inst._installed