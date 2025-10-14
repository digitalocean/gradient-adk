"""
Unit tests for the streaming module.
"""

import pytest
import json
from typing import Iterator, AsyncIterator, Dict, Any
from unittest.mock import Mock

from gradient_adk.streaming import (
    StreamingResponse,
    JSONStreamingResponse,
    ServerSentEventsResponse,
    stream_json,
    stream_events,
)


class TestStreamingResponse:
    """Test cases for the StreamingResponse class."""

    def test_init_with_default_params(self):
        """Test StreamingResponse initialization with default parameters."""

        def content_generator():
            yield "chunk1"
            yield "chunk2"

        response = StreamingResponse(content_generator())

        assert response.content is not None
        assert response.media_type == "text/plain"
        assert response.headers == {}

    def test_init_with_custom_params(self):
        """Test StreamingResponse initialization with custom parameters."""

        def content_generator():
            yield b"chunk1"
            yield b"chunk2"

        custom_headers = {"X-Custom": "header", "Cache-Control": "no-cache"}
        response = StreamingResponse(
            content_generator(),
            media_type="application/octet-stream",
            headers=custom_headers,
        )

        assert response.media_type == "application/octet-stream"
        assert response.headers == custom_headers

    def test_init_with_none_headers(self):
        """Test StreamingResponse initialization with None headers."""

        def content_generator():
            yield "test"

        response = StreamingResponse(content_generator(), headers=None)

        assert response.headers == {}

    def test_repr(self):
        """Test StreamingResponse string representation."""

        def content_generator():
            yield "test"

        response = StreamingResponse(content_generator(), media_type="application/json")

        assert repr(response) == "StreamingResponse(media_type='application/json')"

    def test_with_string_iterator(self):
        """Test StreamingResponse with string iterator."""

        def string_generator():
            yield "Hello "
            yield "World"

        response = StreamingResponse(string_generator())

        # Convert generator to list to test content
        content_list = list(response.content)
        assert content_list == ["Hello ", "World"]

    def test_with_bytes_iterator(self):
        """Test StreamingResponse with bytes iterator."""

        def bytes_generator():
            yield b"Hello "
            yield b"World"

        response = StreamingResponse(bytes_generator())

        # Convert generator to list to test content
        content_list = list(response.content)
        assert content_list == [b"Hello ", b"World"]

    @pytest.mark.asyncio
    async def test_with_async_iterator(self):
        """Test StreamingResponse with async iterator."""

        async def async_generator():
            yield "async chunk 1"
            yield "async chunk 2"

        response = StreamingResponse(async_generator())

        # Convert async generator to list to test content
        content_list = []
        async for chunk in response.content:
            content_list.append(chunk)

        assert content_list == ["async chunk 1", "async chunk 2"]


class TestJSONStreamingResponse:
    """Test cases for the JSONStreamingResponse class."""

    def test_init_with_dict_iterator(self):
        """Test JSONStreamingResponse initialization with dictionary iterator."""

        def dict_generator():
            yield {"key1": "value1"}
            yield {"key2": "value2"}

        response = JSONStreamingResponse(dict_generator())

        assert response.media_type == "application/x-ndjson"
        assert response.headers == {"Cache-Control": "no-cache"}

    def test_json_formatting(self):
        """Test that dictionaries are properly JSON-formatted."""

        def dict_generator():
            yield {"message": "hello"}
            yield {"number": 42, "bool": True}

        response = JSONStreamingResponse(dict_generator())

        # Convert generator to list to test content
        content_list = list(response.content)

        # Each chunk should be JSON + newline
        assert content_list[0] == '{"message": "hello"}\n'
        assert content_list[1] == '{"number": 42, "bool": true}\n'

    def test_empty_dict(self):
        """Test JSONStreamingResponse with empty dictionary."""

        def dict_generator():
            yield {}

        response = JSONStreamingResponse(dict_generator())

        content_list = list(response.content)
        assert content_list == ["{}\n"]

    def test_complex_dict(self):
        """Test JSONStreamingResponse with complex dictionary structure."""

        def dict_generator():
            yield {"nested": {"key": "value"}, "array": [1, 2, 3], "null": None}

        response = JSONStreamingResponse(dict_generator())

        content_list = list(response.content)
        parsed = json.loads(content_list[0].strip())

        assert parsed["nested"]["key"] == "value"
        assert parsed["array"] == [1, 2, 3]
        assert parsed["null"] is None

    @pytest.mark.asyncio
    async def test_with_async_dict_iterator(self):
        """Test JSONStreamingResponse with async dictionary iterator."""

        async def async_dict_generator():
            yield {"async": "data1"}
            yield {"async": "data2"}

        response = JSONStreamingResponse(async_dict_generator())

        # Should create an async generator
        assert hasattr(response.content, "__aiter__")

        content_list = []
        async for chunk in response.content:
            content_list.append(chunk)

        assert content_list[0] == '{"async": "data1"}\n'
        assert content_list[1] == '{"async": "data2"}\n'

    def test_sync_vs_async_content_detection(self):
        """Test that sync and async iterators are handled differently."""

        def sync_generator():
            yield {"sync": True}

        async def async_generator():
            yield {"async": True}

        sync_response = JSONStreamingResponse(sync_generator())
        async_response = JSONStreamingResponse(async_generator())

        # Sync should create regular iterator
        assert hasattr(sync_response.content, "__iter__")
        assert not hasattr(sync_response.content, "__aiter__")

        # Async should create async iterator
        assert hasattr(async_response.content, "__aiter__")


class TestServerSentEventsResponse:
    """Test cases for the ServerSentEventsResponse class."""

    def test_init_with_dict_iterator(self):
        """Test ServerSentEventsResponse initialization."""

        def dict_generator():
            yield {"event": "message"}

        response = ServerSentEventsResponse(dict_generator())

        assert response.media_type == "text/event-stream"
        assert response.headers == {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

    def test_sse_formatting(self):
        """Test that dictionaries are properly formatted as SSE events."""

        def dict_generator():
            yield {"message": "hello"}
            yield {"type": "notification", "data": "test"}

        response = ServerSentEventsResponse(dict_generator())

        content_list = list(response.content)

        # Each chunk should be formatted as SSE event
        assert content_list[0] == 'data: {"message": "hello"}\n\n'
        assert content_list[1] == 'data: {"type": "notification", "data": "test"}\n\n'

    def test_sse_format_structure(self):
        """Test that SSE format follows specification."""

        def dict_generator():
            yield {"test": "data"}

        response = ServerSentEventsResponse(dict_generator())

        content_list = list(response.content)
        chunk = content_list[0]

        # Should start with "data: "
        assert chunk.startswith("data: ")
        # Should end with double newline
        assert chunk.endswith("\n\n")
        # Should contain valid JSON between
        json_part = chunk[6:-2]  # Remove "data: " and "\n\n"
        parsed = json.loads(json_part)
        assert parsed == {"test": "data"}

    @pytest.mark.asyncio
    async def test_with_async_dict_iterator(self):
        """Test ServerSentEventsResponse with async dictionary iterator."""

        async def async_dict_generator():
            yield {"event": "start"}
            yield {"event": "end"}

        response = ServerSentEventsResponse(async_dict_generator())

        # Should create an async generator
        assert hasattr(response.content, "__aiter__")

        content_list = []
        async for chunk in response.content:
            content_list.append(chunk)

        assert content_list[0] == 'data: {"event": "start"}\n\n'
        assert content_list[1] == 'data: {"event": "end"}\n\n'

    def test_sync_vs_async_sse_content_detection(self):
        """Test that sync and async iterators are handled differently for SSE."""

        def sync_generator():
            yield {"sync": True}

        async def async_generator():
            yield {"async": True}

        sync_response = ServerSentEventsResponse(sync_generator())
        async_response = ServerSentEventsResponse(async_generator())

        # Sync should create regular iterator
        assert hasattr(sync_response.content, "__iter__")
        assert not hasattr(sync_response.content, "__aiter__")

        # Async should create async iterator
        assert hasattr(async_response.content, "__aiter__")

    def test_empty_dict_sse(self):
        """Test ServerSentEventsResponse with empty dictionary."""

        def dict_generator():
            yield {}

        response = ServerSentEventsResponse(dict_generator())

        content_list = list(response.content)
        assert content_list == ["data: {}\n\n"]


class TestConvenienceFunctions:
    """Test cases for the convenience functions."""

    def test_stream_json_function(self):
        """Test stream_json convenience function."""

        def dict_generator():
            yield {"test": "data"}

        response = stream_json(dict_generator())

        assert isinstance(response, JSONStreamingResponse)
        assert response.media_type == "application/x-ndjson"
        assert response.headers == {"Cache-Control": "no-cache"}

    def test_stream_events_function(self):
        """Test stream_events convenience function."""

        def dict_generator():
            yield {"event": "test"}

        response = stream_events(dict_generator())

        assert isinstance(response, ServerSentEventsResponse)
        assert response.media_type == "text/event-stream"
        assert response.headers == {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

    def test_stream_json_output_format(self):
        """Test that stream_json produces correct output format."""

        def dict_generator():
            yield {"message": "test"}
            yield {"number": 123}

        response = stream_json(dict_generator())
        content_list = list(response.content)

        assert content_list[0] == '{"message": "test"}\n'
        assert content_list[1] == '{"number": 123}\n'

    def test_stream_events_output_format(self):
        """Test that stream_events produces correct output format."""

        def dict_generator():
            yield {"type": "message"}
            yield {"type": "done"}

        response = stream_events(dict_generator())
        content_list = list(response.content)

        assert content_list[0] == 'data: {"type": "message"}\n\n'
        assert content_list[1] == 'data: {"type": "done"}\n\n'


class TestIntegration:
    """Integration tests for streaming functionality."""

    def test_all_streaming_types_work_together(self):
        """Test that all streaming response types can be used together."""

        # Test data
        test_data = [{"id": 1, "message": "first"}, {"id": 2, "message": "second"}]

        def data_generator():
            for item in test_data:
                yield item

        # Test all three types
        basic_response = StreamingResponse(
            (json.dumps(item) for item in data_generator()),
            media_type="application/json",
        )

        json_response = stream_json(data_generator())

        sse_response = stream_events(data_generator())

        # All should be different instances
        assert basic_response is not json_response
        assert json_response is not sse_response
        assert basic_response is not sse_response

        # All should have different media types
        assert basic_response.media_type == "application/json"
        assert json_response.media_type == "application/x-ndjson"
        assert sse_response.media_type == "text/event-stream"

    @pytest.mark.asyncio
    async def test_async_generators_work_with_all_types(self):
        """Test that async generators work with all streaming response types."""

        async def async_data_generator():
            yield {"async": True, "id": 1}
            yield {"async": True, "id": 2}

        # Test with basic StreamingResponse
        async def async_string_generator():
            async for item in async_data_generator():
                yield json.dumps(item)

        basic_response = StreamingResponse(async_string_generator())

        # Test with JSONStreamingResponse
        json_response = JSONStreamingResponse(async_data_generator())

        # Test with ServerSentEventsResponse
        sse_response = ServerSentEventsResponse(async_data_generator())

        # All should work with async iteration
        basic_content = []
        async for chunk in basic_response.content:
            basic_content.append(chunk)

        json_content = []
        async for chunk in json_response.content:
            json_content.append(chunk)

        sse_content = []
        async for chunk in sse_response.content:
            sse_content.append(chunk)

        # Verify content was generated
        assert len(basic_content) == 2
        assert len(json_content) == 2
        assert len(sse_content) == 2

        # Verify format differences
        assert '"async": true' in basic_content[0]
        assert json_content[0].endswith("\n")
        assert sse_content[0].startswith("data: ")

    def test_error_handling_in_generators(self):
        """Test how streaming responses handle errors in generators."""

        def error_generator():
            yield {"success": True}
            raise ValueError("Test error")
            yield {"never": "reached"}  # This should never be reached

        response = JSONStreamingResponse(error_generator())

        # Should be able to get first chunk
        content_iter = iter(response.content)
        first_chunk = next(content_iter)
        assert first_chunk == '{"success": true}\n'

        # Second iteration should raise the error
        with pytest.raises(ValueError, match="Test error"):
            next(content_iter)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_generator(self):
        """Test streaming responses with empty generators."""

        def empty_generator():
            return
            yield  # Never reached

        response = StreamingResponse(empty_generator())
        content_list = list(response.content)
        assert content_list == []

    def test_json_response_with_empty_generator(self):
        """Test JSONStreamingResponse with empty generator."""

        def empty_dict_generator():
            return
            yield {}  # Never reached

        response = JSONStreamingResponse(empty_dict_generator())
        content_list = list(response.content)
        assert content_list == []

    def test_sse_response_with_empty_generator(self):
        """Test ServerSentEventsResponse with empty generator."""

        def empty_dict_generator():
            return
            yield {}  # Never reached

        response = ServerSentEventsResponse(empty_dict_generator())
        content_list = list(response.content)
        assert content_list == []

    def test_json_serialization_with_special_values(self):
        """Test JSON serialization with special Python values."""

        def special_values_generator():
            yield {"none": None}
            yield {"bool_true": True}
            yield {"bool_false": False}
            yield {"float": 3.14}
            yield {"empty_list": []}
            yield {"empty_dict": {}}

        response = JSONStreamingResponse(special_values_generator())
        content_list = list(response.content)

        # Parse each JSON chunk to verify serialization
        parsed = [json.loads(chunk.strip()) for chunk in content_list]

        assert parsed[0]["none"] is None
        assert parsed[1]["bool_true"] is True
        assert parsed[2]["bool_false"] is False
        assert parsed[3]["float"] == 3.14
        assert parsed[4]["empty_list"] == []
        assert parsed[5]["empty_dict"] == {}

    def test_headers_immutability(self):
        """Test that headers dictionary is not shared between instances."""

        def gen():
            yield "test"

        response1 = StreamingResponse(gen(), headers={"X-Test": "1"})
        response2 = StreamingResponse(gen(), headers={"X-Test": "2"})

        # Headers should be independent
        assert response1.headers["X-Test"] == "1"
        assert response2.headers["X-Test"] == "2"

        # Modifying one shouldn't affect the other
        response1.headers["X-New"] = "new1"
        assert "X-New" not in response2.headers
