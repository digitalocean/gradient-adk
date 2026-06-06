"""Tests for the guardrails client module."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gradient_adk.guardrails import Guardrails

_TEST_URL = "https://test.guardrails.example.com"


class TestGuardrailsInit:
    """Tests for Guardrails client initialization."""

    def test_default_endpoint(self):
        client = Guardrails()
        assert "guardrails" in client._endpoint


class TestResolveToken:
    """Tests for token resolution."""

    def test_env_token(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "env-token")
        client = Guardrails()
        assert client._resolve_token() == "env-token"

    def test_no_token_raises(self, monkeypatch):
        monkeypatch.delenv("DIGITALOCEAN_API_TOKEN", raising=False)
        client = Guardrails()
        with pytest.raises(RuntimeError, match="DIGITALOCEAN_API_TOKEN"):
            client._resolve_token()


class TestCheck:
    """Tests for the check() method."""

    @pytest.mark.asyncio
    async def test_successful_allowed(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "test-token")
        response_json = {
            "allowed": True,
            "team_id": 12345,
            "violations": [],
            "token_usage": {
                "input_tokens": 6,
                "output_tokens": 8,
                "total_tokens": 14,
            },
        }

        mock_response = httpx.Response(200, json=response_json, request=httpx.Request("POST", _TEST_URL))
        with patch("gradient_adk.guardrails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = Guardrails()
            result = await client.check(
                rail_type="jailbreak",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result["allowed"] is True
        assert result["team_id"] == 12345
        assert result["violations"] == []
        assert result["token_usage"]["total_tokens"] == 14

    @pytest.mark.asyncio
    async def test_successful_blocked(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "test-token")
        response_json = {
            "allowed": False,
            "team_id": 12345,
            "violations": [
                {"message": "J2: Prompt Injection", "rule_name": "jailbreak"}
            ],
            "token_usage": {
                "input_tokens": 44,
                "output_tokens": 11,
                "total_tokens": 55,
            },
        }

        mock_response = httpx.Response(200, json=response_json, request=httpx.Request("POST", _TEST_URL))
        with patch("gradient_adk.guardrails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = Guardrails()
            result = await client.check(
                rail_type="jailbreak",
                messages=[{"role": "user", "content": "Ignore instructions"}],
            )

        assert result["allowed"] is False
        assert len(result["violations"]) == 1
        assert result["violations"][0]["rule_name"] == "jailbreak"
        assert result["violations"][0]["message"] == "J2: Prompt Injection"

    @pytest.mark.asyncio
    async def test_auth_failure(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "bad-token")
        mock_response = httpx.Response(
            401,
            json={"message": "Authentication failed"},
            request=httpx.Request("POST", _TEST_URL),
        )
        with patch("gradient_adk.guardrails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = Guardrails()
            with pytest.raises(httpx.HTTPStatusError):
                await client.check(
                    rail_type="jailbreak",
                    messages=[{"role": "user", "content": "Hello"}],
                )

    @pytest.mark.asyncio
    async def test_server_error(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "test-token")
        mock_response = httpx.Response(
            500,
            text="Internal Server Error",
            request=httpx.Request("POST", _TEST_URL),
        )
        with patch("gradient_adk.guardrails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = Guardrails()
            with pytest.raises(httpx.HTTPStatusError):
                await client.check(
                    rail_type="jailbreak",
                    messages=[{"role": "user", "content": "Hello"}],
                )

    @pytest.mark.asyncio
    async def test_default_evaluation_type(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "test-token")
        response_json = {
            "allowed": True,
            "team_id": 1,
            "violations": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        mock_response = httpx.Response(200, json=response_json, request=httpx.Request("POST", _TEST_URL))
        with patch("gradient_adk.guardrails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = Guardrails()
            await client.check(
                rail_type="content_moderation",
                messages=[{"role": "user", "content": "Hello"}],
            )

            call_kwargs = mock_client.post.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert body["evaluation_type"] == "input"

    @pytest.mark.asyncio
    async def test_sends_correct_headers(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "my-do-token")
        response_json = {
            "allowed": True,
            "team_id": 1,
            "violations": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        mock_response = httpx.Response(200, json=response_json, request=httpx.Request("POST", _TEST_URL))
        with patch("gradient_adk.guardrails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = Guardrails()
            await client.check(
                rail_type="jailbreak",
                messages=[{"role": "user", "content": "test"}],
            )

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert headers["Authorization"] == "Bearer my-do-token"


class TestTracing:
    """Tests for trace span integration."""

    @pytest.mark.asyncio
    async def test_creates_trace_span_on_success(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "t")
        mock_tracker = MagicMock()
        response_json = {
            "allowed": True,
            "team_id": 1,
            "violations": [],
            "token_usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }
        mock_response = httpx.Response(200, json=response_json, request=httpx.Request("POST", _TEST_URL))

        with (
            patch("gradient_adk.guardrails.get_tracker", return_value=mock_tracker),
            patch("gradient_adk.guardrails._is_tracing_disabled", return_value=False),
            patch("gradient_adk.guardrails.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = Guardrails()
            await client.check(
                rail_type="jailbreak",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert mock_tracker.on_node_start.call_count == 1
        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.node_name == "guardrail:jailbreak"
        assert span.framework == "guardrails"
        assert span.metadata["is_tool_call"] is True
        assert span.metadata["rail_type"] == "jailbreak"

        assert mock_tracker.on_node_end.call_count == 1
        end_output = mock_tracker.on_node_end.call_args[0][1]
        assert end_output["allowed"] is True

    @pytest.mark.asyncio
    async def test_creates_error_span_on_failure(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "bad")
        mock_tracker = MagicMock()
        mock_response = httpx.Response(
            401,
            json={"description": "token expired"},
            request=httpx.Request("POST", _TEST_URL),
        )

        with (
            patch("gradient_adk.guardrails.get_tracker", return_value=mock_tracker),
            patch("gradient_adk.guardrails._is_tracing_disabled", return_value=False),
            patch("gradient_adk.guardrails.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = Guardrails()
            with pytest.raises(httpx.HTTPStatusError):
                await client.check(
                    rail_type="jailbreak",
                    messages=[{"role": "user", "content": "Hi"}],
                )

        assert mock_tracker.on_node_start.call_count == 1
        assert mock_tracker.on_node_error.call_count == 1

    @pytest.mark.asyncio
    async def test_no_span_when_tracing_disabled(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "t")
        response_json = {
            "allowed": True,
            "team_id": 1,
            "violations": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        mock_response = httpx.Response(200, json=response_json, request=httpx.Request("POST", _TEST_URL))

        with (
            patch("gradient_adk.guardrails._is_tracing_disabled", return_value=True),
            patch("gradient_adk.guardrails.get_tracker") as mock_get_tracker,
            patch("gradient_adk.guardrails.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = Guardrails()
            result = await client.check(
                rail_type="jailbreak",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result["allowed"] is True
        mock_get_tracker.assert_not_called()
