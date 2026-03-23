"""Tests for EvalRecord integration in the decorator."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from fastapi import FastAPI, Request, HTTPException
from httpx import AsyncClient, ASGITransport

from gradient_adk.evaluation.record import (
    EvalRecord,
    _NoOpEvalRecord,
    _begin_eval_request,
    _end_eval_request,
    _eval_store,
    _pop_eval_record,
    eval_record,
)


@pytest.fixture(autouse=True)
def clean_store():
    _eval_store.clear()
    yield
    _eval_store.clear()


@pytest.fixture
def eval_app():
    """Minimal FastAPI app that mirrors the decorator's eval wiring."""
    app = FastAPI()

    async def agent_logic(data):
        return {"response": f"echo: {data}"}

    @app.post("/run")
    async def run(req: Request):
        body = await req.json()
        is_evaluation = "evaluation-id" in req.headers

        eval_request_id = req.headers.get("x-eval-request-id")
        eval_token = None
        if is_evaluation and eval_request_id:
            eval_token = _begin_eval_request(eval_request_id)

        try:
            result = await agent_logic(body)
        except Exception:
            if eval_token is not None:
                _end_eval_request(eval_token)
            raise HTTPException(status_code=500)

        if eval_token is not None:
            _end_eval_request(eval_token)

        return result

    return app


class TestDecoratorEvalRecord:
    @pytest.mark.asyncio
    async def test_eval_record_set_when_eval_headers_present(self, eval_app):
        transport = ASGITransport(app=eval_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/run",
                json={"msg": "hi"},
                headers={
                    "evaluation-id": "local-eval",
                    "x-eval-request-id": "test-req-1",
                },
            )
            assert resp.status_code == 200

        rec = _pop_eval_record("test-req-1")
        assert rec is not None
        assert isinstance(rec, EvalRecord)

    @pytest.mark.asyncio
    async def test_no_eval_record_without_headers(self, eval_app):
        transport = ASGITransport(app=eval_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/run", json={"msg": "hi"})
            assert resp.status_code == 200

        assert len(_eval_store) == 0

    @pytest.mark.asyncio
    async def test_context_cleaned_up_after_request(self, eval_app):
        transport = ASGITransport(app=eval_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/run",
                json={"msg": "hi"},
                headers={
                    "evaluation-id": "local-eval",
                    "x-eval-request-id": "test-req-2",
                },
            )

        # After request, eval_record() should return NoOp
        rec = eval_record()
        assert isinstance(rec, _NoOpEvalRecord)

    @pytest.mark.asyncio
    async def test_no_leaks_between_requests(self, eval_app):
        transport = ASGITransport(app=eval_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/run",
                json={"msg": "first"},
                headers={
                    "evaluation-id": "local-eval",
                    "x-eval-request-id": "req-a",
                },
            )
            await client.post(
                "/run",
                json={"msg": "second"},
                headers={
                    "evaluation-id": "local-eval",
                    "x-eval-request-id": "req-b",
                },
            )

        rec_a = _pop_eval_record("req-a")
        rec_b = _pop_eval_record("req-b")
        assert rec_a is not None
        assert rec_b is not None
        assert rec_a is not rec_b
