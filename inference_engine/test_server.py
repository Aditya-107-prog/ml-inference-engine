"""
inference_engine/test_server.py

Run with: pytest test_server.py -v

Uses httpx's ASGI transport to talk to the FastAPI app directly in-process --
no real network socket needed, and asgi_lifespan triggers our startup/shutdown
(the background batch worker thread) exactly like a real server boot would.
"""

import asyncio

import httpx
import pytest
from asgi_lifespan import LifespanManager

from server import app


@pytest.fixture
async def client():
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.mark.anyio
async def test_health_endpoint(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_single_request_gets_its_own_response(client):
    r = await client.post("/generate", json={"prompt": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert "hello world" in body["response"]
    assert "id" in body
    assert "wait_time_ms" in body


@pytest.mark.anyio
async def test_concurrent_requests_each_get_correct_response(client):
    """The core correctness property of this whole file: when many requests
    land in the SAME batch, each caller must still get back THEIR OWN answer,
    never someone else's."""
    prompts = [f"unique-prompt-{i}" for i in range(10)]

    responses = await asyncio.gather(*[
        client.post("/generate", json={"prompt": p}) for p in prompts
    ])

    for prompt, response in zip(prompts, responses):
        body = response.json()
        assert prompt in body["response"], (
            f"Response routing bug: sent {prompt!r} but got back {body['response']!r}"
        )


@pytest.mark.anyio
async def test_stats_reflects_queue_depth(client):
    r = await client.get("/stats")
    assert r.status_code == 200
    assert "queue_depth" in r.json()