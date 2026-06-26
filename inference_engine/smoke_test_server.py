"""
Quick manual smoke test for server.py -- not part of the pytest suite,
just a sanity check before writing real tests.

Run with: python smoke_test_server.py
"""

import asyncio
import logging

import httpx
from asgi_lifespan import LifespanManager

from server import app

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def main():
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            print("=== Single request ===")
            r = await client.post("/generate", json={"prompt": "hello world"})
            print(r.json())

            print("\n=== /stats and /health ===")
            print((await client.get("/health")).json())
            print((await client.get("/stats")).json())

            print("\n=== 6 concurrent requests, each must get its OWN prompt back ===")
            prompts = [f"request number {i}" for i in range(6)]
            results = await asyncio.gather(*[
                client.post("/generate", json={"prompt": p}) for p in prompts
            ])
            for p, r in zip(prompts, results):
                body = r.json()
                assert p in body["response"], f"MISMATCH: sent {p!r}, got {body['response']!r}"
                print(f"  sent={p!r:30} -> got={body['response']!r}")
            print("\nAll 6 concurrent requests routed back correctly.")


if __name__ == "__main__":
    asyncio.run(main())