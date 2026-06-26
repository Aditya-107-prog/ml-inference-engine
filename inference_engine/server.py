"""
inference_engine/server.py

Wires the RequestQueue + DynamicBatcher into an actual HTTP service.

THE CORE PROBLEM THIS FILE SOLVES:
Multiple HTTP requests arrive concurrently and get grouped into anonymous
batches. When a batch finishes, how does each ORIGINAL caller get back their
OWN specific answer?

THE SOLUTION: a Future per request.
- When a request comes in, we create a concurrent.futures.Future and store it
  in a dict keyed by the request's id.
- The HTTP handler `await`s that future (via asyncio.wrap_future, which bridges
  a thread-based Future into the async world).
- A background thread continuously forms batches and "processes" them (stubbed
  for now -- real model inference comes in a later week). For each request in
  the batch, it looks up that request's Future by id and calls .set_result(...)
  on it -- which is what wakes up the specific `await` that's been sitting there
  waiting.

This is the same pattern real inference servers (vLLM, TGI) use internally,
just simplified: requests go in anonymously batched, but each one's *result*
still has to find its way back to the exact caller that asked for it.

ARCHITECTURE NOTE: form_batch() is a blocking call (it does time.sleep polling
internally). It CANNOT run on the async event loop -- that would freeze every
other request while it waits. So it runs in its own dedicated background
thread, started once at server startup, looping forever.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from batcher import Batch, DynamicBatcher
from model_runner import QwenModelRunner, StubModelRunner
from request_queue import Request, RequestQueue, SchedulingPolicy

logger = logging.getLogger(__name__)

# --- Shared state, touched by both the async HTTP handlers and the
#     background worker thread. ---
queue = RequestQueue(SchedulingPolicy.FIFO)
batcher = DynamicBatcher(queue, max_batch_size=8, max_wait_time_ms=50)

pending_futures: dict[str, Future] = {}
pending_lock = threading.Lock()  # protects pending_futures, separate from the queue's own lock

# Which model backend to use. Defaults to the real Qwen model -- set
# MODEL_BACKEND=stub (e.g. in test_server.py) to skip loading anything real,
# which keeps the test suite instant and independent of GPU/network access.
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "qwen")
model_runner = None  # set during lifespan startup, once we know which backend to build


def build_model_runner():
    if MODEL_BACKEND == "stub":
        logger.info("MODEL_BACKEND=stub -- using StubModelRunner (no real model loaded)")
        return StubModelRunner()
    logger.info("MODEL_BACKEND=qwen -- loading real Qwen model (this may take a while the first time)")
    return QwenModelRunner()


def process_batch(batch: Batch) -> None:
    """Runs the whole batch through the model in ONE call -- this is the
    actual payoff of batching. The model processes all prompts in parallel
    on the GPU/CPU instead of one at a time, which is the entire reason the
    queue and batcher exist."""
    prompts = [r.prompt for r in batch.requests]
    generated_texts = model_runner.generate_batch(prompts)

    for request, wait_ms, text in zip(batch.requests, batch.wait_times_ms, generated_texts):
        result = {"response": text, "wait_time_ms": round(wait_ms, 1)}
        with pending_lock:
            future = pending_futures.pop(request.id, None)
        if future is not None and not future.done():
            future.set_result(result)

    logger.info("Processed batch of %d (avg_wait=%.1fms)", len(batch), batch.avg_wait_time_ms)


def batch_worker_loop(stop_event: threading.Event) -> None:
    """Runs forever in a background thread: form a batch, process it, repeat.

    Known limitation: form_batch() blocks waiting for the FIRST request with
    no awareness of stop_event, so this won't shut down instantly if the queue
    is empty. Acceptable for now since the thread is a daemon (dies when the
    process exits) -- flagged here rather than solved, since solving it
    properly means giving the batcher a way to be woken up early, which is
    more machinery than this stage needs.
    """
    while not stop_event.is_set():
        batch = batcher.form_batch()
        process_batch(batch)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_runner
    model_runner = build_model_runner()  # blocks startup until the model is ready -- intentional

    stop_event = threading.Event()
    worker = threading.Thread(target=batch_worker_loop, args=(stop_event,), daemon=True)
    worker.start()
    logger.info("Batch worker thread started")
    yield
    stop_event.set()


app = FastAPI(lifespan=lifespan)


class GenerateRequestBody(BaseModel):
    prompt: str
    priority: int = 1


@app.post("/generate")
async def generate(body: GenerateRequestBody):
    request = Request(prompt=body.prompt, priority=body.priority)

    future: Future = Future()
    with pending_lock:
        pending_futures[request.id] = future

    queue.push(request)

    # asyncio.wrap_future bridges a concurrent.futures.Future (set from the
    # OTHER thread) into something this coroutine can `await` natively.
    result = await asyncio.wrap_future(future)

    return {"id": request.id, **result}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return {"queue_depth": len(queue)}