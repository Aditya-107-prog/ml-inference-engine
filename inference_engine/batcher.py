"""
inference_engine/batcher.py

Pulls requests off a RequestQueue and groups them into batches for inference.

Core tradeoff: throughput vs latency.
- Bigger batches -> better GPU utilization -> higher throughput.
- But waiting for a batch to fill up adds latency to whichever request arrived first.

Policy implemented here: close the batch when EITHER condition is met, whichever
comes first:
  (a) it reaches `max_batch_size` requests, OR
  (b) `max_wait_time_ms` has elapsed since the FIRST request in this batch arrived.

This is the same idea (simplified) behind dynamic/continuous batching in real
serving systems like vLLM and TGI -- they don't run one request at a time, and
they don't wait indefinitely for a "perfect" batch either.

NOT thread-safe yet. This assumes a single-threaded producer/consumer loop.
Once requests arrive concurrently (Week 3, behind FastAPI), the RequestQueue
itself needs a lock around push/pop -- flagged with a TODO below, not solved
yet, on purpose: no point adding locking complexity before there's concurrency
that needs it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from request_queue import Request, RequestQueue

logger = logging.getLogger(__name__)


@dataclass
class Batch:
    requests: list[Request]
    formed_at: float = field(default_factory=time.monotonic)

    def __len__(self) -> int:
        return len(self.requests)

    def __repr__(self) -> str:
        ids = [r.id for r in self.requests]
        return f"Batch(size={len(self.requests)}, ids={ids})"

    @property
    def wait_times_ms(self) -> list[float]:
        """How long each request sat in the queue before this batch formed,
        in milliseconds. Useful for the Week 4 dashboard and the Month 3
        benchmarking suite -- capture it here while it's cheap, since this
        information is gone the moment the batch is handed off to the model."""
        return [(self.formed_at - r.timestamp) * 1000 for r in self.requests]

    @property
    def avg_wait_time_ms(self) -> float:
        wait_times = self.wait_times_ms
        return sum(wait_times) / len(wait_times) if wait_times else 0.0


class DynamicBatcher:
    def __init__(
        self,
        queue: RequestQueue,
        max_batch_size: int = 8,
        max_wait_time_ms: float = 50.0,
        poll_interval_s: float = 0.002,
    ):
        """
        Args:
            queue: the RequestQueue to pull from.
            max_batch_size: hard cap on requests per batch.
            max_wait_time_ms: max time to wait (from when the first request in
                this batch arrived) before closing the batch even if it isn't full.
            poll_interval_s: how often to check the queue while waiting. Lower =
                more responsive, more CPU spin. 2ms is a reasonable default --
                imperceptible latency cost, negligible CPU usage.
        """
        self.queue = queue
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.poll_interval_s = poll_interval_s

    def form_batch(self) -> Batch | None:
        """
        Blocks until a batch is ready, per the policy above.
        Returns None if the queue stays empty forever (caller should handle
        shutdown signals separately -- this method has no concept of "stop").
        """
        # TODO(Week 3): when the queue is fed by concurrent request handlers,
        # wrap queue.pop() calls in a lock. Right now this assumes the only
        # thing touching `self.queue` is this batcher's own thread.

        # Wait for at least one request before starting the clock.
        while self.queue.is_empty():
            time.sleep(self.poll_interval_s)

        first_request = self.queue.pop()
        batch_requests = [first_request]
        deadline = time.monotonic() + (self.max_wait_time_ms / 1000.0)
        closed_reason = "max_batch_size"

        while len(batch_requests) < self.max_batch_size:
            if time.monotonic() >= deadline:
                closed_reason = "timeout"
                break  # latency budget exceeded, ship what we have

            next_request = self.queue.pop()
            if next_request is None:
                time.sleep(self.poll_interval_s)
                continue

            batch_requests.append(next_request)

        batch = Batch(requests=batch_requests)
        logger.info(
            "Batch closed (%s): size=%d avg_wait=%.1fms",
            closed_reason, len(batch), batch.avg_wait_time_ms,
        )
        return batch


if __name__ == "__main__":
    # Manual smoke test: simulate a producer trickling in requests while the
    # batcher tries to form batches.
    from request_queue import SchedulingPolicy
    import threading

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    q = RequestQueue(SchedulingPolicy.FIFO)

    def producer():
        for i in range(10):
            q.push(Request(prompt=f"prompt number {i}"))
            time.sleep(0.01)  # ~100 requests/sec arrival rate

    threading.Thread(target=producer, daemon=True).start()

    batcher = DynamicBatcher(q, max_batch_size=4, max_wait_time_ms=30)

    collected = 0
    while collected < 10:
        batch = batcher.form_batch()
        print(f"{batch} avg_wait={batch.avg_wait_time_ms:.1f}ms")
        collected += len(batch)