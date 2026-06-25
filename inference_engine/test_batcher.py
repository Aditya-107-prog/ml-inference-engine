"""
inference_engine/test_batcher.py

Run with: pytest test_batcher.py -v

Note on timing tests: we use small real sleeps rather than mocking time.monotonic,
because the batcher's correctness depends on real wall-clock interleaving with a
producer thread. The wait windows here (20-50ms) are short enough to keep the
suite fast, but long enough to be reliable on a normal dev machine or CI runner.
If these ever get flaky in CI, the fix is to inject a fake clock + fake sleep
into DynamicBatcher rather than widening the windows.
"""

import threading
import time

from batcher import DynamicBatcher
from request_queue import Request, RequestQueue, SchedulingPolicy


def test_batch_closes_on_size_when_requests_arrive_fast():
    """If enough requests are already queued, batch should close immediately
    at max_batch_size without waiting for the timeout."""
    q = RequestQueue(SchedulingPolicy.FIFO)
    for i in range(5):
        q.push(Request(id=str(i), prompt="x"))

    batcher = DynamicBatcher(q, max_batch_size=3, max_wait_time_ms=200)

    start = time.monotonic()
    batch = batcher.form_batch()
    elapsed_ms = (time.monotonic() - start) * 1000

    assert len(batch) == 3
    assert elapsed_ms < 50, "should not wait near the 200ms timeout when already full"


def test_batch_closes_on_timeout_when_requests_are_sparse():
    """If only one request ever arrives, the batch should close at the
    timeout rather than waiting forever for more."""
    q = RequestQueue(SchedulingPolicy.FIFO)
    q.push(Request(id="lonely", prompt="x"))

    batcher = DynamicBatcher(q, max_batch_size=8, max_wait_time_ms=30)

    start = time.monotonic()
    batch = batcher.form_batch()
    elapsed_ms = (time.monotonic() - start) * 1000

    assert len(batch) == 1
    assert elapsed_ms >= 30, "should wait out the full timeout before giving up"
    assert elapsed_ms < 100, "should not wait dramatically longer than the timeout"


def test_no_requests_lost_across_multiple_batches():
    """Simulate a producer trickling in requests slower than the batch can
    fill; verify every request is eventually returned exactly once."""
    q = RequestQueue(SchedulingPolicy.FIFO)
    total_requests = 12

    def producer():
        for i in range(total_requests):
            q.push(Request(id=f"req-{i}", prompt="x"))
            time.sleep(0.005)

    threading.Thread(target=producer, daemon=True).start()

    batcher = DynamicBatcher(q, max_batch_size=4, max_wait_time_ms=20)

    seen_ids = []
    while len(seen_ids) < total_requests:
        batch = batcher.form_batch()
        seen_ids.extend(r.id for r in batch.requests)

    assert sorted(seen_ids) == sorted(f"req-{i}" for i in range(total_requests))
    assert len(seen_ids) == len(set(seen_ids)), "no request should appear twice"


def test_batch_never_exceeds_max_size():
    """Even with a flood of requests already queued, a single batch should
    never exceed max_batch_size."""
    q = RequestQueue(SchedulingPolicy.FIFO)
    for i in range(50):
        q.push(Request(id=str(i), prompt="x"))

    batcher = DynamicBatcher(q, max_batch_size=8, max_wait_time_ms=100)
    batch = batcher.form_batch()

    assert len(batch) == 8
    assert len(q) == 42  # 50 - 8 remain on the queue


def test_wait_time_metric_reflects_actual_queue_time():
    """avg_wait_time_ms should roughly match how long requests actually sat
    in the queue before the batch closed."""
    q = RequestQueue(SchedulingPolicy.FIFO)
    q.push(Request(id="only", prompt="x"))

    batcher = DynamicBatcher(q, max_batch_size=8, max_wait_time_ms=40)
    batch = batcher.form_batch()

    # request waited ~40ms (the full timeout, since nothing else arrived)
    assert 35 <= batch.avg_wait_time_ms <= 80
    assert len(batch.wait_times_ms) == 1