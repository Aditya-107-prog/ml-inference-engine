"""
inference_engine/test_queue.py

Run with: pytest test_queue.py -v
"""

import time

from queue import Request, RequestQueue, SchedulingPolicy


def test_fifo_preserves_arrival_order():
    q = RequestQueue(SchedulingPolicy.FIFO)
    for i in range(5):
        q.push(Request(id=str(i), prompt="x"))
        time.sleep(0.001)  # ensure distinct timestamps

    order = [q.pop().id for _ in range(5)]
    assert order == ["0", "1", "2", "3", "4"]


def test_sjf_picks_shortest_first():
    q = RequestQueue(SchedulingPolicy.SHORTEST_JOB_FIRST)
    q.push(Request(id="long", prompt="one two three four five"))
    q.push(Request(id="short", prompt="one"))
    q.push(Request(id="medium", prompt="one two three"))

    order = [q.pop().id for _ in range(3)]
    assert order == ["short", "medium", "long"]


def test_priority_picks_highest_first():
    q = RequestQueue(SchedulingPolicy.PRIORITY)
    q.push(Request(id="low", prompt="x", priority=1))
    q.push(Request(id="high", prompt="x", priority=10))
    q.push(Request(id="mid", prompt="x", priority=5))

    order = [q.pop().id for _ in range(3)]
    assert order == ["high", "mid", "low"]


def test_equal_priority_falls_back_to_fifo():
    """This is the case that crashes the original (buggy) stub implementation:
    two requests with identical sort keys, requiring heapq to compare a third
    element. Our tie-breaker counter prevents that crash."""
    q = RequestQueue(SchedulingPolicy.PRIORITY)
    r1 = Request(id="first", prompt="x", priority=5)
    r2 = Request(id="second", prompt="x", priority=5)
    # force identical timestamps to actually trigger the tie
    r2.timestamp = r1.timestamp

    q.push(r1)
    q.push(r2)

    order = [q.pop().id for _ in range(2)]
    assert order == ["first", "second"]


def test_empty_queue_returns_none():
    q = RequestQueue(SchedulingPolicy.FIFO)
    assert q.pop() is None
    assert q.peek() is None
    assert q.is_empty()


def test_len_tracks_queue_size():
    q = RequestQueue(SchedulingPolicy.FIFO)
    assert len(q) == 0
    q.push(Request(id="a", prompt="x"))
    q.push(Request(id="b", prompt="x"))
    assert len(q) == 2
    q.pop()
    assert len(q) == 1


def test_peek_does_not_remove():
    q = RequestQueue(SchedulingPolicy.FIFO)
    q.push(Request(id="a", prompt="x"))
    peeked = q.peek()
    assert peeked.id == "a"
    assert len(q) == 1  # still there