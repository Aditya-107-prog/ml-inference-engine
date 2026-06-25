"""
inference_engine/request_queue.py

A priority queue for managing incoming inference requests under three
scheduling policies:

- FIFO:                 process in arrival order
- SHORTEST_JOB_FIRST:    process shortest prompt first (estimated by token/word count)
- PRIORITY:              process highest-priority request first, FIFO within same priority

Design notes (worth being able to explain in interviews):
- Python's heapq is a min-heap, so for "highest priority first" we negate priority.
- heapq requires every pushed item to be totally orderable. If two items tie on the
  primary key, heapq compares the *next* tuple element. If that's a non-orderable
  object (like our Request), it crashes. We fix this with a monotonic counter as a
  tie-breaker, so the heap NEVER needs to compare Request objects directly.
- This queue is intentionally synchronous / single-threaded. The batcher (Week 2)
  will be responsible for draining it under a lock if you move to multithreading.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    FIFO = "fifo"
    SHORTEST_JOB_FIRST = "sjf"
    PRIORITY = "priority"


@dataclass
class Request:
    prompt: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    priority: int = 1          # higher = more important (only used by PRIORITY policy)
    timestamp: float = field(default_factory=time.monotonic)
    length: int = field(init=False)

    def __post_init__(self) -> None:
        # Cheap proxy for "job size" until you wire up a real tokenizer.
        # Swap this for `len(tokenizer.encode(self.prompt))` once the
        # server has a tokenizer loaded — word count under/over-estimates
        # true token count, which is worth a sentence in your README.
        self.length = len(self.prompt.split())
        logger.debug("Request created: %s", self)

    def __repr__(self) -> str:
        return f"Request(id={self.id!r}, len={self.length}, priority={self.priority})"


class RequestQueue:
    """A heap-backed queue whose ordering key depends on the chosen policy."""

    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.FIFO):
        self.policy = policy
        self._heap: list[tuple] = []
        self._counter = itertools.count()  # tie-breaker, guarantees total order

    def push(self, request: Request) -> None:
        tie_breaker = next(self._counter)

        if self.policy is SchedulingPolicy.FIFO:
            key = (request.timestamp, tie_breaker)
        elif self.policy is SchedulingPolicy.SHORTEST_JOB_FIRST:
            key = (request.length, tie_breaker)
        elif self.policy is SchedulingPolicy.PRIORITY:
            # negate priority: heapq is a min-heap, we want max-priority first
            key = (-request.priority, request.timestamp, tie_breaker)
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        heapq.heappush(self._heap, (key, request))
        logger.debug("Pushed %s | queue depth now %d", request, len(self._heap))

    def pop(self) -> Optional[Request]:
        """Remove and return the next request to process, or None if empty."""
        if not self._heap:
            return None
        _, request = heapq.heappop(self._heap)
        logger.debug("Popped %s | queue depth now %d", request, len(self._heap))
        return request

    def peek(self) -> Optional[Request]:
        """Look at the next request without removing it."""
        if not self._heap:
            return None
        return self._heap[0][1]

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Quick manual smoke test — run `python request_queue.py`
    print("=== FIFO ===")
    q = RequestQueue(SchedulingPolicy.FIFO)
    q.push(Request(prompt="first in"))
    q.push(Request(prompt="second in"))
    q.push(Request(prompt="third in"))
    while not q.is_empty():
        print(q.pop())

    print("\n=== SHORTEST_JOB_FIRST ===")
    q = RequestQueue(SchedulingPolicy.SHORTEST_JOB_FIRST)
    q.push(Request(prompt="this is a much longer prompt with many words in it"))
    q.push(Request(prompt="hi"))
    q.push(Request(prompt="a medium length prompt here"))
    while not q.is_empty():
        print(q.pop())

    print("\n=== PRIORITY ===")
    q = RequestQueue(SchedulingPolicy.PRIORITY)
    q.push(Request(prompt="low priority", priority=1))
    q.push(Request(prompt="high priority", priority=10))
    q.push(Request(prompt="mid priority", priority=5))
    while not q.is_empty():
        print(q.pop())