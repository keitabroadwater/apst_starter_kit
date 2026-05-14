"""Thread-safe rate and concurrency limiting."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class RateLimiter:
    """Simple minimum-interval limiter."""

    def __init__(self, max_requests_per_second: float = 2.0) -> None:
        self.min_interval = 1.0 / max_requests_per_second if max_requests_per_second > 0 else 0.0
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
                now = time.time()
            self._last_call = now


class RateLimitedExecutor:
    """Apply rate and concurrency limits around a callable."""

    def __init__(
        self,
        *,
        max_concurrent_requests: int = 5,
        max_requests_per_second: float = 2.0,
    ) -> None:
        self.rate_limiter = RateLimiter(max_requests_per_second)
        self.semaphore = threading.Semaphore(max_concurrent_requests)

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.rate_limiter.wait()
        with self.semaphore:
            return func(*args, **kwargs)

