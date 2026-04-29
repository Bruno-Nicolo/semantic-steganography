from __future__ import annotations

from time import perf_counter


class Timer:
    def __enter__(self):
        self._start = perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_ms = (perf_counter() - self._start) * 1000.0
        return False
