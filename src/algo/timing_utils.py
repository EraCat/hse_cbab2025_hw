import sys
import time
from contextlib import ContextDecorator


class timed(ContextDecorator):
    def __init__(self, label=None, stream=None):
        self.label = label or "block"
        self.stream = stream or sys.stderr
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.perf_counter() - self._start
        print(f"[timing] {self.label}: {elapsed:.6f}s", file=self.stream)
        return False
