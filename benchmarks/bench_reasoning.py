import time
import sys

# Fake out atom module requirements so we can test the module
import sys
from unittest.mock import MagicMock
sys.modules['atom'] = MagicMock()

from atom_rust import ReasoningFilter as RustReasoningFilter

import re
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PyReasoningFilter:
    state: int = 0
    buf: str = ""

    def process(self, text: str) -> list:
        results = []

        if self.state == 0:
            self.buf += text
            if "<think>" in self.buf:
                before = self.buf.split("<think>")[0]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf.split("<think>", 1)[1]
                if "</think>" in self.buf:
                    reasoning = self.buf.split("</think>", 1)[0]
                    after = self.buf.split("</think>", 1)[1].lstrip("\n")
                    if reasoning:
                        results.append(("reasoning_content", reasoning))
                    self.state = 2
                    self.buf = ""
                    if after:
                        results.extend(self._process_content(after))
                elif self.buf:
                    results.append(("reasoning_content", self.buf))
                    self.buf = ""
            elif "</think>" in self.buf:
                reasoning = self.buf.split("</think>", 1)[0]
                after = self.buf.split("</think>", 1)[1].lstrip("\n")
                if reasoning:
                    results.append(("reasoning_content", reasoning))
                self.state = 2
                self.buf = ""
                if after:
                    results.extend(self._process_content(after))
            elif len(self.buf) > 7 and "<" not in self.buf:
                results.append(("content", self.buf))
                self.buf = ""

        elif self.state == 1:
            self.buf += text
            if "</think>" in self.buf:
                reasoning = self.buf.split("</think>", 1)[0]
                after = self.buf.split("</think>", 1)[1].lstrip("\n")
                if reasoning:
                    results.append(("reasoning_content", reasoning))
                self.state = 2
                self.buf = ""
                if after:
                    results.extend(self._process_content(after))
            else:
                results.append(("reasoning_content", self.buf))
                self.buf = ""

        else:
            results.extend(self._process_content(text))

        return results

    def _process_content(self, text: str) -> list:
        if text:
            return [("content", text)]
        return []

    def flush(self) -> list:
        results = []
        if self.buf:
            if self.state == 0:
                results.append(("content", self.buf))
            elif self.state == 1:
                results.append(("reasoning_content", self.buf))
            self.buf = ""
        return results

def benchmark():
    tokens = [
        "Sure, I can ", "help with that.\n",
        "<thi", "nk>\n", "This is ", "a reasoning block.\n", "It contains ", "multiple ", "tokens.\n", "</think>\n",
        "Here is the final ", "answer.\n"
    ]

    print("Benchmarking Python ReasoningFilter...")
    start = time.perf_counter()
    for _ in range(50000):
        filter = PyReasoningFilter()
        for t in tokens:
            filter.process(t)
        filter.flush()
    end = time.perf_counter()
    py_time = end - start
    print(f"Time: {py_time:.5f}s")

    print("Benchmarking Rust ReasoningFilter...")
    start = time.perf_counter()
    for _ in range(50000):
        filter = RustReasoningFilter()
        for t in tokens:
            filter.process(t)
        filter.flush()
    end = time.perf_counter()
    rust_time = end - start
    print(f"Time: {rust_time:.5f}s")
    print(f"Speedup: {py_time / rust_time:.2f}x")

    import json
    with open(".jules/verification/rusty/before-benchmark.json", "w") as f:
        json.dump({
            "candidate": "atom/entrypoints/openai/reasoning.py",
            "implementation": "before",
            "command": "python3 benchmarks/bench_reasoning.py",
            "timestamp": "2024-05-24T20:21:00Z",
            "iterations": 50000,
            "input_description": "Streaming thinking tokens",
            "duration_ms": int(py_time * 1000)
        }, f)

    with open(".jules/verification/rusty/after-benchmark.json", "w") as f:
        json.dump({
            "candidate": "atom/entrypoints/openai/reasoning.py",
            "implementation": "after",
            "command": "python3 benchmarks/bench_reasoning.py",
            "timestamp": "2024-05-24T20:21:00Z",
            "iterations": 50000,
            "input_description": "Streaming thinking tokens",
            "duration_ms": int(rust_time * 1000)
        }, f)

if __name__ == "__main__":
    benchmark()
