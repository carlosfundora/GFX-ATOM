import time
import sys
import uuid
import re
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

try:
    from atom_rust import ToolCallStreamParser as RustToolCallStreamParser
except ImportError as e:
    RustToolCallStreamParser = None
    print(f"Failed to import atom_rust: {e}")

@dataclass
class PyToolCallStreamParser:
    state: int = 0
    buf: str = ""
    current_index: int = 0
    _emitted_calls: int = 0

    def process(self, text: str) -> list:
        results = []

        if self.state == 0:
            self.buf += text
            if "<|tool_calls_section_begin|>" in self.buf:
                before = self.buf.split("<|tool_calls_section_begin|>")[0]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf.split("<|tool_calls_section_begin|>", 1)[1]
                results.extend(self._process_buffer())
            elif "<|tool" not in self.buf and len(self.buf) > 30:
                results.append(("content", self.buf))
                self.buf = ""

        elif self.state == 1:
            self.buf += text
            if "<|tool_calls_section_end|>" in self.buf:
                remaining = self.buf.split("<|tool_calls_section_end|>")[0]
                self.buf = remaining
                results.extend(self._process_buffer())
                results.append(("tool_call_end", None))
                self.state = 2
                self.buf = ""
            else:
                results.extend(self._process_buffer())

        return results

    def _process_buffer(self) -> list:
        results = []
        while "<|tool_call_begin|>" in self.buf and "<|tool_call_end|>" in self.buf:
            match = re.search(
                r"<\|tool_call_begin\|>"
                r"functions\.(\w+):(\d+)"
                r"<\|tool_call_argument_begin\|>"
                r"(.*?)"
                r"<\|tool_call_end\|>",
                self.buf,
                re.DOTALL,
            )
            if not match:
                break

            name = match.group(1)
            index = int(match.group(2))
            arguments = match.group(3).strip()

            call_id = f"call_{uuid.uuid4().hex[:8]}"
            results.append(
                (
                    "tool_call_start",
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    },
                )
            )
            if arguments:
                results.append(
                    (
                        "tool_call_args",
                        {"index": index, "function": {"arguments": arguments}},
                    )
                )

            self.buf = self.buf[match.end() :]
            self._emitted_calls += 1

        return results

    def flush(self) -> list:
        results = []
        if self.state == 0 and self.buf:
            results.append(("content", self.buf))
            self.buf = ""
        elif self.state == 1:
            results.extend(self._process_buffer())
            if self._emitted_calls > 0:
                results.append(("tool_call_end", None))
        return results

def benchmark():
    tokens = [
        "Sure, I can ", "call a tool.\n",
        "<|tool_calls_section_be", "gin|>\n",
        "<|tool_call_begin|>functions.my_tool:0<|tool_call_argument_begin|>",
        '{"arg": ', '"value"}',
        "<|tool_call_end|>\n",
        "<|tool_calls_section_end|>\n",
        "Done calling the tool.\n"
    ]

    print("Benchmarking Python ToolCallStreamParser...")
    start = time.perf_counter()
    for _ in range(50000):
        parser = PyToolCallStreamParser()
        for t in tokens:
            parser.process(t)
        parser.flush()
    end = time.perf_counter()
    py_time = end - start
    print(f"Time: {py_time:.5f}s")

    print("Benchmarking Rust ToolCallStreamParser...")
    start = time.perf_counter()
    for _ in range(50000):
        parser = RustToolCallStreamParser()
        for t in tokens:
            parser.process(t)
        parser.flush()
    end = time.perf_counter()
    rust_time = end - start
    print(f"Time: {rust_time:.5f}s")
    print(f"Speedup: {py_time / rust_time:.2f}x")

    import json
    with open(".jules/verification/rusty/before-benchmark.json", "w") as f:
        json.dump({
            "candidate": "atom/entrypoints/openai/tool_parser.py",
            "implementation": "before",
            "command": "python3 benchmarks/bench_tool_parser.py",
            "timestamp": "2024-05-24T20:21:00Z",
            "iterations": 50000,
            "input_description": "Streaming tool tokens",
            "duration_ms": int(py_time * 1000)
        }, f)

    with open(".jules/verification/rusty/after-benchmark.json", "w") as f:
        json.dump({
            "candidate": "atom/entrypoints/openai/tool_parser.py",
            "implementation": "after",
            "command": "python3 benchmarks/bench_tool_parser.py",
            "timestamp": "2024-05-24T20:21:00Z",
            "iterations": 50000,
            "input_description": "Streaming tool tokens",
            "duration_ms": int(rust_time * 1000)
        }, f)

if __name__ == "__main__":
    benchmark()
