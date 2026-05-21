import re
from atom.entrypoints.openai.tool_parser import parse_tool_calls, ToolCall

def test_rust_parse_tool_calls():
    text = "Here are the calls:\n<|tool_calls_section_begin|>\n<|tool_call_begin|>functions.my_tool:1<|tool_call_argument_begin|>{\"arg\": 1}<|tool_call_end|>\n<|tool_calls_section_end|>\nDone."

    content, calls = parse_tool_calls(text)

    assert content == "Here are the calls:\n\nDone."
    assert len(calls) == 1
    assert calls[0].type == "function"
    assert calls[0].function["name"] == "my_tool"
    assert calls[0].function["arguments"] == "{\"arg\": 1}"
    assert calls[0].id.startswith("call_")

def test_rust_parse_no_tool_calls():
    text = "No tool calls here."
    content, calls = parse_tool_calls(text)
    assert content == "No tool calls here."
    assert len(calls) == 0

def test_rust_parse_unclosed_section():
    text = "Start<|tool_calls_section_begin|><|tool_call_begin|>functions.f1:1<|tool_call_argument_begin|>{}<|tool_call_end|>"
    content, calls = parse_tool_calls(text)
    assert content == "Start"
    assert len(calls) == 1
    assert calls[0].function["name"] == "f1"
