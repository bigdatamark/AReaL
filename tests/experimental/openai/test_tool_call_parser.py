# [MODULE DOC]
# Tests for areal.experimental.openai.tool_call_parser.process_tool_calls, which
# extracts tool calls and strips reasoning/tool markup from generated text, returning
# (tool_calls, remaining text, finish_reason). Covers the built-in qwen3_coder XML
# parser (no SGLang needed), and the qwen25 JSON parser via SGLang or via vLLM (with
# vLLM tool and reasoning parsers replaced by fakes and the SGLang path forced to fail).
# Fixtures include a think block followed by two search tool calls, and a variant with a
# spurious tool call inside the think block which must not be parsed.
import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolParam

from areal.experimental.openai import tool_call_parser as parser_module

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "A powerful web search tool for accessing a vast range of external information beyond its training data. "
                "Use this tool when you need detailed information on highly specific, specialized, or niche topics, or when you need to verify information and fact-check claims by finding authoritative sources. "
                "It helps you answer complex questions that require deep knowledge or specific external data. The input should be a clear search query designed to find specific knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A precise search query for information retrieval.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "access",
            "description": (
                "Invokes the Jina AI Reader engine to intelligently access and parse a URL. "
                "This tool takes a webpage URL as input and returns its main article content in a clean Markdown format. "
                "Use this to perform a 'deep dive' on the most relevant link found via web_search to extract detailed evidence and data needed to answer a question. "
                "It automatically ignores advertisements and boilerplate code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the webpage to read.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

TEXT = (
    '<think>\nOkay, so the user is asking whether the director of "Scary Movie" and the director of "The Preacher\'s Wife" are from the same country. Let me think about how to approach this.\n\n'
    'First, I need to confirm the countries of these directors. I remember that "Scary Movie" is directed by Joe Anderson, and "The Preacher\'s Wife" is directed by David Fincher. Wait, but actually, "The Preacher\'s Wife" is directed by Christopher Nolan and David Fincher? No, I think I mixed up. Let me check my memory. \n\n'
    "Wait, no. The Preacher's Wife is a movie directed by Christopher Nolan. And David Fincher directed The Preacher. So the directors are different. Then the user is asking if they are both from the same country. The answer would be no, because the two directors are from different countries. \n\n"
    'But maybe I should verify this to be sure. Since I can use the web search function, I should use the "access" tool to get the URLs of the directors\' websites to confirm. So first, I\'ll search for "director of Scary Movie country" and "director of The Preacher\'s Wife country" to get precise data. Then, analyze the results to see if they have the same country affiliations.\n</think>\n\n'
    '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n'
    '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>'
)

TEXT_WITH_TOOL_CALL_IN_THINKING = (
    '<think>\nOkay, so the user is asking whether the director of "Scary Movie" and the director of "The Preacher\'s Wife" are from the same country. Let me think about how to approach this.\n\n'
    'First, I need to confirm the countries of these directors. I remember that "Scary Movie" is directed by Joe Anderson, and "The Preacher\'s Wife" is directed by David Fincher. Wait, but actually, "The Preacher\'s Wife" is directed by Christopher Nolan and David Fincher? No, I think I mixed up. Let me check my memory. \n\n'
    'Wait, no. The Preacher\'s Wife is a movie directed by Christopher Nolan. And David Fincher directed The Preacher. <tool_call>\n{"name": "search", "arguments": {"query": "aaaa"}}\n</tool_call>\n\n So the directors are different. Then the user is asking if they are both from the same country. The answer would be no, because the two directors are from different countries. \n\n'
    'But maybe I should verify this to be sure. Since I can use the web search function, I should use the "access" tool to get the URLs of the directors\' websites to confirm. So first, I\'ll search for "director of Scary Movie country" and "director of The Preacher\'s Wife country" to get precise data. Then, analyze the results to see if they have the same country affiliations.\n</think>\n\n'
    '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n'
    '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>'
)

QWEN3_CODER_TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
]

QWEN3_CODER_TEXT = """I'll inspect the repository.

<tool_call>
<function=Bash>
<parameter=command>
ls -la
</parameter>
<parameter=timeout>
120
</parameter>
</function>
</tool_call><|im_end|>"""


def _assert_tool_calls(tool_calls, new_text: str, new_finish_reason: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Asserts finish_reason is tool_calls and exactly two
    # ChatCompletionMessageToolCall objects of type function named search with the two
    # expected JSON argument strings. The new_text argument is accepted but not checked.
    assert new_finish_reason == "tool_calls"
    assert tool_calls is not None, "Tool calls should be detected and returned"
    assert len(tool_calls) == 2, "Two tool calls should be parsed from the text"
    assert isinstance(tool_calls[0], ChatCompletionMessageToolCall)
    assert isinstance(tool_calls[1], ChatCompletionMessageToolCall)
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search"
    assert tool_calls[1].function.name == "search"
    assert (
        tool_calls[0].function.arguments
        == '{"query": "director of Scary Movie country"}'
    )
    assert (
        tool_calls[1].function.arguments
        == '{"query": "director of The Preacher\'s Wife country"}'
    )


def _run_process_tool_calls(text: str):
    # [FUNCTION DOC]
    # HELPER: Runs process_tool_calls on text with the search/access TOOLS, qwen25 tool
    # parser, qwen3 reasoning parser, finish_reason tool_calls, chat-completions mode
    # and a dummy tokenizer object; returns its (tool_calls, text, finish_reason) tuple.
    return parser_module.process_tool_calls(
        text=text,
        tools=TOOLS,
        tool_call_parser="qwen25",
        reasoning_parser="qwen3",
        finish_reason="tool_calls",
        use_responses=False,
        tokenizer=object(),
    )


def test_process_tool_calls_qwen3_coder_xml_parameters_without_sglang():
    # [FUNCTION DOC]
    # Parses a qwen3_coder XML tool call (function=Bash with command and timeout params)
    # and asserts finish_reason becomes tool_calls, one Bash call with arguments command
    # "ls -la" and timeout coerced to integer 120 per the schema, and that the remaining
    # text keeps the preamble but no tool_call markup or end-of-turn token.
    tool_calls, new_text, new_finish_reason = parser_module.process_tool_calls(
        text=QWEN3_CODER_TEXT,
        tools=QWEN3_CODER_TOOLS,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        finish_reason="stop",
        use_responses=False,
        tokenizer=object(),
    )

    assert new_finish_reason == "tool_calls"
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "Bash"
    assert json.loads(tool_calls[0].function.arguments) == {
        "command": "ls -la",
        "timeout": 120,
    }
    assert "<tool_call>" not in new_text
    assert "<|im_end|>" not in new_text
    assert "I'll inspect the repository." in new_text


def test_qwen3_coder_xml_empty_args_returns_no_parse():
    # [FUNCTION DOC]
    # Asserts the private qwen3_coder XML parser returns (None, original text, original
    # finish_reason) when the function block has no parameters.
    text = "<tool_call>\n<function=Bash>\n</function>\n</tool_call>"

    tool_calls, new_text, finish_reason = (
        parser_module._process_tool_calls_qwen3_coder_xml(
            text=text,
            tools=QWEN3_CODER_TOOLS,
            finish_reason="stop",
        )
    )

    assert tool_calls is None
    assert new_text == text
    assert finish_reason == "stop"


def test_qwen3_coder_xml_literal_closing_tag_is_not_silently_truncated():
    # [FUNCTION DOC]
    # Asserts that a parameter value containing a literal closing parameter tag makes
    # the XML parser refuse to parse (None, unchanged text, unchanged finish_reason)
    # rather than truncating the content.
    text = (
        "<tool_call>\n<function=Write>\n"
        "<parameter=file_path>\na.xml\n</parameter>\n"
        "<parameter=content>\nsome XML text: </parameter> more content\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    tool_calls, new_text, finish_reason = (
        parser_module._process_tool_calls_qwen3_coder_xml(
            text=text,
            tools=QWEN3_CODER_TOOLS,
            finish_reason="stop",
        )
    )

    assert tool_calls is None
    assert new_text == text
    assert finish_reason == "stop"


@pytest.mark.sglang
def test_process_tool_calls_qwen25_chat_completions_sglang():
    # [FUNCTION DOC]
    # SGLang path (skipped without SGLang): asserts both search calls after the think
    # block are extracted and no tool_call markup remains in the text.
    pytest.importorskip(
        "sglang.srt.function_call.function_call_parser",
        reason="sglang is required for sglang parser tests",
    )
    pytest.importorskip(
        "sglang.srt.parser.reasoning_parser",
        reason="sglang is required for sglang parser tests",
    )

    tool_calls, new_text, new_finish_reason = _run_process_tool_calls(TEXT)

    _assert_tool_calls(tool_calls, new_text, new_finish_reason)
    assert "<tool_call>" not in new_text


@pytest.mark.sglang
def test_process_tool_calls_qwen25_chat_completions_with_tool_call_in_thinking_sglang():
    # [FUNCTION DOC]
    # SGLang path: with a tool call embedded inside the think block, asserts only the
    # two calls after the reasoning are extracted and the embedded markup stays in the
    # text.
    pytest.importorskip(
        "sglang.srt.function_call.function_call_parser",
        reason="sglang is required for sglang parser tests",
    )
    pytest.importorskip(
        "sglang.srt.parser.reasoning_parser",
        reason="sglang is required for sglang parser tests",
    )

    tool_calls, new_text, new_finish_reason = _run_process_tool_calls(
        TEXT_WITH_TOOL_CALL_IN_THINKING
    )

    _assert_tool_calls(tool_calls, new_text, new_finish_reason)
    assert "<tool_call>" in new_text


def _raise_module_not_found(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Always raises ModuleNotFoundError; used to force the SGLang code path to
    # fail so the vLLM fallback is exercised.
    raise ModuleNotFoundError


class FakeReasoningParser:
    # [CLASS DOC]
    # HELPER: Fake vLLM reasoning parser exposing think start/end tokens; constructor
    # ignores its arguments.
    start_token = "<think>"
    end_token = "</think>"

    def __init__(self, tokenizer, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Accepts and ignores tokenizer and any other arguments.
        pass


def _patch_vllm_parsers(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Skips unless vllm.tool_parsers and vllm.reasoning import. Monkeypatches
    # the module's _process_tool_calls_sglang to raise ModuleNotFoundError and makes
    # vLLM's ReasoningParserManager.get_reasoning_parser return FakeReasoningParser.
    # Returns the vllm.tool_parsers module.
    tool_parsers = pytest.importorskip(
        "vllm.tool_parsers",
        reason="vllm is required for vllm parser tests",
    )
    reasoning_mod = pytest.importorskip(
        "vllm.reasoning",
        reason="vllm is required for vllm parser tests",
    )
    monkeypatch.setattr(
        parser_module, "_process_tool_calls_sglang", _raise_module_not_found
    )
    monkeypatch.setattr(
        reasoning_mod.ReasoningParserManager,
        "get_reasoning_parser",
        staticmethod(lambda name: FakeReasoningParser),
    )
    return tool_parsers


@pytest.mark.vllm
def test_process_tool_calls_qwen25_chat_completions_vllm(
    monkeypatch: pytest.MonkeyPatch,
):
    # [FUNCTION DOC]
    # vLLM path: installs a fake qwen25 tool parser that requires skip_special_tokens
    # True on the request, removes the two known tool-call strings from content and
    # returns two search calls. Asserts the parsed calls and that no tool_call markup
    # remains.
    tool_parsers = _patch_vllm_parsers(monkeypatch)

    class FakeParser:
        # [CLASS DOC]
        # HELPER: Fake vLLM tool parser returning canned results for the TEXT fixture.
        def __init__(self, tokenizer):
            # [FUNCTION DOC]
            # HELPER: Stores the tokenizer.
            self.tokenizer = tokenizer

        def extract_tool_calls(self, content_text, request):
            # [FUNCTION DOC]
            # HELPER: Asserts request.skip_special_tokens is True and returns a
            # namespace with tools_called True, content with the two trailing tool-call
            # strings removed, and two search tool calls with fixed arguments.
            assert request.skip_special_tokens is True
            return SimpleNamespace(
                tools_called=True,
                content=content_text.replace(
                    '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n',
                    "",
                ).replace(
                    '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>',
                    "",
                ),
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="search",
                            arguments='{"query": "director of Scary Movie country"}',
                        )
                    ),
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="search",
                            arguments='{"query": "director of The Preacher\'s Wife country"}',
                        )
                    ),
                ],
            )

    monkeypatch.setattr(
        tool_parsers.ToolParserManager,
        "get_tool_parser",
        staticmethod(lambda name: FakeParser),
    )

    tool_calls, new_text, new_finish_reason = _run_process_tool_calls(TEXT)

    _assert_tool_calls(tool_calls, new_text, new_finish_reason)
    assert "<tool_call>" not in new_text


@pytest.mark.vllm
def test_process_tool_calls_qwen25_chat_completions_with_tool_call_in_thinking_vllm(
    monkeypatch: pytest.MonkeyPatch,
):
    # [FUNCTION DOC]
    # vLLM path with a tool call inside the think block: asserts the two real calls are
    # returned and the embedded tool-call markup remains in the text (reasoning is split
    # off before tool parsing).
    tool_parsers = _patch_vllm_parsers(monkeypatch)

    class FakeParser:
        # [CLASS DOC]
        # HELPER: Fake vLLM tool parser returning canned results for the in-thinking
        # fixture.
        def __init__(self, tokenizer):
            # [FUNCTION DOC]
            # HELPER: Stores the tokenizer.
            self.tokenizer = tokenizer

        def extract_tool_calls(self, content_text, request):
            # [FUNCTION DOC]
            # HELPER: Asserts request.skip_special_tokens is True and returns
            # tools_called True, content with the two trailing tool-call strings
            # removed, and two fixed search calls.
            assert request.skip_special_tokens is True
            return SimpleNamespace(
                tools_called=True,
                content=content_text.replace(
                    '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n',
                    "",
                ).replace(
                    '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>',
                    "",
                ),
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="search",
                            arguments='{"query": "director of Scary Movie country"}',
                        )
                    ),
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="search",
                            arguments='{"query": "director of The Preacher\'s Wife country"}',
                        )
                    ),
                ],
            )

    monkeypatch.setattr(
        tool_parsers.ToolParserManager,
        "get_tool_parser",
        staticmethod(lambda name: FakeParser),
    )

    tool_calls, new_text, new_finish_reason = _run_process_tool_calls(
        TEXT_WITH_TOOL_CALL_IN_THINKING
    )

    _assert_tool_calls(tool_calls, new_text, new_finish_reason)
    assert "<tool_call>" in new_text
