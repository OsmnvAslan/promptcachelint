from __future__ import annotations

from promptcachelint.model import Usage
from promptcachelint.providers import detect_provider, get_provider, register_provider
from promptcachelint.providers.anthropic import AnthropicProvider
from promptcachelint.providers.openai import OpenAIProvider
from tests.conftest import anthropic_body, tool


class TestAnthropic:
    p = AnthropicProvider()

    def test_render_order_and_paths(self) -> None:
        body = anthropic_body(
            tools=[tool("a"), tool("b")],
            history=[{"role": "assistant", "content": "hi"}],
            mark_last=True,
        )
        segs = self.p.segments(body)
        assert [s.path for s in segs] == [
            "tools[0]",
            "tools[1]",
            "system[0]",
            "messages[0]",
            "messages[1].content[0]",
        ]
        assert [s.kind for s in segs] == ["tool", "tool", "system", "message", "message"]
        assert segs[3].text == "hi" and segs[3].role == "assistant"
        assert segs[4].text == "hello" and segs[4].role == "user"

    def test_markers_are_stripped_from_text_but_kept_as_flags(self) -> None:
        marked = anthropic_body(mark_system=True, ttl="1h")
        plain = anthropic_body(mark_system=False)
        a, b = self.p.segments(marked), self.p.segments(plain)
        assert [s.text for s in a] == [s.text for s in b]
        assert a[0].breakpoint and a[0].ttl == "1h"
        assert not b[0].breakpoint

    def test_cacheable_prefix_follows_the_last_marker(self) -> None:
        body = anthropic_body(history=[{"role": "assistant", "content": "hi"}], mark_last=True)
        segs = self.p.segments(body)
        assert self.p.cacheable_segments(body, segs) == 3
        no_marks = anthropic_body(mark_system=False)
        assert self.p.cacheable_segments(no_marks, self.p.segments(no_marks)) == 0

    def test_top_level_cache_control_means_everything_is_cacheable(self) -> None:
        body = anthropic_body(mark_system=False, top_level=True)
        segs = self.p.segments(body)
        assert self.p.cacheable_segments(body, segs) == len(segs)

    def test_tool_marker_and_string_system(self) -> None:
        t = tool("a") | {"cache_control": {"type": "ephemeral"}}
        body = {"model": "m", "system": "sys", "tools": [t], "messages": []}
        segs = self.p.segments(body)
        assert segs[0].breakpoint and "cache_control" not in segs[0].text
        assert segs[1].path == "system" and segs[1].text == "sys"

    def test_usage(self) -> None:
        u = self.p.usage(
            {
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 900,
                    "cache_creation_input_tokens": 100,
                    "output_tokens": 5,
                }
            }
        )
        assert u == Usage(10, 900, 100, 5)
        assert u.prompt_tokens == 1010
        assert abs(u.hit_ratio - 900 / 1010) < 1e-9
        assert self.p.usage({"usage": {}}) is None

    def test_usage_from_sse(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 3, "cache_read_input_tokens": 50}},
            },
            {"type": "content_block_delta"},
            {"type": "message_delta", "usage": {"output_tokens": 7}},
        ]
        assert self.p.usage_from_sse(events) == Usage(3, 50, 0, 7)
        assert self.p.usage_from_sse([{"type": "ping"}]) is None

    def test_min_prefix_tokens_per_model(self) -> None:
        assert self.p.min_prefix_tokens("claude-opus-5") == 512
        assert self.p.min_prefix_tokens("claude-fable-5-1") == 512
        assert self.p.min_prefix_tokens("claude-sonnet-5") == 1024
        assert self.p.min_prefix_tokens("claude-opus-4-7") == 2048
        assert self.p.min_prefix_tokens("claude-haiku-4-5") == 4096
        assert self.p.min_prefix_tokens(None) == 1024

    def test_scope(self) -> None:
        body = anthropic_body() | {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        assert self.p.scope(body) == {
            "model": '"claude-opus-5"',
            "thinking": '{"type":"adaptive"}',
            "output_config.effort": '"high"',
        }

    def test_matches(self) -> None:
        assert self.p.matches("https://api.anthropic.com/v1/messages")
        assert self.p.matches("https://api.anthropic.com/v1/messages?beta=true")
        assert not self.p.matches("https://api.anthropic.com/v1/models")


class TestOpenAI:
    p = OpenAIProvider()

    def test_chat_segments(self) -> None:
        body = {
            "model": "gpt-5",
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            ],
        }
        segs = self.p.segments(body)
        assert [s.path for s in segs] == [
            "tools[0]",
            "messages[0]",
            "messages[1].content[0]",
            "messages[2]",
        ]
        assert segs[1].kind == "system" and segs[1].text == "sys"
        assert segs[2].text == "hi" and segs[2].role == "user"
        assert "c1" in segs[3].text and segs[3].block == "tool_calls"
        assert self.p.cacheable_segments(body, segs) == 4

    def test_responses_segments(self) -> None:
        body = {
            "model": "gpt-5",
            "instructions": "sys",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "function_call_output", "call_id": "c", "output": "42"},
            ],
        }
        segs = self.p.segments(body)
        assert [s.path for s in segs] == ["instructions", "input[0].content[0]", "input[1]"]
        assert segs[1].text == "hi" and segs[1].role == "user"

    def test_usage_normalization(self) -> None:
        chat = self.p.usage(
            {
                "usage": {
                    "prompt_tokens": 1200,
                    "prompt_tokens_details": {"cached_tokens": 1024},
                    "completion_tokens": 9,
                }
            }
        )
        assert chat == Usage(176, 1024, 0, 9)
        responses = self.p.usage(
            {
                "usage": {
                    "input_tokens": 500,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                }
            }
        )
        assert responses == Usage(500, 0, 0, 1)

    def test_usage_from_sse(self) -> None:
        chunks = [
            {"choices": [{}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 8}},
            },
        ]
        assert self.p.usage_from_sse(chunks) == Usage(2, 8, 0, None)
        done = [
            {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 4, "input_tokens_details": {"cached_tokens": 0}}
                },
            }
        ]
        assert self.p.usage_from_sse(done) == Usage(4, 0, 0, None)

    def test_matches_and_scope(self) -> None:
        assert self.p.matches("https://api.openai.com/v1/chat/completions")
        assert self.p.matches("https://api.openai.com/v1/responses")
        assert not self.p.matches("https://api.openai.com/v1/embeddings")
        assert self.p.scope({"model": "gpt-5", "prompt_cache_key": "k"}) == {
            "model": '"gpt-5"',
            "prompt_cache_key": '"k"',
        }


def test_registry() -> None:
    assert get_provider("anthropic").name == "anthropic"
    assert detect_provider("https://api.openai.com/v1/responses") is not None
    assert detect_provider("https://example.com/x") is None
    try:
        get_provider("nope")
    except ValueError as e:
        assert "unknown provider" in str(e)
    else:
        raise AssertionError

    class Custom(AnthropicProvider):
        name = "gateway"

        def matches(self, url: str) -> bool:
            return "gateway.local" in url

    register_provider(Custom())
    assert detect_provider("https://gateway.local/v1/messages") is not None
