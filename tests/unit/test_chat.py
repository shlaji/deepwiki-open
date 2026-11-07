import json
from collections.abc import AsyncIterator
from contextlib import aclosing

import httpx
import pytest
from openai import AsyncOpenAI

from api.chat import ChatStreamer
from api.chat._stream import (
    AnthropicChatStreamer,
    AzureChatStreamer,
    BedrockChatStreamer,
    DashScopeChatStreamer,
    GoogleGenerativeChatStreamer,
    LiteLLMChatStreamer,
    OllamaChatStreamer,
    OpenAIChatStreamer,
    OpenRouterChatStreamer,
)


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("ollama", OllamaChatStreamer),
        ("openrouter", OpenRouterChatStreamer),
        ("openai", OpenAIChatStreamer),
        ("azure", AzureChatStreamer),
        ("bedrock", BedrockChatStreamer),
        ("dashscope", DashScopeChatStreamer),
        ("google", GoogleGenerativeChatStreamer),
        ("litellm", LiteLLMChatStreamer),
        ("anthropic", AnthropicChatStreamer),
    ],
)
def test_every_provider_is_registered(provider, expected):
    assert ChatStreamer._registry[provider] is expected


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("ollama", OllamaChatStreamer),
        ("openrouter", OpenRouterChatStreamer),
        ("openai", OpenAIChatStreamer),
        ("azure", AzureChatStreamer),
        ("bedrock", BedrockChatStreamer),
        ("dashscope", DashScopeChatStreamer),
        ("google", GoogleGenerativeChatStreamer),
        ("litellm", LiteLLMChatStreamer),
        ("anthropic", AnthropicChatStreamer),
    ],
)
def test_create_returns_correct_subclass(monkeypatch, provider, expected):
    monkeypatch.setattr(expected, "__init__", lambda self, **kw: None)
    s = ChatStreamer.create(provider=provider, model="m", model_config={"model": "m"})
    assert isinstance(s, expected)


def test_create_unknown_provider_raises():
    with pytest.raises(RuntimeError, match="not registered"):
        ChatStreamer.create(provider="nope", model=None, model_config={})


def test_create_passes_initialize_kwargs_to_provider_streamer(monkeypatch):
    captured = {}

    def fake_init(self, *, model, model_config, initialize_kwargs):
        captured["model"] = model
        captured["model_config"] = model_config
        captured["initialize_kwargs"] = initialize_kwargs

    monkeypatch.setattr(OpenAIChatStreamer, "__init__", fake_init)
    client_kwargs = {"base_url": "https://openai.example.test/v1"}

    ChatStreamer.create(
        provider="openai",
        model="test-model",
        model_config={"model": "test-model"},
        initialize_kwargs=client_kwargs,
    )

    assert captured == {
        "model": "test-model",
        "model_config": {"model": "test-model"},
        "initialize_kwargs": client_kwargs,
    }


class _SSEStream(httpx.AsyncByteStream):
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self.events:
            yield f"event: {json.loads(event).get('type', 'message')}\ndata: {event}\n\n".encode()

    async def aclose(self) -> None:
        self.closed = True


async def _collect_sse(
    events: list[str], provider: str = "openai", *, close_after_first: bool = False
) -> list[str]:
    """Exercise the real AdalFlow client and OpenAI SDK using wire-level SSE."""
    body = _SSEStream(events)

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["stream"] is True
        assert payload["temperature"] == 1.0
        assert payload["top_p"] == 0.9
        if provider == "openai":
            assert request.url.path == "/v1/responses"
            assert payload["input"] == "Generate wiki XML"
            assert "messages" not in payload
        else:
            assert request.url.path == "/v1/chat/completions"
            assert payload["messages"][0]["content"] == "Generate wiki XML"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body
        )

    streamer = ChatStreamer.create(
        provider=provider,
        model="test-model",
        model_config={"temperature": 1.0, "top_p": 0.9},
        initialize_kwargs={"api_key": "test-key", "base_url": "https://test.invalid/v1"},
    )
    assert isinstance(streamer, (OpenAIChatStreamer, LiteLLMChatStreamer))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
        async with AsyncOpenAI(
            api_key="test-key", base_url="https://test.invalid/v1", http_client=transport
        ) as client:
            streamer.client.async_client = client
            try:
                async with aclosing(streamer.respond_stream("Generate wiki XML")) as output:
                    if close_after_first:
                        return [await anext(output)]
                    return [chunk async for chunk in output]
            finally:
                # Check before client teardown can hide a leaked response.
                assert body.closed, "Adapter left the upstream HTTP stream open"


@pytest.mark.asyncio
async def test_responses_stream_emits_only_incremental_output_text():
    response = {
        "id": "resp_test", "object": "response", "created_at": 0,
        "model": "test-model", "output": [], "status": "in_progress",
        "error": None, "incomplete_details": None, "instructions": None,
        "metadata": {}, "parallel_tool_calls": True, "temperature": 1.0,
        "tool_choice": "auto", "tools": [], "top_p": 1.0,
    }
    events = [
        json.dumps({"type": "response.created", "sequence_number": 0, "response": response}),
        json.dumps({"type": "response.in_progress", "sequence_number": 1, "response": response}),
        '{"type":"response.reasoning_summary_text.delta","delta":"private reasoning","item_id":"rs_1","output_index":0,"summary_index":0,"sequence_number":2}',
        '{"type":"response.output_text.delta","delta":"<wiki>","item_id":"msg_1","output_index":1,"content_index":0,"logprobs":[],"sequence_number":3}',
        '{"type":"response.output_text.delta","delta":"page</wiki>","item_id":"msg_1","output_index":1,"content_index":0,"logprobs":[],"sequence_number":4}',
        '{"type":"response.output_text.done","text":"<wiki>page</wiki>","item_id":"msg_1","output_index":1,"content_index":0,"logprobs":[],"sequence_number":5}',
        json.dumps({"type": "response.completed", "sequence_number": 6, "response": {**response, "status": "completed"}}),
        '{"type":"response.output_text.delta","delta":"unread after completion","item_id":"msg_1","output_index":1,"content_index":0,"logprobs":[],"sequence_number":7}',
    ]

    assert await _collect_sse(events) == ["<wiki>", "page</wiki>"]


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_text", [False, True])
@pytest.mark.parametrize(
    "event, message",
    [
        ('{"type":"error","code":"server_error","message":"Upstream unavailable","param":null,"sequence_number":1}', "Upstream unavailable"),
        ('{"type":"response.failed","response":{"error":{"code":"server_error","message":"Generation failed"}},"sequence_number":1}', "Generation failed"),
        ('{"type":"response.failed","response":{"error":null},"sequence_number":1}', "response.failed"),
        ('{"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}},"sequence_number":1}', "max_output_tokens"),
        ('{"type":"response.incomplete","response":{"incomplete_details":{"reason":"content_filter"}},"sequence_number":1}', "content_filter"),
        ('{"type":"response.incomplete","response":{"incomplete_details":null},"sequence_number":1}', "response.incomplete"),
    ],
)
async def test_responses_stream_raises_on_error_or_unsuccessful_terminal_event(
    event: str, message: str, partial_text: bool
):
    events = []
    if partial_text:
        events.append('{"type":"response.output_text.delta","delta":"<wiki>","item_id":"msg_1","output_index":0,"content_index":0,"logprobs":[],"sequence_number":0}')
    events.append(event)
    with pytest.raises(RuntimeError, match=message):
        await _collect_sse(events)


@pytest.mark.asyncio
async def test_chat_completions_stream_preserves_text_and_ignores_empty_choices():
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]},
        {"choices": [{"index": 0, "delta": {"content": "<wiki>page</wiki>"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    ]
    events = [json.dumps({"id": "chatcmpl_test", "object": "chat.completion.chunk", "created": 0, "model": "test-model", **chunk}) for chunk in chunks]

    assert await _collect_sse(events, "litellm") == ["<wiki>page</wiki>"]


@pytest.mark.asyncio
async def test_responses_stream_closes_upstream_when_consumer_closes_generator():
    events = [
        '{"type":"response.output_text.delta","delta":"<wiki>","item_id":"msg_1","output_index":0,"content_index":0,"logprobs":[],"sequence_number":0}',
        '{"type":"response.output_text.delta","delta":"unread","item_id":"msg_1","output_index":0,"content_index":0,"logprobs":[],"sequence_number":1}',
    ]

    assert await _collect_sse(events, close_after_first=True) == ["<wiki>"]


@pytest.mark.asyncio
@pytest.mark.parametrize("events", [[], [
    '{"type":"response.output_text.delta","delta":"<wiki>","item_id":"msg_1","output_index":0,"content_index":0,"logprobs":[],"sequence_number":0}',
]])
async def test_responses_stream_rejects_eof_before_completion(events: list[str]):
    with pytest.raises(RuntimeError, match="before response.completed"):
        await _collect_sse(events)
