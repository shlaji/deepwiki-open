import pytest

import api.services.research as research
from api.schemas import ChatCompletionRequest, ChatMessage


class _FakeMemory:
    def add_dialog_turn(self, user_query, assistant_response):
        del user_query, assistant_response

    def __call__(self):
        return {}


class _FakeRag:
    memory = _FakeMemory()

    async def acall(self, query, language):
        del query, language
        return []


class _FakeStreamer:
    provider = "openai"
    error_hint = None

    async def respond_stream(self, prompt):
        del prompt
        yield "answer"


@pytest.mark.asyncio
async def test_research_chat_passes_initialize_kwargs_to_streamer(monkeypatch):
    captured = {}

    async def fake_prepare_repo_index(request):
        del request
        return _FakeRag()

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeStreamer()

    monkeypatch.setattr(research, "prepare_repo_index", fake_prepare_repo_index)
    monkeypatch.setattr(research, "repo_index_exist", lambda repo: True)
    monkeypatch.setattr(
        research,
        "get_model_config",
        lambda provider, model: {
            "model_kwargs": {"model": model, "temperature": 0.2},
            "initialize_kwargs": {"base_url": "https://openai.example.test/v1"},
        },
    )
    monkeypatch.setattr(research.ChatStreamer, "create", fake_create)

    request = ChatCompletionRequest(
        repo_url="https://github.com/example/repository",
        type="github",
        token=None,
        provider="openai",
        model="test-model",
        language="en",
        messages=[ChatMessage(role="user", content="What does this do?")],
    )

    response = await research.research_chat(request)
    assert [chunk async for chunk in response] == ["answer"]
    assert captured["initialize_kwargs"] == {
        "base_url": "https://openai.example.test/v1"
    }
