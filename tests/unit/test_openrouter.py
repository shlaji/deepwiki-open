from api.clients.openrouter import OpenRouterClient
from api.chat._stream import OpenRouterChatStreamer


def test_openrouter_client_uses_constructor_configuration():
    client = OpenRouterClient(
        api_key="test-api-key",
        base_url="https://openrouter.example.test/v1",
    )

    assert client.sync_client == {
        "api_key": "test-api-key",
        "base_url": "https://openrouter.example.test/v1",
    }


def test_openrouter_streamer_does_not_warn_for_configured_api_key(monkeypatch, caplog):
    import api.clients
    import api.chat._stream as stream

    monkeypatch.setattr(stream, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(api.clients, "OpenRouterClient", lambda **kwargs: object())

    OpenRouterChatStreamer(
        model="test-model",
        model_config={"temperature": 0.2},
        initialize_kwargs={"api_key": "test-api-key"},
    )

    assert "OPENROUTER_API_KEY not configured" not in caplog.text
