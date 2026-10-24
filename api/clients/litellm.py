import os
from typing import Any, Dict, Optional

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    CompletionUsage,
    GeneratorOutput,
    ModelType,
)
from openai import AsyncOpenAI, OpenAI

from api.logger import get_logger

log = get_logger(__name__)


class LiteLLMClient(ModelClient):
    """
    LiteLLM OpenAI-compatible client using Chat Completions API.

    Some LiteLLM-backed models (for example minimax/kimi) do not support the
    OpenAI Responses API that adalflow's OpenAIClient now requires. This client
    keeps using the Chat Completions API surface so those models still work.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        input_type: str = "text",
        base_url: Optional[str] = None,
        env_api_key_name: str = "LITELLM_API_KEY",
    ):
        resolved_base_url = base_url or os.getenv(
            "LITELLM_BASE_URL", "http://localhost:4000"
        )
        if not resolved_base_url.endswith("/v1"):
            resolved_base_url = f"{resolved_base_url.rstrip('/')}/v1"
        super().__init__()
        self._api_key = api_key
        self.base_url = resolved_base_url
        self._env_api_key_name = env_api_key_name
        self._input_type = input_type
        self.sync_client = self.init_sync_client()
        self.async_client = None

    def init_sync_client(self):
        """
        Initialize synchronous LiteLLM OpenAI-compatible client.
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name, "dummy")
        return OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

    def init_async_client(self):
        """
        Initialize asynchronous LiteLLM OpenAI-compatible client.
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name, "dummy")
        return AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        final_model_kwargs = model_kwargs.copy()

        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM or model_type == ModelType.LLM_REASONING:
            if isinstance(input, str):
                final_model_kwargs["messages"] = [
                    {"role": "user", "content": input}
                ]
            else:
                final_model_kwargs["messages"] = input
        else:
            raise ValueError(f"model_type {model_type} is not supported")

        return final_model_kwargs

    def parse_chat_completion(
        self,
        completion: Any,
    ) -> GeneratorOutput:
        try:
            if hasattr(completion, "choices") and len(completion.choices) > 0:
                message = completion.choices[0].message
                content = getattr(message, "content", None) or ""
                usage = getattr(completion, "usage", None)
                return GeneratorOutput(
                    data=content,
                    usage=CompletionUsage(
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        total_tokens=getattr(usage, "total_tokens", 0),
                    ),
                    raw_response=str(completion),
                )
            return GeneratorOutput(data=str(completion))
        except Exception as e:
            log.error("Error parsing LiteLLM chat completion: %s", e)
            return GeneratorOutput(data=str(completion))

    def track_completion_usage(
        self,
        completion: Any,
    ) -> CompletionUsage:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
        return CompletionUsage(
            completion_tokens=getattr(usage, "completion_tokens", 0),
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        if model_type == ModelType.LLM or model_type == ModelType.LLM_REASONING:
            if api_kwargs.get("stream", False):
                return self.sync_client.chat.completions.create(**api_kwargs)
            return self.parse_chat_completion(
                self.sync_client.chat.completions.create(**api_kwargs)
            )
        raise ValueError(f"model_type {model_type} is not supported")

    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        if self.async_client is None:
            self.async_client = self.init_async_client()

        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings.create(**api_kwargs)
        if model_type == ModelType.LLM or model_type == ModelType.LLM_REASONING:
            if api_kwargs.get("stream", False):
                return await self.async_client.chat.completions.create(**api_kwargs)
            return self.parse_chat_completion(
                await self.async_client.chat.completions.create(**api_kwargs)
            )
        raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_key": self._api_key,
            "base_url": self.base_url,
            "input_type": self._input_type,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        if "sync_client" in state:
            del state["sync_client"]
        if "async_client" in state:
            del state["async_client"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sync_client = self.init_sync_client()
        self.async_client = None
