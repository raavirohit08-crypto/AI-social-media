
"""Thin wrapper around a chat-completions API.

Backends:
  - Default: Hugging Face Inference Providers (HF_TOKEN)
  - LLM_BASE_URL set: OpenAI-compatible servers such as Groq,
    Ollama and LM Studio.
"""

import os
import time

from huggingface_hub import InferenceClient


class LLMError(RuntimeError):
    """Raised when the AI service fails."""
    pass


_FRIENDLY = {
    401: "The API key/token was rejected. Check HF_TOKEN in your .env file.",
    402: (
        "The provider says you are out of credits. Add credits, "
        "wait for the monthly reset, or switch to another provider."
    ),
    403: (
        "Access to this model was denied. Check your token permissions "
        "or choose another MODEL_NAME."
    ),
    404: "The model was not found. Check MODEL_NAME in your .env file.",
}


def _status_code(error: Exception) -> int | None:
    code = getattr(error, "status_code", None)

    if code is None:
        code = getattr(
            getattr(error, "response", None),
            "status_code",
            None
        )

    return code if isinstance(code, int) else None


def _is_reasoning_model(model_name: str) -> bool:
    return "gpt-oss" in model_name.lower()


REASONING_HEADROOM = 1000


class LLM:

    def __init__(
        self,
        token: str,
        model_name: str,
        provider: str = "auto",
        base_url: str | None = None
    ):
        if not token:
            raise LLMError(
                "API key is missing. Add HF_TOKEN to your .env file."
            )

        self.model_name = model_name
        self.base_url = base_url or os.getenv("LLM_BASE_URL")

        if self.base_url:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=token
            )
        else:
            self.client = InferenceClient(
                provider=provider,
                token=token
            )

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 600,
        temperature: float = 0.2,
        attempts: int = 3
    ) -> str:
        """Send messages and return the complete response."""

        last_error = None
        reasoning = _is_reasoning_model(self.model_name)

        extra = (
            {"reasoning_effort": "low"}
            if reasoning and self.base_url
            else None
        )

        budget = max_tokens + (
            REASONING_HEADROOM if reasoning else 0
        )

        for attempt in range(1, attempts + 1):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": budget,
                    "temperature": temperature,
                }

                if extra:
                    kwargs["extra_body"] = extra

                response = self.client.chat.completions.create(
                    **kwargs
                )

                result = response.choices[0].message.content

                if not result or not result.strip():
                    raise ValueError(
                        "The model returned an empty response."
                    )

                return result.strip()

            except Exception as error:
                last_error = error
                code = _status_code(error)

                if code == 400 and extra:
                    extra = None
                    continue

                if code in _FRIENDLY:
                    raise LLMError(_FRIENDLY[code]) from error

                if attempt < attempts:
                    time.sleep(2 * attempt)

        raise LLMError(
            f"The AI service is unavailable: {last_error}"
        )

    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Send a system prompt and user prompt."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.chat(messages, **kwargs)

    def stream(
        self,
        messages: list[dict],
        max_tokens: int = 600,
        temperature: float = 0.2
    ):
        """Yield response text as the model generates it."""

        reasoning = _is_reasoning_model(self.model_name)

        budget = max_tokens + (
            REASONING_HEADROOM if reasoning else 0
        )

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": budget,
            "temperature": temperature,
            "stream": True,
        }

        if reasoning and self.base_url:
            kwargs["extra_body"] = {"reasoning_effort": "low"}

        try:
            response = self.client.chat.completions.create(
                **kwargs
            )

            for chunk in response:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    yield content

        except Exception as error:
            code = _status_code(error)

            if code in _FRIENDLY:
                raise LLMError(_FRIENDLY[code]) from error

            raise LLMError(
                f"Streaming failed: {error}"
            ) from error

    def ask_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.2
    ):
        """Stream a response using system and user prompts."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        yield from self.stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
