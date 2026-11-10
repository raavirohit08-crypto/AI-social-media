"""Thin wrapper around a chat-completions API.

Two backends, chosen by the environment:
  * default            -> Hugging Face Inference Providers (HF_TOKEN)
  * LLM_BASE_URL set   -> any OpenAI-compatible server: Groq, Ollama, LM Studio, ...
"""
import os
import time

from huggingface_hub import InferenceClient


class LLMError(RuntimeError):
    pass


# Errors that retrying cannot fix: fail fast with a clear message.
_FRIENDLY = {
    401: "The API key/token was rejected. Check HF_TOKEN in your .env file.",
    402: "The provider says you are out of credits (402 Payment Required). Add credits, "
         "wait for the monthly reset, or switch to Groq/Ollama via LLM_BASE_URL in .env.",
    403: "Access to this model was denied. Check your token permissions or choose another MODEL_NAME.",
    404: "The model was not found. Check MODEL_NAME in your .env file.",
}


def _status_code(error: Exception) -> int | None:
    code = getattr(error, "status_code", None)
    if code is None:
        code = getattr(getattr(error, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def _is_reasoning_model(model_name: str) -> bool:
    """gpt-oss models 'think' first; that thinking counts against max_tokens."""
    return "gpt-oss" in model_name.lower()


# Extra tokens so short answers are not eaten by the model's hidden reasoning.
REASONING_HEADROOM = 1000


class LLM:
    def __init__(self, token: str, model_name: str, provider: str = "auto",
                 base_url: str | None = None):
        if not token:
            raise LLMError("API key is missing. Add HF_TOKEN to your .env file.")
        self.model_name = model_name
        base_url = base_url or os.getenv("LLM_BASE_URL")
        self.base_url = base_url
        if base_url:
            from openai import OpenAI      # pip install openai
            self.client = OpenAI(base_url=base_url, api_key=token)
        else:
            self.client = InferenceClient(provider=provider, token=token)

    # ------------------------------------------------------------------ helpers
    def _settings(self, max_tokens: int):
        reasoning = _is_reasoning_model(self.model_name)
        extra = {"reasoning_effort": "low"} if (reasoning and self.base_url) else None
        budget = max_tokens + (REASONING_HEADROOM if reasoning else 0)
        return extra, budget

    @staticmethod
    def _kwargs(model, messages, budget, temperature, extra, stream=False):
        kwargs = dict(model=model, messages=messages, max_tokens=budget, temperature=temperature)
        if extra:
            kwargs["extra_body"] = extra
        if stream:
            kwargs["stream"] = True
        return kwargs

    def chat(self, messages: list[dict], max_tokens: int = 600,
             temperature: float = 0.2, attempts: int = 3) -> str:
        """Send chat messages, retry temporary failures, return the text."""
        last_error = None
        extra, budget = self._settings(max_tokens)

        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    **self._kwargs(self.model_name, messages, budget, temperature, extra))
                text = response.choices[0].message.content
                if not text or not text.strip():
                    raise ValueError("The model returned an empty response.")
                return text.strip()
            except Exception as error:
                last_error = error
                code = _status_code(error)
                if code == 400 and extra:          # server rejected reasoning_effort
                    extra = None                    # retry once without it
                    continue
                if code in _FRIENDLY:                      # no point retrying
                    raise LLMError(_FRIENDLY[code]) from error
                if attempt < attempts:
                    time.sleep(2 * attempt)
        raise LLMError(f"The AI service is unavailable: {last_error}")

    def chat_stream(self, messages: list[dict], max_tokens: int = 600,
                    temperature: float = 0.2, attempts: int = 3):
        """Yield the reply piece by piece (streaming).

        A failure BEFORE the first piece arrives is retried like chat(). A failure after
        text has already been shown cannot be retried, so it raises LLMError.
        """
        last_error = None
        extra, budget = self._settings(max_tokens)

        for attempt in range(1, attempts + 1):
            started = False
            try:
                stream = self.client.chat.completions.create(
                    **self._kwargs(self.model_name, messages, budget, temperature, extra, stream=True))
                for chunk in stream:
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    piece = getattr(getattr(choices[0], "delta", None), "content", None)
                    if piece:                       # reasoning-only chunks have no content
                        started = True
                        yield piece
                return
            except Exception as error:
                last_error = error
                code = _status_code(error)
                if started:
                    raise LLMError(f"The reply was interrupted: {error}") from error
                if code == 400 and extra:
                    extra = None
                    continue
                if code in _FRIENDLY:
                    raise LLMError(_FRIENDLY[code]) from error
                if attempt < attempts:
                    time.sleep(2 * attempt)
        raise LLMError(f"The AI service is unavailable: {last_error}")

    def ask(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        return self.chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            **kwargs,
        )

    def ask_streaming(self, system_prompt: str, user_prompt: str, on_token=None, **kwargs) -> str:
        """Like ask(), but calls on_token(piece) as text arrives and returns the full text.

        If the stream produces no text (some providers/models do not stream), fall back
        to a normal request so the caller still gets an answer.
        """
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        parts = []
        for piece in self.chat_stream(messages, **kwargs):
            parts.append(piece)
            if on_token:
                on_token(piece)
        text = "".join(parts).strip()
        if not text:
            text = self.chat(messages, **kwargs)
            if on_token:
                on_token(text)
        return text
