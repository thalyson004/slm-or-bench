"""Minimal Ollama and OpenRouter clients with auditable retry behavior."""

from __future__ import annotations

import time
from typing import Any, Protocol

import httpx

from .errors import ModelRequestError
from .records import ModelResponse, Provider


class ModelClient(Protocol):
    model: str
    provider: Provider

    def generate(self, system: str, prompt: str) -> ModelResponse: ...

    def close(self) -> None: ...


class _HttpClient:
    RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        model: str,
        provider: Provider,
        client: httpx.Client,
        timeout_seconds: float,
        retry_timeouts: bool,
        max_retries: int,
        sleep=time.sleep,
        clock=time.perf_counter,
    ) -> None:
        self.model = model
        self.provider = provider
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._retry_timeouts = retry_timeouts
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[httpx.Response, int, float]:
        started = self._clock()
        for attempt in range(1, self._max_retries + 2):
            try:
                response = self._client.post(path, json=payload)
            except httpx.ConnectTimeout as error:
                if attempt <= self._max_retries:
                    self._sleep(min(2 ** attempt, 60))
                    continue
                raise ModelRequestError(
                    code="transport_error",
                    message=f"{self.provider} connection timed out: {error}",
                    provider=self.provider,
                    model=self.model,
                    attempts=attempt,
                    latency_seconds=self._clock() - started,
                    details={"exception_type": type(error).__name__},
                ) from None
            except httpx.TimeoutException as error:
                if self._retry_timeouts and attempt <= self._max_retries:
                    self._sleep(min(2 ** attempt, 60))
                    continue
                raise ModelRequestError(
                    code="request_timeout",
                    message=(
                        f"{self.provider} request exceeded the configured "
                        f"{self._timeout_seconds:g}-second timeout."
                    ),
                    provider=self.provider,
                    model=self.model,
                    attempts=attempt,
                    latency_seconds=self._clock() - started,
                    details={"exception_type": type(error).__name__},
                ) from None
            except httpx.RequestError as error:
                if attempt > self._max_retries:
                    raise ModelRequestError(
                        code="transport_error",
                        message=f"{self.provider} transport failure: {error}",
                        provider=self.provider,
                        model=self.model,
                        attempts=attempt,
                        latency_seconds=self._clock() - started,
                        details={"exception_type": type(error).__name__},
                    ) from None
                self._sleep(min(2 ** attempt, 60))
                continue
            if response.is_success:
                return response, attempt, self._clock() - started
            if response.status_code not in self.RETRYABLE or attempt > self._max_retries:
                code = (
                    "rate_limit_error" if response.status_code == 429
                    else "authentication_error" if response.status_code in {401, 403}
                    else "provider_server_error" if response.status_code >= 500
                    else "provider_http_error"
                )
                raise ModelRequestError(
                    code=code,
                    message=(
                        f"{self.provider} rejected the request "
                        f"(HTTP {response.status_code})."
                    ),
                    provider=self.provider,
                    model=self.model,
                    attempts=attempt,
                    latency_seconds=self._clock() - started,
                    status_code=response.status_code,
                    details={"response_excerpt": response.text[-2000:]},
                )
            retry_after = response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after is not None else 2 ** attempt
            except ValueError:
                delay = 2 ** attempt
            if not 0 <= delay <= 60:
                raise ModelRequestError(
                    code="invalid_retry_delay",
                    message=f"{self.provider} returned an invalid retry delay: {delay}.",
                    provider=self.provider,
                    model=self.model,
                    attempts=attempt,
                    latency_seconds=self._clock() - started,
                    status_code=response.status_code,
                )
            self._sleep(delay)
        raise ModelRequestError(
            code="transport_error",
            message=f"{self.provider} retries exhausted.",
            provider=self.provider,
            model=self.model,
            attempts=self._max_retries + 1,
            latency_seconds=self._clock() - started,
        )

    def close(self) -> None:
        self._client.close()


class OllamaClient(_HttpClient):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 600,
        retry_timeouts: bool = False,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        seed: int = 0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        clock=time.perf_counter,
    ) -> None:
        http_client = client or httpx.Client(
            base_url=base_url.rstrip("/") + "/", timeout=timeout_seconds
        )
        super().__init__(
            model=model,
            provider="ollama",
            client=http_client,
            timeout_seconds=timeout_seconds,
            retry_timeouts=retry_timeouts,
            max_retries=max_retries,
            sleep=sleep,
            clock=clock,
        )
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed

    def generate(self, system: str, prompt: str) -> ModelResponse:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": self._temperature,
                "seed": self._seed,
                "num_predict": self._max_tokens,
            },
        }
        response, attempts, latency = self._post("api/chat", payload)
        try:
            data = response.json()
        except ValueError:
            raise ModelRequestError(
                code="malformed_response",
                message="Ollama returned a non-JSON response.",
                provider="ollama",
                model=self.model,
                attempts=attempts,
                latency_seconds=latency,
                details={"response_excerpt": response.text[-2000:]},
            ) from None
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError):
            raise ModelRequestError(
                code="malformed_response",
                message="Ollama returned a response without message.content.",
                provider="ollama",
                model=self.model,
                attempts=attempts,
                latency_seconds=latency,
                details={"raw_response": data},
            ) from None
        if not isinstance(content, str) or not content.strip():
            raise ModelRequestError(
                code="empty_response",
                message="Ollama returned empty message content.",
                provider="ollama",
                model=self.model,
                attempts=attempts,
                latency_seconds=latency,
                details={"raw_response": data},
            )
        return ModelResponse(
            content=content,
            model=self.model,
            provider="ollama",
            latency_seconds=latency,
            request_attempts=attempts,
            prompt_tokens=_optional_int(data.get("prompt_eval_count")),
            completion_tokens=_optional_int(data.get("eval_count")),
            finish_reason=_optional_string(data.get("done_reason")),
            provider_metadata={
                "total_duration_ns": data.get("total_duration"),
                "load_duration_ns": data.get("load_duration"),
                "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
                "eval_duration_ns": data.get("eval_duration"),
                "thinking": (data.get("message") or {}).get("thinking"),
            },
            raw_response=data,
        )


class OpenRouterClient(_HttpClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 600,
        retry_timeouts: bool = False,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        seed: int = 0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        clock=time.perf_counter,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter.")
        http_client = client or httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/thalyson004/code-or-auto-solver-small-models-models",
                "X-OpenRouter-Title": "OR Small Models Benchmark",
                "X-OpenRouter-Metadata": "enabled",
            },
        )
        super().__init__(
            model=model,
            provider="openrouter",
            client=http_client,
            timeout_seconds=timeout_seconds,
            retry_timeouts=retry_timeouts,
            max_retries=max_retries,
            sleep=sleep,
            clock=clock,
        )
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed

    def generate(self, system: str, prompt: str) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "seed": self._seed,
            "max_completion_tokens": self._max_tokens,
            "provider": {"order": ["openai"], "allow_fallbacks": False},
        }
        response, attempts, latency = self._post("chat/completions", payload)
        try:
            data = response.json()
        except ValueError:
            raise ModelRequestError(
                code="malformed_response",
                message="OpenRouter returned a non-JSON response.",
                provider="openrouter",
                model=self.model,
                attempts=attempts,
                latency_seconds=latency,
                details={"response_excerpt": response.text[-2000:]},
            ) from None
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelRequestError(
                code="malformed_response",
                message="OpenRouter returned a response without choices[0].message.content.",
                provider="openrouter",
                model=self.model,
                attempts=attempts,
                latency_seconds=latency,
                details={"raw_response": data},
            ) from None
        if not isinstance(content, str) or not content.strip():
            raise ModelRequestError(
                code="empty_response",
                message="OpenRouter returned empty message content.",
                provider="openrouter",
                model=self.model,
                attempts=attempts,
                latency_seconds=latency,
                details={"raw_response": data},
            )
        usage = data.get("usage") or {}
        return ModelResponse(
            content=content,
            model=str(data.get("model") or self.model),
            provider="openrouter",
            latency_seconds=latency,
            request_attempts=attempts,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            finish_reason=_optional_string(choice.get("finish_reason")),
            provider_metadata={
                "response_id": data.get("id"),
                "system_fingerprint": data.get("system_fingerprint"),
                "openrouter_metadata": data.get("openrouter_metadata"),
            },
            raw_response=data,
        )


def ollama_inventory(
    base_url: str = "http://localhost:11434", timeout_seconds: float = 30
) -> dict[str, dict[str, Any]]:
    with httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout_seconds) as client:
        response = client.get("api/tags")
        response.raise_for_status()
        payload = response.json()
    return {
        str(item["name"]): {
            "digest": str(item.get("digest", "")),
            "size_bytes": item.get("size"),
            "modified_at": item.get("modified_at"),
            "details": item.get("details") or {},
        }
        for item in payload.get("models", [])
        if isinstance(item, dict) and item.get("name")
    }


def openrouter_inventory(
    base_url: str = "https://openrouter.ai/api/v1", timeout_seconds: float = 30
) -> set[str]:
    with httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout_seconds) as client:
        response = client.get("models")
        response.raise_for_status()
        payload = response.json()
    return {
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None
