"""
Provider abstraction for LLM access.

The enrichment pipeline does not know or care which vendor answers a request.
It hands a provider a system prompt, a user message, and a JSON schema, and
gets back text plus token usage. Everything that makes enrichment trustworthy —
schema validation, taxonomy checks, grounding verification, id reconciliation —
sits above this line and is identical for every provider.

That boundary is what makes a cross-provider benchmark meaningful: the *only*
thing that varies between two runs is the model.

Two implementations:

* ``AnthropicProvider``       — first-party SDK, native Batch API.
* ``OpenAICompatibleProvider`` — the OpenAI SDK pointed at any compatible
  ``base_url``. One adapter covers OpenRouter, Groq, Together, Fireworks,
  DeepSeek, local Ollama and vLLM, because they all speak the same wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from config.settings import ModelProfile


#: HTTP statuses that will never succeed on retry, because the problem is the
#: account rather than the request: bad key, exhausted credit, forbidden model.
#: Retrying these wastes time and, on a rate-limited account, makes things worse.
NON_RETRYABLE_STATUSES: frozenset[int] = frozenset({400, 401, 402, 403, 404})


class ProviderError(RuntimeError):
    """A provider could not fulfil a request. Recorded, never silently swallowed.

    ``retryable`` distinguishes a transient failure (timeout, 5xx, rate limit)
    from a systemic one (no credit, bad key). The orchestrator retries the
    first and aborts on the second -- without that distinction a single
    account-level problem fans out into one doomed retry per review.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        if retryable is None:
            retryable = status_code not in NON_RETRYABLE_STATUSES if status_code else True
        self.retryable = retryable


def classify_error(exc: Exception) -> ProviderError:
    """Wrap a provider SDK exception, preserving whether a retry could help.

    SDKs expose the status inconsistently, so it is read from an attribute
    where available and otherwise scraped from the message. Defaulting to
    retryable=True is the safe direction: a needless retry costs one request,
    whereas wrongly aborting loses the run.
    """
    import re

    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if not isinstance(status, int):
        match = re.search(r"\b(4\d{2}|5\d{2})\b", str(exc))
        status = int(match.group(1)) if match else None

    return ProviderError(str(exc), status_code=status)


@dataclass(frozen=True)
class CompletionResult:
    """One model response, normalised across providers."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def cache_read_tokens(self) -> int:
        return self.usage.get("cache_read_input_tokens", 0)


@runtime_checkable
class LLMProvider(Protocol):
    """What the enrichment pipeline needs from any vendor."""

    #: Registry key, e.g. ``"anthropic"`` or ``"openrouter"``.
    name: str

    #: True when the provider offers a native async batch endpoint at reduced
    #: cost. False means the pipeline falls back to concurrent live requests.
    supports_batch: bool

    def complete(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_message: str,
        response_schema: dict[str, Any],
        effort: str | None = None,
        max_tokens: int = 8000,
    ) -> CompletionResult:
        """Send one request and return the model's text plus token usage."""
        ...


def normalise_usage(raw: Any) -> dict[str, int]:
    """Pull token counts out of whichever usage object a provider returned.

    Providers name these fields differently and some omit them entirely, so
    every key is looked up defensively. Missing usage is not an error — it just
    means the run report cannot show a token count for that call.
    """
    if raw is None:
        return {}

    def read(*names: str) -> int:
        for name in names:
            value = getattr(raw, name, None)
            if value is None and isinstance(raw, dict):
                value = raw.get(name)
            if isinstance(value, int):
                return value
        return 0

    usage = {
        "input_tokens": read("input_tokens", "prompt_tokens"),
        "output_tokens": read("output_tokens", "completion_tokens"),
        "cache_creation_input_tokens": read("cache_creation_input_tokens"),
        "cache_read_input_tokens": read("cache_read_input_tokens", "cached_tokens"),
    }
    return {key: value for key, value in usage.items() if value}
