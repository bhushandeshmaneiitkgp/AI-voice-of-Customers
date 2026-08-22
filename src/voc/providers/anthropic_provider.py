"""
Anthropic provider.

Kept alongside the OpenRouter adapter so ``EVALUATION.md`` can compare an open
model against a frontier one on the same reviews, with the same prompt and the
same validators. Without a second provider that comparison is not possible, and
the comparison is the point of the configurable-model architecture.

This is also the only provider with a native Batch API (50% cost reduction),
which is why the batch path lives here rather than in the shared orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import ModelProfile, Settings
from voc.providers.base import (
    CompletionResult,
    ProviderError,
    classify_error,
    normalise_usage,
)

logger = logging.getLogger(__name__)


def resolve_effort(profile: ModelProfile, override: str | None = None) -> str | None:
    """Effort level for a request, or None when the model does not support it."""
    if not profile.supports_effort:
        return None
    return override or profile.default_effort


class AnthropicProvider:
    """First-party Anthropic SDK access."""

    name = "anthropic"
    supports_batch = True

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise ProviderError(
                "The 'anthropic' package is required for the anthropic provider. "
                "Install it with: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=settings.require_api_key("anthropic"))

    @property
    def client(self) -> Any:
        return self._client

    def build_params(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_message: str,
        response_schema: dict[str, Any],
        effort: str | None = None,
        max_tokens: int = 8000,
    ) -> dict[str, Any]:
        """Assemble one Messages API request, adapted to the selected model.

        The system prompt carries ``cache_control`` because it is byte-identical
        across every request in a run (~4,400 tokens of taxonomy). Cache hits
        cut input cost roughly tenfold on the repeated prefix, which is most of
        the spend.
        """
        output_config: dict[str, Any] = {"format": response_schema}

        params: dict[str, Any] = {
            "model": profile.model_id,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_message}],
        }

        if profile.thinking_style == "adaptive":
            # Current models: thinking depth is controlled by effort, and
            # budget_tokens is rejected outright.
            params["thinking"] = {"type": "adaptive"}
            chosen = resolve_effort(profile, effort)
            if chosen:
                output_config["effort"] = chosen
        else:
            # Older models (e.g. Haiku 4.5) predate adaptive thinking and return
            # a 400 if given `effort`. Classification does not need thinking, so
            # it is omitted rather than configured with a token budget.
            logger.debug("Model %s predates adaptive thinking; running without it.", profile.model_id)

        params["output_config"] = output_config
        return params

    def complete(
        self,
        profile: ModelProfile,
        system_prompt: str,
        user_message: str,
        response_schema: dict[str, Any],
        effort: str | None = None,
        max_tokens: int = 8000,
    ) -> CompletionResult:
        params = self.build_params(
            profile, system_prompt, user_message, response_schema, effort, max_tokens
        )
        try:
            message = self._client.messages.create(**params)
        except Exception as exc:  # noqa: BLE001 - surfaced as ProviderError
            raise classify_error(exc) from exc

        return CompletionResult(
            text=self.extract_text(message),
            usage=normalise_usage(getattr(message, "usage", None)),
        )

    @staticmethod
    def extract_text(message: Any) -> str:
        """Pull the JSON payload out of a response.

        With ``output_config.format`` set the model returns one text block of
        valid JSON, but responses may also carry thinking blocks, so the block
        type must be checked rather than assuming ``content[0]``.
        """
        for block in message.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise ProviderError(
            "No text block in response. Block types: "
            f"{[getattr(b, 'type', '?') for b in message.content]}"
        )
