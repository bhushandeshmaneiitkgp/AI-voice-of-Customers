"""
OpenAI-compatible provider — covers every open-source option.

OpenRouter, Groq, Together, Fireworks, DeepSeek, local Ollama and vLLM all
expose the same ``/chat/completions`` wire format, so one adapter reaches all of
them. Switching between them is a ``base_url`` change in ``config/models.yaml``,
not a new integration.

**Structured output varies, and the pipeline degrades gracefully.** Providers
support three levels, declared per model in the registry:

``json_schema``  the schema is enforced by the provider (best)
``json_object``  valid JSON is guaranteed, the schema is only a suggestion
``none``         nothing is guaranteed; the schema is described in the prompt

At ``json_object`` and ``none`` the schema is also appended to the user message,
because a model that cannot be constrained can still be instructed. Whatever
comes back is validated identically either way — the taxonomy validator and the
grounding check do not trust any provider's promises.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config.settings import ModelProfile, Settings
from voc.providers.base import CompletionResult, ProviderError, normalise_usage

logger = logging.getLogger(__name__)

#: Sent by OpenRouter to attribute traffic. Harmless elsewhere.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/bhushandeshmaneiitkgp/AI-voice-of-Customers",
    "X-Title": "Quick-Commerce VoC Copilot",
}


def schema_instruction(response_schema: dict[str, Any]) -> str:
    """Describe the required shape in the prompt, for providers that cannot enforce it."""
    schema = response_schema.get("schema", response_schema)
    return (
        "\n\nReturn ONLY a JSON object matching this schema exactly. "
        "No markdown fences, no commentary before or after.\n\n"
        f"{json.dumps(schema, indent=2)}"
    )


def strip_code_fences(text: str) -> str:
    """Remove markdown fences some models wrap JSON in despite instructions.

    Cheap to do and it converts a whole class of "unparseable response" into a
    successful one, which matters more on open models than on frontier ones.
    """
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


class OpenAICompatibleProvider:
    """Any provider speaking the OpenAI chat-completions API."""

    supports_batch = False  # none of these offer a discounted async batch endpoint

    def __init__(
        self,
        settings: Settings,
        provider_name: str,
        base_url: str,
        client: Any | None = None,
    ) -> None:
        self.name = provider_name
        self.base_url = base_url

        if client is not None:
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise ProviderError(
                f"The 'openai' package is required for the {provider_name} provider. "
                "Install it with: pip install openai"
            ) from exc

        self._client = OpenAI(
            api_key=settings.require_api_key(provider_name),
            base_url=base_url,
            default_headers=_ATTRIBUTION_HEADERS if provider_name == "openrouter" else None,
        )

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
        """Assemble a chat-completions request for this model's capability level."""
        mode = profile.structured_output

        if mode == "json_schema":
            response_format: dict[str, Any] | None = {
                "type": "json_schema",
                "json_schema": {
                    "name": "review_enrichment",
                    "strict": True,
                    "schema": response_schema.get("schema", response_schema),
                },
            }
            content = user_message
        elif mode == "json_object":
            response_format = {"type": "json_object"}
            content = user_message + schema_instruction(response_schema)
        else:
            response_format = None
            content = user_message + schema_instruction(response_schema)

        params: dict[str, Any] = {
            "model": profile.model_id,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            # Classification wants the most likely label, not a creative one.
            "temperature": 0.0,
        }
        if response_format is not None:
            params["response_format"] = response_format
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
            completion = self._client.chat.completions.create(**params)
        except Exception as exc:  # noqa: BLE001 - surfaced as ProviderError
            raise ProviderError(str(exc)) from exc

        if not completion.choices:
            raise ProviderError("Provider returned no choices")

        text = completion.choices[0].message.content
        if not text:
            raise ProviderError("Provider returned an empty message")

        return CompletionResult(
            text=strip_code_fences(text),
            usage=normalise_usage(getattr(completion, "usage", None)),
        )
