"""Provider registry and factory.

Which vendor serves a model is declared in ``config/models.yaml``, so adding a
provider means adding a YAML entry, not editing pipeline code.
"""

from __future__ import annotations

from typing import Any

from config.settings import ModelProfile, Settings
from voc.providers.anthropic_provider import AnthropicProvider, resolve_effort
from voc.providers.base import (
    CompletionResult,
    LLMProvider,
    ProviderError,
    normalise_usage,
)
from voc.providers.openai_compatible import (
    OpenAICompatibleProvider,
    schema_instruction,
    strip_code_fences,
)

__all__ = [
    "AnthropicProvider",
    "CompletionResult",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
    "get_provider",
    "normalise_usage",
    "resolve_effort",
    "schema_instruction",
    "strip_code_fences",
]


def get_provider(
    profile: ModelProfile, settings: Settings, client: Any | None = None
) -> LLMProvider:
    """Construct the provider a model profile declares.

    ``client`` is an injection point for tests, which pass a mock rather than
    reaching the network.
    """
    if profile.provider == "anthropic":
        return AnthropicProvider(settings, client=client)

    if not profile.base_url:
        raise ProviderError(
            f"Model {profile.key!r} uses provider {profile.provider!r} but declares no "
            "base_url. OpenAI-compatible providers need one in config/models.yaml."
        )

    return OpenAICompatibleProvider(
        settings,
        provider_name=profile.provider,
        base_url=profile.base_url,
        client=client,
    )
