"""Tests for the provider abstraction.

No network. Every provider is constructed with an injected mock client, so the
whole suite runs offline and without any API key.

What matters here is that the *only* thing varying between two benchmark runs
is the model. If a provider quietly changed the prompt, the schema, or the
validation, a cross-provider comparison would measure the adapter rather than
the model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config.settings import Settings, load_model_registry
from voc.enrichment_schemas import build_response_schema
from voc.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    ProviderError,
    get_provider,
    normalise_usage,
    resolve_effort,
)
from voc.providers.openai_compatible import schema_instruction, strip_code_fences
from voc.taxonomy import get_taxonomy


@pytest.fixture(scope="module")
def schema():
    return build_response_schema(get_taxonomy())


@pytest.fixture()
def settings():
    return Settings(_env_file=None, openrouter_api_key="test-key", anthropic_api_key="test-key")


def _openai_provider(client=None, name: str = "openrouter") -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        settings=MagicMock(),
        provider_name=name,
        base_url="https://example.test/v1",
        client=client or MagicMock(),
    )


def _completion(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=340),
    )


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_every_profile_declares_a_provider() -> None:
    for key, profile in load_model_registry().items():
        assert profile.provider, f"{key} has no provider"


def test_openai_compatible_profiles_declare_a_base_url() -> None:
    """Anthropic is the only provider with a built-in endpoint."""
    for key, profile in load_model_registry().items():
        if profile.provider != "anthropic":
            assert profile.base_url, f"{key} uses {profile.provider} but has no base_url"


def test_factory_returns_the_declared_provider(settings) -> None:
    registry = load_model_registry()
    assert isinstance(
        get_provider(registry["opus"], settings, client=MagicMock()), AnthropicProvider
    )
    assert isinstance(
        get_provider(registry["llama70b"], settings, client=MagicMock()),
        OpenAICompatibleProvider,
    )


def test_factory_rejects_openai_provider_without_base_url(settings) -> None:
    broken = load_model_registry()["llama70b"].model_copy(update={"base_url": None})
    with pytest.raises(ProviderError, match="base_url"):
        get_provider(broken, settings, client=MagicMock())


def test_only_anthropic_advertises_batch(settings) -> None:
    """Batch pricing must not be claimed for providers that lack the endpoint."""
    registry = load_model_registry()
    assert get_provider(registry["opus"], settings, client=MagicMock()).supports_batch is True
    assert get_provider(registry["llama70b"], settings, client=MagicMock()).supports_batch is False


# ---------------------------------------------------------------------------
# OpenAI-compatible request construction
# ---------------------------------------------------------------------------


def test_json_schema_mode_sends_the_schema(schema) -> None:
    profile = load_model_registry()["llama70b"]
    params = _openai_provider().build_params(profile, "sys", "user", schema)

    assert params["response_format"]["type"] == "json_schema"
    assert params["response_format"]["json_schema"]["strict"] is True
    # The schema is enforced by the API, so it need not bloat the prompt.
    assert params["messages"][1]["content"] == "user"


def test_json_object_mode_puts_the_schema_in_the_prompt(schema) -> None:
    """A model that cannot be constrained can still be instructed."""
    profile = load_model_registry()["deepseek"]
    params = _openai_provider().build_params(profile, "sys", "user", schema)

    assert params["response_format"] == {"type": "json_object"}
    assert "json" in params["messages"][1]["content"].lower()
    assert len(params["messages"][1]["content"]) > len("user")


def test_unconstrained_mode_sends_no_response_format(schema) -> None:
    profile = load_model_registry()["llama70b"].model_copy(
        update={"structured_output": "none"}
    )
    params = _openai_provider().build_params(profile, "sys", "user", schema)

    assert "response_format" not in params
    assert "schema" in params["messages"][1]["content"].lower()


def test_temperature_is_zero_for_classification(schema) -> None:
    """Classification wants the most likely label, not a creative one."""
    profile = load_model_registry()["llama70b"]
    params = _openai_provider().build_params(profile, "sys", "user", schema)
    assert params["temperature"] == 0.0


def test_system_prompt_goes_in_the_system_role(schema) -> None:
    profile = load_model_registry()["llama70b"]
    params = _openai_provider().build_params(profile, "SYSTEM TEXT", "user", schema)

    assert params["messages"][0] == {"role": "system", "content": "SYSTEM TEXT"}


def test_effort_is_never_sent_to_open_models(schema) -> None:
    """Open models here have no effort parameter; sending one would error."""
    profile = load_model_registry()["llama70b"]
    params = _openai_provider().build_params(profile, "sys", "user", schema, effort="high")

    assert "effort" not in json.dumps(params)
    assert resolve_effort(profile, "high") is None


# ---------------------------------------------------------------------------
# OpenAI-compatible response handling
# ---------------------------------------------------------------------------


def test_complete_returns_text_and_usage(schema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _completion('{"results": []}')
    profile = load_model_registry()["llama70b"]

    result = _openai_provider(client).complete(profile, "sys", "user", schema)

    assert result.text == '{"results": []}'
    assert result.usage["input_tokens"] == 120
    assert result.usage["output_tokens"] == 340


@pytest.mark.parametrize(
    "wrapped",
    [
        '```json\n{"results": []}\n```',
        '```\n{"results": []}\n```',
        '  {"results": []}  ',
    ],
)
def test_code_fences_are_stripped(wrapped: str) -> None:
    """Open models wrap JSON in markdown despite instructions not to.

    Stripping converts a whole class of 'unparseable response' into a success,
    which matters far more on open models than on frontier ones.
    """
    assert json.loads(strip_code_fences(wrapped)) == {"results": []}


def test_empty_response_raises_provider_error(schema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _completion("")
    profile = load_model_registry()["llama70b"]

    with pytest.raises(ProviderError, match="empty"):
        _openai_provider(client).complete(profile, "sys", "user", schema)


def test_no_choices_raises_provider_error(schema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[], usage=None)
    profile = load_model_registry()["llama70b"]

    with pytest.raises(ProviderError, match="no choices"):
        _openai_provider(client).complete(profile, "sys", "user", schema)


def test_transport_errors_become_provider_errors(schema) -> None:
    """The orchestrator catches ProviderError; a raw SDK exception would escape."""
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("connection reset")
    profile = load_model_registry()["llama70b"]

    with pytest.raises(ProviderError, match="connection reset"):
        _openai_provider(client).complete(profile, "sys", "user", schema)


def test_schema_instruction_contains_the_schema(schema) -> None:
    text = schema_instruction(schema)
    assert "product_area" in text
    assert "results" in text


# ---------------------------------------------------------------------------
# Usage normalisation
# ---------------------------------------------------------------------------


def test_usage_normalises_openai_field_names() -> None:
    usage = normalise_usage(SimpleNamespace(prompt_tokens=10, completion_tokens=20))
    assert usage == {"input_tokens": 10, "output_tokens": 20}


def test_usage_normalises_anthropic_field_names() -> None:
    usage = normalise_usage(
        SimpleNamespace(input_tokens=10, output_tokens=20, cache_read_input_tokens=5)
    )
    assert usage["input_tokens"] == 10
    assert usage["cache_read_input_tokens"] == 5


def test_usage_handles_missing_data() -> None:
    """Providers that omit usage are not an error; the report just cannot show it."""
    assert normalise_usage(None) == {}
    assert normalise_usage(SimpleNamespace()) == {}


# ---------------------------------------------------------------------------
# Cross-provider consistency -- what makes the benchmark valid
# ---------------------------------------------------------------------------


def test_both_providers_receive_the_identical_schema(schema) -> None:
    """The schema must not be altered per provider, or labels are not comparable."""
    registry = load_model_registry()

    anthropic_params = AnthropicProvider(MagicMock(), client=MagicMock()).build_params(
        registry["opus"], "sys", "user", schema
    )
    openai_params = _openai_provider().build_params(registry["llama70b"], "sys", "user", schema)

    anthropic_schema = anthropic_params["output_config"]["format"]["schema"]
    openai_schema = openai_params["response_format"]["json_schema"]["schema"]
    assert anthropic_schema == openai_schema


def test_both_providers_receive_the_identical_system_prompt(schema) -> None:
    registry = load_model_registry()
    prompt = "THE TAXONOMY PROMPT"

    anthropic_params = AnthropicProvider(MagicMock(), client=MagicMock()).build_params(
        registry["opus"], prompt, "user", schema
    )
    openai_params = _openai_provider().build_params(registry["llama70b"], prompt, "user", schema)

    assert anthropic_params["system"][0]["text"] == prompt
    assert openai_params["messages"][0]["content"] == prompt


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


def test_provider_specific_key_is_used(settings) -> None:
    assert settings.require_api_key("openrouter") == "test-key"
    assert settings.require_api_key("anthropic") == "test-key"


def test_generic_key_covers_other_openai_compatible_providers() -> None:
    settings = Settings(_env_file=None, openai_compatible_api_key="generic")
    assert settings.require_api_key("groq") == "generic"
    assert settings.require_api_key("together") == "generic"


def test_local_providers_need_no_key() -> None:
    """Ollama and vLLM accept any token; blocking the run would be wrong."""
    settings = Settings(_env_file=None)
    assert settings.require_api_key("ollama") == "not-needed"


def test_missing_key_message_names_the_variable_and_url() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(RuntimeError) as excinfo:
        settings.require_api_key("openrouter")

    message = str(excinfo.value)
    assert "OPENROUTER_API_KEY" in message
    assert "openrouter.ai/keys" in message


def test_missing_key_message_for_unknown_provider_is_still_actionable() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        settings.require_api_key("some_new_provider")
