"""
Anthropic API access for the enrichment pipeline.

One request shape, built once, used by both transports:

* **Sync** (`client.messages.create`) for small samples and debugging.
* **Batch** (`client.messages.batches`) for the full corpus at 50% cost.

Both take the identical params dict, so what is tested on 20 reviews is exactly
what runs on 4,620. Two code paths that drift is a classic way to have a
validated sample and a broken production run.

Per-model API differences come from ``config/models.yaml``, not from branches on
a model name. Adaptive thinking, effort levels, and token budgets are not
uniform across models — Haiku 4.5 rejects ``effort`` outright — so the registry
records the capability and this module reads it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from config.settings import ModelProfile, Settings

logger = logging.getLogger(__name__)

# Enough for a group of enrichments plus reasoning. Structured output is
# compact, but truncation mid-JSON loses the whole response, so this is
# deliberately generous rather than tight.
DEFAULT_MAX_TOKENS = 8000

# Reviews sent per API call. Groups amortise the ~2,700-token taxonomy prompt:
# one review per call would spend ~12M input tokens on the corpus, five per call
# spends ~2.7M. Every returned review_id is checked against what was requested,
# and anything missing is retried individually, so grouping costs reliability
# nothing.
DEFAULT_REVIEWS_PER_REQUEST = 5


@dataclass(frozen=True)
class CostEstimate:
    requests: int
    input_tokens: int
    output_tokens: int
    usd_standard: float
    usd_batch: float
    model_id: str

    def summary(self, use_batch: bool) -> str:
        price = self.usd_batch if use_batch else self.usd_standard
        mode = "Batch API (50% off)" if use_batch else "standard"
        return (
            f"{self.requests:,} requests · ~{self.input_tokens:,} input + "
            f"~{self.output_tokens:,} output tokens · {self.model_id} · "
            f"{mode} · estimated ${price:,.2f}"
        )


def create_client(settings: Settings):
    """Construct the SDK client.

    The key is read from the environment by ``require_api_key`` and passed
    explicitly, so a missing key fails with our actionable message rather than
    the SDK's generic one.
    """
    import anthropic

    return anthropic.Anthropic(api_key=settings.require_api_key())


#: Rough multipliers on billed output tokens by effort level. Thinking tokens
#: are billed as output and dominate the bill at high effort, so an estimator
#: that ignores them badly understates cost. These are approximations for
#: planning only -- the run report records what was actually spent.
EFFORT_OUTPUT_MULTIPLIER: dict[str, float] = {
    "low": 1.0,
    "medium": 1.8,
    "high": 3.0,
    "xhigh": 4.5,
    "max": 6.0,
}


def resolve_effort(profile: ModelProfile, override: str | None = None) -> str | None:
    """Effort level for a request, or None when the model does not support it."""
    if not profile.supports_effort:
        return None
    return override or profile.default_effort


def build_request_params(
    profile: ModelProfile,
    system_prompt: str,
    user_message: str,
    response_schema: dict[str, Any],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str | None = None,
) -> dict[str, Any]:
    """Assemble one Messages API request, adapted to the selected model.

    The system prompt carries ``cache_control`` because it is byte-identical
    across every request in a run (~2,700 tokens of taxonomy). Cache hits cut
    input cost by roughly 10x on the repeated prefix, which is most of the spend.
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
        # Older models (e.g. Haiku 4.5) predate adaptive thinking and return a
        # 400 if given `effort`. Classification does not need thinking, so it is
        # simply omitted rather than configured with a token budget.
        logger.debug("Model %s uses budget-style thinking; running without it.", profile.model_id)

    params["output_config"] = output_config
    return params


def estimate_cost(
    profile: ModelProfile,
    n_reviews: int,
    system_prompt: str,
    reviews_per_request: int = DEFAULT_REVIEWS_PER_REQUEST,
    avg_review_tokens: int = 90,
    avg_output_tokens_per_review: int = 260,
    effort: str | None = None,
) -> CostEstimate:
    """Estimate spend before committing to a run.

    Two opposing approximations, stated plainly rather than hidden:

    * **Ignores prompt-cache savings**, which pushes the estimate UP. The
      taxonomy prefix is ~4,400 tokens and repeats on every request, so real
      input cost should land well below this.
    * **Models thinking tokens** via an effort multiplier, because they are
      billed as output and dominate the bill at high effort. Without this the
      estimate would be roughly 3x too low on a frontier model.

    Treat the result as a planning figure; the run report records actual spend.
    """
    requests = max(1, -(-n_reviews // reviews_per_request))  # ceiling division
    system_tokens = len(system_prompt) // 4  # ~4 chars/token for English

    input_tokens = requests * system_tokens + n_reviews * avg_review_tokens

    chosen_effort = resolve_effort(profile, effort)
    multiplier = EFFORT_OUTPUT_MULTIPLIER.get(chosen_effort or "", 1.0)
    output_tokens = int(n_reviews * avg_output_tokens_per_review * multiplier)

    return CostEstimate(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd_standard=profile.estimate_cost_usd(input_tokens, output_tokens, use_batch=False),
        usd_batch=profile.estimate_cost_usd(input_tokens, output_tokens, use_batch=True),
        model_id=profile.model_id,
    )


def extract_json_text(message: Any) -> str:
    """Pull the JSON payload out of a response.

    With ``output_config.format`` set the model returns one text block of valid
    JSON, but responses may also carry thinking blocks, so the block type must
    be checked rather than assuming ``content[0]``.
    """
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError(
        "No text block in response. Content block types: "
        f"{[getattr(b, 'type', '?') for b in message.content]}"
    )


def describe_usage(message: Any) -> dict[str, int]:
    """Extract token usage, including cache performance.

    ``cache_read_input_tokens`` staying at zero across a run means a silent
    cache invalidator crept into the prompt prefix — worth noticing, since it
    is the difference between a $5 run and a $45 one.
    """
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }
