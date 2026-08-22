"""
Provider-neutral LLM concerns: sizing, batching policy, and cost.

Everything vendor-specific lives in ``voc.providers``. What stays here is the
arithmetic that is true regardless of who serves the model, so a cost
comparison across providers is apples to apples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import ModelProfile, Settings
from voc.providers import LLMProvider, get_provider, resolve_effort

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "max_tokens_for",
    "DEFAULT_REVIEWS_PER_REQUEST",
    "EFFORT_OUTPUT_MULTIPLIER",
    "CostEstimate",
    "create_provider",
    "estimate_cost",
    "resolve_effort",
]

# Measured output per enriched review, from the 100-review benchmark:
# llama70b 251 tokens/review, qwen72b 881. Sized for the VERBOSE case with
# headroom -- undersizing truncates mid-JSON and loses the whole response,
# which is far worse than reserving a few hundred unused tokens.
#
# Right-sizing is not tidiness. Providers RESERVE credit against max_tokens,
# not against actual usage, so an oversized value makes requests unaffordable
# on a low balance and they fail with 402 before the model ever runs. A flat
# 8000 did exactly that on a free-tier account: "you requested up to 8000
# tokens, but can only afford 5580".
OUTPUT_TOKENS_PER_REVIEW = 900

# Fixed allowance for the JSON envelope around the per-review results.
OUTPUT_TOKENS_OVERHEAD = 500

# Fallback when the caller does not know the group size.
DEFAULT_MAX_TOKENS = 4000


def max_tokens_for(reviews_per_request: int) -> int:
    """Size the output budget to the actual work in one request.

    Truncation mid-JSON loses the whole response, so this errs generous -- but
    generous relative to measured output, not to a round number.
    """
    return OUTPUT_TOKENS_OVERHEAD + OUTPUT_TOKENS_PER_REVIEW * max(1, reviews_per_request)

# Reviews sent per API call. Groups amortise the ~4,400-token taxonomy prompt:
# one review per call would spend ~12M input tokens on the corpus, five per call
# spends ~2.7M. Every returned review_id is checked against what was requested,
# and anything missing is retried individually, so grouping costs reliability
# nothing.
DEFAULT_REVIEWS_PER_REQUEST = 5

#: Rough multipliers on billed output tokens by effort level. Thinking tokens
#: are billed as output and dominate the bill at high effort, so an estimator
#: that ignores them badly understates cost. Only applies to models that expose
#: an effort setting; open models here run without extended thinking.
EFFORT_OUTPUT_MULTIPLIER: dict[str, float] = {
    "low": 1.0,
    "medium": 1.8,
    "high": 3.0,
    "xhigh": 4.5,
    "max": 6.0,
}


@dataclass(frozen=True)
class CostEstimate:
    requests: int
    input_tokens: int
    output_tokens: int
    usd_standard: float
    usd_batch: float
    model_id: str
    provider: str = "anthropic"
    batch_available: bool = True

    def summary(self, use_batch: bool) -> str:
        effective_batch = use_batch and self.batch_available
        price = self.usd_batch if effective_batch else self.usd_standard
        mode = "Batch API (50% off)" if effective_batch else "live requests"
        return (
            f"{self.requests:,} requests · ~{self.input_tokens:,} input + "
            f"~{self.output_tokens:,} output tokens · {self.provider}/{self.model_id} · "
            f"{mode} · estimated ${price:,.2f}"
        )


def create_provider(
    profile: ModelProfile, settings: Settings, client=None
) -> LLMProvider:
    """Construct the provider this model profile declares."""
    return get_provider(profile, settings, client=client)


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
      input cost should land below this where caching is supported.
    * **Models thinking tokens** via an effort multiplier, because they are
      billed as output and dominate the bill on models that think. Without it
      the estimate would be roughly 3x too low on a frontier model at high
      effort.

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
        provider=profile.provider,
        batch_available=profile.provider == "anthropic",
    )
