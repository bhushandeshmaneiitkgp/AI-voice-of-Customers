"""
Layer 14 -- the Streamlit dashboard.

    streamlit run app/main.py

Deliberately thin. Every function that could be wrong lives in ``app/loaders.py``
and is tested; this file arranges what those return.

One editorial rule runs through it: **a number never appears without what
qualifies it.** This is the first place someone meets these figures without
having read the report around them, and a table strips context better than any
other format. So rates carry intervals, platform differences carry
"established / not established" rather than a bare p-value, the roadmap says in
its heading whether effort was ever estimated, and any page whose inputs are
missing prints the command that produces them instead of an empty chart.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pandas as pd
import streamlit as st

from app.loaders import (
    build_state,
    corpus_summary,
    format_rate,
    load_report,
    load_table,
    scoring_mode,
    significance_label,
    split_packed,
)
from config.settings import Paths, get_settings

# expanded: the sidebar is the only navigation, and Streamlit collapses it by
# default on narrower viewports -- which leaves the app looking like a
# single-page report with six pages the reader never discovers.
st.set_page_config(
    page_title="Quick-Commerce VOC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _table(key: str) -> pd.DataFrame:
    return load_table(build_state(Paths).artefacts[key])


@st.cache_data(show_spinner=False)
def _report(key: str) -> dict:
    return load_report(build_state(Paths).artefacts[key])


@st.cache_resource(show_spinner="Loading the embedding model…")
def _retriever():
    """Built once per session: constructing it loads a transformer."""
    from voc.retrieval import Retriever

    return Retriever.from_paths(
        Paths.faiss_index, _table("reviews"), get_settings().embedding_model, _table("labels")
    )


def require(*keys: str) -> bool:
    """Render a "run this first" message when inputs are missing."""
    state = build_state(Paths)
    if state.available(*keys):
        return True
    st.warning("This page needs output that has not been generated yet.")
    for command in state.instructions_for(*keys):
        st.code(command, language="bash")
    return False


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    st.title("Voice of Customer — Quick Commerce")
    if not require("reviews", "labels"):
        return

    summary = corpus_summary(_table("reviews"), _table("labels"), _report("enrichment_report"))

    a, b, c, d = st.columns(4)
    a.metric("Reviews enriched", f"{summary['reviews']:,}")
    b.metric("Labels", f"{summary['labels']:,}", f"{summary['areas_per_review']} per review")
    c.metric("Coverage", f"{summary['coverage_pct']}%" if summary["coverage_pct"] else "—")
    d.metric("Grounding", f"{summary['grounding_pct']}%" if summary["grounding_pct"] else "—")

    st.caption(
        f"{summary['platforms']} platforms · {summary['months']} months present, but only "
        f"**{summary['comparable_months']}** are comparable "
        f"({summary['comparable_reviews']:,} reviews). Everything competitive on this "
        "dashboard is scoped to that window — outside it, review volume tracks when "
        "the data was scraped rather than what customers did."
    )

    st.divider()
    st.subheader("What this is, and what it is not")
    st.markdown(
        """
- **Labels are model output, not ground truth.** Grounding is verified — every
  quoted span was checked to appear verbatim in the review — but no
  hand-labelled gold set exists yet, so accuracy is unmeasured.
- **Reviews are not users.** App-store reviewers are a small, self-selecting,
  annoyed slice. Rates here compare platforms *against each other*, not against
  reality.
- **There is no trend analysis.** Three comparable months is too short, and the
  earlier attempt produced 197× "growth" that was pure collection artefact.
        """
    )


def page_pain_points() -> None:
    st.title("Pain points")
    if not require("pain_points"):
        return

    pain_points = _table("pain_points")
    st.caption(
        "Scored on volume, severity, escalation, churn and negativity. The score "
        "**ranks, it does not measure** — a different weighting gives a different "
        "order. Weights live in `src/voc/painpoints.py`."
    )

    top = st.slider("Show top", 5, min(50, len(pain_points)), min(15, len(pain_points)))
    view = pain_points.head(top)

    st.dataframe(
        view[["rank", "product_area", "issue_type", "volume", "mean_severity",
              "escalation_rate", "churn_rate", "score"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "escalation_rate": st.column_config.ProgressColumn(
                "Escalation", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
            "churn_rate": st.column_config.NumberColumn("Churn", format="%.1f%%"),
            "score": st.column_config.NumberColumn("Score", format="%.3f"),
            "mean_severity": st.column_config.NumberColumn("Severity", format="%.2f"),
        },
    )

    st.divider()
    st.subheader("Evidence")
    choice = st.selectbox(
        "Pain point",
        view.index,
        format_func=lambda i: f"{view.loc[i, 'product_area']} / {view.loc[i, 'issue_type']}",
    )
    row = view.loc[choice]
    st.caption(
        f"{int(row['volume']):,} reviews · severity {row['mean_severity']:.2f} · "
        f"{row['escalation_rate'] * 100:.0f}% drove a support contact"
    )
    for quote in split_packed(row.get("evidence")):
        st.markdown(f"> {quote}")


def page_competitive() -> None:
    st.title("Competitive")
    if not require("platform_metrics", "comparisons"):
        return

    metrics, comparisons = _table("platform_metrics"), _table("comparisons")
    st.caption(
        "Rates, never counts — December holds three times October's reviews. "
        "Differences are corrected for multiplicity with Benjamini–Hochberg: "
        "read **established / not established**, not the raw p-value."
    )

    metric = st.selectbox(
        "Metric", metrics["metric"].unique(),
        format_func=lambda k: metrics[metrics["metric"] == k].iloc[0]["label"],
    )
    subset = metrics[metrics["metric"] == metric].sort_values("rate", ascending=False)

    columns = st.columns(len(subset))
    for column, row in zip(columns, subset.itertuples()):
        column.metric(row.platform, f"{row.rate * 100:.1f}%")
        column.caption(f"95% CI {row.ci_low * 100:.1f}–{row.ci_high * 100:.1f}%")

    st.divider()
    pairs = comparisons[comparisons["metric"] == metric].copy()
    pairs["verdict"] = pairs["significant"].map(significance_label)
    pairs["difference_pp"] = (pairs["difference"] * 100).round(1)
    st.dataframe(
        pairs[["platform_a", "platform_b", "difference_pp", "p_value", "verdict"]],
        use_container_width=True, hide_index=True,
        column_config={
            "difference_pp": st.column_config.NumberColumn("Difference (pp)", format="%+.1f"),
            "p_value": st.column_config.NumberColumn("p", format="%.4f"),
        },
    )

    if not pairs["significant"].all():
        st.info(
            "Rows marked *not established* have point estimates that differ, but "
            "the evidence does not support saying so. That is a result, not a gap."
        )

    if build_state(Paths).available("area_rates"):
        st.divider()
        st.subheader("Where each platform over-indexes")
        areas = _table("area_rates")
        flagged = areas[areas["significant"]].sort_values("lift", ascending=False)
        st.dataframe(
            flagged[["product_area", "platform", "rate", "corpus_rate", "lift", "p_value"]],
            use_container_width=True, hide_index=True,
            column_config={
                "rate": st.column_config.NumberColumn("Rate", format="%.1f%%"),
                "corpus_rate": st.column_config.NumberColumn("Corpus", format="%.1f%%"),
                "lift": st.column_config.NumberColumn("Lift", format="%.2fx"),
                "p_value": st.column_config.NumberColumn("p", format="%.4f"),
            },
        )


def page_themes() -> None:
    st.title("Themes")
    if not require("clusters"):
        return

    clusters = _table("clusters")
    st.caption(
        "Discovered by clustering review embeddings, not declared by the taxonomy. "
        "Separation is weak — treat these as a reading aid over the ranked pain "
        "points, not a partition of the corpus."
    )
    st.dataframe(
        clusters[["cluster_id", "size", "share_pct", "dominant_area", "dominant_issue",
                  "mean_severity", "escalation_rate", "top_platform"]],
        use_container_width=True, hide_index=True,
    )

    for row in clusters.itertuples():
        with st.expander(f"Cluster {row.cluster_id} — {row.dominant_area} ({row.size:,} reviews)"):
            for text in split_packed(row.exemplar_texts):
                st.markdown(f"> {text}")


def page_root_causes() -> None:
    st.title("Root causes")
    if not require("root_causes"):
        return

    causes = _table("root_causes")
    st.caption(
        "**Hypotheses, not findings.** Each is a candidate mechanism generated "
        "from customer text, and every citation was verified against the reviews "
        "the model was actually shown. The *proposed check* is the actionable "
        "part — it is what would confirm or kill the hypothesis."
    )

    area = st.selectbox(
        "Pain point",
        causes[["product_area", "issue_type"]].drop_duplicates().index,
        format_func=lambda i: f"{causes.loc[i, 'product_area']} / {causes.loc[i, 'issue_type']}",
    )
    selected = causes[
        (causes["product_area"] == causes.loc[area, "product_area"])
        & (causes["issue_type"] == causes.loc[area, "issue_type"])
    ]

    for n, row in enumerate(selected.itertuples(), start=1):
        st.markdown(f"### {n}. {row.hypothesis}")
        st.markdown(f"**Mechanism.** {row.mechanism}")
        st.success(f"**Check this.** {row.proposed_check}")
        if str(row.disconfirming_evidence).strip():
            st.warning(f"**Against it.** {row.disconfirming_evidence}")
        st.caption(
            f"Cited reviews: {', '.join(split_packed(row.supporting_review_ids))} · "
            f"model confidence {row.confidence:.2f} (self-reported, not measured)"
        )
        st.divider()


def page_roadmap() -> None:
    st.title("Roadmap")
    if not require("rice"):
        return

    rice = _table("rice")
    mode = scoring_mode(rice)

    if mode == "ric":
        st.error(
            "**This is a RIC ranking, not RICE — no effort estimates were "
            "supplied.** Reach, impact and confidence describe the *problem*, "
            "which reviews can speak to. Effort describes the *solution and your "
            "codebase*, which they cannot. A guessed denominator would rank work "
            "by fiction while looking quantitative."
        )
        st.code(
            "python scripts/09_build_roadmap.py --write-effort-template\n"
            "# fill in effort_person_weeks, then\n"
            "python scripts/09_build_roadmap.py --effort data/processed/effort_template.csv",
            language="bash",
        )
    else:
        st.success("Effort estimates supplied — this is a real RICE ranking.")

    score_column = "rice" if mode == "rice" else "ric"
    st.dataframe(
        rice[["rank", "product_area", "issue_type", "reach_per_month", "impact_label",
              "confidence", "effort_person_weeks", score_column]],
        use_container_width=True, hide_index=True,
        column_config={
            "reach_per_month": st.column_config.NumberColumn("Reach/mo", format="%.0f"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "effort_person_weeks": st.column_config.NumberColumn("Effort (wks)", format="%.1f"),
            score_column: st.column_config.NumberColumn(score_column.upper(), format="%.1f"),
        },
    )
    st.caption(
        "Reach is **reviews per month**, not customers. Confidence is derived "
        "from grounding, sample size, label confidence, whether a mechanism was "
        "found, and whether a competitive difference survived correction."
    )

    if build_state(Paths).available("opportunities", "experiments"):
        st.divider()
        st.subheader("Opportunities and experiments")
        opportunities, experiments = _table("opportunities"), _table("experiments")
        for row in opportunities.itertuples():
            with st.expander(f"{row.title}  ·  {row.product_area}/{row.issue_type}"):
                st.markdown(f"**Change.** {row.change}")
                st.markdown(f"**Addresses.** {row.addresses_hypothesis}")
                st.markdown(
                    f"**Success.** `{row.primary_metric}` should **{row.expected_direction}**."
                )
                st.warning(f"**Risk if wrong.** {row.risk_if_wrong}")

                plan = experiments[experiments["title"] == row.title]
                if not plan.empty:
                    p = plan.iloc[0]
                    left, right = st.columns(2)
                    left.metric("Sample per arm", f"{int(p['sample_per_arm']):,}")
                    right.metric("Duration", f"{p['months_required']:.1f} mo")
                    if not p["practical"]:
                        st.error(
                            "**Underpowered at review volume.** A flat result here "
                            "would mean the test could not have seen the effect, "
                            "not that there was none. Note this is reviews — "
                            "product telemetry would reach power far sooner."
                        )


def page_evidence() -> None:
    st.title("Evidence explorer")
    if not require("reviews", "faiss"):
        return

    st.caption(
        "Semantic search over the corpus, using the same index the root-cause "
        "layer retrieves from. Returns reviews — never a generated answer."
    )
    query = st.text_input("Search customer reviews", placeholder="wallet balance stuck at checkout")
    if not query.strip():
        return

    reviews = _table("reviews")
    left, right = st.columns(2)
    platform = left.selectbox("Platform", ["any", *sorted(reviews["platform"].unique())])
    k = right.slider("Results", 3, 25, 8)

    result = _retriever().search(
        query, k=k, platform=None if platform == "any" else platform
    )
    if result.truncated_by_filters:
        st.info(f"Only {len(result.hits)} review(s) matched those filters.")

    for hit in result.hits:
        st.markdown(
            f"**`{hit.review_id}`** · {hit.platform} · {hit.rating}★ · {hit.year_month} "
            f"· similarity {hit.score:.3f}"
        )
        st.markdown(f"> {hit.text}")
        if hit.product_areas:
            st.caption("Areas: " + ", ".join(f"`{a}`" for a in hit.product_areas))
        st.divider()


PAGES = {
    "Overview": page_overview,
    "Pain points": page_pain_points,
    "Competitive": page_competitive,
    "Themes": page_themes,
    "Root causes": page_root_causes,
    "Roadmap": page_roadmap,
    "Evidence explorer": page_evidence,
}


def main() -> None:
    st.sidebar.title("Quick-Commerce VOC")
    choice = st.sidebar.radio("Page", list(PAGES))

    state = build_state(Paths)
    st.sidebar.divider()
    st.sidebar.caption("Pipeline status")
    for phase in ("1", "3", "4", "5", "6", "7"):
        items = [a for a in state.artefacts.values() if a.phase == phase]
        done = all(a.available for a in items)
        st.sidebar.write(f"{'✅' if done else '⬜'} Phase {phase}")

    PAGES[choice]()


if __name__ == "__main__":
    main()
