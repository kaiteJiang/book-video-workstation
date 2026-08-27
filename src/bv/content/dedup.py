from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bv.content.topics import EpisodeBrief, TopicDecision


_PUNCTUATION = re.compile(r"[\W_]+", re.UNICODE)


def normalize_life_expression(value: str) -> str:
    """Normalize deterministic text equality without claiming semantic equivalence."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _PUNCTUATION.sub("", " ".join(normalized.split()))


def deterministic_dedup(
    candidate: EpisodeBrief,
    protected: list[EpisodeBrief],
) -> TopicDecision:
    """Apply the deterministic first three topic-deduplication layers."""
    from bv.content.topics import TopicDecision

    if not candidate.supporting_claim_cluster_ids or not candidate.source_claim_ids:
        return TopicDecision(accepted=False, reasons=["empty_evidence"])

    reasons: set[str] = set()
    matched: set[str] = set()
    highest_cluster = 0.0
    highest_claim = 0.0
    for prior in protected:
        comparison_reasons: set[str] = set()
        if _same_reader_transformation(candidate, prior):
            comparison_reasons.add("same_reader_transformation")
        if normalize_life_expression(candidate.life_connection) == normalize_life_expression(
            prior.life_connection
        ):
            comparison_reasons.add("same_life_expression")

        if prior.episode_kind == "value_angle":
            cluster_overlap = _jaccard(
                candidate.supporting_claim_cluster_ids,
                prior.supporting_claim_cluster_ids,
            )
            claim_overlap = _jaccard(candidate.source_claim_ids, prior.source_claim_ids)
            highest_cluster = max(highest_cluster, cluster_overlap)
            highest_claim = max(highest_claim, claim_overlap)
            if cluster_overlap >= 0.5:
                comparison_reasons.add("cluster_overlap_high")
            if claim_overlap >= 0.5:
                comparison_reasons.add("claim_overlap_high")
        if comparison_reasons:
            matched.add(prior.episode_id)
            reasons.update(comparison_reasons)

    return TopicDecision(
        accepted=not reasons,
        reasons=sorted(reasons),
        matched_episode_ids=sorted(matched),
        cluster_overlap=highest_cluster,
        claim_overlap=highest_claim,
    )


def _same_reader_transformation(candidate: EpisodeBrief, prior: EpisodeBrief) -> bool:
    return (
        normalize_life_expression(candidate.target_reader)
        == normalize_life_expression(prior.target_reader)
        and normalize_life_expression(candidate.central_life_tension)
        == normalize_life_expression(prior.central_life_tension)
        and normalize_life_expression(candidate.reader_before)
        == normalize_life_expression(prior.reader_before)
        and normalize_life_expression(candidate.reader_after)
        == normalize_life_expression(prior.reader_after)
    ) or normalize_life_expression(candidate.practical_value) == normalize_life_expression(
        prior.practical_value
    )


def _jaccard(left: list[str], right: list[str]) -> float:
    left_values = set(left)
    right_values = set(right)
    if not left_values or not right_values:
        return 0.0
    return len(left_values & right_values) / len(left_values | right_values)
