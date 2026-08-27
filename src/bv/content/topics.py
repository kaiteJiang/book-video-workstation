from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bv.content.dedup import deterministic_dedup
from bv.content.synthesis import WholeBookValue
from bv.models.contracts import (
    ModelCompletionError,
    ReviewSkipped,
    StructuredModel,
    empty_request_directory,
    is_redirected as _is_redirected,
)
from bv.models.prompts import compose_source_prompt
from bv.state.locks import EpisodeLock, LockHeldError


EpisodeKind = Literal["whole_book_value", "value_angle"]
EpisodeStatus = Literal[
    "reserved", "script_approved", "produced", "published", "rejected", "released"
]
SemanticLabel = Literal["distinct", "related_but_close", "too_close", "duplicate"]
GrokLabel = Literal[
    "distinct", "related_but_close", "too_close", "duplicate", "grok_review_skipped"
]
_PROTECTED_STATUSES = {"reserved", "script_approved", "produced", "published"}
_ALLOWED_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "reserved": {"script_approved", "rejected", "released"},
    "script_approved": {"produced"},
    "produced": {"published"},
    "published": set(),
    "rejected": {"released"},
    "released": set(),
}
_UNSUPPORTED_ATTRIBUTION = ("书中说", "书里说", "作者说", "作者认为", "这本书告诉我们", "这本书证明")
_SCOPE_EXPANSION = ("保证", "一定能", "彻底解决", "治愈", "解决所有")
_FABRICATED_TESTIMONY = ("我朋友", "读者小", "真实案例", "亲身经历")
_ISOLATED_CONCEPTS = ("课题分离", "一个概念", "单一概念")
_TRUSTED_REJECTION_CODES = {
    "blank_topic_text",
    "claim_cluster_mismatch",
    "claim_overlap_high",
    "cluster_overlap_high",
    "codex_semantic_failed",
    "codex_semantic_not_distinct",
    "duplicate_claim_id",
    "duplicate_cluster_id",
    "empty_evidence",
    "fabricated_testimony",
    "grok_reader_repeat",
    "invalid_cluster_count",
    "invalid_evidence_identifier",
    "invalid_later_episode_identity",
    "invalid_topic_status",
    "isolated_concept_framing",
    "same_life_expression",
    "same_reader_transformation",
    "scope_expansion",
    "topic_model_failed",
    "unknown_claim_id",
    "unknown_cluster_id",
    "unsupported_attribution",
}


class EpisodeBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    episode_kind: EpisodeKind
    topic_name: str
    book_value_thesis: str
    target_reader: str
    reader_before: str
    reader_after: str
    central_life_tension: str
    supporting_claim_cluster_ids: list[str]
    source_claim_ids: list[str]
    life_connection: str
    practical_value: str
    reading_reason: str
    constructed_scene: bool
    status: EpisodeStatus


class TopicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    matched_episode_ids: list[str] = Field(default_factory=list)
    cluster_overlap: float = 0.0
    claim_overlap: float = 0.0
    semantic_label: SemanticLabel | None = None
    grok_label: GrokLabel | None = None


class SemanticTopicReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: SemanticLabel
    reasons: list[str] = Field(default_factory=list)


class GrokTopicReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["distinct", "related_but_close", "too_close", "duplicate"]
    reasons: list[str] = Field(default_factory=list)


class _TopicLedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: EpisodeBrief
    decision: TopicDecision | None = None


class TopicGenerationError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class TopicService:
    """Create exactly one locked E001, then explicitly create later value angles."""

    def __init__(
        self,
        book_root: Path,
        *,
        model: StructuredModel,
        grok_model: StructuredModel | None = None,
        topic_prompt_text: str,
        dedup_prompt_text: str,
        grok_prompt_text: str,
    ) -> None:
        self.book_root = Path(book_root)
        self.model = model
        self.grok_model = grok_model
        self.topic_prompt_text = _required_prompt(topic_prompt_text)
        self.dedup_prompt_text = _required_prompt(dedup_prompt_text)
        self.grok_prompt_text = _required_prompt(grok_prompt_text)
        self.ledger_path = self.book_root / "ledger" / "topics.jsonl"
        self.lock_path = self.ledger_path.parent / ".topics.lock"
        self.request_root = self.book_root / ".private" / "requests"

    def generate_first_topic(self, value: WholeBookValue) -> EpisodeBrief:
        """Generate and persist the canonical whole-book-value E001 once."""
        self._prepare_storage()
        try:
            with EpisodeLock(self.lock_path):
                topics = self._read_topics()
                existing = _find_e001(topics)
                if existing is not None:
                    if _first_identity_matches(existing, value):
                        return existing
                    raise TopicGenerationError("e001_conflict")

                response = self._complete(
                    self.model,
                    _first_prompt(value, self.topic_prompt_text),
                    EpisodeBrief,
                    "first",
                )
                if not isinstance(response, EpisodeBrief):
                    raise TopicGenerationError("topic_model_failed")
                brief = _validate_first_candidate(response, value)
                decision = self._grok_decision(brief, topics)
                self._append(_TopicLedgerEntry(brief=brief, decision=decision))
                return brief
        except LockHeldError as error:
            raise TopicGenerationError("topic_lock_held") from error

    def generate_next_topic(self, value: WholeBookValue) -> EpisodeBrief:
        """Explicitly create one later distinct value angle; never precompute a pool."""
        self._prepare_storage()
        try:
            with EpisodeLock(self.lock_path):
                topics = self._read_topics()
                if not any(
                    item.episode_id == "E001" and item.status in _PROTECTED_STATUSES
                    for item in topics
                ):
                    raise TopicGenerationError("missing_active_e001")
                expected_id = _next_episode_id(topics)
                protected = [item for item in topics if item.status in _PROTECTED_STATUSES]
                rejected: list[dict[str, object]] = []
                for attempt in range(3):
                    response = self._complete_or_none(
                        self.model,
                        _next_prompt(
                            value,
                            protected,
                            expected_id,
                            rejected,
                            self.topic_prompt_text,
                        ),
                        EpisodeBrief,
                        f"next-{attempt + 1}",
                    )
                    if not isinstance(response, EpisodeBrief):
                        rejected.append({"reason_codes": ["topic_model_failed"]})
                        continue
                    code = _validate_later_candidate(response, value, expected_id)
                    if code is not None:
                        rejected.append({"reason_codes": [code]})
                        continue
                    deterministic = deterministic_dedup(
                        response,
                        protected,
                    )
                    if not deterministic.accepted:
                        rejected.append({"reason_codes": deterministic.reasons})
                        continue
                    semantic = self._semantic_decision(response, protected)
                    if not semantic.accepted:
                        rejected.append({"reason_codes": semantic.reasons})
                        continue
                    grok = self._grok_decision(response, protected)
                    if grok.grok_label not in (None, "distinct", "grok_review_skipped"):
                        rejected.append({"reason_codes": ["grok_reader_repeat"]})
                        continue
                    decision = TopicDecision(
                        accepted=True,
                        matched_episode_ids=semantic.matched_episode_ids,
                        cluster_overlap=deterministic.cluster_overlap,
                        claim_overlap=deterministic.claim_overlap,
                        semantic_label="distinct",
                        grok_label=grok.grok_label,
                    )
                    self._append(_TopicLedgerEntry(brief=response, decision=decision))
                    return response
        except LockHeldError as error:
            raise TopicGenerationError("topic_lock_held") from error
        raise TopicGenerationError("insufficient_distinct_topic")

    def list_topics(self) -> list[EpisodeBrief]:
        self._check_storage_paths()
        return self._read_topics()

    def transition_status(
        self,
        episode_id: str,
        new_status: EpisodeStatus,
    ) -> EpisodeBrief:
        """Append one validated lifecycle transition without changing topic identity."""
        self._prepare_storage()
        try:
            with EpisodeLock(self.lock_path):
                entries = self._read_latest_entries()
                current = entries.get(episode_id)
                if current is None:
                    raise TopicGenerationError("topic_not_found")
                if new_status not in _ALLOWED_STATUS_TRANSITIONS[current.brief.status]:
                    raise TopicGenerationError("invalid_topic_transition")
                updated = current.brief.model_copy(update={"status": new_status})
                self._append(
                    _TopicLedgerEntry(
                        brief=updated,
                        decision=current.decision,
                    )
                )
                return updated
        except LockHeldError as error:
            raise TopicGenerationError("topic_lock_held") from error

    def _prepare_storage(self) -> None:
        self._check_storage_paths()
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.request_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TopicGenerationError("unsafe_topic_storage") from error
        self._check_storage_paths()

    def _check_storage_paths(self) -> None:
        paths = (self.book_root, self.ledger_path.parent, self.ledger_path, self.lock_path, self.request_root)
        if any(_redirect_in_existing_chain(path) or _is_redirected(path) for path in paths):
            raise TopicGenerationError("unsafe_topic_storage")

    def _read_topics(self) -> list[EpisodeBrief]:
        return [entry.brief for entry in self._read_latest_entries().values()]

    def _read_latest_entries(self) -> dict[str, _TopicLedgerEntry]:
        if not self.ledger_path.exists():
            return {}
        latest_by_id: dict[str, _TopicLedgerEntry] = {}
        try:
            with self.ledger_path.open("r", encoding="utf-8", newline="\n") as stream:
                for line in stream:
                    if not line.strip():
                        raise TopicGenerationError("corrupted_topics_ledger")
                    entry = _TopicLedgerEntry.model_validate_json(line)
                    brief = entry.brief
                    if not _valid_episode_id(brief.episode_id):
                        raise TopicGenerationError("corrupted_topics_ledger")
                    previous = latest_by_id.get(brief.episode_id)
                    if previous is not None and (
                        previous.brief.status == brief.status
                        or not _same_topic_identity(previous.brief, brief)
                        or previous.decision != entry.decision
                    ):
                        raise TopicGenerationError("corrupted_topics_ledger")
                    latest_by_id[brief.episode_id] = entry
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
            raise TopicGenerationError("corrupted_topics_ledger") from error
        return latest_by_id

    def _append(self, entry: _TopicLedgerEntry) -> None:
        self._check_storage_paths()
        encoded = entry.model_dump_json(ensure_ascii=False) + "\n"
        try:
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise TopicGenerationError("topic_ledger_write_failed") from error

    def _complete(
        self,
        model: StructuredModel,
        prompt: str,
        schema_type: type[BaseModel],
        prefix: str,
    ) -> BaseModel | ReviewSkipped:
        try:
            request_dir = Path(tempfile.mkdtemp(prefix=f"topic-{prefix}-", dir=self.request_root))
        except OSError as error:
            raise TopicGenerationError("unsafe_request_directory") from error
        if not empty_request_directory(request_dir) or _is_redirected(request_dir):
            raise TopicGenerationError("unsafe_request_directory")
        try:
            return model.complete(prompt, schema_type, request_dir)
        except ModelCompletionError as error:
            raise TopicGenerationError("topic_model_failed") from error
        except Exception as error:
            raise TopicGenerationError("topic_model_failed") from error

    def _complete_or_none(
        self,
        model: StructuredModel,
        prompt: str,
        schema_type: type[BaseModel],
        prefix: str,
    ) -> BaseModel | ReviewSkipped | None:
        try:
            return self._complete(model, prompt, schema_type, prefix)
        except TopicGenerationError as error:
            if error.error_code == "topic_model_failed":
                return None
            raise

    def _semantic_decision(
        self, candidate: EpisodeBrief, protected: list[EpisodeBrief]
    ) -> TopicDecision:
        response = self._complete_or_none(
            self.model,
            _semantic_prompt(candidate, protected, self.dedup_prompt_text),
            SemanticTopicReview,
            "semantic",
        )
        matched = [item.episode_id for item in protected]
        if not isinstance(response, SemanticTopicReview):
            return TopicDecision(accepted=False, reasons=["codex_semantic_failed"], matched_episode_ids=matched)
        if response.label != "distinct":
            return TopicDecision(
                accepted=False,
                reasons=["codex_semantic_not_distinct"],
                matched_episode_ids=matched,
                semantic_label=response.label,
            )
        return TopicDecision(accepted=True, matched_episode_ids=matched, semantic_label="distinct")

    def _grok_decision(
        self, candidate: EpisodeBrief, protected: list[EpisodeBrief]
    ) -> TopicDecision:
        if self.grok_model is None:
            return TopicDecision(accepted=True, grok_label="grok_review_skipped")
        response = self._complete_or_none(
            self.grok_model,
            _grok_prompt(candidate, protected, self.grok_prompt_text),
            GrokTopicReview,
            "grok",
        )
        if isinstance(response, ReviewSkipped) or not isinstance(response, GrokTopicReview):
            return TopicDecision(accepted=True, grok_label="grok_review_skipped")
        return TopicDecision(
            accepted=response.label == "distinct",
            reasons=[] if response.label == "distinct" else ["grok_reader_repeat"],
            grok_label=response.label,
        )


def _validate_first_candidate(candidate: EpisodeBrief, value: WholeBookValue) -> EpisodeBrief:
    code = _validate_brief_shape(candidate, require_whole_book=True)
    if code is not None:
        raise TopicGenerationError(code)
    expected = _bound_first_brief(value, candidate)
    locked = (
        "episode_id", "episode_kind", "book_value_thesis", "target_reader", "reader_before",
        "reader_after", "central_life_tension", "supporting_claim_cluster_ids", "source_claim_ids",
        "practical_value", "reading_reason",
    )
    if any(getattr(candidate, field) != getattr(expected, field) for field in locked):
        raise TopicGenerationError("locked_field_mismatch")
    return expected


def _validate_later_candidate(candidate: EpisodeBrief, value: WholeBookValue, expected_id: str) -> str | None:
    code = _validate_brief_shape(candidate, require_whole_book=False)
    if code is not None:
        return code
    if candidate.episode_id != expected_id or candidate.episode_kind != "value_angle":
        return "invalid_later_episode_identity"
    known_clusters = [cluster.cluster_id for cluster in value.supporting_claim_clusters]
    known_claims = {claim_id for cluster in value.supporting_claim_clusters for claim_id in cluster.claim_ids}
    if any(cluster not in known_clusters for cluster in candidate.supporting_claim_cluster_ids):
        return "unknown_cluster_id"
    if any(claim not in known_claims for claim in candidate.source_claim_ids):
        return "unknown_claim_id"
    if len(set(candidate.supporting_claim_cluster_ids)) < 2:
        return "isolated_concept_framing"
    claims_by_cluster = {
        cluster.cluster_id: set(cluster.claim_ids)
        for cluster in value.supporting_claim_clusters
    }
    selected_claims = {
        cluster_id: claims_by_cluster[cluster_id]
        for cluster_id in candidate.supporting_claim_cluster_ids
    }
    candidate_claims = set(candidate.source_claim_ids)
    if any(not (candidate_claims & claims) for claims in selected_claims.values()):
        return "claim_cluster_mismatch"
    if not candidate_claims <= set().union(*selected_claims.values()):
        return "claim_cluster_mismatch"
    return None


def _validate_brief_shape(candidate: EpisodeBrief, *, require_whole_book: bool) -> str | None:
    if candidate.status != "reserved":
        return "invalid_topic_status"
    if not _canonical_text(candidate.topic_name) or not _canonical_text(candidate.life_connection):
        return "blank_topic_text"
    if len(candidate.supporting_claim_cluster_ids) not in (2, 3):
        return "isolated_concept_framing" if len(candidate.supporting_claim_cluster_ids) < 2 else "invalid_cluster_count"
    if len(candidate.source_claim_ids) < 2:
        return "isolated_concept_framing"
    if any(not _canonical_identifier(item) for item in candidate.supporting_claim_cluster_ids + candidate.source_claim_ids):
        return "invalid_evidence_identifier"
    if len(candidate.supporting_claim_cluster_ids) != len(set(candidate.supporting_claim_cluster_ids)):
        return "duplicate_cluster_id"
    if len(candidate.source_claim_ids) != len(set(candidate.source_claim_ids)):
        return "duplicate_claim_id"
    contents = "\n".join(candidate.model_dump().get(field, "") for field in (
        "topic_name", "book_value_thesis", "target_reader", "reader_before", "reader_after",
        "central_life_tension", "life_connection", "practical_value", "reading_reason",
    ))
    if any(token in contents for token in _UNSUPPORTED_ATTRIBUTION):
        return "unsupported_attribution"
    if any(token in contents for token in _SCOPE_EXPANSION):
        return "scope_expansion"
    if any(token in contents for token in _FABRICATED_TESTIMONY):
        return "fabricated_testimony"
    if require_whole_book and any(token in candidate.topic_name for token in _ISOLATED_CONCEPTS):
        return "isolated_concept_framing"
    return None


def _bound_first_brief(value: WholeBookValue, candidate: EpisodeBrief) -> EpisodeBrief:
    return EpisodeBrief(
        episode_id="E001",
        episode_kind="whole_book_value",
        topic_name=candidate.topic_name,
        book_value_thesis=value.value_thesis,
        target_reader=value.target_reader,
        reader_before=value.reader_before,
        reader_after=value.reader_after,
        central_life_tension=value.central_life_tension,
        supporting_claim_cluster_ids=[item.cluster_id for item in value.supporting_claim_clusters],
        source_claim_ids=sorted({claim_id for item in value.supporting_claim_clusters for claim_id in item.claim_ids}),
        life_connection=candidate.life_connection,
        practical_value=value.practical_value,
        reading_reason=value.reading_reason,
        constructed_scene=candidate.constructed_scene,
        status="reserved",
    )


def _first_identity_matches(brief: EpisodeBrief, value: WholeBookValue) -> bool:
    expected = _bound_first_brief(value, brief)
    return brief.model_copy(update={"status": "reserved"}) == expected


def _find_e001(topics: list[EpisodeBrief]) -> EpisodeBrief | None:
    first = [item for item in topics if item.episode_id == "E001"]
    if len(first) > 1:
        raise TopicGenerationError("corrupted_topics_ledger")
    return first[0] if first else None


def _same_topic_identity(left: EpisodeBrief, right: EpisodeBrief) -> bool:
    return left.model_dump(exclude={"status"}) == right.model_dump(exclude={"status"})


def _valid_episode_id(value: str) -> bool:
    return len(value) == 4 and value.startswith("E") and value[1:].isdigit()


def _next_episode_id(topics: list[EpisodeBrief]) -> str:
    numbers = [int(item.episode_id[1:]) for item in topics if item.episode_id.startswith("E") and item.episode_id[1:].isdigit()]
    return f"E{max(numbers, default=1) + 1:03d}"


def _redirect_in_existing_chain(path: Path) -> bool:
    current = Path(path)
    while True:
        if current.exists() or current.is_symlink():
            if _is_redirected(current):
                return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _canonical_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _canonical_identifier(value: object) -> bool:
    return _canonical_text(value)


def _prompt(instructions: str, payload: dict[str, object]) -> str:
    return compose_source_prompt(instructions, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _first_prompt(value: WholeBookValue, prompt_text: str) -> str:
    return _prompt(
        f"{prompt_text.rstrip()}\n\nGenerate exactly one E001 whole_book_value brief. Keep the whole-book evidence, reader transformation, and boundaries unchanged. Stable rejection codes: locked_field_mismatch, isolated_concept_framing, unsupported_attribution, scope_expansion.",
        {"value": value.model_dump(mode="json")},
    )


def _next_prompt(
    value: WholeBookValue,
    protected: list[EpisodeBrief],
    expected_id: str,
    rejected: list[dict[str, object]],
    prompt_text: str,
) -> str:
    rejection_codes = _validated_rejection_codes(rejected)
    repair_instruction = ""
    if rejection_codes:
        repair_instruction = (
            "\n\nPrevious attempts failed these trusted validation codes: "
            f"{json.dumps(rejection_codes, ensure_ascii=False)}. "
            "Correct those failures without weakening any gate."
        )
    return _prompt(
        f"{prompt_text.rstrip()}\n\nGenerate exactly one {expected_id} value_angle brief. It must be distinct from protected episodes. Stable rejection codes: empty_evidence, claim_cluster_mismatch, cluster_overlap_high, claim_overlap_high, same_reader_transformation, same_life_expression, codex_semantic_not_distinct.{repair_instruction}",
        {
            "value": value.model_dump(mode="json"),
            "protected_briefs": [item.model_dump(mode="json") for item in protected],
            "rejected_attempt_count": len(rejected),
        },
    )


def _semantic_prompt(
    candidate: EpisodeBrief,
    protected: list[EpisodeBrief],
    prompt_text: str,
) -> str:
    return _prompt(
        f"{prompt_text.rstrip()}\n\nCompare the candidate with protected briefs. Return distinct only when their reader value and final takeaway are semantically different.",
        {"candidate": candidate.model_dump(mode="json"), "protected_briefs": [item.model_dump(mode="json") for item in protected]},
    )


def _grok_prompt(
    candidate: EpisodeBrief,
    protected: list[EpisodeBrief],
    prompt_text: str,
) -> str:
    return _prompt(
        f"{prompt_text.rstrip()}\n\nReview only reader-repeat risk in the structured briefs. Do not change evidence or locked fields.",
        {"candidate": candidate.model_dump(mode="json"), "protected_briefs": [item.model_dump(mode="json") for item in protected]},
    )


def _required_prompt(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TopicGenerationError("topic_prompt_missing")
    return value


def _validated_rejection_codes(rejected: list[dict[str, object]]) -> list[str]:
    codes: set[str] = set()
    for attempt in rejected:
        values = attempt.get("reason_codes", [])
        if not isinstance(values, list) or any(
            not isinstance(code, str) or code not in _TRUSTED_REJECTION_CODES
            for code in values
        ):
            raise TopicGenerationError("invalid_rejection_code")
        codes.update(values)
    return sorted(codes)
