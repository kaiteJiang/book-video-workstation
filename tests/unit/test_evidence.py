import pytest
from pydantic import ValidationError

from bv.content.evidence import (
    ChapterAnalysis,
    EvidenceCard,
    verify_evidence_excerpt,
)


def make_card(**changes: object) -> EvidenceCard:
    values: dict[str, object] = {
        "claim_id": "C01-001",
        "claim_type": "author_argument",
        "paraphrase": "A paraphrase",
        "evidence_excerpt": "The supported sentence.",
        "chapter_id": "chapter_001",
        "start_paragraph": 1,
        "end_paragraph": 1,
        "confidence": "high",
    }
    values.update(changes)
    return EvidenceCard.model_validate(values)


def test_fabricated_quote_is_rejected() -> None:
    card = make_card(evidence_excerpt="This sentence is absent")

    result = verify_evidence_excerpt(card, "The actual chapter text.")

    assert result.valid is False
    assert result.error_code == "evidence_excerpt_not_found"


def test_unicode_whitespace_and_punctuation_spacing_only_are_normalized() -> None:
    card = make_card(evidence_excerpt="Ａ supported\n sentence , yes ！")

    result = verify_evidence_excerpt(
        card,
        "A   supported sentence,yes!",
        expected_chapter_id="chapter_001",
    )

    assert result.valid is True
    assert result.error_code is None


def test_word_rewrite_is_rejected() -> None:
    card = make_card(evidence_excerpt="The supported claim.")

    result = verify_evidence_excerpt(card, "The supported sentence.")

    assert result.valid is False
    assert result.error_code == "evidence_excerpt_not_found"


def test_excerpt_outside_claimed_paragraph_range_is_rejected() -> None:
    card = make_card(
        evidence_excerpt="Only the second paragraph supports this.",
        start_paragraph=1,
        end_paragraph=1,
    )
    chapter = "First paragraph has no support.\n\nOnly the second paragraph supports this."

    result = verify_evidence_excerpt(card, chapter)

    assert result.valid is False
    assert result.error_code == "evidence_excerpt_not_found"


def test_single_newline_paragraphs_enforce_the_claimed_range() -> None:
    card = make_card(
        evidence_excerpt="Second EPUB paragraph.",
        start_paragraph=1,
        end_paragraph=1,
    )

    result = verify_evidence_excerpt(
        card,
        "First EPUB paragraph.\nSecond EPUB paragraph.",
    )

    assert result.valid is False
    assert result.error_code == "evidence_excerpt_not_found"


def test_wrong_chapter_id_is_rejected() -> None:
    card = make_card(chapter_id="chapter_002")

    result = verify_evidence_excerpt(
        card,
        "The supported sentence.",
        expected_chapter_id="chapter_001",
    )

    assert result.valid is False
    assert result.error_code == "evidence_chapter_mismatch"


@pytest.mark.parametrize(
    ("start_paragraph", "end_paragraph"),
    [(0, 1), (2, 1), (1, 3)],
)
def test_invalid_paragraph_range_is_rejected(
    start_paragraph: int,
    end_paragraph: int,
) -> None:
    card = make_card(
        start_paragraph=start_paragraph,
        end_paragraph=end_paragraph,
    )

    result = verify_evidence_excerpt(
        card,
        "The supported sentence.\n\nSecond paragraph.",
    )

    assert result.valid is False
    assert result.error_code == "evidence_range_invalid"


def test_empty_excerpt_is_rejected() -> None:
    card = make_card(evidence_excerpt=" \n\t ")

    result = verify_evidence_excerpt(card, "The supported sentence.")

    assert result.valid is False
    assert result.error_code == "evidence_excerpt_empty"


@pytest.mark.parametrize(
    "claim_type",
    [
        "author_argument",
        "narrator_statement",
        "character_view",
        "translator_note",
        "editor_material",
        "derived_summary",
    ],
)
def test_all_approved_claim_types_are_accepted(claim_type: str) -> None:
    card = make_card(claim_type=claim_type)

    assert card.claim_type == claim_type


def test_unapproved_claim_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_card(claim_type="invented_claim")


def test_evidence_schema_rejects_model_supplied_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_card(output_path="outside.json")


def test_chapter_analysis_schema_rejects_model_supplied_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ChapterAnalysis.model_validate(
            {
                "chapter_id": "chapter_001",
                "cards": [],
                "analysis_sha256": "model-controlled",
            }
        )
