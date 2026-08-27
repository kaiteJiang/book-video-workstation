from __future__ import annotations

from typing import Protocol


_START = "<!-- BV:VOICEOVER:START -->"
_END = "<!-- BV:VOICEOVER:END -->"


class _ReviewablePackage(Protocol):
    semantic_lock: object
    voiceover: str
    validation: object
    grok_status: str
    adjudication: object
    fact_diff: object


def build_script_review(package: _ReviewablePackage) -> str:
    """Render a deterministic human-review artifact without private sources or model output."""
    lock = package.semantic_lock
    validation = package.validation
    coverage = getattr(package, "coverage", None)
    if coverage is None:
        coverage = type("Coverage", (), {
            "thesis_span": lock.value_thesis, "reader_after_span": lock.reader_after,
            "life_connection_span": lock.life_connection,
            "boundary_span": lock.practical_boundary, "reading_reason_span": lock.reading_reason,
            "cluster_spans": {},
        })()
    lines = [
        "# E001 全书价值口播审阅", "", f"- 全书价值：{lock.value_thesis}",
        f"- 目标读者：{lock.target_reader}", f"- 读前状态：{lock.reader_before}",
        f"- 读后变化：{lock.reader_after}", f"- 生活张力：{lock.central_life_tension}",
        f"- 回到生活：{lock.life_connection}",
        f"- 实践边界：{lock.practical_boundary}", f"- 阅读理由：{lock.reading_reason}", "",
        "## 观点簇追溯", "",
    ]
    for cluster_id in lock.required_cluster_ids:
        claims = ", ".join(lock.allowed_claim_ids_by_cluster[cluster_id])
        regions = ", ".join(lock.chapter_regions_by_cluster[cluster_id])
        span = coverage.cluster_spans.get(cluster_id, "")
        lines.append(f"- {cluster_id}：claims={claims}；regions={regions}；覆盖={span}")
    lines.extend([
        "", "## 精确文本覆盖", "",
        f"- 价值：{coverage.thesis_span}", f"- 读后变化：{coverage.reader_after_span}",
        f"- 回到生活：{coverage.life_connection_span}",
        f"- 边界：{coverage.boundary_span}", f"- 阅读理由：{coverage.reading_reason_span}", "",
        "## 审阅状态", "", f"- Grok：{package.grok_status}",
        f"- Codex 接受建议：{', '.join(package.adjudication.accepted_suggestions) or '无'}",
        f"- Codex 拒绝建议：{', '.join(package.adjudication.rejected_suggestions) or '无'}",
        f"- 风险：{', '.join(package.fact_diff.violations) or '无'}",
        f"- 非空白字符数：{validation.character_count}",
        f"- 长度提醒：{', '.join(validation.warnings) or '无'}",
        "- 估计时长：仅供审阅参考，未渲染音频，非权威时长。", "",
        _START + package.voiceover + _END, "",
    ])
    return "\n".join(lines)


def extract_review_voiceover(review: str) -> str:
    if not isinstance(review, str):
        raise ValueError("invalid_review_markers")
    start_count, end_count = review.count(_START), review.count(_END)
    if start_count != 1 or end_count != 1:
        raise ValueError("invalid_review_markers")
    start = review.find(_START)
    end = review.find(_END)
    if start < 0 or end < 0 or end <= start + len(_START):
        raise ValueError("invalid_review_markers")
    value = review[start + len(_START):end]
    if not value.strip() or _START in value or _END in value:
        raise ValueError("invalid_review_markers")
    return value
