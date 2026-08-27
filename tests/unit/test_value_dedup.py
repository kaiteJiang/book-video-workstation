from bv.content.dedup import deterministic_dedup, normalize_life_expression
from bv.content.topics import EpisodeBrief


def _brief(
    episode_id: str,
    *,
    clusters: list[str] | None = None,
    claims: list[str] | None = None,
    target_reader: str = "经常为了评价而犹豫的成年人",
    reader_before: str = "把他人的评价当成选择是否正确的证明",
    reader_after: str = "能区分自己的选择和他人的评价",
    tension: str = "想按自己的判断生活又害怕不被认可",
    practical_value: str = "提供重新理解选择和关系的视角",
    life_connection: str = "从反复解释和迎合转向为自己的选择承担责任",
    status: str = "reserved",
) -> EpisodeBrief:
    return EpisodeBrief(
        episode_id=episode_id,
        episode_kind="whole_book_value" if episode_id == "E001" else "value_angle",
        topic_name=f"{episode_id} 的读者价值",
        book_value_thesis="这本书帮助读者重看选择、关系与责任的边界",
        target_reader=target_reader,
        reader_before=reader_before,
        reader_after=reader_after,
        central_life_tension=tension,
        supporting_claim_cluster_ids=clusters if clusters is not None else ["problem", "reframe", "application"],
        source_claim_ids=claims if claims is not None else ["C01-001", "C02-001", "C03-001"],
        life_connection=life_connection,
        practical_value=practical_value,
        reading_reason="视频只呈现价值主线，完整论证和边界需要回到原书",
        constructed_scene=True,
        status=status,
    )


def test_dedup_rejects_same_reader_transformation_before_semantic_review() -> None:
    old = _brief("E001")
    new = _brief(
        "E002",
        clusters=["other-problem", "other-reframe"],
        claims=["C04-001", "C05-001"],
        life_connection="从沉默回避转向说清自己的期待",
    )

    decision = deterministic_dedup(new, [old])

    assert decision.accepted is False
    assert decision.reasons == ["same_reader_transformation"]
    assert decision.matched_episode_ids == ["E001"]
    assert decision.semantic_label is None


def test_dedup_rejects_cluster_and_claim_jaccard_at_half() -> None:
    old = _brief("E002", clusters=["a", "b", "d"], claims=["C01", "C02", "C04"])
    new = _brief(
        "E003",
        clusters=["a", "b", "c"],
        claims=["C01", "C02", "C03"],
        target_reader="正在经历关系摩擦的成年人",
        reader_before="把沉默当成避免冲突的办法",
        reader_after="能用清楚表达代替回避",
        tension="想维持关系又害怕冲突",
        practical_value="帮助判断何时需要表达自己的边界",
        life_connection="从吞下委屈转向提出具体请求",
    )

    decision = deterministic_dedup(new, [old])

    assert decision.accepted is False
    assert decision.reasons == ["claim_overlap_high", "cluster_overlap_high"]
    assert decision.claim_overlap == 0.5
    assert decision.cluster_overlap == 0.5


def test_dedup_accepts_below_threshold_structural_difference_for_semantic_review() -> None:
    old = _brief("E002", clusters=["a", "b", "c"], claims=["C01", "C02", "C03"])
    new = _brief(
        "E002",
        clusters=["a", "d", "e"],
        claims=["C01", "C04", "C05"],
        target_reader="在工作中回避承担责任的成年人",
        reader_before="把拖延当成降低风险的办法",
        reader_after="能把可承担的行动拆成具体步骤",
        tension="想避免失败又希望推进重要选择",
        practical_value="帮助把抽象焦虑转成可承担的行动",
        life_connection="从反复等待确定感转向完成下一步可控动作",
    )

    decision = deterministic_dedup(new, [old])

    assert decision.accepted is True
    assert decision.reasons == []
    assert decision.claim_overlap == 0.2
    assert decision.cluster_overlap == 0.2


def test_dedup_rejects_normalized_identical_life_expression() -> None:
    old = _brief("E001", life_connection="从反复解释和迎合，转向为自己的选择承担责任。")
    new = _brief(
        "E002",
        clusters=["other-a", "other-b"],
        claims=["C04", "C05"],
        target_reader="正在处理职场冲突的成年人",
        reader_before="把沉默当成唯一的安全策略",
        reader_after="能提出清楚的工作边界",
        tension="想合作又怕被持续消耗",
        practical_value="帮助判断协作中的责任边界",
        life_connection=" 从反复解释和迎合 转向为自己的选择承担责任！ ",
    )

    decision = deterministic_dedup(new, [old])

    assert decision.accepted is False
    assert decision.reasons == ["same_life_expression"]
    assert normalize_life_expression(new.life_connection) == normalize_life_expression(
        old.life_connection
    )


def test_dedup_rejects_empty_evidence_sets_instead_of_calling_them_distinct() -> None:
    old = _brief("E001")
    new = _brief(
        "E002",
        clusters=[],
        claims=[],
        target_reader="正在学习表达需要的成年人",
        reader_before="不知道怎样说明自己的需要",
        reader_after="能说出可讨论的具体请求",
        tension="想被理解又害怕表达",
        practical_value="帮助把模糊委屈转成可以沟通的问题",
        life_connection="从猜测他人转向提出自己的需要",
    )

    decision = deterministic_dedup(new, [old])

    assert decision.accepted is False
    assert decision.reasons == ["empty_evidence"]
