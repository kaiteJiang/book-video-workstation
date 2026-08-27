import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bv.cli import app
from bv.state.models import BookState, EpisodeState
from bv.state.store import StateStore
from bv.workflow.orchestrator import WorkflowView
from bv.workflow.runtime import (
    ExternalAuthorizationError,
    RuntimeAuthorization,
    invoke_external,
)


runner = CliRunner()

_READONLY_COMMANDS = ("status", "open", "import-video", "retry", "add")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _write_human_writing_skill(skill_root: Path) -> Path:
    assets = {
        "SKILL.md": "skill",
        "references/reality.md": "reality",
        "references/formats.md": "formats",
        "references/revision.md": "revision",
        "scripts/check_prose.py": "checker",
    }
    for relative, content in assets.items():
        path = skill_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return skill_root / "SKILL.md"


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


class _RecordingWorkflow:
    def next(self, book_id: str, episode_id: str = "E001") -> WorkflowView:
        del book_id, episode_id
        return WorkflowView(status="source_imported", next_command="bv next book-demo")

    def approve_gate(self, book_id: str, episode_id: str, gate: str) -> WorkflowView:
        del book_id, episode_id, gate
        return WorkflowView(status="script_approved", next_command="bv next book-demo")


def test_runtime_authorization_defaults_to_denied_and_is_frozen() -> None:
    authorization = RuntimeAuthorization()

    assert authorization.allow_external is False
    with pytest.raises(FrozenInstanceError):
        authorization.allow_external = True  # type: ignore[misc]


def test_runtime_bindings_cover_the_content_stages_through_script_approval(
    tmp_path: Path,
) -> None:
    """The CMD runtime must not stop at ``stage_not_configured`` before review."""
    from bv.config import AppConfig
    from bv.workflow.runtime import build_runtime_bindings

    bindings = build_runtime_bindings(
        AppConfig(workspace_dir=tmp_path / "workspace"),
        RuntimeAuthorization(),
    )

    assert set(bindings.stages) >= {
        "parse_source",
        "analyze_chapters",
        "synthesize_book_value",
        "generate_episode_brief",
        "draft_script",
    }
    assert set(bindings.gate_approvers) >= {"script"}


def test_content_runtime_uses_complete_human_writing_bundle_identity(
    tmp_path: Path,
) -> None:
    """Changing a required supporting file must invalidate the stage identity."""
    from bv.content.human_writing import combined_skill_sha256
    import bv.workflow.content_runtime as content_runtime

    skill_root = tmp_path / "human-writing"
    skill_path = _write_human_writing_skill(skill_root)

    assert hasattr(content_runtime, "_human_writing_skill_sha256")
    identity = content_runtime._human_writing_skill_sha256(  # type: ignore[attr-defined]
        skill_path
    )
    assert identity == combined_skill_sha256(skill_root)

    (skill_root / "references" / "revision.md").write_text(
        "revised", encoding="utf-8"
    )
    assert (
        content_runtime._human_writing_skill_sha256(  # type: ignore[attr-defined]
            skill_path
        )
        != identity
    )


def test_script_draft_stage_detects_supporting_skill_change_during_generation(
    tmp_path: Path,
) -> None:
    from bv.content.synthesis import ClaimCluster, WholeBookValue
    from bv.content.topics import EpisodeBrief
    from bv.workflow.content_runtime import ContentRuntimeError, ScriptDraftStage

    store = StateStore(tmp_path / "workspace")
    episode_root = (
        store.root / "books" / "book-demo" / "episodes" / "E001"
    )
    analysis = episode_root / "analysis"
    topic = episode_root / "topic"
    analysis.mkdir(parents=True)
    topic.mkdir(parents=True)
    value = WholeBookValue(
        value_thesis="整本书帮助读者重新看见生活仍可继续。",
        target_reader="正在经历长期低谷的读者。",
        reader_before="注意力被眼前的失去占满。",
        reader_after="读者开始留意仍能做出的选择。",
        central_life_tension="事实无法逆转，人仍要继续生活。",
        supporting_claim_clusters=[
            ClaimCluster(
                cluster_id="cluster_problem",
                summary="困境会把注意力压缩到眼前的失去。",
                narrative_role="problem",
                claim_ids=["claim_001"],
                chapter_regions=["early"],
            ),
            ClaimCluster(
                cluster_id="cluster_reframe",
                summary="重新看见关系与行动，生活才有继续的可能。",
                narrative_role="reframe",
                claim_ids=["claim_002"],
                chapter_regions=["middle"],
            ),
        ],
        practical_value="这种理解不能替代现实行动。",
        reading_reason="原书保留了完整经历，值得回去阅读。",
        coverage_exception=None,
    )
    (analysis / "book_value_card.json").write_text(
        value.model_dump_json(), encoding="utf-8"
    )
    brief = EpisodeBrief(
        episode_id="E001",
        episode_kind="whole_book_value",
        topic_name="重新看见生活",
        book_value_thesis=value.value_thesis,
        target_reader=value.target_reader,
        reader_before=value.reader_before,
        reader_after=value.reader_after,
        central_life_tension=value.central_life_tension,
        supporting_claim_cluster_ids=["cluster_problem", "cluster_reframe"],
        source_claim_ids=["claim_001", "claim_002"],
        life_connection="在低谷里先辨认眼前还能做什么。",
        practical_value=value.practical_value,
        reading_reason=value.reading_reason,
        constructed_scene=True,
        status="reserved",
    )
    (topic / "episode_brief.json").write_text(
        brief.model_dump_json(), encoding="utf-8"
    )
    skill_path = _write_human_writing_skill(tmp_path / "human-writing")

    stage: ScriptDraftStage

    class MutatingService:
        prompt_hashes: dict[str, str]

        def __init__(self) -> None:
            self.prompt_hashes = dict(stage._prompt_hashes)

        def create(self, episode: EpisodeBrief, current_value: WholeBookValue):
            del episode, current_value
            (skill_path.parent / "references" / "revision.md").write_text(
                "changed during generation", encoding="utf-8"
            )
            return object()

    stage = ScriptDraftStage(
        store,
        authorization=RuntimeAuthorization(allow_external=True),
        service_factory=lambda _root: MutatingService(),
        human_writing_skill_path=skill_path,
    )

    context = StageContext(
        book_id="book-demo",
        episode_id="E001",
        episode_root=episode_root,
        episode_state=EpisodeState(book_id="book-demo", episode_id="E001"),
    )
    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(context)

    assert caught.value.error_code == "human_writing_skill_changed"


@pytest.mark.parametrize(
    ("mutate_supporting_file", "expected_error", "expected_approval_calls"),
    [
        (False, "script_approval_failed", 1),
        (True, "human_writing_skill_changed", 0),
    ],
)
def test_script_approval_stage_uses_bundle_identity_before_downstream_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate_supporting_file: bool,
    expected_error: str,
    expected_approval_calls: int,
) -> None:
    from types import SimpleNamespace

    from bv.content.human_writing import combined_skill_sha256
    import bv.workflow.content_runtime as content_runtime
    from bv.workflow.content_runtime import ContentRuntimeError, ScriptApprovalStage

    store = StateStore(tmp_path / "workspace")
    episode_root = (
        store.root / "books" / "book-demo" / "episodes" / "E001"
    )
    script_root = episode_root / "script"
    script_root.mkdir(parents=True)
    (script_root / "script_package.json").write_text("{}", encoding="utf-8")
    (script_root / "review.md").write_text("review", encoding="utf-8")
    skill_path = _write_human_writing_skill(tmp_path / "human-writing")
    stage = ScriptApprovalStage(
        store,
        topic_service_factory=lambda _root: object(),
        human_writing_skill_path=skill_path,
    )
    package = SimpleNamespace(
        voiceover="reviewed text",
        semantic_lock=object(),
        fact_diff=object(),
        human_writing_checker_passed=True,
        original_draft=object(),
        content_hashes={
            f"prompt_{name}": digest
            for name, digest in stage._prompt_hashes.items()
        },
        human_writing_sha256=combined_skill_sha256(skill_path.parent),
    )
    monkeypatch.setattr(
        content_runtime.ScriptPackage,
        "model_validate_json",
        classmethod(lambda _cls, _raw: package),
    )
    monkeypatch.setattr(
        content_runtime, "extract_review_voiceover", lambda _text: package.voiceover
    )
    monkeypatch.setattr(
        content_runtime, "_base_script_package_is_anchored", lambda *_args: True
    )
    approval_calls: list[str] = []

    def fake_approve_script(*args, **kwargs):
        del args, kwargs
        approval_calls.append("called")
        raise RuntimeError("downstream sentinel")

    monkeypatch.setattr(content_runtime, "approve_script", fake_approve_script)
    if mutate_supporting_file:
        (skill_path.parent / "references" / "revision.md").write_text(
            "changed before approval", encoding="utf-8"
        )

    context = StageContext(
        book_id="book-demo",
        episode_id="E001",
        episode_root=episode_root,
        episode_state=EpisodeState(book_id="book-demo", episode_id="E001"),
    )
    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(context)

    assert caught.value.error_code == expected_error
    assert len(approval_calls) == expected_approval_calls


def test_script_approval_stage_routes_a_bound_reviewed_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import bv.workflow.content_runtime as content_runtime
    from bv.content.human_writing import combined_skill_sha256
    from bv.workflow.content_runtime import ContentRuntimeError, ScriptApprovalStage
    from bv.workflow.stages import StageContext

    store = StateStore(tmp_path / "workspace")
    episode_root = store.root / "books" / "book-demo" / "episodes" / "E001"
    script_root = episode_root / "script"
    script_root.mkdir(parents=True)
    (script_root / "script_package.json").write_text("{}", encoding="utf-8")
    (script_root / "review.md").write_text("review", encoding="utf-8")
    (episode_root / ".private" / "requests").mkdir(parents=True)
    skill_path = _write_human_writing_skill(tmp_path / "human-writing")
    stage = ScriptApprovalStage(
        store,
        topic_service_factory=lambda _root: object(),
        human_writing_skill_path=skill_path,
    )
    revised_text = "人工审阅后的口播"
    reviewed_draft = SimpleNamespace(recommended_voiceover=revised_text)
    revision_fact_diff = object()
    revision_manifest = SimpleNamespace(
        prompt_sha256={
            name: digest for name, digest in stage._prompt_hashes.items()
        },
        human_writing_sha256=combined_skill_sha256(skill_path.parent),
    )
    package = SimpleNamespace(
        voiceover="原自动稿",
        semantic_lock=object(),
        fact_diff=object(),
        human_writing_checker_passed=True,
        original_draft=object(),
        content_hashes={
            **{f"prompt_{name}": digest for name, digest in stage._prompt_hashes.items()},
        },
        human_writing_sha256=combined_skill_sha256(skill_path.parent),
    )
    monkeypatch.setattr(
        content_runtime.ScriptPackage,
        "model_validate_json",
        classmethod(lambda _cls, _raw: package),
    )
    monkeypatch.setattr(
        content_runtime, "extract_review_voiceover", lambda _text: revised_text
    )
    monkeypatch.setattr(
        content_runtime, "_base_script_package_is_anchored", lambda *_args: True
    )
    monkeypatch.setattr(
        content_runtime,
        "_load_prepared_script_revision",
        lambda **_kwargs: (reviewed_draft, revision_fact_diff, revision_manifest),
    )
    checker_calls: list[str] = []
    monkeypatch.setattr(
        content_runtime,
        "check_human_writing_candidate",
        lambda **kwargs: checker_calls.append(kwargs["voiceover"])
        or package.human_writing_sha256,
    )
    captured: dict[str, object] = {}

    def fake_approve_script(voiceover, _lock, fact_diff, *_args, **kwargs):
        captured.update(
            voiceover=voiceover,
            fact_diff=fact_diff,
            human_revision=kwargs["human_revision"],
            revision_draft=kwargs["revision_draft"],
        )
        raise RuntimeError("downstream sentinel")

    monkeypatch.setattr(content_runtime, "approve_script", fake_approve_script)
    context = StageContext(
        book_id="book-demo",
        episode_id="E001",
        episode_root=episode_root,
        episode_state=EpisodeState(book_id="book-demo", episode_id="E001"),
    )

    with pytest.raises(ContentRuntimeError, match="script_approval_failed"):
        stage.run(context)

    assert checker_calls == [revised_text]
    assert captured == {
        "voiceover": revised_text,
        "fact_diff": revision_fact_diff,
        "human_revision": True,
        "revision_draft": reviewed_draft,
    }


def test_review_edit_can_be_prepared_and_approved_from_real_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import json

    from bv.content.human_writing import combined_skill_sha256
    from bv.content.reviews import build_script_review
    from bv.content.topics import TopicService
    from bv.core.atomic import atomic_write_json
    from bv.core.hashing import sha256_file
    from bv.state.models import ArtifactRef, StageManifest
    from bv.workflow import content_runtime
    from bv.workflow.content_runtime import (
        ScriptApprovalStage,
        prepare_script_revision,
    )
    from bv.workflow.stages import StageContext
    from test_value_scripts import (
        _FakeHumanWriting,
        _FakeModel,
        _adjudication,
        _brief,
        _diff,
        _draft,
        _human_revision_draft,
        _service,
        _value,
    )

    skill_path = _write_human_writing_skill(tmp_path / "human-writing")
    skill_hash = combined_skill_sha256(skill_path.parent)
    value = _value()
    brief = _brief(value)
    service, *_ = _service(
        tmp_path,
        [_draft(), _adjudication(), _diff()],
        human_writing=_FakeHumanWriting(skill_sha256=skill_hash),
    )
    package = service.create(brief, value)
    revision = _human_revision_draft()

    store = StateStore(tmp_path / "workspace")
    episode_root = store.root / "books" / "book" / "episodes" / "E001"
    script_root = episode_root / "script"
    package_path = script_root / "script_package.json"
    review_path = script_root / "review.md"
    atomic_write_json(package_path, package.model_dump(mode="json"))
    review_path.write_text(
        build_script_review(package).replace(
            package.voiceover, revision.recommended_voiceover
        ),
        encoding="utf-8",
    )
    package_ref = ArtifactRef(
        path=str(package_path),
        sha256=sha256_file(package_path),
        size_bytes=package_path.stat().st_size,
    )
    episode = EpisodeState(
        book_id="book",
        episode_id="E001",
        status="awaiting_script_review",
        completed_stages=["draft_script"],
        stage_manifests={
            "draft_script": StageManifest(
                stage="draft_script",
                status="completed",
                inputs={},
                outputs={"script_package": package_ref},
                config_sha256="0" * 64,
            )
        },
    )
    store.save_episode(episode)

    prompt_text = {
        "draft": "DRAFT_ASSET_MARKER: draft the value voiceover.",
        "fact_diff": "DIFF_ASSET_MARKER: compare claims and scope.",
        "grok": "GROK_ASSET_MARKER: critique reader clarity only.",
    }
    prompt_paths: dict[str, Path] = {}
    for name, text in prompt_text.items():
        path = tmp_path / "prompts" / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        prompt_paths[name] = path

    monkeypatch.setattr(
        content_runtime,
        "check_human_writing_candidate",
        lambda **_kwargs: skill_hash,
    )
    prepare_script_revision(
        store=store,
        book_id="book",
        episode_id="E001",
        reviewed_draft=revision,
        fact_diff=_diff(
            script_sha256=hashlib.sha256(
                revision.recommended_voiceover.encode("utf-8")
            ).hexdigest()
        ),
        human_writing_skill_path=skill_path,
        prompt_paths=prompt_paths,
    )
    topic = TopicService(
        store.root / "books" / "book",
        model=_FakeModel([brief]),
        topic_prompt_text="topic",
        dedup_prompt_text="dedup",
        grok_prompt_text="grok",
    )
    topic.generate_first_topic(value)
    stage = ScriptApprovalStage(
        store,
        topic_service_factory=lambda _root: topic,
        human_writing_skill_path=skill_path,
        draft_prompt_path=prompt_paths["draft"],
        fact_diff_prompt_path=prompt_paths["fact_diff"],
        grok_prompt_path=prompt_paths["grok"],
    )
    current = store.load_episode("book", "E001")
    outcome = stage.run(
        StageContext(
            book_id="book",
            episode_id="E001",
            episode_root=episode_root,
            episode_state=current,
        )
    )

    assert outcome.outputs["approved_script"].read_text(encoding="utf-8") == revision.recommended_voiceover
    manifest = json.loads(outcome.outputs["script_manifest"].read_text(encoding="utf-8"))
    assert manifest["human_revision"] is True
    assert manifest["reviewed_draft_sha256"]


def test_self_rehashed_package_mutation_is_rejected_by_draft_manifest_anchor(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    from bv.core.hashing import sha256_file
    from bv.state.models import ArtifactRef, StageManifest
    from bv.workflow.content_runtime import _base_script_package_is_anchored

    package_path = tmp_path / "script_package.json"
    package_path.write_text('{"voiceover":"original"}', encoding="utf-8")
    original = ArtifactRef(
        path=str(package_path),
        sha256=sha256_file(package_path),
        size_bytes=package_path.stat().st_size,
    )
    episode = EpisodeState(
        book_id="book",
        episode_id="E001",
        stage_manifests={
            "draft_script": StageManifest(
                stage="draft_script",
                status="completed",
                inputs={},
                outputs={"script_package": original},
                config_sha256="0" * 64,
            )
        },
    )
    mutated = {
        "voiceover": "forged",
        "content_hashes": {
            "voiceover": hashlib.sha256(b"forged").hexdigest()
        },
    }
    package_path.write_text(
        json.dumps(mutated, sort_keys=True), encoding="utf-8"
    )

    assert _base_script_package_is_anchored(episode, package_path) is False


@pytest.mark.parametrize(
    ("book_id", "episode_id"),
    [("..", "E001"), ("book/escape", "E001"), ("book", "..\\escape")],
)
def test_prepare_script_revision_rejects_unsafe_path_segments_before_file_access(
    tmp_path: Path,
    book_id: str,
    episode_id: str,
) -> None:
    from bv.workflow.content_runtime import (
        ContentRuntimeError,
        prepare_script_revision,
    )
    from test_value_scripts import _diff, _human_revision_draft

    with pytest.raises(ContentRuntimeError, match="parse_context_unsafe"):
        prepare_script_revision(
            store=StateStore(tmp_path / "workspace"),
            book_id=book_id,
            episode_id=episode_id,
            reviewed_draft=_human_revision_draft(),
            fact_diff=_diff(),
            human_writing_skill_path=tmp_path / "missing" / "SKILL.md",
        )


def test_runtime_stops_at_first_external_stage_after_local_title_snapshot(
    tmp_path: Path,
) -> None:
    from bv.books.service import create_book_from_metadata
    from bv.cli import build_orchestrator
    from bv.config import AppConfig
    from bv.workflow.orchestrator import WorkflowError

    store = StateStore(tmp_path / "workspace")
    receipt = create_book_from_metadata("示例书", ["示例作者"], store)
    workflow = build_orchestrator(
        AppConfig(workspace_dir=store.root), RuntimeAuthorization()
    )

    with pytest.raises(WorkflowError, match="external_not_authorized"):
        workflow.next(receipt.book_id)

    episode = store.load_episode(receipt.book_id, "E001")
    assert episode.failure_summary == "external_not_authorized"
    assert "parse_source" in episode.completed_stages
    assert "analyze_chapters" not in episode.completed_stages
    assert (store.root / "books" / receipt.book_id / "episodes" / "E001" / "analysis" / "title_metadata.json").is_file()


def test_external_authorization_error_has_stable_public_code() -> None:
    error = ExternalAuthorizationError()

    assert error.error_code == "external_not_authorized"
    assert str(error) == "external_not_authorized"
    assert error.args == ("external_not_authorized",)


def test_authorization_guard_raises_before_external_callable() -> None:
    calls: list[str] = []

    def fake_external_model() -> str:
        calls.append("invoked")
        return "private-prompt\\secret\\path"

    with pytest.raises(ExternalAuthorizationError) as caught:
        invoke_external(RuntimeAuthorization(allow_external=False), fake_external_model)

    assert calls == []
    assert caught.value.error_code == "external_not_authorized"
    assert "private-prompt" not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert "\\" not in str(caught.value)


def test_next_and_approve_pass_fresh_current_command_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bv import cli

    captured: list[RuntimeAuthorization | None] = []
    workflow = _RecordingWorkflow()

    def fake_build(config, authorization=None):
        del config
        captured.append(authorization)
        return workflow

    monkeypatch.setattr(cli, "build_orchestrator", fake_build)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("workspace_dir: workspace\n", encoding="utf-8")

    default_next = runner.invoke(app, ["next", "book-demo"])
    allowed_next = runner.invoke(app, ["next", "book-demo", "E001", "--allow-external"])
    default_approve = runner.invoke(app, ["approve", "book-demo", "E001", "script"])
    allowed_approve = runner.invoke(
        app, ["approve", "book-demo", "E001", "script", "--allow-external"]
    )

    assert default_next.exit_code == allowed_next.exit_code == 0
    assert default_approve.exit_code == allowed_approve.exit_code == 0
    assert [auth.allow_external for auth in captured] == [False, True, False, True]
    assert all(isinstance(auth, RuntimeAuthorization) for auth in captured)
    assert captured[0] is not captured[1]
    assert captured[2] is not captured[3]
    assert "--allow-external" in _plain(runner.invoke(app, ["next", "--help"]).stdout)
    assert "--allow-external" in _plain(runner.invoke(app, ["approve", "--help"]).stdout)


def test_allow_external_default_is_false_and_is_not_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("workspace_dir: workspace\n", encoding="utf-8")
    store = StateStore(tmp_path / "workspace")
    (store.root / "books" / "book-demo" / "episodes" / "E001").mkdir(parents=True)
    store.save_book(BookState(book_id="book-demo", title="Demo"))
    store.save_episode(EpisodeState(book_id="book-demo", episode_id="E001"))
    original_config = config_path.read_text(encoding="utf-8")

    denied = runner.invoke(app, ["next", "book-demo"])
    allowed = runner.invoke(app, ["next", "book-demo", "--allow-external"])

    assert "No such option: --allow-external" not in (allowed.stdout + (allowed.stderr or ""))
    assert denied.exit_code in {0, 1}
    assert allowed.exit_code in {0, 1}
    assert config_path.read_text(encoding="utf-8") == original_config
    assert "allow_external" not in original_config
    assert "allow-external" not in original_config
    for path in (tmp_path / "workspace").rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "allow_external" not in text
        assert "allow-external" not in text


@pytest.mark.parametrize("command", _READONLY_COMMANDS)
def test_other_commands_do_not_accept_allow_external(command: str) -> None:
    help_result = runner.invoke(app, [command, "--help"])
    help_text = _plain(help_result.stdout) + _plain(help_result.stderr or "")

    assert help_result.exit_code == 0
    assert "--allow-external" not in help_text

# --- Task 6B1a: TitleOrFileParseStage (local title/file parse, no bindings) ---

import hashlib
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from bv.core.hashing import sha256_file
from bv.state.models import BookState
from bv.state.store import StateStore
from bv.workflow.content_runtime import (
    ContentRuntimeError,
    TitleMetadataSnapshot,
    TitleOrFileParseStage,
)
from bv.workflow.stages import StageContext

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _identity_sha256(book_id: str, title: str, authors: list[str], source_mode: str) -> str:
    payload = {
        "authors": authors,
        "book_id": book_id,
        "source_mode": source_mode,
        "title": title,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_book_json(store_root: Path, book: BookState, *, dir_book_id: str | None = None) -> Path:
    book_id = dir_book_id or book.book_id
    book_dir = store_root / "books" / book_id
    book_dir.mkdir(parents=True, exist_ok=True)
    path = book_dir / "book.json"
    path.write_text(book.model_dump_json(), encoding="utf-8")
    return path


def _episode_root(store_root: Path, book_id: str, episode_id: str) -> Path:
    root = store_root / "books" / book_id / "episodes" / episode_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _context(book_id: str, episode_id: str, episode_root: Path) -> StageContext:
    return StageContext(book_id=book_id, episode_id=episode_id, episode_root=episode_root)


def _assert_input_hashes(outcome, *paths: Path) -> None:
    assert outcome.inputs
    assert all(_SHA256_RE.fullmatch(value) for value in outcome.inputs.values())
    actual = set(outcome.inputs.values())
    for path in paths:
        assert sha256_file(path) in actual


def _write_source_file(store_root: Path, book_id: str, payload: bytes, suffix: str = ".txt") -> Path:
    source_dir = store_root / "books" / book_id / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    scratch = source_dir / f"_scratch{suffix}"
    scratch.write_bytes(payload)
    digest = sha256_file(scratch)
    dest = source_dir / f"{digest}{suffix}"
    scratch.replace(dest)
    return dest


_ORCHESTRATOR_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_PUBLIC_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def _chapter_index_item(
    chapter_id: str,
    title: str,
    *,
    sha256: str | None = None,
) -> dict[str, object]:
    digest = sha256 if sha256 is not None else hashlib.sha256(
        f"{chapter_id}\n{title}".encode("utf-8")
    ).hexdigest()
    return {
        "chapter_id": chapter_id,
        "title": title,
        "sha256": digest,
        "anchor": {
            "source_type": "txt",
            "source_path": "book.txt",
            "chapter_id": chapter_id,
            "start_line": 1,
            "end_line": 1,
        },
    }


def _persist_ok(
    episode_root: Path, *, chapters: list[dict[str, object]] | None = None
) -> str:
    items = chapters if chapters is not None else [_chapter_index_item("ch-001", "One")]
    analysis = episode_root / "analysis"
    chapter_dir = episode_root / "chapters"
    analysis.mkdir(parents=True, exist_ok=True)
    chapter_dir.mkdir(parents=True, exist_ok=True)
    (analysis / "source_quality.json").write_text('{"status":"ok"}', encoding="utf-8")
    (chapter_dir / "index.json").write_text(
        json.dumps({"chapters": items}),
        encoding="utf-8",
    )
    for index, item in enumerate(items, start=1):
        name = f"{index:03d}.md"
        title = item.get("title", name)
        (chapter_dir / name).write_text(f"# {title}\ntext\n", encoding="utf-8")
    return "source_parsed"


def _assert_identifier_keys(outcome) -> None:
    for key in (*outcome.inputs, *outcome.outputs):
        assert _ORCHESTRATOR_IDENTIFIER.fullmatch(key), key


def test_content_runtime_error_stores_and_stringifies_only_code() -> None:
    err = ContentRuntimeError("source_missing")
    assert err.error_code == "source_missing"
    assert str(err) == "source_missing"
    assert f"{err}" == "source_missing"


def test_content_runtime_error_rejects_invalid_public_codes() -> None:
    invalid = (
        "",
        "Source_missing",
        "source-missing",
        "1abc",
        "a" * 65,
        "has space",
        "book.json",
        "source.missing",
        "_leading_underscore",
    )
    for code in invalid:
        with pytest.raises(ValueError):
            ContentRuntimeError(code)
    assert _PUBLIC_ERROR_CODE.fullmatch("source_missing")
    assert ContentRuntimeError("a" * 64).error_code == "a" * 64


def test_title_metadata_snapshot_is_frozen_strict_and_forbids_extra() -> None:
    snap = TitleMetadataSnapshot(
        book_id="book-1",
        title="Notes",
        authors=("Ada Lovelace",),
        source_mode="title_author",
        identity_sha256="a" * 64,
    )
    assert snap.authors == ("Ada Lovelace",)
    with pytest.raises(ValidationError):
        snap.title = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        TitleMetadataSnapshot(
            book_id="book-1",
            title="Notes",
            authors=("Ada Lovelace",),
            source_mode="title_author",
            identity_sha256="a" * 64,
            extra_field="nope",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        TitleMetadataSnapshot(
            book_id="book-1",
            title="Notes",
            authors=("Ada Lovelace",),
            source_mode="file",  # type: ignore[arg-type]
            identity_sha256="a" * 64,
        )


def test_title_metadata_is_exact_and_hash_valid(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book = BookState(
        book_id="book-1",
        title="三体",
        authors=["刘慈欣"],
        source_mode="title_author",
        source_ids=[],
    )
    book_json = _write_book_json(store_root, book)
    episode_id = "ep-1"
    episode_root = _episode_root(store_root, book.book_id, episode_id)

    def _boom(*_args, **_kwargs):
        raise AssertionError("title mode must not parse a file or call models")

    stage = TitleOrFileParseStage(store, parser=_boom, persister=_boom)
    outcome = stage.run(_context(book.book_id, episode_id, episode_root))

    meta_path = episode_root / "analysis" / "title_metadata.json"
    assert outcome.outputs["title_metadata"] == meta_path
    assert meta_path.is_file()
    snap = TitleMetadataSnapshot.model_validate_json(meta_path.read_text(encoding="utf-8"))
    expected_hash = _identity_sha256("book-1", "三体", ["刘慈欣"], "title_author")
    assert snap.book_id == "book-1"
    assert snap.title == "三体"
    assert snap.authors == ("刘慈欣",)
    assert snap.source_mode == "title_author"
    assert snap.identity_sha256 == expected_hash
    assert set(outcome.inputs) == {"book_state", "title_metadata"}
    assert outcome.inputs["book_state"] == sha256_file(book_json)
    assert outcome.inputs["title_metadata"] == sha256_file(meta_path)
    _assert_input_hashes(outcome, book_json, meta_path)
    _assert_identifier_keys(outcome)


def test_title_mode_rejects_blank_title_and_authors_and_source_ids(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    episode_id = "ep-1"

    cases = [
        BookState(
            book_id="book-blank-title",
            title="   ",
            authors=["Ada"],
            source_mode="title_author",
            source_ids=[],
        ),
        BookState(
            book_id="book-blank-authors",
            title="Notes",
            authors=["  "],
            source_mode="title_author",
            source_ids=[],
        ),
        BookState(
            book_id="book-has-source",
            title="Notes",
            authors=["Ada"],
            source_mode="title_author",
            source_ids=["abc"],
        ),
        BookState(
            book_id="book-padded-title",
            title=" Notes",
            authors=["Ada"],
            source_mode="title_author",
            source_ids=[],
        ),
        BookState(
            book_id="book-padded-author",
            title="Notes",
            authors=["Ada "],
            source_mode="title_author",
            source_ids=[],
        ),
        BookState(
            book_id="book-blank-author-kept",
            title="Notes",
            authors=["Ada", ""],
            source_mode="title_author",
            source_ids=[],
        ),
        BookState(
            book_id="book-empty-authors",
            title="Notes",
            authors=[],
            source_mode="title_author",
            source_ids=[],
        ),
    ]
    for book in cases:
        _write_book_json(store_root, book)
        episode_root = _episode_root(store_root, book.book_id, episode_id)
        stage = TitleOrFileParseStage(store)
        with pytest.raises(ContentRuntimeError) as caught:
            stage.run(_context(book.book_id, episode_id, episode_root))
        assert caught.value.error_code == "title_metadata_invalid"
        assert str(caught.value) == "title_metadata_invalid"


def test_file_mode_local_parse_persists_named_outputs_and_hashes(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book_id = "book-file"
    episode_id = "ep-file"
    source = _write_source_file(store_root, book_id, b"chapter one\n")
    book = BookState(
        book_id=book_id,
        title="Local File",
        authors=["Author"],
        source_mode="file",
        source_ids=[source.stem],
    )
    book_json = _write_book_json(store_root, book)
    episode_root = _episode_root(store_root, book_id, episode_id)
    parsed_holder: dict[str, object] = {}

    def fake_parser(path: Path):
        parsed_holder["path"] = path
        return {"parsed": True, "path": str(path)}

    def fake_persister(parsed, destination: Path):
        parsed_holder["parsed"] = parsed
        parsed_holder["destination"] = destination
        return _persist_ok(
            destination,
            chapters=[
                _chapter_index_item("ch-001", "One"),
                _chapter_index_item("ch-002", "Two"),
            ],
        )

    stage = TitleOrFileParseStage(store, parser=fake_parser, persister=fake_persister)
    outcome = stage.run(_context(book_id, episode_id, episode_root))

    assert parsed_holder["path"] == source
    assert parsed_holder["destination"] == episode_root
    assert outcome.outputs["source_quality"] == episode_root / "analysis" / "source_quality.json"
    assert outcome.outputs["chapters_index"] == episode_root / "chapters" / "index.json"
    assert outcome.outputs["chapter_001"] == episode_root / "chapters" / "001.md"
    assert outcome.outputs["chapter_002"] == episode_root / "chapters" / "002.md"
    assert set(outcome.inputs) == {"book_state", "source"}
    assert outcome.inputs["book_state"] == sha256_file(book_json)
    assert outcome.inputs["source"] == sha256_file(source)
    _assert_input_hashes(outcome, book_json, source)
    _assert_identifier_keys(outcome)


def test_file_mode_missing_source_fails(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book = BookState(
        book_id="book-missing",
        title="Missing",
        authors=["A"],
        source_mode="file",
        source_ids=["a" * 64],
    )
    _write_book_json(store_root, book)
    (store_root / "books" / book.book_id / "source").mkdir(parents=True, exist_ok=True)
    episode_root = _episode_root(store_root, book.book_id, "ep-1")
    stage = TitleOrFileParseStage(store, parser=lambda *_a, **_k: None, persister=lambda *_a, **_k: "source_parsed")
    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))
    assert caught.value.error_code == "source_missing"
    assert str(caught.value) == "source_missing"


def test_file_mode_multiple_sources_fails(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book_id = "book-multi"
    first = _write_source_file(store_root, book_id, b"one\n", suffix=".txt")
    second_dir = store_root / "books" / book_id / "source"
    extra = second_dir / "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.pdf"
    extra.write_bytes(b"two\n")
    book = BookState(
        book_id=book_id,
        title="Multi",
        authors=["A"],
        source_mode="file",
        source_ids=[first.stem],
    )
    _write_book_json(store_root, book)
    episode_root = _episode_root(store_root, book_id, "ep-1")
    stage = TitleOrFileParseStage(store, parser=lambda *_a, **_k: None, persister=lambda *_a, **_k: "source_parsed")
    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book_id, "ep-1", episode_root))
    assert caught.value.error_code == "source_ambiguous"
    assert str(caught.value) == "source_ambiguous"


def test_book_identity_mismatch_fails(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book = BookState(
        book_id="other-book",
        title="Notes",
        authors=["Ada"],
        source_mode="title_author",
        source_ids=[],
    )
    _write_book_json(store_root, book, dir_book_id="book-1")
    episode_root = _episode_root(store_root, "book-1", "ep-1")
    stage = TitleOrFileParseStage(store)
    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-1", "ep-1", episode_root))
    assert caught.value.error_code == "book_identity_mismatch"
    assert str(caught.value) == "book_identity_mismatch"


def test_mismatched_episode_root_is_unsafe(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book = BookState(
        book_id="book-1",
        title="Notes",
        authors=["Ada"],
        source_mode="title_author",
        source_ids=[],
    )
    _write_book_json(store_root, book)
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    stage = TitleOrFileParseStage(store)
    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", other_root))
    assert caught.value.error_code == "parse_context_unsafe"
    assert str(caught.value) == "parse_context_unsafe"


def test_file_mode_empty_and_malformed_artifacts_fail(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book_id = "book-artifacts"
    source = _write_source_file(store_root, book_id, b"body\n")
    book = BookState(
        book_id=book_id,
        title="Artifacts",
        authors=["A"],
        source_mode="file",
        source_ids=[source.stem],
    )
    _write_book_json(store_root, book)
    episode_id = "ep-1"

    def parser(_path: Path):
        return {"ok": True}

    def persist_empty_quality(_parsed, destination: Path) -> str:
        analysis = destination / "analysis"
        chapters = destination / "chapters"
        analysis.mkdir(parents=True, exist_ok=True)
        chapters.mkdir(parents=True, exist_ok=True)
        (analysis / "source_quality.json").write_text("", encoding="utf-8")
        (chapters / "index.json").write_text(
            json.dumps({"chapters": [_chapter_index_item("ch-001", "One")]}),
            encoding="utf-8",
        )
        (chapters / "001.md").write_text("# one\n", encoding="utf-8")
        return "source_parsed"

    def persist_malformed_index(_parsed, destination: Path) -> str:
        analysis = destination / "analysis"
        chapters = destination / "chapters"
        analysis.mkdir(parents=True, exist_ok=True)
        chapters.mkdir(parents=True, exist_ok=True)
        (analysis / "source_quality.json").write_text('{"ok":true}', encoding="utf-8")
        (chapters / "index.json").write_text("{not-json", encoding="utf-8")
        return "source_parsed"

    def persist_empty_index(_parsed, destination: Path) -> str:
        analysis = destination / "analysis"
        chapters = destination / "chapters"
        analysis.mkdir(parents=True, exist_ok=True)
        chapters.mkdir(parents=True, exist_ok=True)
        (analysis / "source_quality.json").write_text('{"ok":true}', encoding="utf-8")
        (chapters / "index.json").write_text(json.dumps({"chapters": []}), encoding="utf-8")
        return "source_parsed"

    def persist_missing_chapter(_parsed, destination: Path) -> str:
        analysis = destination / "analysis"
        chapters = destination / "chapters"
        analysis.mkdir(parents=True, exist_ok=True)
        chapters.mkdir(parents=True, exist_ok=True)
        (analysis / "source_quality.json").write_text('{"ok":true}', encoding="utf-8")
        (chapters / "index.json").write_text(
            json.dumps(
                {
                    "chapters": [
                        _chapter_index_item("ch-001", "One"),
                        _chapter_index_item("ch-002", "Two"),
                    ]
                }
            ),
            encoding="utf-8",
        )
        (chapters / "001.md").write_text("# one\n", encoding="utf-8")
        return "source_parsed"

    def persist_unexpected_chapter(_parsed, destination: Path) -> str:
        analysis = destination / "analysis"
        chapters = destination / "chapters"
        analysis.mkdir(parents=True, exist_ok=True)
        chapters.mkdir(parents=True, exist_ok=True)
        (analysis / "source_quality.json").write_text('{"ok":true}', encoding="utf-8")
        (chapters / "index.json").write_text(
            json.dumps({"chapters": [_chapter_index_item("ch-001", "One")]}),
            encoding="utf-8",
        )
        (chapters / "001.md").write_text("# one\n", encoding="utf-8")
        (chapters / "002.md").write_text("# two\n", encoding="utf-8")
        return "source_parsed"

    def persist_string_chapter_names(_parsed, destination: Path) -> str:
        analysis = destination / "analysis"
        chapters = destination / "chapters"
        analysis.mkdir(parents=True, exist_ok=True)
        chapters.mkdir(parents=True, exist_ok=True)
        (analysis / "source_quality.json").write_text('{"ok":true}', encoding="utf-8")
        (chapters / "index.json").write_text(
            json.dumps({"chapters": ["001.md"]}),
            encoding="utf-8",
        )
        (chapters / "001.md").write_text("# one\n", encoding="utf-8")
        return "source_parsed"

    def persist_blank_chapter_fields(_parsed, destination: Path) -> str:
        analysis = destination / "analysis"
        chapters = destination / "chapters"
        analysis.mkdir(parents=True, exist_ok=True)
        chapters.mkdir(parents=True, exist_ok=True)
        (analysis / "source_quality.json").write_text('{"ok":true}', encoding="utf-8")
        (chapters / "index.json").write_text(
            json.dumps(
                {
                    "chapters": [
                        {
                            "chapter_id": "  ",
                            "title": "One",
                            "sha256": "a" * 64,
                            "anchor": {
                                "source_type": "txt",
                                "source_path": "x",
                                "chapter_id": "c",
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (chapters / "001.md").write_text("# one\n", encoding="utf-8")
        return "source_parsed"

    def persist_uppercase_sha256(_parsed, destination: Path) -> str:
        analysis = destination / "analysis"
        chapters = destination / "chapters"
        analysis.mkdir(parents=True, exist_ok=True)
        chapters.mkdir(parents=True, exist_ok=True)
        (analysis / "source_quality.json").write_text('{"ok":true}', encoding="utf-8")
        item = _chapter_index_item("ch-001", "One")
        item["sha256"] = "A" * 64
        (chapters / "index.json").write_text(
            json.dumps({"chapters": [item]}),
            encoding="utf-8",
        )
        (chapters / "001.md").write_text("# one\n", encoding="utf-8")
        return "source_parsed"

    def persist_duplicate_chapter_id(_parsed, destination: Path) -> str:
        analysis = destination / "analysis"
        chapters = destination / "chapters"
        analysis.mkdir(parents=True, exist_ok=True)
        chapters.mkdir(parents=True, exist_ok=True)
        (analysis / "source_quality.json").write_text('{"ok":true}', encoding="utf-8")
        first = _chapter_index_item("ch-dup", "One")
        second = _chapter_index_item("ch-dup", "Two")
        (chapters / "index.json").write_text(
            json.dumps({"chapters": [first, second]}),
            encoding="utf-8",
        )
        (chapters / "001.md").write_text("# one\n", encoding="utf-8")
        (chapters / "002.md").write_text("# two\n", encoding="utf-8")
        return "source_parsed"

    def persist_invalid_status(_parsed, destination: Path) -> str:
        _persist_ok(destination)
        return "source_invalid"

    cases = [
        persist_empty_quality,
        persist_malformed_index,
        persist_empty_index,
        persist_missing_chapter,
        persist_unexpected_chapter,
        persist_string_chapter_names,
        persist_blank_chapter_fields,
        persist_uppercase_sha256,
        persist_duplicate_chapter_id,
    ]
    for persister in cases:
        episode_root = _episode_root(store_root, book_id, episode_id)
        for leftover in (episode_root / "analysis", episode_root / "chapters"):
            if leftover.exists():
                for child in leftover.rglob("*"):
                    if child.is_file():
                        child.unlink()
        stage = TitleOrFileParseStage(store, parser=parser, persister=persister)
        with pytest.raises(ContentRuntimeError) as caught:
            stage.run(_context(book_id, episode_id, episode_root))
        assert caught.value.error_code == "parsed_artifact_invalid"
        assert str(caught.value) == "parsed_artifact_invalid"

    episode_root = _episode_root(store_root, book_id, episode_id)
    stage = TitleOrFileParseStage(store, parser=parser, persister=persist_invalid_status)
    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book_id, episode_id, episode_root))
    assert caught.value.error_code == "source_invalid"
    assert str(caught.value) == "source_invalid"


def test_file_mode_source_integrity_mismatch_fails(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book_id = "book-integrity"
    source_dir = store_root / "books" / book_id / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    fake_name = source_dir / ("b" * 64 + ".txt")
    fake_name.write_bytes(b"not-the-hash\n")
    book = BookState(
        book_id=book_id,
        title="Integrity",
        authors=["A"],
        source_mode="file",
        source_ids=["b" * 64],
    )
    _write_book_json(store_root, book)
    episode_root = _episode_root(store_root, book_id, "ep-1")
    stage = TitleOrFileParseStage(store, parser=lambda *_a, **_k: None, persister=lambda *_a, **_k: "source_parsed")
    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book_id, "ep-1", episode_root))
    assert caught.value.error_code == "source_integrity_mismatch"
    assert str(caught.value) == "source_integrity_mismatch"


def test_file_mode_ignores_artifact_supplied_path(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book_id = "book-path"
    episode_id = "ep-1"
    source = _write_source_file(store_root, book_id, b"chapter\n")
    book = BookState(
        book_id=book_id,
        title="Path Trap",
        authors=["A"],
        source_mode="file",
        source_ids=[source.stem],
    )
    _write_book_json(store_root, book)
    episode_root = _episode_root(store_root, book_id, episode_id)

    def persist_with_supplied_path(_parsed, destination: Path) -> str:
        item = _chapter_index_item("ch-001", "One")
        item["file"] = "../escape.md"
        item["path"] = "nested/evil.md"
        item["name"] = "not-the-real.md"
        return _persist_ok(destination, chapters=[item])

    stage = TitleOrFileParseStage(
        store, parser=lambda *_a, **_k: {"ok": True}, persister=persist_with_supplied_path
    )
    outcome = stage.run(_context(book_id, episode_id, episode_root))

    assert outcome.outputs["chapter_001"] == episode_root / "chapters" / "001.md"
    assert "escape.md" not in {path.name for path in outcome.outputs.values()}
    assert "evil.md" not in {path.name for path in outcome.outputs.values()}
    assert "not-the-real.md" not in {path.name for path in outcome.outputs.values()}
    _assert_identifier_keys(outcome)


def test_file_mode_real_persist_parsed_book_contract(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book_id = "book-real"
    episode_id = "ep-real"
    fixture = fixtures_dir / "books" / "utf8.txt"
    payload = fixture.read_bytes()
    padding = "".join(f"补充段落{index:03d}独立正文。" for index in range(80)).encode("utf-8")
    source = _write_source_file(store_root, book_id, payload + b"\n" + padding, suffix=".txt")
    book = BookState(
        book_id=book_id,
        title="UTF-8 Fixture",
        authors=["Fixture Author"],
        source_mode="file",
        source_ids=[source.stem],
    )
    _write_book_json(store_root, book)
    episode_root = _episode_root(store_root, book_id, episode_id)

    stage = TitleOrFileParseStage(store)
    outcome = stage.run(_context(book_id, episode_id, episode_root))

    assert outcome.outputs["chapters_index"] == episode_root / "chapters" / "index.json"
    assert "chapter_001" in outcome.outputs
    assert outcome.outputs["chapter_001"] == episode_root / "chapters" / "001.md"
    payload = json.loads(
        (episode_root / "chapters" / "index.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    assert isinstance(payload["chapters"], list)
    assert payload["chapters"]
    first = payload["chapters"][0]
    assert set(first) >= {"chapter_id", "title", "sha256", "anchor"}
    assert "file" not in first
    assert "path" not in first
    assert "name" not in first
    _assert_identifier_keys(outcome)


def test_stage_outcome_keys_match_orchestrator_identifier(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)

    title_book = BookState(
        book_id="book-keys-title",
        title="Notes",
        authors=["Ada"],
        source_mode="title_author",
        source_ids=[],
    )
    _write_book_json(store_root, title_book)
    title_root = _episode_root(store_root, title_book.book_id, "ep-1")
    title_outcome = TitleOrFileParseStage(store).run(
        _context(title_book.book_id, "ep-1", title_root)
    )
    assert set(title_outcome.inputs) == {"book_state", "title_metadata"}
    assert set(title_outcome.outputs) == {"title_metadata"}
    _assert_identifier_keys(title_outcome)

    file_id = "book-keys-file"
    source = _write_source_file(store_root, file_id, b"one\n")
    file_book = BookState(
        book_id=file_id,
        title="File",
        authors=["A"],
        source_mode="file",
        source_ids=[source.stem],
    )
    _write_book_json(store_root, file_book)
    file_root = _episode_root(store_root, file_id, "ep-1")
    file_outcome = TitleOrFileParseStage(
        store,
        parser=lambda *_a, **_k: {"ok": True},
        persister=lambda _p, destination: _persist_ok(destination),
    ).run(_context(file_id, "ep-1", file_root))
    assert set(file_outcome.inputs) == {"book_state", "source"}
    assert "chapters_index" in file_outcome.outputs
    assert "chapter_001" in file_outcome.outputs
    _assert_identifier_keys(file_outcome)


def test_title_mode_rejects_non_string_title_or_author(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book = BookState(
        book_id="book-typed",
        title="Notes",
        authors=["Ada"],
        source_mode="title_author",
        source_ids=[],
    )
    _write_book_json(store_root, book)
    episode_root = _episode_root(store_root, book.book_id, "ep-1")
    stage = TitleOrFileParseStage(store)

    class _BadTitle:
        book_id = book.book_id
        title = 12
        authors = ["Ada"]
        source_mode = "title_author"
        source_ids: list[str] = []

    class _BadAuthor:
        book_id = book.book_id
        title = "Notes"
        authors = ["Ada", None]
        source_mode = "title_author"
        source_ids: list[str] = []

    for fake in (_BadTitle(), _BadAuthor()):
        store.load_book = lambda _book_id, payload=fake: payload  # type: ignore[method-assign]
        with pytest.raises(ContentRuntimeError) as caught:
            stage.run(_context(book.book_id, "ep-1", episode_root))
        assert caught.value.error_code == "title_metadata_invalid"


def test_file_mode_rejects_source_id_before_selecting_source(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book_id = "book-bad-source-id"
    source_dir = store_root / "books" / book_id / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    trap = source_dir / "not-a-hash.txt"
    trap.write_bytes(b"trap\n")
    invalid_ids = ["abc", "A" * 64, "g" * 64, ""]
    for source_id in invalid_ids:
        book = BookState(
            book_id=book_id,
            title="Bad Source",
            authors=["A"],
            source_mode="file",
            source_ids=[source_id],
        )
        _write_book_json(store_root, book)
        episode_root = _episode_root(store_root, book_id, "ep-1")
        called: list[str] = []

        def parser(path: Path) -> dict[str, str]:
            called.append(str(path))
            return {"ok": True}

        stage = TitleOrFileParseStage(
            store, parser=parser, persister=lambda *_a, **_k: "source_parsed"
        )
        with pytest.raises(ContentRuntimeError) as caught:
            stage.run(_context(book_id, "ep-1", episode_root))
        assert caught.value.error_code == "source_invalid"
        assert str(caught.value) == "source_invalid"
        assert called == []


# --- Task 6B1b: BookAnalysisStage (title research / file evidence, no bindings) ---

from types import SimpleNamespace

from pydantic import ValidationError as _PydanticValidationError

from bv.content.evidence import (
    ChapterAnalysis,
    ChapterAnalysisCacheManifest,
    ChapterAnalysisRunResult,
    EvidenceCard,
)
from bv.content.research import BookResearch, BookResearchResult
from bv.ebook.models import Chapter, ParsedBook, SourceAnchor, SourceQualityReport
from bv.ebook.service import parse_book
from bv.models.prompts import load_prompt
from bv.workflow.content_runtime import BookAnalysisStage, EvidenceBundle
from bv.workflow.runtime import ExternalAuthorizationError, RuntimeAuthorization


_RESEARCH_PROMPT = (
    "Research the exact named book as a whole.\n"
    "Verify identity and cite URLs.\n"
)
_CHAPTER_PROMPT = "Analyze the supplied chapter only.\n"


def _write_prompt(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _analysis_prompt_paths(
    tmp_path: Path,
    *,
    research: str = _RESEARCH_PROMPT,
    chapter: str = _CHAPTER_PROMPT,
) -> tuple[Path, Path]:
    return (
        _write_prompt(tmp_path / "prompts" / "book_research" / "v1.md", research),
        _write_prompt(tmp_path / "prompts" / "chapter_analysis" / "v1.md", chapter),
    )


def _allowed() -> RuntimeAuthorization:
    return RuntimeAuthorization(allow_external=True)


def _denied() -> RuntimeAuthorization:
    return RuntimeAuthorization(allow_external=False)


def _research_payload(title: str, authors: list[str]) -> dict[str, object]:
    return {
        "title": title,
        "authors": authors,
        "identity_confidence": "high",
        "central_question": "What whole-book question does the work answer?",
        "argument_or_narrative_arc": ["The argument develops across the book."],
        "key_ideas": ["First key idea of the whole book.", "Second key idea of the whole book."],
        "life_connections": ["A concrete life situation the book addresses."],
        "reader_value": ["A reader can change a specific judgment or action."],
        "misunderstandings_and_boundaries": ["The book is not a license to ignore others."],
        "sources": [
            {
                "url": "https://example.com/publisher/catalog",
                "source_type": "publisher",
                "supports": ["identity"],
            }
        ],
        "uncertainties": [],
    }


def _write_research_artifacts(
    output_root: Path, title: str, authors: list[str]
) -> BookResearch:
    research = BookResearch.model_validate(_research_payload(title, authors))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "book_research.json").write_text(
        research.model_dump_json(), encoding="utf-8"
    )
    (output_root / "book_research.md").write_text("# research\n", encoding="utf-8")
    return research


def _accepted_research_result(
    output_root: Path, research: BookResearch, prompt_text: str
) -> BookResearchResult:
    return BookResearchResult(
        status="accepted",
        research=research,
        json_path=output_root / "book_research.json",
        markdown_path=output_root / "book_research.md",
        prompt_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
    )


class _RecordingResearcher:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict[str, object]] = []

    def __call__(self, *, book, model, output_root, prompt_text):
        self.calls.append(
            {
                "book": book,
                "model": model,
                "output_root": Path(output_root),
                "prompt_text": prompt_text,
            }
        )
        return self.handler(
            book=book, model=model, output_root=output_root, prompt_text=prompt_text
        )


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def complete(self, prompt: str, schema_type: type, request_dir: Path) -> object:
        self.calls.append(
            {"prompt": prompt, "schema_type": schema_type, "request_dir": Path(request_dir)}
        )
        raise AssertionError("external model must not be invoked")


def _make_chapter(chapter_id: str, title: str, text: str) -> Chapter:
    return Chapter(
        chapter_id=chapter_id,
        title=title,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        anchor=SourceAnchor(
            source_type="txt",
            source_path="book.txt",
            chapter_id=chapter_id,
            start_line=1,
            end_line=1,
        ),
    )


def _parsed_book(source: Path, chapters: list[Chapter]) -> ParsedBook:
    return ParsedBook(
        source_path=str(source),
        encoding="utf-8",
        full_text="\n".join(chapter.text for chapter in chapters),
        chapters=chapters,
        quality=SourceQualityReport(blocking=False, codes=[], warnings=[]),
    )


def _index_items(chapters: list[Chapter]) -> list[dict[str, object]]:
    return [
        _chapter_index_item(chapter.chapter_id, chapter.title, sha256=chapter.sha256)
        for chapter in chapters
    ]


def _evidence_card(chapter_id: str, claim_id: str, excerpt: str) -> EvidenceCard:
    return EvidenceCard.model_validate(
        {
            "claim_id": claim_id,
            "claim_type": "author_argument",
            "paraphrase": f"Paraphrase of {claim_id}",
            "evidence_excerpt": excerpt,
            "chapter_id": chapter_id,
            "start_paragraph": 1,
            "end_paragraph": 1,
            "confidence": "high",
        }
    )


class _RecordingAnalyzer:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        chapters: list[Chapter],
        *,
        model,
        output_dir: Path,
        prompt_text: str,
        prompt_sha256: str,
        source_sha256: str,
        model_identifier: str,
    ):
        self.calls.append(
            {
                "chapters": list(chapters),
                "model": model,
                "output_dir": Path(output_dir),
                "prompt_text": prompt_text,
                "prompt_sha256": prompt_sha256,
                "source_sha256": source_sha256,
                "model_identifier": model_identifier,
            }
        )
        return self.handler(
            chapters,
            model=model,
            output_dir=output_dir,
            prompt_text=prompt_text,
            prompt_sha256=prompt_sha256,
            source_sha256=source_sha256,
            model_identifier=model_identifier,
        )


def _write_chapter_analysis_artifacts(
    output_dir: Path,
    chapters: list[Chapter],
    cards_by_id: dict[str, list[EvidenceCard]],
    *,
    source_sha256: str,
    prompt_sha256: str,
    model_identifier: str,
    **_ignored: object,
) -> ChapterAnalysisRunResult:
    chapter_dir = Path(output_dir) / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    for chapter in chapters:
        analysis = ChapterAnalysis(
            chapter_id=chapter.chapter_id,
            cards=list(cards_by_id.get(chapter.chapter_id, [])),
        )
        analysis_path = chapter_dir / f"{chapter.chapter_id}.analysis.json"
        manifest_path = chapter_dir / f"{chapter.chapter_id}.manifest.json"
        analysis_path.write_text(analysis.model_dump_json(), encoding="utf-8")
        manifest = ChapterAnalysisCacheManifest.model_validate(
            {
                "cache_key": {
                    "source_sha256": source_sha256,
                    "chapter_sha256": chapter.sha256,
                    "prompt_sha256": prompt_sha256,
                    "schema_version": "1",
                    "model_identifier": model_identifier,
                },
                "analysis_sha256": sha256_file(analysis_path),
            }
        )
        manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
        completed.append(chapter.chapter_id)
    return ChapterAnalysisRunResult(status="book_analysis_ready", completed_chapters=completed)


def _analysis_cache_kwargs(kwargs: dict[str, object]) -> dict[str, str]:
    return {
        "source_sha256": str(kwargs["source_sha256"]),
        "prompt_sha256": str(kwargs["prompt_sha256"]),
        "model_identifier": str(kwargs["model_identifier"]),
    }


def _ok_analysis_handler(cards_by_id: dict[str, list[EvidenceCard]]):
    def handler(parsed_chapters, **kwargs):
        return _write_chapter_analysis_artifacts(
            Path(kwargs["output_dir"]),
            parsed_chapters,
            cards_by_id,
            **_analysis_cache_kwargs(kwargs),
        )

    return handler


def _assert_stable_public_error(caught, error_code: str, *secrets: object) -> None:
    assert caught.value.error_code == error_code
    assert str(caught.value) == error_code
    assert caught.value.args == (error_code,)
    text = str(caught.value)
    for secret in secrets:
        if secret is None:
            continue
        assert str(secret) not in text


def _title_book_ready(
    store_root: Path, book_id: str, title: str, authors: list[str]
) -> tuple[StateStore, BookState, Path, Path]:
    store = StateStore(store_root)
    book = BookState(
        book_id=book_id,
        title=title,
        authors=authors,
        source_mode="title_author",
        source_ids=[],
    )
    book_json = _write_book_json(store_root, book)
    episode_root = _episode_root(store_root, book_id, "ep-1")
    TitleOrFileParseStage(store).run(_context(book_id, "ep-1", episode_root))
    return store, book, book_json, episode_root


def _file_book_ready(
    store_root: Path,
    book_id: str,
    chapters: list[Chapter],
    payload: bytes = b"chapter body\n",
) -> tuple[StateStore, Path, Path, Path]:
    store = StateStore(store_root)
    source = _write_source_file(store_root, book_id, payload)
    book = BookState(
        book_id=book_id,
        title="Local File",
        authors=["Author"],
        source_mode="file",
        source_ids=[source.stem],
    )
    book_json = _write_book_json(store_root, book)
    episode_root = _episode_root(store_root, book_id, "ep-1")
    _persist_ok(episode_root, chapters=_index_items(chapters))
    return store, book_json, source, episode_root


def _make_analysis_stage(
    store: StateStore,
    tmp_path: Path,
    *,
    authorization: RuntimeAuthorization | None = None,
    researcher=None,
    analyzer=None,
    parser=None,
    research_model=None,
    chapter_model=None,
    model_identifier: str = "chapter-analysis-test",
    research_text: str = _RESEARCH_PROMPT,
    chapter_text: str = _CHAPTER_PROMPT,
) -> BookAnalysisStage:
    research_path, chapter_path = _analysis_prompt_paths(
        tmp_path, research=research_text, chapter=chapter_text
    )
    kwargs: dict[str, object] = {
        "authorization": authorization if authorization is not None else _denied(),
        "research_prompt_path": research_path,
        "chapter_prompt_path": chapter_path,
        "model_identifier": model_identifier,
    }
    if researcher is not None:
        kwargs["researcher"] = researcher
    if analyzer is not None:
        kwargs["analyzer"] = analyzer
    if parser is not None:
        kwargs["parser"] = parser
    if research_model is not None:
        kwargs["research_model"] = research_model
    if chapter_model is not None:
        kwargs["chapter_model"] = chapter_model
    return BookAnalysisStage(store, **kwargs)  # type: ignore[arg-type]


def _bundle_kwargs(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "source_sha256": "a" * 64,
        "evidence": (_evidence_card("ch-001", "C01-001", "The supported sentence."),),
        "chapter_region_by_id": {"ch-001": "whole"},
        "chapter_sha256_by_id": {"ch-001": "b" * 64},
        "prompt_sha256": "c" * 64,
    }
    values.update(changes)
    return values


def test_evidence_bundle_is_frozen_strict_and_forbids_extra() -> None:
    bundle = EvidenceBundle.model_validate(_bundle_kwargs())
    assert bundle.source_sha256 == "a" * 64
    assert bundle.chapter_region_by_id["ch-001"] == "whole"
    with pytest.raises(_PydanticValidationError):
        bundle.source_sha256 = "d" * 64  # type: ignore[misc]
    with pytest.raises(_PydanticValidationError):
        EvidenceBundle.model_validate({**_bundle_kwargs(), "extra_field": "nope"})


def test_evidence_bundle_rejects_empty_duplicate_unknown_and_uppercase() -> None:
    with pytest.raises(_PydanticValidationError):
        EvidenceBundle.model_validate(_bundle_kwargs(evidence=()))
    duplicate = (
        _evidence_card("ch-001", "C01-001", "One."),
        _evidence_card("ch-001", "C01-001", "Two."),
    )
    with pytest.raises(_PydanticValidationError):
        EvidenceBundle.model_validate(_bundle_kwargs(evidence=duplicate))
    with pytest.raises(_PydanticValidationError):
        EvidenceBundle.model_validate(
            _bundle_kwargs(
                evidence=(_evidence_card("ch-999", "C99-001", "Elsewhere."),)
            )
        )
    with pytest.raises(_PydanticValidationError):
        EvidenceBundle.model_validate(_bundle_kwargs(source_sha256=("A" * 64)))
    with pytest.raises(_PydanticValidationError):
        EvidenceBundle.model_validate(_bundle_kwargs(prompt_sha256=("C" * 64)))
    with pytest.raises(_PydanticValidationError):
        EvidenceBundle.model_validate(
            _bundle_kwargs(chapter_sha256_by_id={"ch-001": "B" * 64})
        )


def test_title_mode_authorization_runs_before_researcher_and_model(
    tmp_path: Path,
) -> None:
    store, _book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-auth", "Notes", ["Ada"]
    )
    model = _RecordingModel()

    def handler(*, book, model, output_root, prompt_text):
        del book, model, output_root, prompt_text
        raise AssertionError("researcher must not run when unauthorized")

    researcher = _RecordingResearcher(handler)
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_denied(),
        researcher=researcher,
        research_model=model,
    )

    with pytest.raises(ExternalAuthorizationError) as caught:
        stage.run(_context("book-title-auth", "ep-1", episode_root))

    assert caught.value.error_code == "external_not_authorized"
    assert str(caught.value) == "external_not_authorized"
    assert researcher.calls == []
    assert model.calls == []
    assert "Notes" not in str(caught.value)
    assert "Ada" not in str(caught.value)


def test_title_mode_denied_default_researcher_does_not_invoke_model(
    tmp_path: Path,
) -> None:
    from bv.content.research import research_book

    store, _book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-default-auth", "Notes", ["Ada"]
    )
    model = _RecordingModel()
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_denied(),
        researcher=research_book,
        research_model=model,
    )

    with pytest.raises(ExternalAuthorizationError):
        stage.run(_context("book-title-default-auth", "ep-1", episode_root))

    assert model.calls == []


def test_title_mode_prompt_changed_before_run_rejects_without_research(
    tmp_path: Path,
) -> None:
    store, _book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-prompt", "Notes", ["Ada"]
    )
    researcher = _RecordingResearcher(
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("researcher invoked"))
    )
    research_path, chapter_path = _analysis_prompt_paths(tmp_path)
    stage = BookAnalysisStage(
        store,
        authorization=_allowed(),
        researcher=researcher,
        research_prompt_path=research_path,
        chapter_prompt_path=chapter_path,
    )
    research_path.write_text(_RESEARCH_PROMPT + "changed\n", encoding="utf-8")

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-title-prompt", "ep-1", episode_root))

    assert caught.value.error_code == "content_prompt_changed"
    assert str(caught.value) == "content_prompt_changed"
    assert researcher.calls == []
    assert caught.value.args == ("content_prompt_changed",)


def test_title_mode_identity_mismatch_rejects_without_research(tmp_path: Path) -> None:
    store, book, book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-identity", "Notes", ["Ada"]
    )
    changed = BookState(
        book_id=book.book_id,
        title="Other Title",
        authors=list(book.authors),
        source_mode="title_author",
        source_ids=[],
    )
    book_json.write_text(changed.model_dump_json(), encoding="utf-8")
    researcher = _RecordingResearcher(
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("researcher invoked"))
    )
    stage = _make_analysis_stage(
        store, tmp_path, authorization=_allowed(), researcher=researcher
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    assert caught.value.error_code == "book_identity_mismatch"
    assert str(caught.value) == "book_identity_mismatch"
    assert researcher.calls == []
    assert "Notes" not in str(caught.value)
    assert "Other Title" not in str(caught.value)


def test_title_mode_accepted_research_writes_outputs_and_input_hashes(
    tmp_path: Path,
) -> None:
    store, book, book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-ok", "Notes", ["Ada"]
    )
    model = object()

    def handler(*, book, model, output_root, prompt_text):
        research = _write_research_artifacts(Path(output_root), book.title, list(book.authors))
        return _accepted_research_result(Path(output_root), research, prompt_text)

    researcher = _RecordingResearcher(handler)
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        researcher=researcher,
        research_model=model,
    )
    outcome = stage.run(_context(book.book_id, "ep-1", episode_root))

    research_json = episode_root / "analysis" / "book_research.json"
    research_md = episode_root / "analysis" / "book_research.md"
    metadata = episode_root / "analysis" / "title_metadata.json"
    prompt_path = tmp_path / "prompts" / "book_research" / "v1.md"
    assert researcher.calls[0]["output_root"] == episode_root / "analysis"
    assert researcher.calls[0]["model"] is model
    assert outcome.outputs["book_research"] == research_json
    assert outcome.outputs["book_research_md"] == research_md
    assert set(outcome.inputs) == {"book_state", "title_metadata", "book_research_prompt"}
    assert outcome.inputs["book_state"] == sha256_file(book_json)
    assert outcome.inputs["title_metadata"] == sha256_file(metadata)
    assert outcome.inputs["book_research_prompt"] == load_prompt(prompt_path).sha256
    loaded = BookResearch.model_validate_json(research_json.read_text(encoding="utf-8"))
    assert loaded.title == "Notes"
    assert list(loaded.authors) == ["Ada"]
    _assert_identifier_keys(outcome)
    _assert_input_hashes(outcome, book_json, metadata, prompt_path)


def test_title_mode_research_mismatch_fails_closed(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-mismatch", "Notes", ["Ada"]
    )

    def handler(*, book, model, output_root, prompt_text):
        del book, model
        research = _write_research_artifacts(Path(output_root), "Wrong Title", ["Ada"])
        return _accepted_research_result(Path(output_root), research, prompt_text)

    researcher = _RecordingResearcher(handler)
    stage = _make_analysis_stage(
        store, tmp_path, authorization=_allowed(), researcher=researcher
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    assert caught.value.error_code == "book_research_identity_mismatch"
    assert str(caught.value) == "book_research_identity_mismatch"
    assert "Wrong Title" not in str(caught.value)
    assert "Notes" not in str(caught.value)
    assert "example.com" not in str(caught.value)


def test_title_mode_empty_research_artifacts_fail_closed(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-empty", "Notes", ["Ada"]
    )

    def handler(*, book, model, output_root, prompt_text):
        del book, model
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "book_research.json").write_text("", encoding="utf-8")
        (root / "book_research.md").write_text("", encoding="utf-8")
        return SimpleNamespace(
            status="accepted",
            json_path=root / "book_research.json",
            markdown_path=root / "book_research.md",
            prompt_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        )

    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        researcher=_RecordingResearcher(handler),
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    assert caught.value.error_code == "book_research_invalid"
    assert str(caught.value) == "book_research_invalid"


def test_file_mode_local_work_happens_before_authorization_denial(
    tmp_path: Path,
) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-deny", chapters
    )
    parsed = _parsed_book(source, chapters)
    parser_calls: list[Path] = []

    def parser(path: Path) -> ParsedBook:
        parser_calls.append(Path(path))
        return parsed

    analyzer = _RecordingAnalyzer(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("analyzer invoked"))
    )
    chapter_model = _RecordingModel()
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_denied(),
        parser=parser,
        analyzer=analyzer,
        chapter_model=chapter_model,
    )

    with pytest.raises(ExternalAuthorizationError) as caught:
        stage.run(_context("book-file-deny", "ep-1", episode_root))

    assert caught.value.error_code == "external_not_authorized"
    assert parser_calls == [source]
    assert analyzer.calls == []
    assert chapter_model.calls == []


def test_file_mode_denied_default_analyzer_does_not_invoke_model(tmp_path: Path) -> None:
    from bv.content.evidence import analyze_chapters

    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-default-deny", chapters
    )
    chapter_model = _RecordingModel()
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_denied(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=analyze_chapters,
        chapter_model=chapter_model,
    )

    with pytest.raises(ExternalAuthorizationError):
        stage.run(_context("book-file-default-deny", "ep-1", episode_root))

    assert chapter_model.calls == []


def test_file_mode_stale_chapter_index_fails_before_authorization(
    tmp_path: Path,
) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-stale", chapters
    )
    stale = [_make_chapter("ch-001", "One", "Changed paragraph.")]
    _persist_ok(episode_root, chapters=_index_items(stale))
    analyzer = _RecordingAnalyzer(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("analyzer invoked"))
    )
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=analyzer,
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-stale", "ep-1", episode_root))

    assert caught.value.error_code == "chapter_index_stale"
    assert str(caught.value) == "chapter_index_stale"
    assert analyzer.calls == []


def test_file_mode_prompt_changed_before_run_rejects_without_analysis(
    tmp_path: Path,
) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-prompt", chapters
    )
    analyzer = _RecordingAnalyzer(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("analyzer invoked"))
    )
    research_path, chapter_path = _analysis_prompt_paths(tmp_path)
    stage = BookAnalysisStage(
        store,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=analyzer,
        research_prompt_path=research_path,
        chapter_prompt_path=chapter_path,
    )
    chapter_path.write_text(_CHAPTER_PROMPT + "changed\n", encoding="utf-8")

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-prompt", "ep-1", episode_root))

    assert caught.value.error_code == "content_prompt_changed"
    assert str(caught.value) == "content_prompt_changed"
    assert analyzer.calls == []
    assert caught.value.args == ("content_prompt_changed",)


def test_file_mode_writes_evidence_bundle_regions_and_identifier_safe_outputs(
    tmp_path: Path,
) -> None:
    chapters = [
        _make_chapter("ch-001", "One", "First paragraph."),
        _make_chapter("ch-002", "Two", "Second paragraph."),
        _make_chapter("ch-003", "Three", "Third paragraph."),
    ]
    store, book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-ok", chapters
    )
    cards = {
        "ch-001": [_evidence_card("ch-001", "C01-001", "First paragraph.")],
        "ch-002": [_evidence_card("ch-002", "C02-001", "Second paragraph.")],
        "ch-003": [_evidence_card("ch-003", "C03-001", "Third paragraph.")],
    }
    analyzer = _RecordingAnalyzer(_ok_analysis_handler(cards))
    chapter_model = object()
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=analyzer,
        chapter_model=chapter_model,
        model_identifier="chapter-analysis-test",
    )
    outcome = stage.run(_context("book-file-ok", "ep-1", episode_root))

    bundle_path = episode_root / "analysis" / "evidence_bundle.json"
    index_path = episode_root / "chapters" / "index.json"
    prompt_path = tmp_path / "prompts" / "chapter_analysis" / "v1.md"
    assert analyzer.calls[0]["output_dir"] == episode_root / "analysis"
    assert analyzer.calls[0]["model"] is chapter_model
    assert analyzer.calls[0]["model_identifier"] == "chapter-analysis-test"
    assert analyzer.calls[0]["source_sha256"] == sha256_file(source)
    assert outcome.outputs["evidence_bundle"] == bundle_path
    assert outcome.outputs["chapter_001_analysis"] == (
        episode_root / "analysis" / "chapters" / "ch-001.analysis.json"
    )
    assert outcome.outputs["chapter_001_manifest"] == (
        episode_root / "analysis" / "chapters" / "ch-001.manifest.json"
    )
    assert outcome.outputs["chapter_002_analysis"].name == "ch-002.analysis.json"
    assert outcome.outputs["chapter_003_manifest"].name == "ch-003.manifest.json"
    assert set(outcome.inputs) == {
        "book_state",
        "source",
        "chapters_index",
        "chapter_analysis_prompt",
    }
    assert outcome.inputs["book_state"] == sha256_file(book_json)
    assert outcome.inputs["source"] == sha256_file(source)
    assert outcome.inputs["chapters_index"] == sha256_file(index_path)
    assert outcome.inputs["chapter_analysis_prompt"] == load_prompt(prompt_path).sha256
    bundle = EvidenceBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    assert bundle.source_sha256 == sha256_file(source)
    assert bundle.prompt_sha256 == load_prompt(prompt_path).sha256
    assert bundle.chapter_region_by_id == {
        "ch-001": "early",
        "ch-002": "middle",
        "ch-003": "late",
    }
    assert bundle.chapter_sha256_by_id == {
        chapter.chapter_id: chapter.sha256 for chapter in chapters
    }
    assert [card.claim_id for card in bundle.evidence] == ["C01-001", "C02-001", "C03-001"]
    _assert_identifier_keys(outcome)
    _assert_input_hashes(outcome, book_json, source, index_path, prompt_path)


@pytest.mark.parametrize(
    ("count", "regions"),
    [
        (1, ("whole",)),
        (2, ("early", "late")),
        (4, ("early", "early", "middle", "late")),
    ],
)
def test_file_mode_chapter_regions_split_deterministically(
    tmp_path: Path, count: int, regions: tuple[str, ...]
) -> None:
    chapters = [
        _make_chapter(f"ch-{index:03d}", f"C{index}", f"Paragraph {index}.")
        for index in range(1, count + 1)
    ]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", f"book-file-regions-{count}", chapters
    )
    cards = {
        chapter.chapter_id: [
            _evidence_card(chapter.chapter_id, f"C{index:02d}-001", chapter.text)
        ]
        for index, chapter in enumerate(chapters, start=1)
    }
    analyzer = _RecordingAnalyzer(_ok_analysis_handler(cards))
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=analyzer,
    )
    outcome = stage.run(
        _context(f"book-file-regions-{count}", "ep-1", episode_root)
    )
    bundle = EvidenceBundle.model_validate_json(
        outcome.outputs["evidence_bundle"].read_text(encoding="utf-8")
    )
    assert tuple(bundle.chapter_region_by_id[chapter.chapter_id] for chapter in chapters) == regions
    if count >= 3:
        assert set(bundle.chapter_region_by_id.values()) == {"early", "middle", "late"}


def test_file_mode_duplicate_claims_fail_closed(tmp_path: Path) -> None:
    chapters = [
        _make_chapter("ch-001", "One", "First paragraph."),
        _make_chapter("ch-002", "Two", "Second paragraph."),
    ]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-dup", chapters
    )
    cards = {
        "ch-001": [_evidence_card("ch-001", "C-DUP", "First paragraph.")],
        "ch-002": [_evidence_card("ch-002", "C-DUP", "Second paragraph.")],
    }
    analyzer = _RecordingAnalyzer(_ok_analysis_handler(cards))
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=analyzer,
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-dup", "ep-1", episode_root))

    assert caught.value.error_code == "claim_id_duplicate"
    assert str(caught.value) == "claim_id_duplicate"
    assert not (episode_root / "analysis" / "evidence_bundle.json").exists()


def test_file_mode_malformed_analysis_artifact_fails_closed(tmp_path: Path) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-bad-analysis", chapters
    )

    def handler(parsed_chapters, **kwargs):
        del parsed_chapters
        chapter_dir = Path(kwargs["output_dir"]) / "chapters"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        (chapter_dir / "ch-001.analysis.json").write_text("{not-json", encoding="utf-8")
        (chapter_dir / "ch-001.manifest.json").write_text("{}", encoding="utf-8")
        return ChapterAnalysisRunResult(status="book_analysis_ready", completed_chapters=["ch-001"])

    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=_RecordingAnalyzer(handler),
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-bad-analysis", "ep-1", episode_root))

    assert caught.value.error_code == "book_analysis_invalid"
    assert str(caught.value) == "book_analysis_invalid"


def test_file_mode_real_parser_validates_current_index_then_builds_bundle(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    store_root = tmp_path / "store"
    store = StateStore(store_root)
    book_id = "book-file-real-parse"
    fixture = fixtures_dir / "books" / "utf8.txt"
    padding = "".join(f"padding-{index:03d}-text " for index in range(80)).encode("utf-8")
    source = _write_source_file(
        store_root, book_id, fixture.read_bytes() + b"\n" + padding, suffix=".txt"
    )
    book = BookState(
        book_id=book_id,
        title="UTF-8 Fixture",
        authors=["Fixture Author"],
        source_mode="file",
        source_ids=[source.stem],
    )
    book_json = _write_book_json(store_root, book)
    episode_root = _episode_root(store_root, book_id, "ep-1")
    TitleOrFileParseStage(store).run(_context(book_id, "ep-1", episode_root))
    parsed = parse_book(source)

    def handler(parsed_chapters, **kwargs):
        cards = {
            chapter.chapter_id: [
                _evidence_card(
                    chapter.chapter_id,
                    f"C{index:02d}-001",
                    chapter.text.splitlines()[0] if chapter.text.splitlines() else chapter.text[:20],
                )
            ]
            for index, chapter in enumerate(parsed_chapters, start=1)
        }
        return _write_chapter_analysis_artifacts(
            Path(kwargs["output_dir"]),
            parsed_chapters,
            cards,
            **_analysis_cache_kwargs(kwargs),
        )

    analyzer = _RecordingAnalyzer(handler)
    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=parse_book,
        analyzer=analyzer,
    )
    outcome = stage.run(_context(book_id, "ep-1", episode_root))

    bundle = EvidenceBundle.model_validate_json(
        outcome.outputs["evidence_bundle"].read_text(encoding="utf-8")
    )
    index_payload = json.loads(
        (episode_root / "chapters" / "index.json").read_text(encoding="utf-8")
    )
    expected_hashes = {
        item["chapter_id"]: item["sha256"] for item in index_payload["chapters"]
    }
    assert bundle.source_sha256 == sha256_file(source)
    assert bundle.chapter_sha256_by_id == expected_hashes
    assert set(bundle.chapter_sha256_by_id) == {chapter.chapter_id for chapter in parsed.chapters}
    assert analyzer.calls[0]["source_sha256"] == sha256_file(source)
    assert outcome.inputs["book_state"] == sha256_file(book_json)
    _assert_identifier_keys(outcome)


def _write_chapter_analysis_with_manifest(
    output_dir: Path,
    chapter: Chapter,
    *,
    manifest_text: str,
) -> ChapterAnalysisRunResult:
    chapter_dir = Path(output_dir) / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    analysis = ChapterAnalysis(
        chapter_id=chapter.chapter_id,
        cards=[_evidence_card(chapter.chapter_id, "C01-001", chapter.text)],
    )
    (chapter_dir / f"{chapter.chapter_id}.analysis.json").write_text(
        analysis.model_dump_json(), encoding="utf-8"
    )
    (chapter_dir / f"{chapter.chapter_id}.manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    return ChapterAnalysisRunResult(
        status="book_analysis_ready", completed_chapters=[chapter.chapter_id]
    )


def _well_formed_manifest_json(
    *,
    source_sha256: str,
    chapter_sha256: str,
    prompt_sha256: str,
    model_identifier: str,
    analysis_sha256: str,
) -> str:
    return ChapterAnalysisCacheManifest.model_validate(
        {
            "cache_key": {
                "source_sha256": source_sha256,
                "chapter_sha256": chapter_sha256,
                "prompt_sha256": prompt_sha256,
                "schema_version": "1",
                "model_identifier": model_identifier,
            },
            "analysis_sha256": analysis_sha256,
        }
    ).model_dump_json()


def test_file_mode_empty_manifest_object_is_invalid(tmp_path: Path) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-empty-manifest", chapters
    )

    def handler(parsed_chapters, **kwargs):
        return _write_chapter_analysis_with_manifest(
            kwargs["output_dir"], parsed_chapters[0], manifest_text="{}"
        )

    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=_RecordingAnalyzer(handler),
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-empty-manifest", "ep-1", episode_root))

    _assert_stable_public_error(caught, "book_analysis_invalid")
    assert not (episode_root / "analysis" / "evidence_bundle.json").exists()


@pytest.mark.parametrize(
    "broken_field",
    ("source_sha256", "prompt_sha256", "analysis_sha256"),
)
def test_file_mode_well_formed_manifest_wrong_hash_is_stale(
    tmp_path: Path, broken_field: str
) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", f"book-file-stale-{broken_field}", chapters
    )
    wrong = "0" * 64

    def handler(parsed_chapters, **kwargs):
        chapter = parsed_chapters[0]
        chapter_dir = Path(kwargs["output_dir"]) / "chapters"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        analysis = ChapterAnalysis(
            chapter_id=chapter.chapter_id,
            cards=[_evidence_card(chapter.chapter_id, "C01-001", chapter.text)],
        )
        analysis_path = chapter_dir / f"{chapter.chapter_id}.analysis.json"
        analysis_path.write_text(analysis.model_dump_json(), encoding="utf-8")
        fields = {
            "source_sha256": str(kwargs["source_sha256"]),
            "chapter_sha256": chapter.sha256,
            "prompt_sha256": str(kwargs["prompt_sha256"]),
            "model_identifier": str(kwargs["model_identifier"]),
            "analysis_sha256": sha256_file(analysis_path),
        }
        fields[broken_field] = wrong
        (chapter_dir / f"{chapter.chapter_id}.manifest.json").write_text(
            _well_formed_manifest_json(**fields),
            encoding="utf-8",
        )
        return ChapterAnalysisRunResult(
            status="book_analysis_ready", completed_chapters=[chapter.chapter_id]
        )

    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=_RecordingAnalyzer(handler),
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(f"book-file-stale-{broken_field}", "ep-1", episode_root))

    _assert_stable_public_error(caught, "chapter_manifest_stale")
    assert not (episode_root / "analysis" / "evidence_bundle.json").exists()


def test_title_mode_book_json_mutation_during_research_fails_closed(tmp_path: Path) -> None:
    store, book, book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-mut-book", "Notes", ["Ada"]
    )
    research_path = tmp_path / "prompts" / "book_research" / "v1.md"

    def handler(*, book, model, output_root, prompt_text):
        del model
        research = _write_research_artifacts(Path(output_root), book.title, list(book.authors))
        mutated = BookState(
            book_id=book.book_id,
            title="Mutated Title",
            authors=list(book.authors),
            source_mode="title_author",
            source_ids=[],
        )
        book_json.write_text(mutated.model_dump_json(), encoding="utf-8")
        return _accepted_research_result(Path(output_root), research, prompt_text)

    researcher = _RecordingResearcher(handler)
    stage = _make_analysis_stage(
        store, tmp_path, authorization=_allowed(), researcher=researcher
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(
        caught,
        "content_input_changed",
        "Mutated Title",
        "Notes",
        book_json,
        research_path,
        _RESEARCH_PROMPT,
    )
    assert researcher.calls


def test_title_mode_title_metadata_mutation_during_research_fails_closed(
    tmp_path: Path,
) -> None:
    store, book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-mut-meta", "Notes", ["Ada"]
    )
    metadata = episode_root / "analysis" / "title_metadata.json"

    def handler(*, book, model, output_root, prompt_text):
        del model
        research = _write_research_artifacts(Path(output_root), book.title, list(book.authors))
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload["title"] = "Mutated Metadata"
        metadata.write_text(json.dumps(payload), encoding="utf-8")
        return _accepted_research_result(Path(output_root), research, prompt_text)

    researcher = _RecordingResearcher(handler)
    stage = _make_analysis_stage(
        store, tmp_path, authorization=_allowed(), researcher=researcher
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(
        caught, "content_input_changed", "Mutated Metadata", "Notes", metadata
    )
    assert researcher.calls


def test_title_mode_prompt_mutation_during_research_fails_closed(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-mut-prompt", "Notes", ["Ada"]
    )
    research_path, chapter_path = _analysis_prompt_paths(tmp_path)
    secret = "SECRET_PROMPT_BODY"

    def handler(*, book, model, output_root, prompt_text):
        del model
        research = _write_research_artifacts(Path(output_root), book.title, list(book.authors))
        research_path.write_text(_RESEARCH_PROMPT + secret + "\n", encoding="utf-8")
        return _accepted_research_result(Path(output_root), research, prompt_text)

    researcher = _RecordingResearcher(handler)
    stage = BookAnalysisStage(
        store,
        authorization=_allowed(),
        researcher=researcher,
        research_prompt_path=research_path,
        chapter_prompt_path=chapter_path,
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(
        caught, "content_prompt_changed", secret, research_path, _RESEARCH_PROMPT
    )
    assert researcher.calls


def test_title_mode_result_prompt_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-result-prompt", "Notes", ["Ada"]
    )

    def handler(*, book, model, output_root, prompt_text):
        del model
        research = _write_research_artifacts(Path(output_root), book.title, list(book.authors))
        result = _accepted_research_result(Path(output_root), research, prompt_text)
        return result.model_copy(update={"prompt_sha256": "0" * 64})

    researcher = _RecordingResearcher(handler)
    stage = _make_analysis_stage(
        store, tmp_path, authorization=_allowed(), researcher=researcher
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(caught, "content_prompt_changed")
    assert researcher.calls


def test_file_mode_book_json_mutation_during_analysis_fails_closed(tmp_path: Path) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-mut-book", chapters
    )
    cards = {"ch-001": [_evidence_card("ch-001", "C01-001", "First paragraph.")]}

    def handler(parsed_chapters, **kwargs):
        result = _write_chapter_analysis_artifacts(
            Path(kwargs["output_dir"]),
            parsed_chapters,
            cards,
            **_analysis_cache_kwargs(kwargs),
        )
        mutated = BookState(
            book_id="book-file-mut-book",
            title="Mutated File Book",
            authors=["Author"],
            source_mode="file",
            source_ids=json.loads(book_json.read_text(encoding="utf-8")).get("source_ids", []),
        )
        book_json.write_text(mutated.model_dump_json(), encoding="utf-8")
        return result

    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=_RecordingAnalyzer(handler),
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-mut-book", "ep-1", episode_root))

    _assert_stable_public_error(caught, "content_input_changed", "Mutated File Book", book_json)
    assert not (episode_root / "analysis" / "evidence_bundle.json").exists()


def test_file_mode_source_mutation_during_analysis_fails_closed(tmp_path: Path) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-mut-source", chapters
    )
    cards = {"ch-001": [_evidence_card("ch-001", "C01-001", "First paragraph.")]}

    def handler(parsed_chapters, **kwargs):
        result = _write_chapter_analysis_artifacts(
            Path(kwargs["output_dir"]),
            parsed_chapters,
            cards,
            **_analysis_cache_kwargs(kwargs),
        )
        source.write_bytes(source.read_bytes() + b"\nmutated-source\n")
        return result

    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=_RecordingAnalyzer(handler),
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-mut-source", "ep-1", episode_root))

    _assert_stable_public_error(caught, "content_input_changed", source, "mutated-source")
    assert not (episode_root / "analysis" / "evidence_bundle.json").exists()


def test_file_mode_index_mutation_during_analysis_fails_closed(tmp_path: Path) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-mut-index", chapters
    )
    cards = {"ch-001": [_evidence_card("ch-001", "C01-001", "First paragraph.")]}
    index_path = episode_root / "chapters" / "index.json"

    def handler(parsed_chapters, **kwargs):
        result = _write_chapter_analysis_artifacts(
            Path(kwargs["output_dir"]),
            parsed_chapters,
            cards,
            **_analysis_cache_kwargs(kwargs),
        )
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        payload["chapters"][0]["title"] = "Mutated Index"
        index_path.write_text(json.dumps(payload), encoding="utf-8")
        return result

    stage = _make_analysis_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=_RecordingAnalyzer(handler),
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-mut-index", "ep-1", episode_root))

    _assert_stable_public_error(caught, "content_input_changed", "Mutated Index", index_path)
    assert not (episode_root / "analysis" / "evidence_bundle.json").exists()


def test_file_mode_prompt_mutation_during_analysis_fails_closed(tmp_path: Path) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-mut-prompt", chapters
    )
    cards = {"ch-001": [_evidence_card("ch-001", "C01-001", "First paragraph.")]}
    research_path, chapter_path = _analysis_prompt_paths(tmp_path)
    secret = "SECRET_CHAPTER_PROMPT"

    def handler(parsed_chapters, **kwargs):
        result = _write_chapter_analysis_artifacts(
            Path(kwargs["output_dir"]),
            parsed_chapters,
            cards,
            **_analysis_cache_kwargs(kwargs),
        )
        chapter_path.write_text(_CHAPTER_PROMPT + secret + "\n", encoding="utf-8")
        return result

    stage = BookAnalysisStage(
        store,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=_RecordingAnalyzer(handler),
        research_prompt_path=research_path,
        chapter_prompt_path=chapter_path,
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-mut-prompt", "ep-1", episode_root))

    _assert_stable_public_error(
        caught, "content_prompt_changed", secret, chapter_path, _CHAPTER_PROMPT
    )
    assert not (episode_root / "analysis" / "evidence_bundle.json").exists()


def test_title_mode_prompt_missing_after_construction_is_stable(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-prompt-missing", "Notes", ["Ada"]
    )
    research_path, chapter_path = _analysis_prompt_paths(tmp_path)
    researcher = _RecordingResearcher(
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("researcher invoked"))
    )
    stage = BookAnalysisStage(
        store,
        authorization=_allowed(),
        researcher=researcher,
        research_prompt_path=research_path,
        chapter_prompt_path=chapter_path,
    )
    research_path.unlink()

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(
        caught, "content_prompt_changed", research_path, _RESEARCH_PROMPT
    )
    assert researcher.calls == []


def test_title_mode_prompt_missing_during_research_is_stable(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_book_ready(
        tmp_path / "store", "book-title-prompt-gone", "Notes", ["Ada"]
    )
    research_path, chapter_path = _analysis_prompt_paths(tmp_path)

    def handler(*, book, model, output_root, prompt_text):
        del model
        research = _write_research_artifacts(Path(output_root), book.title, list(book.authors))
        research_path.unlink()
        return _accepted_research_result(Path(output_root), research, prompt_text)

    researcher = _RecordingResearcher(handler)
    stage = BookAnalysisStage(
        store,
        authorization=_allowed(),
        researcher=researcher,
        research_prompt_path=research_path,
        chapter_prompt_path=chapter_path,
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(
        caught, "content_prompt_changed", research_path, _RESEARCH_PROMPT
    )
    assert researcher.calls


def test_file_mode_prompt_missing_after_construction_is_stable(tmp_path: Path) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-prompt-missing", chapters
    )
    research_path, chapter_path = _analysis_prompt_paths(tmp_path)
    analyzer = _RecordingAnalyzer(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("analyzer invoked"))
    )
    stage = BookAnalysisStage(
        store,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=analyzer,
        research_prompt_path=research_path,
        chapter_prompt_path=chapter_path,
    )
    chapter_path.unlink()

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-prompt-missing", "ep-1", episode_root))

    _assert_stable_public_error(
        caught, "content_prompt_changed", chapter_path, _CHAPTER_PROMPT
    )
    assert analyzer.calls == []
    assert not (episode_root / "analysis" / "evidence_bundle.json").exists()


def test_file_mode_prompt_missing_during_analysis_is_stable(tmp_path: Path) -> None:
    chapters = [_make_chapter("ch-001", "One", "First paragraph.")]
    store, _book_json, source, episode_root = _file_book_ready(
        tmp_path / "store", "book-file-prompt-gone", chapters
    )
    cards = {"ch-001": [_evidence_card("ch-001", "C01-001", "First paragraph.")]}
    research_path, chapter_path = _analysis_prompt_paths(tmp_path)

    def handler(parsed_chapters, **kwargs):
        result = _write_chapter_analysis_artifacts(
            Path(kwargs["output_dir"]),
            parsed_chapters,
            cards,
            **_analysis_cache_kwargs(kwargs),
        )
        chapter_path.unlink()
        return result

    stage = BookAnalysisStage(
        store,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        analyzer=_RecordingAnalyzer(handler),
        research_prompt_path=research_path,
        chapter_prompt_path=chapter_path,
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-prompt-gone", "ep-1", episode_root))

    _assert_stable_public_error(
        caught, "content_prompt_changed", chapter_path, _CHAPTER_PROMPT
    )
    assert not (episode_root / "analysis" / "evidence_bundle.json").exists()


# --- Task 6B1c: BookValueStage (file synthesis / title research value, no bindings) ---

from bv.content.synthesis import (
    BookReport,
    BookSynthesisResult,
    ClaimCluster,
    WholeBookValue,
    synthesize_book,
)
from bv.core.atomic import atomic_write_json
from bv.workflow.content_runtime import (
    BookValueStage,
    TitleResearchClaim,
    TitleValueProvenance,
    build_title_research_claim_catalog,
)


_FILE_VALUE_PROMPT = "Synthesize whole-book reader value from verified claim cards.\n"
_TITLE_VALUE_PROMPT = (
    "Derive the whole-book reader value from accepted web research.\n"
    "E001 is not an isolated concept.\n"
)
_FORBIDDEN_TITLE_KEYS = (
    "evidence_excerpt",
    "chapter",
    "page",
    "chapter_id",
    "start_paragraph",
    "end_paragraph",
    "start_page",
    "end_page",
)


def _value_prompt_paths(
    tmp_path: Path,
    *,
    file_text: str = _FILE_VALUE_PROMPT,
    title_text: str = _TITLE_VALUE_PROMPT,
) -> tuple[Path, Path]:
    return (
        _write_prompt(tmp_path / "prompts" / "book_synthesis" / "v2.md", file_text),
        _write_prompt(tmp_path / "prompts" / "title_book_synthesis" / "v1.md", title_text),
    )


def _json_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(value)
        for item in value.values():
            keys.update(_json_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_json_keys(item))
    return keys


def _file_whole_book_value() -> WholeBookValue:
    return WholeBookValue(
        value_thesis="帮助读者把反复内耗的问题重新理解为可判断的选择。",
        target_reader="在重要选择前反复拖延的人",
        reader_before="把不确定感误当成自己没有能力决定。",
        reader_after="能区分需要继续收集信息与可以承担的选择。",
        central_life_tension="想避免选错，又不愿一直停在原地。",
        supporting_claim_clusters=[
            ClaimCluster(
                cluster_id="problem",
                summary="说明选择焦虑如何形成。",
                narrative_role="problem",
                claim_ids=["C01-001"],
                chapter_regions=["early"],
            ),
            ClaimCluster(
                cluster_id="reframe",
                summary="重看不确定感和责任。",
                narrative_role="reframe",
                claim_ids=["C02-001"],
                chapter_regions=["middle"],
            ),
            ClaimCluster(
                cluster_id="application",
                summary="把判断带回具体行动。",
                narrative_role="application",
                claim_ids=["C03-001"],
                chapter_regions=["late"],
            ),
        ],
        practical_value="提供重新理解选择和行动节奏的视角。",
        reading_reason="视频只呈现主线，完整论证、案例和适用边界仍需回到原书。",
        coverage_exception=None,
    )


def _title_whole_book_value() -> WholeBookValue:
    return WholeBookValue(
        value_thesis="帮助读者把对外认可的依赖重看为自己可以承担的选择。",
        target_reader="在关系里反复猜测评价的人",
        reader_before="把他人反应当成自己该不该行动的标准。",
        reader_after="能分开自己该做的事和他人如何评价。",
        central_life_tension="想保持关系，又不想把决定权交给别人。",
        supporting_claim_clusters=[
            ClaimCluster(
                cluster_id="problem",
                summary="认可焦虑如何把生活停在猜测里。",
                narrative_role="problem",
                claim_ids=["central_question_001", "life_connection_001"],
                chapter_regions=["central_question", "life_connections"],
            ),
            ClaimCluster(
                cluster_id="reframe",
                summary="把课题交还到自己可承担的范围。",
                narrative_role="reframe",
                claim_ids=["arc_001", "key_idea_001"],
                chapter_regions=["arc", "key_ideas"],
            ),
            ClaimCluster(
                cluster_id="application",
                summary="在边界内改变行动，而不是切断关系。",
                narrative_role="application",
                claim_ids=["reader_value_001", "boundary_001"],
                chapter_regions=["reader_value", "boundaries"],
            ),
        ],
        practical_value="提供区分责任和评价的判断框架。",
        reading_reason="公开材料只能标出主线，完整论证仍需回到原书。",
        coverage_exception=None,
    )


class _ScriptedValueModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, prompt: str, schema_type: type, request_dir: Path) -> object:
        self.calls.append(
            {
                "prompt": prompt,
                "schema_type": schema_type,
                "request_dir": Path(request_dir),
            }
        )
        assert Path(request_dir).is_dir()
        assert not any(Path(request_dir).iterdir())
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _RecordingFileSynthesizer:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        evidence,
        *,
        chapter_region_by_id,
        model,
        output_root,
        prompt_text,
    ):
        self.calls.append(
            {
                "evidence": list(evidence),
                "chapter_region_by_id": dict(chapter_region_by_id),
                "model": model,
                "output_root": Path(output_root),
                "prompt_text": prompt_text,
            }
        )
        return self.handler(
            evidence,
            chapter_region_by_id=chapter_region_by_id,
            model=model,
            output_root=output_root,
            prompt_text=prompt_text,
        )


def _write_file_evidence_bundle(
    episode_root: Path,
    source: Path,
    chapters: list,
    cards: list[EvidenceCard],
    *,
    prompt_sha256: str,
) -> EvidenceBundle:
    regions = {
        1: {"ch-001": "whole"},
        2: {"ch-001": "early", "ch-002": "late"},
        3: {"ch-001": "early", "ch-002": "middle", "ch-003": "late"},
    }[len(chapters)]
    bundle = EvidenceBundle(
        source_sha256=sha256_file(source),
        evidence=tuple(cards),
        chapter_region_by_id=regions,
        chapter_sha256_by_id={chapter.chapter_id: chapter.sha256 for chapter in chapters},
        prompt_sha256=prompt_sha256,
    )
    path = Path(episode_root) / "analysis" / "evidence_bundle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    return bundle


def _file_value_ready(
    store_root: Path, book_id: str
) -> tuple[StateStore, Path, Path, Path, list, EvidenceBundle]:
    chapters = [
        _make_chapter("ch-001", "One", "First paragraph."),
        _make_chapter("ch-002", "Two", "Second paragraph."),
        _make_chapter("ch-003", "Three", "Third paragraph."),
    ]
    store, book_json, source, episode_root = _file_book_ready(store_root, book_id, chapters)
    cards = [
        _evidence_card("ch-001", "C01-001", "First paragraph."),
        _evidence_card("ch-002", "C02-001", "Second paragraph."),
        _evidence_card("ch-003", "C03-001", "Third paragraph."),
    ]
    bundle = _write_file_evidence_bundle(
        episode_root,
        source,
        chapters,
        cards,
        prompt_sha256="c" * 64,
    )
    return store, book_json, source, episode_root, chapters, bundle


def _title_value_ready(
    store_root: Path, book_id: str, title: str = "Notes", authors: list[str] | None = None
) -> tuple[StateStore, BookState, Path, Path]:
    names = ["Ada"] if authors is None else authors
    store, book, book_json, episode_root = _title_book_ready(store_root, book_id, title, names)
    _write_research_artifacts(episode_root / "analysis", book.title, list(book.authors))
    return store, book, book_json, episode_root


def _make_value_stage(
    store: StateStore,
    tmp_path: Path,
    *,
    authorization: RuntimeAuthorization | None = None,
    file_model=None,
    title_model=None,
    parser=None,
    file_synthesizer=None,
    title_synthesizer=None,
    file_text: str = _FILE_VALUE_PROMPT,
    title_text: str = _TITLE_VALUE_PROMPT,
) -> BookValueStage:
    file_path, title_path = _value_prompt_paths(
        tmp_path, file_text=file_text, title_text=title_text
    )
    kwargs: dict[str, object] = {
        "authorization": authorization if authorization is not None else _denied(),
        "file_prompt_path": file_path,
        "title_prompt_path": title_path,
    }
    if file_model is not None:
        kwargs["file_model"] = file_model
    if title_model is not None:
        kwargs["title_model"] = title_model
    if parser is not None:
        kwargs["parser"] = parser
    if file_synthesizer is not None:
        kwargs["file_synthesizer"] = file_synthesizer
    if title_synthesizer is not None:
        kwargs["title_synthesizer"] = title_synthesizer
    return BookValueStage(store, **kwargs)  # type: ignore[arg-type]


def _provenance_kwargs(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "book_research_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "source_urls": ("https://example.com/publisher/catalog",),
        "uncertainties": ("Edition details remain incomplete.",),
        "catalog": (
            TitleResearchClaim(
                claim_id="central_question_001",
                research_region="central_question",
                text="What whole-book question does the work answer?",
            ),
        ),
    }
    values.update(changes)
    return values


def test_title_research_claim_catalog_ids_and_regions_are_stable() -> None:
    research = BookResearch.model_validate(_research_payload("Notes", ["Ada"]))

    catalog = build_title_research_claim_catalog(research)

    assert [(item.claim_id, item.research_region) for item in catalog] == [
        ("central_question_001", "central_question"),
        ("arc_001", "arc"),
        ("key_idea_001", "key_ideas"),
        ("key_idea_002", "key_ideas"),
        ("life_connection_001", "life_connections"),
        ("reader_value_001", "reader_value"),
        ("boundary_001", "boundaries"),
    ]
    assert catalog[0].text == "What whole-book question does the work answer?"
    assert catalog[2].text == "First key idea of the whole book."


def test_title_value_provenance_is_frozen_strict_and_has_no_ebook_fields() -> None:
    provenance = TitleValueProvenance.model_validate(_provenance_kwargs())
    fields = set(TitleValueProvenance.model_fields)
    assert provenance.source_urls == ("https://example.com/publisher/catalog",)
    assert "evidence_excerpt" not in fields
    assert "chapter" not in fields
    assert "page" not in fields
    assert "chapter_id" not in fields
    assert "start_paragraph" not in fields
    assert "end_paragraph" not in fields
    with pytest.raises(_PydanticValidationError):
        provenance.prompt_sha256 = "c" * 64  # type: ignore[misc]
    with pytest.raises(_PydanticValidationError):
        TitleValueProvenance.model_validate(
            {**_provenance_kwargs(), "evidence_excerpt": "quoted page 12"}
        )
    with pytest.raises(_PydanticValidationError):
        TitleValueProvenance.model_validate({**_provenance_kwargs(), "chapter": "ch-001"})
    with pytest.raises(_PydanticValidationError):
        TitleValueProvenance.model_validate({**_provenance_kwargs(), "page": 12})
    with pytest.raises(_PydanticValidationError):
        TitleValueProvenance.model_validate(_provenance_kwargs(book_research_sha256="A" * 64))


def test_file_mode_authorization_runs_before_synthesizer_and_model(tmp_path: Path) -> None:
    store, _book_json, source, episode_root, chapters, _bundle = _file_value_ready(
        tmp_path / "store", "book-file-value-auth"
    )
    model = _RecordingModel()
    synthesizer = _RecordingFileSynthesizer(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("file synthesizer must not run when unauthorized")
        )
    )
    stage = _make_value_stage(
        store,
        tmp_path,
        authorization=_denied(),
        parser=lambda _path: _parsed_book(source, chapters),
        file_synthesizer=synthesizer,
        file_model=model,
    )

    with pytest.raises(ExternalAuthorizationError) as caught:
        stage.run(_context("book-file-value-auth", "ep-1", episode_root))

    _assert_stable_public_error(caught, "external_not_authorized")
    assert synthesizer.calls == []
    assert model.calls == []


def test_file_mode_denied_default_synthesizer_does_not_invoke_model(tmp_path: Path) -> None:
    store, _book_json, source, episode_root, chapters, _bundle = _file_value_ready(
        tmp_path / "store", "book-file-value-default-auth"
    )
    model = _RecordingModel()
    stage = _make_value_stage(
        store,
        tmp_path,
        authorization=_denied(),
        parser=lambda _path: _parsed_book(source, chapters),
        file_synthesizer=synthesize_book,
        file_model=model,
    )

    with pytest.raises(ExternalAuthorizationError):
        stage.run(_context("book-file-value-default-auth", "ep-1", episode_root))

    assert model.calls == []


def test_file_mode_uses_actual_evidence_card_bundle_and_writes_exact_card(
    tmp_path: Path,
) -> None:
    store, book_json, source, episode_root, chapters, bundle = _file_value_ready(
        tmp_path / "store", "book-file-value-ok"
    )
    model = _ScriptedValueModel([_file_whole_book_value()])
    captured: dict[str, object] = {}

    def wrapper(evidence, **kwargs):
        captured["evidence"] = list(evidence)
        captured["regions"] = dict(kwargs["chapter_region_by_id"])
        return synthesize_book(evidence, **kwargs)

    stage = _make_value_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        file_synthesizer=wrapper,
        file_model=model,
    )
    outcome = stage.run(_context("book-file-value-ok", "ep-1", episode_root))

    analysis = episode_root / "analysis"
    card_path = analysis / "book_value_card.json"
    evidence = captured["evidence"]
    assert isinstance(evidence, list)
    assert evidence
    assert all(isinstance(card, EvidenceCard) for card in evidence)
    assert [card.claim_id for card in evidence] == ["C01-001", "C02-001", "C03-001"]
    assert [card.evidence_excerpt for card in evidence] == [
        "First paragraph.",
        "Second paragraph.",
        "Third paragraph.",
    ]
    assert captured["regions"] == bundle.chapter_region_by_id
    assert outcome.outputs["book_value_card"] == card_path
    assert outcome.outputs["book_report"] == analysis / "book_report.json"
    assert outcome.outputs["book_report_md"] == analysis / "book_report.md"
    assert outcome.outputs["concept_map"] == analysis / "concept_map.json"
    assert outcome.outputs["value_synthesis"] == analysis / "value_synthesis.json"
    assert set(outcome.inputs) == {
        "book_state",
        "source",
        "evidence_bundle",
        "chapters_index",
        "book_synthesis_prompt",
    }
    assert outcome.inputs["book_state"] == sha256_file(book_json)
    assert outcome.inputs["source"] == sha256_file(source)
    assert outcome.inputs["evidence_bundle"] == sha256_file(analysis / "evidence_bundle.json")
    assert outcome.inputs["chapters_index"] == sha256_file(episode_root / "chapters" / "index.json")
    assert outcome.inputs["book_synthesis_prompt"] == load_prompt(
        tmp_path / "prompts" / "book_synthesis" / "v2.md"
    ).sha256
    card = WholeBookValue.model_validate_json(card_path.read_text(encoding="utf-8"))
    assert card.model_dump(mode="json") == _file_whole_book_value().model_dump(mode="json")
    _assert_identifier_keys(outcome)
    _assert_input_hashes(
        outcome,
        book_json,
        source,
        analysis / "evidence_bundle.json",
        episode_root / "chapters" / "index.json",
        tmp_path / "prompts" / "book_synthesis" / "v2.md",
    )


def test_title_mode_authorization_runs_before_model(tmp_path: Path) -> None:
    store, _book, _book_json, episode_root = _title_value_ready(
        tmp_path / "store", "book-title-value-auth"
    )
    model = _RecordingModel()
    stage = _make_value_stage(
        store,
        tmp_path,
        authorization=_denied(),
        title_model=model,
        file_synthesizer=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("file synthesizer must not run in title mode")
        ),
    )

    with pytest.raises(ExternalAuthorizationError) as caught:
        stage.run(_context("book-title-value-auth", "ep-1", episode_root))

    _assert_stable_public_error(caught, "external_not_authorized")
    assert model.calls == []


def test_title_mode_does_not_create_or_use_evidence_card_or_ebook_keys(
    tmp_path: Path,
) -> None:
    store, book, book_json, episode_root = _title_value_ready(
        tmp_path / "store", "book-title-value-ok"
    )
    model = _ScriptedValueModel([_title_whole_book_value()])
    file_calls: list[str] = []

    def file_synthesizer(*_args, **_kwargs):
        file_calls.append("invoked")
        raise AssertionError("title mode must not call synthesize_book")

    stage = _make_value_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        title_model=model,
        file_synthesizer=file_synthesizer,
    )
    outcome = stage.run(_context(book.book_id, "ep-1", episode_root))

    analysis = episode_root / "analysis"
    card_path = analysis / "book_value_card.json"
    provenance_path = analysis / "title_value_provenance.json"
    assert file_calls == []
    assert model.calls[0]["schema_type"] is WholeBookValue
    prompt = str(model.calls[0]["prompt"])
    assert "BEGIN_SOURCE_DATA" in prompt
    trusted, encoded = prompt.split("BEGIN_SOURCE_DATA\n", maxsplit=1)
    payload = json.loads(json.loads(encoded.split("END_SOURCE_DATA", maxsplit=1)[0]))
    payload_keys = _json_keys(payload)
    for key in _FORBIDDEN_TITLE_KEYS:
        assert key not in payload_keys
        assert f'"{key}"' not in json.dumps(payload, ensure_ascii=False)
    assert "EvidenceCard" not in prompt
    assert "https://example.com/publisher/catalog" in json.dumps(payload, ensure_ascii=False)
    assert payload["sources"][0]["supports"] == ["identity"]
    assert set(outcome.outputs) == {"book_value_card", "title_value_provenance"}
    assert outcome.outputs["book_value_card"] == card_path
    assert outcome.outputs["title_value_provenance"] == provenance_path
    assert set(outcome.inputs) == {
        "book_state",
        "title_metadata",
        "book_research",
        "title_book_synthesis_prompt",
    }
    assert outcome.inputs["book_state"] == sha256_file(book_json)
    assert outcome.inputs["title_metadata"] == sha256_file(analysis / "title_metadata.json")
    assert outcome.inputs["book_research"] == sha256_file(analysis / "book_research.json")
    assert outcome.inputs["title_book_synthesis_prompt"] == load_prompt(
        tmp_path / "prompts" / "title_book_synthesis" / "v1.md"
    ).sha256
    card = WholeBookValue.model_validate_json(card_path.read_text(encoding="utf-8"))
    assert card.model_dump(mode="json") == _title_whole_book_value().model_dump(mode="json")
    provenance = TitleValueProvenance.model_validate_json(
        provenance_path.read_text(encoding="utf-8")
    )
    assert provenance.source_urls == ("https://example.com/publisher/catalog",)
    assert provenance.book_research_sha256 == sha256_file(analysis / "book_research.json")
    assert provenance.prompt_sha256 == load_prompt(
        tmp_path / "prompts" / "title_book_synthesis" / "v1.md"
    ).sha256
    assert [item.claim_id for item in provenance.catalog] == [
        "central_question_001",
        "arc_001",
        "key_idea_001",
        "key_idea_002",
        "life_connection_001",
        "reader_value_001",
        "boundary_001",
    ]
    assert _json_keys(json.loads(provenance_path.read_text(encoding="utf-8"))).isdisjoint(
        _FORBIDDEN_TITLE_KEYS
    )
    _assert_identifier_keys(outcome)
    _assert_input_hashes(
        outcome,
        book_json,
        analysis / "title_metadata.json",
        analysis / "book_research.json",
        tmp_path / "prompts" / "title_book_synthesis" / "v1.md",
    )
    del trusted


def test_title_mode_rejects_unknown_claim(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_value_ready(
        tmp_path / "store", "book-title-unknown-claim"
    )
    value = _title_whole_book_value()
    value.supporting_claim_clusters[0].claim_ids = ["unknown_claim_001"]
    value.supporting_claim_clusters[0].chapter_regions = ["central_question"]
    model = _ScriptedValueModel([value])
    stage = _make_value_stage(
        store, tmp_path, authorization=_allowed(), title_model=model
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(caught, "unknown_claim_id", "unknown_claim_001")
    assert not (episode_root / "analysis" / "book_value_card.json").exists()
    assert not (episode_root / "analysis" / "title_value_provenance.json").exists()


def test_title_mode_rejects_research_region_mismatch(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_value_ready(
        tmp_path / "store", "book-title-region-mismatch"
    )
    value = _title_whole_book_value()
    value.supporting_claim_clusters[0].chapter_regions = ["arc"]
    model = _ScriptedValueModel([value])
    stage = _make_value_stage(
        store, tmp_path, authorization=_allowed(), title_model=model
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(caught, "cluster_regions_mismatch")
    assert not (episode_root / "analysis" / "book_value_card.json").exists()


def test_title_mode_rejects_isolated_one_cluster(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_value_ready(
        tmp_path / "store", "book-title-isolated"
    )
    value = _title_whole_book_value()
    value.supporting_claim_clusters = value.supporting_claim_clusters[:1]
    model = _ScriptedValueModel([value])
    stage = _make_value_stage(
        store, tmp_path, authorization=_allowed(), title_model=model
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(caught, "isolated_concept_framing")
    assert not (episode_root / "analysis" / "book_value_card.json").exists()


def test_title_mode_prompt_mutation_during_call_is_rejected(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_value_ready(
        tmp_path / "store", "book-title-prompt-mut"
    )
    file_path, title_path = _value_prompt_paths(tmp_path)

    def mutator() -> None:
        title_path.write_text(_TITLE_VALUE_PROMPT + "changed\n", encoding="utf-8")

    class _MutatingModel(_ScriptedValueModel):
        def complete(self, prompt: str, schema_type: type, request_dir: Path) -> object:
            mutator()
            return super().complete(prompt, schema_type, request_dir)

    model = _MutatingModel([_title_whole_book_value()])
    stage = BookValueStage(
        store,
        authorization=_allowed(),
        title_model=model,
        file_prompt_path=file_path,
        title_prompt_path=title_path,
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(caught, "content_prompt_changed")
    assert not (episode_root / "analysis" / "book_value_card.json").exists()


def test_title_mode_research_mutation_during_call_is_rejected(tmp_path: Path) -> None:
    store, book, _book_json, episode_root = _title_value_ready(
        tmp_path / "store", "book-title-research-mut"
    )
    research_path = episode_root / "analysis" / "book_research.json"

    class _MutatingModel(_ScriptedValueModel):
        def complete(self, prompt: str, schema_type: type, request_dir: Path) -> object:
            mutated = BookResearch.model_validate(
                _research_payload("Notes", ["Ada"]) | {"central_question": "Changed question."}
            )
            research_path.write_text(mutated.model_dump_json(), encoding="utf-8")
            return super().complete(prompt, schema_type, request_dir)

    model = _MutatingModel([_title_whole_book_value()])
    stage = _make_value_stage(
        store, tmp_path, authorization=_allowed(), title_model=model
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context(book.book_id, "ep-1", episode_root))

    _assert_stable_public_error(caught, "content_input_changed")
    assert not (episode_root / "analysis" / "book_value_card.json").exists()


def test_file_mode_stale_evidence_bundle_source_hash_fails_before_synthesizer(
    tmp_path: Path,
) -> None:
    store, _book_json, source, episode_root, chapters, _bundle = _file_value_ready(
        tmp_path / "store", "book-file-value-stale"
    )
    payload = json.loads(
        (episode_root / "analysis" / "evidence_bundle.json").read_text(encoding="utf-8")
    )
    payload["source_sha256"] = "d" * 64
    (episode_root / "analysis" / "evidence_bundle.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    synthesizer = _RecordingFileSynthesizer(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("synthesizer invoked"))
    )
    stage = _make_value_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        file_synthesizer=synthesizer,
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-value-stale", "ep-1", episode_root))

    _assert_stable_public_error(caught, "evidence_bundle_stale")
    assert synthesizer.calls == []


def test_file_mode_prompt_changed_before_run_rejects_without_synthesis(
    tmp_path: Path,
) -> None:
    store, _book_json, source, episode_root, chapters, _bundle = _file_value_ready(
        tmp_path / "store", "book-file-value-prompt"
    )
    synthesizer = _RecordingFileSynthesizer(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("synthesizer invoked"))
    )
    file_path, title_path = _value_prompt_paths(tmp_path)
    stage = BookValueStage(
        store,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        file_synthesizer=synthesizer,
        file_prompt_path=file_path,
        title_prompt_path=title_path,
    )
    file_path.write_text(_FILE_VALUE_PROMPT + "changed\n", encoding="utf-8")

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-value-prompt", "ep-1", episode_root))

    _assert_stable_public_error(caught, "content_prompt_changed")
    assert synthesizer.calls == []


def test_file_mode_bundle_or_source_mutation_during_call_is_rejected(tmp_path: Path) -> None:
    store, _book_json, source, episode_root, chapters, _bundle = _file_value_ready(
        tmp_path / "store", "book-file-value-mut"
    )
    bundle_path = episode_root / "analysis" / "evidence_bundle.json"

    def handler(evidence, **kwargs):
        del evidence
        bundle_path.write_text(bundle_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return BookSynthesisResult(
            status="book_value_ready",
            report=BookReport(
                value=_file_whole_book_value(),
                concepts=[],
            ),
        )

    stage = _make_value_stage(
        store,
        tmp_path,
        authorization=_allowed(),
        parser=lambda _path: _parsed_book(source, chapters),
        file_synthesizer=_RecordingFileSynthesizer(handler),
        file_model=_ScriptedValueModel([_file_whole_book_value()]),
    )

    with pytest.raises(ContentRuntimeError) as caught:
        stage.run(_context("book-file-value-mut", "ep-1", episode_root))

    _assert_stable_public_error(caught, "content_input_changed")
    assert not (episode_root / "analysis" / "book_value_card.json").exists()
