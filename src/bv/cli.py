from pathlib import Path
from typing import Optional

import typer

from bv.books.service import create_book_from_metadata, import_book
from bv.content.story import (
    StoryEvidence,
    StoryFactualReview,
    StoryMetadata,
    StorySection,
    import_story_candidate,
)
from bv.config import AppConfig, load_config
from bv.state.store import StateStore
from bv.workflow.doctor import render_doctor_report, run_doctor
from bv.workflow.orchestrator import Orchestrator, WorkflowError, WorkflowView
from bv.workflow.runtime import RuntimeAuthorization, build_runtime_bindings
from bv.voice.catalog import load_voice_catalog, select_story_voices


app = typer.Typer(no_args_is_help=True, help="BV Workstation")


def version_callback(value: bool) -> None:
    if value:
        typer.echo("0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
) -> None:
    return None


def _stub(name: str) -> None:
    typer.echo(f"{name}: not yet available")


def build_orchestrator(
    config: AppConfig,
    authorization: RuntimeAuthorization | None = None,
) -> Orchestrator:
    """Create a coordinator from the current command's authorization only."""
    bindings = build_runtime_bindings(
        config,
        authorization if authorization is not None else RuntimeAuthorization(),
    )
    return Orchestrator(
        store=StateStore(config.workspace_dir),
        stages=bindings.stages,
        gate_approvers=bindings.gate_approvers,
    )


def _render_view(view: WorkflowView) -> None:
    typer.echo(f"Status: {view.status}")
    typer.echo(f"Present shots: {', '.join(view.present_shot_ids) or 'none'}")
    typer.echo(f"Missing shots: {', '.join(view.missing_shot_ids) or 'none'}")
    if view.failure_code:
        typer.echo(f"Failure: {view.failure_code}")
    typer.echo(f"Next: {view.next_command}")


def _workflow_call(action):
    try:
        return action()
    except WorkflowError as exc:
        typer.echo(f"Error: {exc.error_code}")
        raise typer.Exit(code=1) from None


def load_config_default() -> AppConfig:
    for candidate in (Path("config.yaml"), Path("config.example.yaml")):
        if candidate.is_file():
            return load_config(candidate)
    raise FileNotFoundError(
        "neither config.yaml nor config.example.yaml exists in the current directory"
    )


@app.command()
def doctor(
    live: bool = typer.Option(
        False, "--live", help="Run real model and cloud probes"
    ),
) -> None:
    report = run_doctor(load_config_default(), live=live)
    render_doctor_report(report)
    if report.has_blocking_missing:
        raise typer.Exit(code=1)


@app.command()
def new(
    book_file: Path | None = typer.Argument(None),
    title: str | None = typer.Option(None, "--title"),
    author: list[str] = typer.Option([], "--author"),
) -> None:
    if book_file is None:
        if title is None or not str(title).strip():
            typer.echo("Error: book_source_required")
            raise typer.Exit(code=2)
        if not any(str(value).strip() for value in author):
            typer.echo("Error: book_author_required")
            raise typer.Exit(code=2)
        config = load_config_default()
        result = create_book_from_metadata(
            title,
            author,
            StateStore(config.workspace_dir),
        )
        typer.echo(f"Created: {result.title}")
        typer.echo(f"Book ID: {result.book_id}")
        typer.echo("Episode: E001")
        typer.echo(f"Next: bv next {result.book_id}")
        return

    config = load_config_default()
    result = import_book(
        book_file,
        StateStore(config.workspace_dir),
        title=title if title is not None else book_file.stem,
        authors=author,
    )
    typer.echo(f"Imported: {result.title}")
    typer.echo(f"Book ID: {result.book_id}")
    typer.echo("Episode: E001")
    typer.echo(f"Next: bv next {result.book_id}")


@app.command("next")
def next_command(
    book_id: str,
    episode_id: str = typer.Argument("E001"),
    allow_external: bool = typer.Option(
        False,
        "--allow-external",
        help="Authorize network, cloud, or paid model calls for this command only",
    ),
) -> None:
    workflow = build_orchestrator(
        load_config_default(),
        RuntimeAuthorization(allow_external=allow_external),
    )
    _render_view(_workflow_call(lambda: workflow.next(book_id, episode_id)))


@app.command("import-story")
def import_story(
    book_id: str,
    episode_id: str,
    story_file: Path,
    contract_file: Path,
    candidate_name: str = typer.Option("candidate.txt", "--candidate-name"),
) -> None:
    """Import a directly authored, evidence-bound story as a review candidate."""
    try:
        text = story_file.read_text(encoding="utf-8")
        import json

        contract = json.loads(contract_file.read_text(encoding="utf-8"))
        result = import_story_candidate(
            store=StateStore(load_config_default().workspace_dir),
            book_id=book_id,
            episode_id=episode_id,
            text=text,
            metadata=StoryMetadata.model_validate(contract["metadata"]),
            evidence=[StoryEvidence.model_validate(item) for item in contract["evidence"]],
            sections=[StorySection.model_validate(item) for item in contract["sections"]],
            factual_review=StoryFactualReview.model_validate(contract["factual_review"]),
            candidate_filename=candidate_name,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=2) from None
    typer.echo("Status: awaiting_script_review")
    typer.echo(f"Candidate: {result.candidate_path}")
    typer.echo(f"Manifest: {result.manifest_path}")
    typer.echo(f"Warnings: {', '.join(result.warnings) or 'none'}")
    typer.echo(f"Next: bv approve {book_id} {episode_id} script")


@app.command("voice-pool")
def voice_pool(
    gender: str = typer.Option("any", "--gender"),
    tag: list[str] = typer.Option([], "--tag"),
) -> None:
    """List up to three catalog candidates without probing or synthesizing."""
    normalized_gender = gender.strip().lower()
    if normalized_gender not in {"male", "female", "any"}:
        typer.echo("Error: voice_gender_invalid")
        raise typer.Exit(code=2)
    try:
        selections = select_story_voices(
            load_voice_catalog(),
            narrator_gender=normalized_gender,
            desired_story_tags=tuple(tag or ["natural"]),
            limit=3,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=2) from None
    if not selections:
        typer.echo("No matching voices")
        return
    for selection in selections:
        typer.echo(f"Name: {selection.voice.display_name}")
        typer.echo(f"Voice ID: {selection.voice.voice_id}")
        typer.echo(f"Reasons: {'; '.join(selection.reasons)}")
        typer.echo(f"Audition: {selection.voice.audition_status}")
        typer.echo("Live availability: unknown")


@app.command()
def status(book_id: str | None = None, episode_id: str | None = None) -> None:
    if book_id is None:
        typer.echo("Error: book_id_required")
        raise typer.Exit(code=2)
    workflow = build_orchestrator(load_config_default())
    _render_view(_workflow_call(lambda: workflow.status_view(book_id, episode_id or "E001")))


@app.command("open")
def open_artifact(book_id: str, episode_id: str, target: str) -> None:
    workflow = build_orchestrator(load_config_default())
    opened = _workflow_call(lambda: workflow.open_target(book_id, episode_id, target))
    typer.echo(opened.listing)


@app.command()
def approve(
    book_id: str,
    episode_id: str,
    gate: str,
    allow_external: bool = typer.Option(
        False,
        "--allow-external",
        help="Authorize network, cloud, or paid model calls for this command only",
    ),
) -> None:
    workflow = build_orchestrator(
        load_config_default(),
        RuntimeAuthorization(allow_external=allow_external),
    )
    _render_view(_workflow_call(lambda: workflow.approve_gate(book_id, episode_id, gate)))


@app.command("import-video")
def import_video(book_id: str, episode_id: str, shot_id: str, video_file: Path) -> None:
    workflow = build_orchestrator(load_config_default())
    _render_view(
        _workflow_call(
            lambda: workflow.import_video(book_id, episode_id, shot_id, video_file)
        )
    )


@app.command()
def retry(book_id: str, episode_id: str = "E001") -> None:
    workflow = build_orchestrator(load_config_default())
    _render_view(_workflow_call(lambda: workflow.retry(book_id, episode_id)))


@app.command()
def add(book_id: str) -> None:
    workflow = build_orchestrator(load_config_default())
    episode_id = _workflow_call(lambda: workflow.add(book_id))
    typer.echo(f"Episode: {episode_id}")
