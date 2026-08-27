from pathlib import Path
from typing import Optional

import typer

from bv.books.service import create_book_from_metadata, import_book
from bv.config import AppConfig, load_config
from bv.state.store import StateStore
from bv.workflow.doctor import render_doctor_report, run_doctor
from bv.workflow.orchestrator import Orchestrator, WorkflowError, WorkflowView
from bv.workflow.runtime import RuntimeAuthorization, build_runtime_bindings


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
