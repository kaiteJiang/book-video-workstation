from typer.testing import CliRunner

from bv.cli import app


runner = CliRunner()


def test_help_lists_core_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "doctor",
        "new",
        "next",
        "status",
        "open",
        "approve",
        "import-video",
        "retry",
        "add",
    ):
        assert command in result.stdout


def test_version_is_available() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"
