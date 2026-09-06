from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from bv.state.store import StateStore
from bv.workflow.media_runtime import MediaProductionService, MediaWorkflowError
from bv.workflow.media_stages import PrepareIllustrationsStage, StyleSelectionStage
from bv.workflow.runtime import RuntimeAuthorization


CATALOG = Path(
    "vendor/story_to_handdrawn_video/references/handdrawn-style-library.json"
).resolve()


class _Images:
    pass


class _SelfReportedCatalog:
    catalog_path = CATALOG

    def run(self, _context):
        raise AssertionError


class _UntrustedStyleSelection(StyleSelectionStage):
    def run(self, _context):
        raise AssertionError


class _UntrustedPrepare(PrepareIllustrationsStage):
    def run(self, _context):
        raise AssertionError


def _style_stage(catalog: Path) -> StyleSelectionStage:
    return StyleSelectionStage(
        model=object(),  # type: ignore[arg-type]
        authorization=RuntimeAuthorization(),
        catalog_path=catalog,
        prompt_path=Path("prompts/illustration/style_select.md"),
    )


def _untrusted_style_stage(catalog: Path) -> _UntrustedStyleSelection:
    return _UntrustedStyleSelection(
        model=object(),  # type: ignore[arg-type]
        authorization=RuntimeAuthorization(),
        catalog_path=catalog,
        prompt_path=Path("prompts/illustration/style_select.md"),
    )


def _service(
    tmp_path: Path,
    *,
    select_catalog: Path | None = None,
    prepare_catalog: Path | None = None,
    explicit_catalog: Path | None = None,
) -> MediaProductionService:
    stages = {}
    if select_catalog is not None:
        stages["select_style"] = _style_stage(select_catalog)
    if prepare_catalog is not None:
        stages["prepare_representatives"] = PrepareIllustrationsStage(
            catalog_path=prepare_catalog
        )
    return MediaProductionService(
        store=StateStore(tmp_path / "workspace"),
        stages=stages,
        images=_Images(),  # type: ignore[arg-type]
        restyle_catalog_path=explicit_catalog,
    )


@pytest.mark.parametrize("explicit", (False, True))
def test_service_derives_matching_catalog_from_production_stages(
    tmp_path: Path,
    explicit: bool,
) -> None:
    service = _service(
        tmp_path,
        select_catalog=CATALOG,
        prepare_catalog=CATALOG,
        explicit_catalog=CATALOG if explicit else None,
    )

    assert service.restyle_catalog_path == CATALOG


def test_service_allows_catalogless_constructor_for_legacy(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service.restyle_catalog_path is None


def test_service_ignores_catalog_self_reported_by_unknown_stages(
    tmp_path: Path,
) -> None:
    service = MediaProductionService(
        store=StateStore(tmp_path / "workspace"),
        stages={
            "select_style": _SelfReportedCatalog(),
            "prepare_representatives": _SelfReportedCatalog(),
        },
        images=_Images(),  # type: ignore[arg-type]
    )

    assert service.restyle_catalog_path is None


def test_service_ignores_catalog_self_reported_by_production_subclasses(
    tmp_path: Path,
) -> None:
    service = MediaProductionService(
        store=StateStore(tmp_path / "workspace"),
        stages={
            "select_style": _untrusted_style_stage(CATALOG),
            "prepare_representatives": _UntrustedPrepare(catalog_path=CATALOG),
        },
        images=_Images(),  # type: ignore[arg-type]
    )

    assert service.restyle_catalog_path is None


@pytest.mark.parametrize("subclass", ("select", "prepare"))
def test_service_rejects_exact_provider_paired_with_untrusted_subclass(
    tmp_path: Path,
    subclass: str,
) -> None:
    select = (
        _untrusted_style_stage(CATALOG)
        if subclass == "select"
        else _style_stage(CATALOG)
    )
    prepare = (
        _UntrustedPrepare(catalog_path=CATALOG)
        if subclass == "prepare"
        else PrepareIllustrationsStage(catalog_path=CATALOG)
    )

    with pytest.raises(
        MediaWorkflowError, match="media_catalog_configuration_invalid"
    ):
        MediaProductionService(
            store=StateStore(tmp_path / "workspace"),
            stages={
                "select_style": select,
                "prepare_representatives": prepare,
            },
            images=_Images(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("missing", ("select", "prepare"))
def test_service_rejects_incomplete_production_catalog_providers(
    tmp_path: Path,
    missing: str,
) -> None:
    kwargs = {
        "select_catalog": None if missing == "select" else CATALOG,
        "prepare_catalog": None if missing == "prepare" else CATALOG,
    }

    with pytest.raises(
        MediaWorkflowError, match="media_catalog_configuration_invalid"
    ):
        _service(tmp_path, **kwargs)


def test_service_rejects_stage_catalog_disagreement(tmp_path: Path) -> None:
    other = tmp_path / "other-catalog.json"
    other.write_bytes(CATALOG.read_bytes())

    with pytest.raises(
        MediaWorkflowError, match="media_catalog_configuration_invalid"
    ):
        _service(
            tmp_path,
            select_catalog=CATALOG,
            prepare_catalog=other,
        )


def test_service_rejects_explicit_catalog_conflict(tmp_path: Path) -> None:
    other = tmp_path / "other-catalog.json"
    other.write_bytes(CATALOG.read_bytes())

    with pytest.raises(
        MediaWorkflowError, match="media_catalog_configuration_invalid"
    ):
        _service(
            tmp_path,
            select_catalog=CATALOG,
            prepare_catalog=CATALOG,
            explicit_catalog=other,
        )


def test_service_rejects_missing_explicit_catalog(tmp_path: Path) -> None:
    with pytest.raises(
        MediaWorkflowError, match="media_catalog_configuration_invalid"
    ):
        _service(tmp_path, explicit_catalog=tmp_path / "missing-catalog.json")


def test_service_rejects_catalog_symlink(tmp_path: Path) -> None:
    link = tmp_path / "catalog-link.json"
    try:
        os.symlink(CATALOG, link)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")

    with pytest.raises(
        MediaWorkflowError, match="media_catalog_configuration_invalid"
    ):
        _service(tmp_path, explicit_catalog=link)


def test_service_rejects_catalog_under_windows_junction(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction contract")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "catalog.json").write_bytes(CATALOG.read_bytes())
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside", encoding="utf-8")
    junction = tmp_path / "catalog-junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("junction unavailable")

    with pytest.raises(
        MediaWorkflowError, match="media_catalog_configuration_invalid"
    ):
        _service(tmp_path, explicit_catalog=junction / "catalog.json")
    assert sentinel.read_text(encoding="utf-8") == "outside"
