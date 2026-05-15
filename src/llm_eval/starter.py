"""Starter workspace scaffolding for pip-installed APST users."""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path


def init_starter_workspace(target_dir: str | Path, *, force: bool = False) -> Path:
    """Copy packaged starter assets into a runnable workspace."""

    target = Path(target_dir).expanduser().resolve()
    template = resources.files("llm_eval").joinpath("starter_template")
    if not template.is_dir():
        raise RuntimeError("Packaged starter template is missing")

    target.mkdir(parents=True, exist_ok=True)
    for item in template.iterdir():
        destination = target / item.name
        if destination.exists() and not force:
            raise FileExistsError(
                f"{destination} already exists. Re-run with --force to overwrite starter files."
            )
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
    return target
