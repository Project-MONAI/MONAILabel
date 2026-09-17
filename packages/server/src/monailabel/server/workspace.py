"""Filesystem defaults for the single-workspace server process."""

import os
from pathlib import Path


def workspace_dir() -> Path:
    return Path(os.environ.get("MONAILABEL_DATA_DIR", "workspace")).expanduser()


def configure_workspace(directory: Path | None) -> Path:
    """Set runtime integration paths before providers or viewers are instantiated.

    Explicit cache overrides remain supported. Called by the process entry point,
    not Services, so constructing an isolated test service never changes process paths.
    """
    directory = (directory or workspace_dir()).expanduser().resolve()
    os.environ["MONAILABEL_DATA_DIR"] = str(directory)
    cache = Path(os.environ.setdefault("MONAILABEL_CACHE_DIR", str(directory / ".cache")))
    for key, path in {
        "MONAILABEL_DATASETS_DIR": cache / "datasets",
        "MONAILABEL_MODELS_DIR": cache / "models",
        "MONAILABEL_TOOLS_DIR": cache / "tools",
        "MONAILABEL_QUPATH_DATA": directory / "viewers" / "qupath",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
    }.items():
        os.environ.setdefault(key, str(path.expanduser().resolve()))
    return directory
