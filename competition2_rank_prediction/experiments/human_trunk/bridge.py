"""Loads pinned upstream HumanSL modules without changing installed packages.

Upstream files use legacy top-level imports. Temporary compatibility aliases
are restored after loading; our callers retain fully named module objects.
"""

import builtins
import collections
import importlib.util
import pathlib
import sys
import types
from typing import Any

import torch

MODULES = (
    "modelconfigs",
    "board",
    "features",
    "sgfmetadata",
    "model_pytorch",
    "gamestate",
)


def load_assets(
    root: pathlib.Path,
) -> tuple[dict[str, types.ModuleType], dict[str, Any]]:
    """Loads the pinned source and official checkpoint with bounded CPU threads."""
    torch.set_num_threads(4)
    return (
        load_vendor(root / "vendor/KataGo-human-v1.15.0/python"),
        checkpoint(root / "models/human_trunk/b18c384nbt-humanv0.ckpt"),
    )


def load_vendor(root: pathlib.Path) -> dict[str, types.ModuleType]:
    """Loads the pinned vendor code under isolated fully qualified names.

    Args:
        root: The upstream release's python source directory.

    Raises:
        ImportError: Required source or its loader is unavailable.
    """
    previous = {name: sys.modules.get(name) for name in MODULES}
    loaded = {}
    try:
        for name in MODULES:
            full_name = f"aicup_humansl_vendor.{name}"
            specification = importlib.util.spec_from_file_location(
                full_name, root / f"{name}.py"
            )
            if specification is None or specification.loader is None:
                raise ImportError(f"Cannot load pinned vendor module {name!r}")
            module = importlib.util.module_from_spec(specification)
            sys.modules[full_name] = module
            sys.modules[name] = module
            specification.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return loaded


def checkpoint(path: pathlib.Path) -> dict[str, Any]:
    """Loads the official checkpoint using only inspected standard containers."""
    with torch.serialization.safe_globals(
        [
            collections.defaultdict,
            builtins.float,
            builtins.set,
        ]
    ):
        return torch.load(path, map_location="cpu", weights_only=True)


def build_model(
    modules: dict[str, types.ModuleType],
    payload: dict[str, Any],
    device: str,
    use_swa: bool,
) -> torch.nn.Module:
    """Loads the official raw or SWA weights into an unmodified upstream model."""
    model = modules["model_pytorch"].Model(payload["config"], 19)
    model.initialize()
    values = payload["swa_model"] if use_swa else payload["model"]
    state = {}
    for key, value in values.items():
        while key.startswith("module."):
            key = key[7:]
        if key == "n_averaged":
            continue
        if "score_belief_offset" in key or "score_belief_parity" in key:
            continue
        state[key] = value
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
