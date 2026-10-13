"""Small shared helpers for persisting experiment result JSON."""

import json
import pathlib


def write_json(output_path: pathlib.Path, payload: dict[str, object]) -> None:
    """Creates parent directories and writes an indented JSON result payload."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
