# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Reports either the declared inner-gate stop or a complete policy pilot.

Run with PYTHONPATH=competition2_rank_prediction MPLBACKEND=Agg python -m
plus this path. Incomplete staged runs never produce an outer utility score.
"""

import hashlib
import json
import pathlib

from competition2_rank_prediction.experiments.human_finetune import (
    evaluate as hidden_evaluate,
)


def main() -> None:
    """Audits recorded sources and refuses to score missing seeds or failed gates."""
    root = pathlib.Path.cwd()
    directory = root / "production/policy_adaptation_20261109"
    report = directory / "report.json"
    if report.exists():
        raise FileExistsError(f"Refusing existing policy report: {report}")
    for path, expected in json.loads(
        (directory / "input_hashes.json").read_text()
    ).items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed training input/source: {path}")
    if (directory / "STOPPED_AT_INNER_GATE").exists():
        gate = json.loads((directory / "inner_gate.json").read_text())
        if gate["passes"] or (directory / "COMPLETE").exists():
            raise ValueError("Inconsistent stage-stop markers")
        result = {
            "scope": "inner-only staged feasibility screen; no outer utility",
            "inner_gate": gate,
            "full_three_seed_comparison_completed": False,
            "outer_utility_computed": False,
            "screen_pass": False,
            "official_submission_created": False,
        }
    else:
        if not (directory / "COMPLETE").exists():
            raise RuntimeError("Policy training is still incomplete")
        if not json.loads((directory / "inner_gate.json").read_text())[
            "passes"
        ]:
            raise ValueError("Completed run did not pass the fixed inner gate")
        predictions, metadata = hidden_evaluate.load(directory)
        result = hidden_evaluate.coverage_evaluate.summarize(
            predictions, metadata
        )
        result["scope"] = (
            "exploratory legal-policy adaptation own branch; not Public"
        )
        result["arms"] = {"2": "frozen_policy", "8": "adapted_policy"}
        result["full_three_seed_comparison_completed"] = True
    result["evaluation_source_sha256"] = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()
    ).hexdigest()
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
