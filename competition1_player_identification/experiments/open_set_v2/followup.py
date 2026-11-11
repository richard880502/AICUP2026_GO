"""Waits on a specific live extractor, then runs verified downstream
experiments.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.followup \
      --engine-pid 12345

This command does not start or restart extraction. An incomplete terminated
extractor is an error. Logs and stage state stay inside the experiment folder.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset

PACKAGE = "competition1_player_identification.experiments.open_set_v2"


def extractor_alive(process: int) -> bool:
    """Checks the exact Python extractor process and rejects PID reuse."""
    directory = pathlib.Path("/proc") / str(process)
    try:
        command = (directory / "cmdline").read_bytes()
        status = (directory / "status").read_text()
    except FileNotFoundError:
        return False
    if "State:\tZ" in status:
        return False
    if (
        b"competition1_player_identification.extract_submission_features"
        not in command
    ):
        raise RuntimeError(f"PID {process} no longer belongs to the extractor")
    return True


def wait_for_extraction(processes: list[int]) -> None:
    """Observes the live process without treating a long quiet shard as failure."""
    while any(extractor_alive(process) for process in processes):
        artifacts.write_report(
            dataset.ROOT / "followup_status.json",
            {
                "stage": "waiting_for_extraction",
                "engine_pids": processes,
                "engine_live": True,
                "observed_at_unix": time.time(),
                "complete_shards": len(
                    list(
                        (dataset.ROOT / "engine_features/candidates").glob(
                            "shard_*.npz"
                        )
                    )
                ),
            },
        )
        time.sleep(30)


def run_stage(name: str, arguments: list[str]) -> None:
    """Runs a checked downstream command and retains its full local log."""
    artifacts.write_report(
        dataset.ROOT / "followup_status.json",
        {
            "stage": name,
            "started_at_unix": time.time(),
            "pid": os.getpid(),
        },
    )
    with (dataset.ROOT / "logs" / f"{name}.log").open("w") as output:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                f"{PACKAGE}.{arguments[0]}",
                *arguments[1:],
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    """Runs engine profiles and two embedding dimensions after extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-pid",
        type=int,
        nargs="+",
        required=True,
    )
    arguments = parser.parse_args()
    wait_for_extraction(arguments.engine_pid)
    run_stage(
        "engine_profiles",
        [
            "engine_profiles",
            "--workers",
            "8",
        ],
    )
    run_stage("fisher_baseline", ["profile_baseline"])
    for dimension in (512, 1024):
        run_stage(
            f"train_engine_{dimension}",
            [
                "train",
                "--kind",
                "engine",
                "--device",
                "cuda:0",
                "--dimension",
                str(dimension),
                "--output",
                str(dataset.ROOT / f"encoder_engine_{dimension}"),
            ],
        )
    run_stage("fusion", ["fusion"])
    artifacts.write_report(
        dataset.ROOT / "followup_status.json",
        {"stage": "engine_encoders_complete", "completed_at_unix": time.time()},
        emit=True,
    )


if __name__ == "__main__":
    main()
