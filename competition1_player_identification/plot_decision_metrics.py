"""Plots retrieval metrics from one player-retrieval experiment.

Example:
    .venv/bin/python plot_decision_metrics.py \
      --metrics experiments/10_katago_decision/results/holdout_metrics.json \
      --output experiments/10_katago_decision/results/holdout_metrics.png
"""

import argparse
import json
import pathlib

import matplotlib.pyplot


PLOT_METRICS = ("top1", "top5", "top5_exponential_score", "mrr")


def parse_arguments() -> argparse.Namespace:
    """Parses the metrics JSON file and a figure destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def load_metrics(path: pathlib.Path) -> dict[str, dict[str, float]]:
    """Loads retrieval metrics from a decision or ArcFace result artifact."""
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if "query_metrics" in payload:
        return payload["query_metrics"]
    if "metrics" in payload:
        return payload["metrics"]
    raise ValueError(f"Result file has no recognized metric section: {path!s}")


def main() -> None:
    """Renders a compact summary of retrieval quality by query-game count."""
    arguments = parse_arguments()
    metrics = load_metrics(arguments.metrics)
    query_counts = sorted(metrics, key=int)
    figure, axes = matplotlib.pyplot.subplots(2, 2, figsize=(10, 7))
    for axis, metric_name in zip(axes.flat, PLOT_METRICS, strict=True):
        values = [metrics[count][metric_name] for count in query_counts]
        axis.plot(query_counts, values, marker="o", color="#176b87")
        axis.set_xlabel("black query games aggregated")
        axis.set_ylabel(metric_name.replace("_", " "))
        axis.set_title(metric_name.upper())
        axis.grid(alpha=0.25)
        axis.set_ylim(bottom=0)
    figure.suptitle("Black-only 400-way player retrieval")
    figure.tight_layout()
    figure.savefig(arguments.output, dpi=180)


if __name__ == "__main__":
    main()
