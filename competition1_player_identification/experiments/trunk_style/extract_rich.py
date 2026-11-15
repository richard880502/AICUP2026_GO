"""Richer per-move tokens from the frozen HumanSL trunk (pilot).

Reproduces Competition 1 style submission v9, step 2 of 5 (after
original_match.py): `fit` writes the PCA projection, `run` writes the 960-D
`*_rich` token shards of the training corpus and the four pools that
rich_full.py and rich_pipeline.py read. Competition 2's opponent_identity
chain also runs it on the opponent side of its game lists.

Per own move, at the played point:
  A  final trunk activation (384)                       -- the existing token
  B  final trunk activation averaged over the 3x3 neighbourhood
  C  output of residual block 12
  D  output of residual block 6
B, C, D are reduced to 192 principal components each (fitted once on a sample
of training games), giving a 960-D token.

Fit:     extract_rich.py fit  --games <csv> --output <dir> --limit 1024 ...
Extract: extract_rich.py run  --games <csv> --output <dir> ...   (extract.py options)

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.extract_rich \
      run --games <csv> --output <dir> --samples 160 --batch-games 8 --workers 8
"""
import pathlib
import sys

import numpy as np
import torch

from competition1_player_identification.experiments.trunk_style import extract

PCA = pathlib.Path("production/trunk_style_20261113/rich_pca.npz")
NAMES = ("trunkfinal", "rconv12.out", "rconv6.out")
MODE = ""
STATE: dict = {}


def infer(  # pylint: disable=unused-argument,too-many-locals
    model, modules, batch, local_only
):
    """Returns [positions, 960] (or raw 1536 when fitting) half-precision tokens."""
    arrays = [
        torch.as_tensor(np.concatenate([item[0][column] for item in batch]), device="cuda:0")
        for column in range(3)
    ]
    points = torch.as_tensor(np.concatenate([item[1] for item in batch]), device="cuda:0")
    extra = modules["model_pytorch"].ExtraOutputs(list(NAMES))
    rows = torch.arange(len(points), device="cuda:0")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        model(*arrays, extra_outputs=extra)
        missing = [name for name in NAMES if name not in extra.returned]
        if missing:
            raise ValueError(f"Missing activations {missing}; available {extra.available[:60]}")
        trunk = extra.returned["trunkfinal"].float()
        # torch builtin; pylint cannot see that it is callable.
        area = torch.nn.functional.avg_pool2d(  # pylint: disable=not-callable
            trunk, 3, 1, 1, count_include_pad=False
        )
        blocks = [trunk, area, extra.returned[NAMES[1]].float(), extra.returned[NAMES[2]].float()]
        local = [
            block.flatten(start_dim=2).transpose(1, 2)[rows, points] for block in blocks
        ]
        if MODE == "fit":
            tokens = torch.cat(local, 1)
        else:
            if not STATE:
                with np.load(PCA) as saved:
                    STATE["mean"] = torch.from_numpy(saved["mean"]).to("cuda:0")
                    STATE["components"] = torch.from_numpy(saved["components"]).to("cuda:0")
            reduced = [
                (local[index + 1] - STATE["mean"][index]) @ STATE["components"][index].T
                for index in range(3)
            ]
            tokens = torch.cat([local[0]] + reduced, 1)
    values = tokens.cpu().numpy().astype(np.float16)
    if not np.isfinite(values).all():
        raise ValueError("Malformed or overflowing trunk tokens")
    return values


def main() -> None:
    """Fits the projection (`fit`) or extracts 960-D tokens (`run`)."""
    global MODE  # pylint: disable=global-statement
    MODE = sys.argv.pop(1)
    extract.infer = infer
    extract.DIMENSION = 1536 if MODE == "fit" else 960
    if MODE == "run" and not PCA.exists():
        raise SystemExit("Fit the projection first")
    extract.main()
    if MODE == "fit":
        output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
        parts, kept = [], []
        for path in sorted(output.glob("shard_*.npz")):
            with np.load(path) as arrays:
                # NpzFile values are arrays; pylint infers them as unsubscriptable.
                parts.append(
                    arrays["tokens"][  # pylint: disable=unsubscriptable-object
                        arrays["masks"]
                    ].astype(np.float32)
                )
        values = np.concatenate(parts)
        generator = np.random.default_rng(0)
        values = values[generator.choice(len(values), min(len(values), 200000), replace=False)]
        means, components = [], []
        for index in range(1, 4):
            block = torch.from_numpy(values[:, index * 384 : (index + 1) * 384]).cuda().double()
            mean = block.mean(0)
            # torch builtin; pylint cannot see that it is callable.
            _, singular, basis = torch.linalg.svd(  # pylint: disable=not-callable
                block - mean, full_matrices=False
            )
            variance = singular**2
            kept.append(float(variance[:192].sum() / variance.sum()))
            means.append(mean.float().cpu().numpy())
            components.append(basis[:192].float().cpu().numpy())
        np.savez(PCA, mean=np.stack(means), components=np.stack(components))
        print(
            "positions", len(values),
            "variance kept by 192 components (B, C, D):", [round(k, 4) for k in kept],
        )


if __name__ == "__main__":
    main()
