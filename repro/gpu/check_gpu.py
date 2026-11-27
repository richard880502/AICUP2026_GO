"""Exercise the competition models with a small synthetic batch on GPU."""

import torch

from competition1_player_identification.experiments.trunk_style import train
from competition2_rank_prediction.transformer_rank import RankTransformer


def check_training() -> None:
    """Run both GPU training paths through loss, backpropagation, and update."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Start the image with GPU access.")

    device = torch.device("cuda:0")
    torch.manual_seed(20261125)
    train.use_all_moves()
    token_width = train.LAYOUT.trunk + train.LAYOUT.engine
    c1 = train.StyleEncoder(2).to(device)
    optimizer = torch.optim.AdamW(c1.parameters(), lr=5e-4)
    tokens = torch.randn(4, train.LAYOUT.length, token_width, device=device)
    valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
    colors = torch.tensor([0, 1, 0, 1], device=device)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = c1(tokens, valid, colors)
    loss = sum(train.losses(c1, embeddings.float(), labels))
    if not torch.isfinite(loss):
        raise RuntimeError("GPU training returned a non-finite loss.")
    loss.backward()
    optimizer.step()
    print(f"Competition 1 GPU optimizer step: loss={loss.item():.4f}")

    model = RankTransformer(15, n_tokens=96).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    tokens = torch.randn(4, 96, 15, device=device)
    valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
    log_probabilities = model(tokens, valid)
    loss = torch.nn.functional.nll_loss(log_probabilities, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("GPU training returned a non-finite loss.")
    loss.backward()
    optimizer.step()
    print(f"Competition 2 GPU optimizer step: loss={loss.item():.4f}")

    print(f"PyTorch {torch.__version__} / CUDA {torch.version.cuda} / {torch.cuda.get_device_name(device)}")


if __name__ == "__main__":
    check_training()
