import argparse
import os
import yaml

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pretrain import PretrainConfig, init_train_state, create_dataloader


def collect_hidden_states(model, dataloader, device, max_tokens=None):
    """Iterate one epoch and collect hidden states for all layers.

    Returns two tensors shaped (num_layers, ``N``, hidden_size) for the high and
    low level modules respectively, where ``N`` is the total number of token
    positions collected across the epoch. To avoid exhausting GPU memory, states
    are moved to CPU and, if ``max_tokens`` is provided, collection stops once
    roughly that many token positions have been gathered.
    """
    high_states = []
    low_states = []
    collected = 0

    with torch.no_grad():
        for _set_name, batch, _ in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = model.initial_carry(batch)  # type: ignore
            seq_len = batch["inputs"].shape[1]
            while True:
                carry, _, _, outputs, all_finish = model(
                    carry=carry,
                    batch=batch,
                    return_hidden_states=True,
                    return_keys=["hidden_states_high", "hidden_states_low"],
                )
                z_h = outputs["hidden_states_high"].to(device="cpu", dtype=torch.float32)
                z_l = outputs["hidden_states_low"].to(device="cpu", dtype=torch.float32)
                # Keep only positions corresponding to output tokens
                z_h = z_h[:, :, -seq_len:, :].reshape(z_h.shape[0], -1, z_h.shape[-1])
                z_l = z_l[:, :, -seq_len:, :].reshape(z_l.shape[0], -1, z_l.shape[-1])
                high_states.append(z_h)
                low_states.append(z_l)
                collected += z_h.shape[1]
                if all_finish or (max_tokens and collected >= max_tokens):
                    break
            if max_tokens and collected >= max_tokens:
                break

    high_states = torch.cat(high_states, dim=1) if high_states else torch.empty(0)
    low_states = torch.cat(low_states, dim=1) if low_states else torch.empty(0)
    if max_tokens:
        high_states = high_states[:, :max_tokens]
        low_states = low_states[:, :max_tokens]
    return high_states, low_states


def pca_scatter(states, ax, title, max_points=10000):
    """Plot PCA scatter of hidden states colored by layer index."""
    num_layers, n_points, hidden = states.shape
    # Flatten across layers but keep track of layer ids
    points = states.permute(1, 0, 2).reshape(-1, hidden)
    layer_ids = (
        torch.arange(num_layers).unsqueeze(1).repeat(1, n_points).reshape(-1)
    )
    if points.shape[0] > max_points:
        idx = torch.randperm(points.shape[0])[:max_points]
        points = points[idx]
        layer_ids = layer_ids[idx]

    # PCA to two components
    _, _, v = torch.pca_lowrank(points, q=2)
    coords = (points @ v[:, :2]).cpu().numpy()
    layer_ids = layer_ids.cpu().numpy()

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=layer_ids, cmap="viridis", s=4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    return sc


def layer_similarity(states, ax, title):
    """Plot cosine similarity between layer-wise averaged hidden states."""
    mean_states = states.mean(dim=1)
    sim = F.cosine_similarity(
        mean_states.unsqueeze(1), mean_states.unsqueeze(0), dim=-1
    )
    im = ax.imshow(sim.cpu().numpy(), vmin=-1, vmax=1, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    return im


def main():
    parser = argparse.ArgumentParser(
        description="Run one epoch and visualise PCA of hidden states."
    )
    parser.add_argument("checkpoint", help="Path to a model checkpoint")
    parser.add_argument(
        "--output", default="hidden_state_pca.png", help="Output image file"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10000,
        help="Max token positions to collect for analysis",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(
        os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml"), "r"
    ) as f:
        cfg = PretrainConfig(**yaml.safe_load(f))

    dataloader, metadata = create_dataloader(
        cfg,
        "train",
        rank=0,
        world_size=1,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=cfg.global_batch_size,
    )

    train_state = init_train_state(cfg, metadata, world_size=1)
    model = train_state.model.to(device)
    state = torch.load(args.checkpoint, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    model.eval()

    high_states, low_states = collect_hidden_states(
        model, dataloader, device, max_tokens=args.max_points
    )

    if high_states.numel() == 0 or low_states.numel() == 0:
        raise RuntimeError("No hidden states were collected. Check dataset and model.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sc_high = pca_scatter(
        high_states, axes[0, 0], "High module PCA", args.max_points
    )
    sc_low = pca_scatter(
        low_states, axes[0, 1], "Low module PCA", args.max_points
    )
    im_high = layer_similarity(high_states, axes[1, 0], "High module similarity")
    im_low = layer_similarity(low_states, axes[1, 1], "Low module similarity")

    fig.colorbar(sc_high, ax=axes[0, 0], label="Layer index")
    fig.colorbar(sc_low, ax=axes[0, 1], label="Layer index")
    fig.colorbar(im_high, ax=axes[1, 0])
    fig.colorbar(im_low, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(args.output)
    plt.show()


if __name__ == "__main__":
    main()
