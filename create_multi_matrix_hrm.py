import argparse
import json
import os
import random
import yaml

import numpy as np
import torch
from tqdm import tqdm

from pretrain import PretrainConfig, init_train_state
from dataset.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID


def number_to_row(number: int, width: int) -> np.ndarray:
    """Return a row of digits for ``number`` padded to ``width``.

    Left padding is filled with ``-1`` so that it can be treated as PAD when
    tokenized, mirroring the dataset generation utility.
    """
    digits = str(number)
    pad = width - len(digits)
    row = np.full(width, -1, dtype=np.int64)
    row[pad:] = np.frombuffer(digits.encode(), dtype=np.uint8) - ord("0")
    return row


def generate_number(digits: int) -> int:
    return random.randint(10 ** (digits - 1), 10 ** digits - 1)


def generate_multiplication_problem(d1: int, d2: int):
    a = generate_number(d1)
    b = generate_number(d2)
    return a, b, a * b


def build_batch(a: int, b: int, metadata: PuzzleDatasetMetadata, device: torch.device):
    width = metadata.seq_len // 3
    if any(len(str(x)) > width for x in (a, b, a * b)):
        raise ValueError("Numbers do not fit in model sequence length")

    row_a = number_to_row(a, width)
    row_b = number_to_row(b, width)
    blank = np.full(width, -1, dtype=np.int64)

    inp = np.vstack([row_a, row_b, blank]) + 1
    inputs = torch.tensor(inp.reshape(1, -1), dtype=torch.int32, device=device)
    labels = torch.full_like(inputs, IGNORE_LABEL_ID)
    batch = {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": torch.full(
            (1,), metadata.blank_identifier_id, dtype=torch.int32, device=device
        ),
    }
    return batch, width


def hrm_predict(model: torch.nn.Module, metadata: PuzzleDatasetMetadata, a: int, b: int, device: torch.device) -> int:
    """Predict ``a * b`` using an HRM model.

    The model runs in inference mode with ACT halting.  The logits from the
    final iteration are reshaped back into three rows matching the dataset
    layout before extracting the digits of the product from the last row.
    """
    batch, width = build_batch(a, b, metadata, device)
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        while True:
            carry, _, _, outputs, all_finish = model(
                return_keys=["logits"], carry=carry, batch=batch
            )
            if bool(all_finish):
                break

    preds = (
        outputs["logits"].argmax(dim=-1).squeeze(0).view(3, width).cpu().numpy() - 1
    )
    digits = preds[2]
    digits = digits[digits != -1]
    return int("".join(map(str, digits)) or "0")


def test_accuracy_for_pair(model, metadata, d1: int, d2: int, n_samples: int, device: torch.device) -> float:
    correct = 0
    for _ in range(n_samples):
        a, b, true_answer = generate_multiplication_problem(d1, d2)
        pred = hrm_predict(model, metadata, a, b, device)
        if pred == true_answer:
            correct += 1
    return (correct / n_samples) * 100.0


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(checkpoint_dir, "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
        config.checkpoint_path = checkpoint_dir
    with open(os.path.join(config.data_path, "train", "dataset.json"), "r") as f:
        metadata = PuzzleDatasetMetadata(**json.load(f))
    train_state = init_train_state(config, metadata, world_size=1)
    state = torch.load(checkpoint_path, map_location=device)
    try:
        train_state.model.load_state_dict(state, assign=True)
    except Exception:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()}, assign=True)
    train_state.model.eval()
    return train_state.model, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate 20x20 accuracy matrix for HRM model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--samples", type=int, default=50, help="Samples per digit pair")
    parser.add_argument("--output", default="accuracy_matrix_hrm.npy", help="Output .npy file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_model(args.checkpoint, device)

    accuracy_matrix = np.zeros((20, 20), dtype=np.float32)
    for i in tqdm(range(20), desc="Digit 1"):
        for j in range(20):
            try:
                acc = test_accuracy_for_pair(model, metadata, i + 1, j + 1, args.samples, device)
            except ValueError:
                acc = float("nan")
            accuracy_matrix[i][j] = acc

    np.save(args.output, accuracy_matrix)
    print(accuracy_matrix)


if __name__ == "__main__":
    main()
