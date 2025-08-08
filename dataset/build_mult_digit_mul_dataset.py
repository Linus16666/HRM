"""Utility to generate a multiplication dataset for HRM."""

import os
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    """Configuration options for dataset generation."""
    output_dir: str = "data/mult-digit-mul"
    num_train: int = 10000
    num_test: int = 2500
    max_digits: int = 20
    print_samples: int = 3


def number_to_row(number: int, width: int) -> np.ndarray:
    """Return a row of digits for ``number`` padded to ``width``.

    Left padding is filled with ``-1`` so that it can be treated as PAD when the
    dataset is serialized.  Actual digits remain in the range ``0-9``.
    """
    digits = str(number)
    pad = width - len(digits)
    row = np.full(width, -1, dtype=np.int8)
    row[pad:] = np.frombuffer(digits.encode(), dtype=np.uint8) - ord("0")
    return row


def generate_dataset(split: str, num_examples: int, cfg: DataProcessConfig):
    """Generate one split of the dataset."""

    rng = np.random.default_rng()
    width = cfg.max_digits * 2

    # Storage for all examples in this split
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices", "max_digits"]}
    # "puzzle_id" and "example_id" track dataset indexing for HRM's loader
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for i in tqdm(range(num_examples)):
        # Generate a new multiplication example
        # Randomly choose digit lengths for the two operands
        digits_a = rng.integers(1, cfg.max_digits + 1)
        digits_b = rng.integers(1, cfg.max_digits + 1)

        # Sample numbers of the chosen length (no leading zeros).  ``np.random``
        # is limited to 64-bit integers, so we build the numbers digit-by-digit to
        # avoid overflow when ``cfg.max_digits`` is large.
        def sample_number(num_digits: int) -> int:
            first = rng.integers(1, 10)
            if num_digits == 1:
                digits = [first]
            else:
                rest = rng.integers(0, 10, size=num_digits - 1)
                digits = np.concatenate([[first], rest])
            return int("".join(str(int(d)) for d in digits))

        a = sample_number(digits_a)
        b = sample_number(digits_b)

        product = a * b

        # Convert numbers to fixed-width digit rows
        row_a = number_to_row(a, width)
        row_b = number_to_row(b, width)
        row_c = number_to_row(product, width)

        # Third row is blank in the input and filled with the product in labels
        inp = np.vstack([row_a, row_b, np.full_like(row_c, -1)])
        out = np.vstack([row_a, row_b, row_c])

        results["inputs"].append(inp)
        results["labels"].append(out)
        results["max_digits"].append(max(digits_a, digits_b))

        if i < cfg.print_samples:
            print(f"[{split}] Example {i + 1}: {a} x {b} = {product}")
            print("Input:")
            for row in inp:
                print(" ".join("." if int(d) == -1 else str(int(d)) for d in row))
            print("Output:")
            for row in out:
                print(" ".join("." if int(d) == -1 else str(int(d)) for d in row))
            print()

        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)

    def _seq_to_numpy(seq):
        """Flatten list of grids to token IDs where -1 denotes padding."""
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= -1) & (arr <= 9))
        # Shift by one so -1 -> 0 (PAD) and digits 0-9 -> 1-10
        return (arr + 1).astype(np.int8)

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
        "max_digits": np.array(results["max_digits"], dtype=np.int32),
    }

    # Metadata describing the dataset for the HRM loader
    metadata = PuzzleDatasetMetadata(
        seq_len=width * 3,
        vocab_size=11,  # PAD + digits 0-9
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    save_dir = os.path.join(cfg.output_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    # Each field is stored as a separate .npy file
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Entry point used from the command line."""

    generate_dataset("train", config.num_train, config)
    generate_dataset("test", config.num_test, config)
    # Only a single identifier is used ("<blank>") for visualization
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    cli()
