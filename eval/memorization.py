import argparse

import numpy as np
import torch

from model.gpt import PretrainedGPT

REFERENCE = "rhysjones/gpt2-774M-fineweb-150B"


def total_memorization(model_bits, data_bits):
    """Return bits memorized about a dataset of known information content (paper Sec. 3.2)."""
    return data_bits - model_bits.sum()


def unintended_memorization(model_bits, reference_bits):
    """Return bits memorized beyond the reference, clipped at zero per sample (paper App. A.13)."""
    return np.maximum(reference_bits - model_bits, 0).sum()


def score_reference(pool_path, out_path, name=REFERENCE, batch_size=64):
    """Score every sequence in the FineWeb pool with the reference model and save the bits."""
    pool = np.load(pool_path).astype(np.int64)
    reference = PretrainedGPT(name).to("cuda" if torch.cuda.is_available() else "cpu")
    bits = reference.bits(pool, batch_size)
    np.save(out_path, bits)
    print(f"{len(bits)} sequences scored, mean {bits.mean() / pool.shape[1]:.3f} bits/token")
    return bits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/fineweb.npy")
    parser.add_argument("--out", default="data/fineweb_reference_bits.npy")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    score_reference(args.pool, args.out, batch_size=args.batch_size)
