import numpy as np
import torch

from data.synthetic import SyntheticData
from eval.memorization import score_reference, total_memorization, unintended_memorization
from model.gpt import GPT, PretrainedGPT


def test_total_memorization_arithmetic():
    """Memorization is the dataset's information content minus the model's code length."""
    assert total_memorization(np.array([700.0, 600.0]), data_bits=1408.0) == 108.0


def test_uniform_model_memorizes_nothing():
    """A model that predicts uniformly has zero memorization of synthetic data."""
    data = SyntheticData(n_train=50, n_test=1)
    model = GPT(2, 64, vocab_size=2048)
    torch.nn.init.zeros_(model.gpt2.transformer.wte.weight)
    assert abs(total_memorization(model.bits(data.train), data.entropy_bits())) < 0.1


def test_unintended_memorization_clips_each_sample():
    """Samples the reference compresses better count as zero, not negative."""
    reference = np.array([10.0, 5.0, 3.0])
    model = np.array([4.0, 8.0, 3.0])
    assert unintended_memorization(model, reference) == 6.0


def test_unintended_memorization_zero_when_equal():
    """A model no better than the reference has memorized nothing."""
    bits = np.array([1.0, 2.0, 3.0])
    assert unintended_memorization(bits, bits) == 0.0


def test_score_reference_saves_bits(tmp_path):
    """Scoring a pool writes one value per sequence, matching the reference's own bits."""
    GPT(2, 64, vocab_size=50257).gpt2.save_pretrained(tmp_path / "ref")
    pool = np.random.default_rng(0).integers(0, 50256, size=(10, 64)).astype(np.uint16)
    np.save(tmp_path / "pool.npy", pool)
    bits = score_reference(tmp_path / "pool.npy", tmp_path / "bits.npy", name=tmp_path / "ref")
    expected = PretrainedGPT(tmp_path / "ref").bits(pool.astype(np.int64))
    assert np.allclose(np.load(tmp_path / "bits.npy"), expected)
    assert np.allclose(bits, expected)
