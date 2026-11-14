import numpy as np

from data.synthetic import SyntheticData


def test_shape_and_range():
    """Train and test sets have the right shape and token range."""
    data = SyntheticData(n_train=1000, n_test=100)
    assert data.train.shape == (1000, 64)
    assert data.test.shape == (100, 64)
    assert data.train.min() >= 0 and data.train.max() < 2048


def test_same_seed_same_data():
    """The same seed reproduces the same data."""
    a = SyntheticData(n_train=100, seed=0)
    b = SyntheticData(n_train=100, seed=0)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)


def test_different_seed_different_data():
    """Different seeds give different data."""
    a = SyntheticData(n_train=100, seed=0)
    b = SyntheticData(n_train=100, seed=1)
    assert not np.array_equal(a.train, b.train)


def test_test_set_independent_of_train_size():
    """For a given seed, the test set is the same for every training set size."""
    a = SyntheticData(n_train=100, seed=0)
    b = SyntheticData(n_train=5000, seed=0)
    assert np.array_equal(a.test, b.test)


def test_train_test_disjoint():
    """No test sequence also appears in the training set."""
    data = SyntheticData(n_train=10_000, n_test=10_000)
    train = {row.tobytes() for row in data.train}
    assert not any(row.tobytes() in train for row in data.test)


def test_entropy_bits():
    """Entropy is N * S * log2(V), matching the paper's Table 3."""
    assert SyntheticData(n_train=4096).entropy_bits() == 4096 * 64 * 11
    assert SyntheticData(n_train=4096, seq_len=4).entropy_bits() == 180_224


def test_tokens_uniform():
    """Token counts pass a chi-square test for uniformity."""
    data = SyntheticData(n_train=4096)
    counts = np.bincount(data.train.ravel(), minlength=2048)
    expected = data.train.size / 2048
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert abs(chi2 - 2047) < 6 * np.sqrt(2 * 2047)
