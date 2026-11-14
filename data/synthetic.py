import numpy as np


class SyntheticData:
    def __init__(self, n_train, n_test=10_000, seq_len=64, vocab_size=2048, seed=0):
        """Sample uniform random token sequences for training and testing."""
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self.test = rng.integers(0, vocab_size, size=(n_test, seq_len))
        self.train = rng.integers(0, vocab_size, size=(n_train, seq_len))

    def entropy_bits(self):
        """Return the exact information content of the training set in bits."""
        return self.train.size * np.log2(self.vocab_size)
