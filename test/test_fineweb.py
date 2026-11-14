import numpy as np
import pytest

from data.fineweb import FineWebData


def encode(text):
    """Stand-in tokenizer: each whitespace-separated integer is one token."""
    return [int(t) for t in text.split()]


def build(tmp_path, texts, n_sequences, seq_len=4):
    """Build a pool from fake documents and return it with its path."""
    path = tmp_path / "pool.npy"
    pool = FineWebData.build(n_sequences, path, seq_len=seq_len, texts=texts, encode=encode)
    return pool, path


def make_pool(tmp_path, n_sequences=1000):
    """Build a pool of distinct fake documents and return its path."""
    texts = [f"{i} {i + 1} {i + 2} {i + 3} 99" for i in range(n_sequences)]
    return build(tmp_path, texts, n_sequences)[1]


def test_build_truncates_to_first_tokens(tmp_path):
    """Each document contributes only its first seq_len tokens."""
    pool, _ = build(tmp_path, ["1 2 3 4 5 6 7 8", "9 10 11 12 13"], n_sequences=2)
    assert pool.tolist() == [[1, 2, 3, 4], [9, 10, 11, 12]]


def test_build_drops_short_documents(tmp_path):
    """Documents shorter than seq_len are skipped."""
    pool, _ = build(tmp_path, ["1 2 3", "4 5 6 7", "8"], n_sequences=1)
    assert pool.tolist() == [[4, 5, 6, 7]]


def test_build_drops_duplicates_created_by_truncation(tmp_path):
    """Documents that differ only after seq_len tokens become one sequence."""
    texts = ["1 2 3 4 5", "1 2 3 4 6", "1 2 3 4", "5 6 7 8"]
    pool, _ = build(tmp_path, texts, n_sequences=2)
    assert pool.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_build_stops_at_n_sequences(tmp_path):
    """Building stops once enough unique sequences are found."""
    texts = (f"{i} {i} {i} {i}" for i in range(10**9))
    pool, path = build(tmp_path, texts, n_sequences=5)
    assert pool.shape == (5, 4)
    assert np.array_equal(np.load(path), pool)


def test_build_raises_when_not_enough_data(tmp_path):
    """Building fails loudly if the stream runs out."""
    with pytest.raises(ValueError, match="only found 1"):
        build(tmp_path, ["1 2 3 4", "1 2 3 4"], n_sequences=2)


def test_build_stores_gpt2_token_ids(tmp_path):
    """The largest GPT-2 token id survives the uint16 round trip."""
    pool, _ = build(tmp_path, ["50256 50255 0 1"], n_sequences=1)
    assert pool.tolist() == [[50256, 50255, 0, 1]]


def test_split_shapes_and_dtype(tmp_path):
    """Train and test sets have the requested sizes and int64 tokens."""
    data = FineWebData(n_train=500, n_test=100, path=make_pool(tmp_path))
    assert data.train.shape == (500, 4)
    assert data.test.shape == (100, 4)
    assert data.train.dtype == np.int64
    assert data.seq_len == 4


def test_split_train_test_disjoint(tmp_path):
    """No test sequence also appears in the training set."""
    data = FineWebData(n_train=900, n_test=100, path=make_pool(tmp_path))
    train = {row.tobytes() for row in data.train}
    assert not any(row.tobytes() in train for row in data.test)


def test_split_test_set_independent_of_train_size(tmp_path):
    """For a given seed, the test set is the same for every training set size."""
    path = make_pool(tmp_path)
    a = FineWebData(n_train=100, n_test=100, path=path)
    b = FineWebData(n_train=800, n_test=100, path=path)
    assert np.array_equal(a.test, b.test)


def test_split_train_sets_nested(tmp_path):
    """For a given seed, a smaller training set is a prefix of a larger one."""
    path = make_pool(tmp_path)
    a = FineWebData(n_train=100, n_test=100, path=path)
    b = FineWebData(n_train=800, n_test=100, path=path)
    assert np.array_equal(a.train, b.train[:100])


def test_split_seed_changes_split(tmp_path):
    """Different seeds give different splits of the same pool."""
    path = make_pool(tmp_path)
    a = FineWebData(n_train=100, n_test=100, seed=0, path=path)
    b = FineWebData(n_train=100, n_test=100, seed=1, path=path)
    assert not np.array_equal(a.train, b.train)


def test_split_raises_when_pool_too_small(tmp_path):
    """Asking for more sequences than the pool holds fails loudly."""
    with pytest.raises(ValueError, match="pool has 1000"):
        FineWebData(n_train=1000, n_test=1, path=make_pool(tmp_path))


def test_split_indices_point_into_pool(tmp_path):
    """train_idx and test_idx give the pool rows used for each split."""
    path = make_pool(tmp_path)
    data = FineWebData(n_train=300, n_test=100, path=path)
    pool = np.load(path)
    assert np.array_equal(pool[data.train_idx], data.train)
    assert np.array_equal(pool[data.test_idx], data.test)
