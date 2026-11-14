import argparse

import numpy as np
import tiktoken
from datasets import load_dataset


class FineWebData:
    vocab_size = 50257

    def __init__(self, n_train, n_test=10_000, seed=0, path="data/fineweb.npy"):
        """Split a saved pool of unique FineWeb sequences into train and test sets."""
        pool = np.load(path)
        if n_train + n_test > len(pool):
            raise ValueError(f"need {n_train + n_test} sequences, pool has {len(pool)}")
        self.seq_len = pool.shape[1]
        idx = np.random.default_rng(seed).permutation(len(pool))
        self.test = pool[idx[:n_test]].astype(np.int64)
        self.train = pool[idx[n_test : n_test + n_train]].astype(np.int64)

    @staticmethod
    def build(n_sequences, path="data/fineweb.npy", seq_len=64, texts=None, encode=None):
        """Truncate FineWeb documents to their first seq_len tokens, drop duplicates, and save."""
        if texts is None:
            stream = load_dataset(
                "HuggingFaceFW/fineweb", name="sample-350BT", split="train", streaming=True
            )
            texts = (row["text"] for row in stream)
        if encode is None:
            encode = tiktoken.get_encoding("gpt2").encode_ordinary

        seen, pool = set(), []
        n_docs = n_short = n_dup = 0
        for text in texts:
            n_docs += 1
            if n_docs % 100_000 == 0:
                print(f"{n_docs} docs read, {len(pool)} sequences kept")
            tokens = encode(text)
            if len(tokens) < seq_len:
                n_short += 1
                continue
            seq = np.array(tokens[:seq_len], dtype=np.uint16)
            key = seq.tobytes()
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            pool.append(seq)
            if len(pool) == n_sequences:
                break

        if len(pool) < n_sequences:
            raise ValueError(f"only found {len(pool)} unique sequences, wanted {n_sequences}")
        pool = np.stack(pool)
        np.save(path, pool)
        print(f"{n_docs} docs read: {n_short} too short, {n_dup} duplicates, {len(pool)} kept")
        return pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sequences", type=int, required=True)
    parser.add_argument("--path", default="data/fineweb.npy")
    args = parser.parse_args()
    FineWebData.build(args.n_sequences, args.path)
