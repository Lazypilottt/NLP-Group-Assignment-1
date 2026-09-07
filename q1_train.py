from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the fox and the dog run in the park",
    "a quick brown fox jumps over a lazy dog",
    "the house is near the park",
    "the dog runs in the house",
    "a dog and a fox are in the house",
    "the quick fox jumps over the brown dog",
]


def train_trigram_lm(corpus: list[str]) -> dict:
    """Train a tiny trigram language model and return its learned state."""
    token_counts: Counter[str] = Counter()
    trigram_counts: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    vocab: set[str] = set()

    for sentence in corpus:
        tokens = ["<s>"] + sentence.lower().split() + ["</s>"]
        vocab.update(tokens)
        for token in tokens:
            token_counts[token] += 1

        for i in range(len(tokens) - 2):
            trigram = tuple(tokens[i : i + 3])
            trigram_counts[trigram] += 1.0

    return {
        "vocab": sorted(vocab),
        "token_counts": dict(token_counts),
        "trigram_counts": dict(trigram_counts),
        "sentences": list(corpus),
    }


def save_model(path: str | Path | None = None) -> dict:
    artifact_path = Path(path) if path is not None else Path(__file__).with_name("q1_trigram_lm.pkl")
    model = train_trigram_lm(CORPUS)
    with artifact_path.open("wb") as fh:
        pickle.dump(model, fh)
    return model


if __name__ == "__main__":
    save_model()
    print("Saved trained trigram LM to q1_trigram_lm.pkl")
