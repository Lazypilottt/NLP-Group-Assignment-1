from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path

try:
    from nltk.corpus import brown
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "NLTK is required to build the spelling-correction model. Install it with `python -m pip install nltk` and download the Brown corpus."
    ) from exc


def build_models():
    words = [word.lower() for word in brown.words() if word.isalpha()]
    unigram_counts = Counter(words)
    vocabulary = set(words)
    bigram_counts = defaultdict(Counter)

    for w1, w2 in zip(words, words[1:]):
        bigram_counts[w1][w2] += 1

    return vocabulary, unigram_counts, bigram_counts


def save_model(path: str | Path | None = None):
    artifact_path = Path(path) if path is not None else Path(__file__).with_name("q3_spelling_model.pkl")
    vocabulary, unigram_counts, bigram_counts = build_models()
    payload = {
        "vocab": vocabulary,
        "unigram_freq": unigram_counts,
        "bigram_model": bigram_counts,
    }
    with artifact_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return payload


if __name__ == "__main__":
    save_model()
    print("Saved trained spelling model to q3_spelling_model.pkl")
