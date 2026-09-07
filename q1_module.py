from __future__ import annotations

import math
import pickle
import random
import time
from functools import lru_cache
from pathlib import Path


MODEL_PATH = Path(__file__).with_name("q1_trigram_lm.pkl")


def _load_trained_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {MODEL_PATH}. Run q1_train.py first."
        )
    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


trigram_lm = _load_trained_model()


def segment_beam(token: str) -> list[str]:
    """Beam-search segmentation for merged/unknown tokens using the trained vocabulary."""
    text = token.lower().strip()
    if not text:
        return []

    vocab = set(trigram_lm["vocab"])
    token_counts = trigram_lm["token_counts"]
    vocab_size = max(len(vocab), 1)

    @lru_cache(maxsize=None)
    def best(i: int):
        if i == len(text):
            return 0.0, []

        best_score = float("inf")
        best_parts: list[str] | None = None
        max_span = min(len(text) - i, 10)

        for span in range(1, max_span + 1):
            piece = text[i : i + span]
            if piece not in vocab and len(piece) > 1:
                continue

            rest_score, rest_parts = best(i + span)
            piece_weight = math.log((token_counts.get(piece, 1) + 1) / (vocab_size + 1))
            candidate_score = -piece_weight + rest_score

            if candidate_score < best_score:
                best_score = candidate_score
                best_parts = [piece] + rest_parts

        if best_parts is None:
            piece = text[i]
            rest_score, rest_parts = best(i + 1)
            return rest_score + 1.0, [piece] + rest_parts

        return best_score, best_parts

    _, parts = best(0)
    return parts


def pos_tag(words: list[str]) -> list[tuple[str, str]]:
    """A lightweight feature-based POS tagger using lexical rules and suffix heuristics."""
    lexicon = {
        "the": "DET",
        "a": "DET",
        "an": "DET",
        "quick": "ADJ",
        "brown": "ADJ",
        "lazy": "ADJ",
        "fox": "NOUN",
        "dog": "NOUN",
        "house": "NOUN",
        "park": "NOUN",
        "jumps": "VERB",
        "jump": "VERB",
        "run": "VERB",
        "runs": "VERB",
        "is": "AUX",
        "are": "AUX",
        "near": "ADP",
        "in": "ADP",
        "and": "CCONJ",
        "over": "ADP",
    }

    tags: list[tuple[str, str]] = []
    for word in words:
        lower = word.lower()
        if lower in lexicon:
            tag = lexicon[lower]
        elif lower.endswith("ly"):
            tag = "ADV"
        elif lower.endswith("ing"):
            tag = "VERB"
        elif lower.endswith("s"):
            tag = "NOUN"
        elif lower.endswith("ed"):
            tag = "VERB"
        else:
            tag = "NOUN"
        tags.append((word, tag))
    return tags


def simulate_typing(passage: str, p: float = 0.08):
    """Yield tokens from a passage with occasional missed-space merges.

    A merge occurs when the current word is followed by the next word and a random
    draw falls below p. This creates a realistic merged token such as 'thecat'.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")

    words = passage.split()
    i = 0
    while i < len(words):
        current = words[i]
        if i + 1 < len(words) and random.random() < p:
            merged = current + words[i + 1]
            yield merged
            time.sleep(0.18)
            i += 2
        else:
            yield current
            time.sleep(0.18)
            i += 1


__all__ = ["segment_beam", "pos_tag", "trigram_lm", "simulate_typing"]
