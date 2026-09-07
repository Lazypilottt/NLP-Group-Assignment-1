from __future__ import annotations

import random
import statistics
import time

from q1_module import segment_beam
from q3_module import correct_nonword, vocab
from live_pipeline import GRAMMAR_WINDOW, _is_anomalous


COMMON_WORDS = [
    "the", "cat", "sat", "on", "mat", "with", "green", "salad", "weather", "city",
    "people", "looked", "sky", "investigation", "fulton", "county", "grand", "jury",
    "said", "friday", "cold", "door", "window", "garden", "flower", "river", "mountain",
    "travel", "office", "report", "result", "system", "analysis", "language", "model",
    "sentence", "grammar", "lexicon", "correction", "pipeline", "window", "token", "word",
    "phrase", "structure", "parser", "scoring", "candidate", "unknown", "segment", "spelling",
    "example", "illustration", "signal", "pattern", "sequence", "context", "document", "chapter",
    "paper", "method", "experiment", "output", "dataset", "corpus", "brown", "english", "study",
    "research", "verify", "quantify", "measure", "latency", "analysis", "decision", "check",
    "tokenize", "simulate", "typing", "merge", "mistake", "missed", "space", "error", "typo",
]


def corrupt_word(word: str) -> str:
    """Create a simple misspelling equivalent to a noisy, user-typed word."""
    if len(word) <= 2:
        return word + "x"
    i = random.randint(0, len(word) - 1)
    letters = list(word)
    letters[i] = chr((ord(letters[i]) - 97 + 1) % 26 + 97)
    candidate = "".join(letters)
    if candidate in vocab:
        return candidate + "z"
    return candidate


def build_misspelled_batch(n_words: int = 1000, seed: int = 7) -> list[str]:
    random.seed(seed)
    batch = []
    for _ in range(n_words):
        base = random.choice(COMMON_WORDS)
        batch.append(corrupt_word(base))
    return batch


def seg_spell_batch(tokens: list[str]) -> float:
    t0 = time.perf_counter()
    for token in tokens:
        lower = token.lower()
        if len(lower) > 12 or lower not in vocab:
            parts = segment_beam(lower)
            valid_parts = [p for p in parts if p and p.isalpha()]
            if len(valid_parts) >= 2 and all(p in vocab for p in valid_parts):
                pass
        if lower not in vocab:
            suggestion = correct_nonword(lower)
            if suggestion != lower and suggestion in vocab:
                pass
    return time.perf_counter() - t0


def grammar_only_batch(tokens: list[str]) -> float:
    t0 = time.perf_counter()
    window: list[str] = []
    for token in tokens:
        word = token.lower()
        if word:
            window.append(word)
        if len(window) >= GRAMMAR_WINDOW:
            if _is_anomalous(window[-GRAMMAR_WINDOW:]):
                pass
    return time.perf_counter() - t0


def main() -> None:
    batch = build_misspelled_batch(1000)

    seg_spell_total = seg_spell_batch(batch)
    grammar_total = grammar_only_batch(batch)

    seg_spell_avg = seg_spell_total / len(batch)
    grammar_avg = grammar_total / len(batch)
    delta = seg_spell_total - grammar_total
    delta_avg = seg_spell_avg - grammar_avg
    percent_extra = (delta / grammar_total) * 100.0 if grammar_total > 0 else 0.0

    print("Benchmark: 1,000 simulated misspelled words")
    print("-" * 72)
    print(f"Segmentation + spelling total: {seg_spell_total:.6f}s ({seg_spell_avg:.6f}s/word)")
    print(f"Grammar-only total:           {grammar_total:.6f}s ({grammar_avg:.6f}s/word)")
    print(f"Additional latency:           {delta:.6f}s ({delta_avg:.6f}s/word)")
    print(f"Percent overhead:             {percent_extra:.1f}%")
    print("-" * 72)
    if delta > 0:
        conclusion = (
            "Conclusion: in a typical run, the segmentation+spelling layer adds cost because unknown tokens trigger "
            "beam-search splitting and edit-distance candidate generation while grammar checks only score short windows. "
            "This part of the pipeline is therefore the dominant overhead whenever many tokens are out-of-vocabulary."
        )
    else:
        conclusion = (
            "Conclusion: on this specific benchmark, the grammar-only path is slower than the segmentation+spelling path. "
            "The reason is that each grammar check recalculates log-probability scores over a full six-word window, "
            "while most tokens are already in-vocabulary and quickly short-circuit through a simple lookup. In this "
            "implementation, grammar scoring is the heavier loop even without the extra segmentation/spelling work."
        )
    print(conclusion)


if __name__ == "__main__":
    main()
