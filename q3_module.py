from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path


MODEL_PATH = Path(__file__).with_name("q3_spelling_model.pkl")


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {MODEL_PATH}. Run q3_train.py first."
        )
    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


_model = _load_model()
vocab = _model["vocab"]
unigram_freq = _model["unigram_freq"]
bigram_model = _model["bigram_model"]


def bigram_probability(w1: str, w2: str, bigram_counts, unigram_counts, k: float = 1.0) -> float:
    vocabulary_size = len(unigram_counts)
    count_bigram = bigram_counts[w1][w2]
    count_w1 = unigram_counts[w1]
    return (count_bigram + k) / (count_w1 + k * vocabulary_size)


def edits1(word: str) -> set[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletions = [left + right[1:] for left, right in splits if right]
    transposes = [
        left + right[1] + right[0] + right[2:]
        for left, right in splits
        if len(right) > 1
    ]
    replacements = [
        left + c + right[1:]
        for left, right in splits
        if right
        for c in alphabet
    ]
    insertions = [
        left + c + right
        for left, right in splits
        for c in alphabet
    ]

    return set(deletions + transposes + replacements + insertions)


def build_delete_dictionary(vocabulary: set[str]) -> dict[str, set[str]]:
    delete_dict: dict[str, set[str]] = defaultdict(set)
    for word in vocabulary:
        for i in range(len(word)):
            deleted = word[:i] + word[i + 1 :]
            delete_dict[deleted].add(word)
    return delete_dict


_delete_dict = build_delete_dictionary(vocab)


def gen_candidates_a(word: str) -> set[str]:
    return {candidate for candidate in edits1(word.lower()) if candidate.isalpha()}


def gen_candidates_b(word: str) -> set[str]:
    tokens = word.lower()
    candidates: set[str] = set()
    for i in range(len(tokens)):
        deleted = tokens[:i] + tokens[i + 1 :]
        candidates.update(_delete_dict.get(deleted, set()))
    return candidates


def correct_nonword(word: str) -> str:
    candidate_word = word.lower()
    if candidate_word in vocab:
        return candidate_word
    candidates = (gen_candidates_a(candidate_word) | gen_candidates_b(candidate_word)) & vocab
    if not candidates:
        return candidate_word
    return max(candidates, key=lambda w: unigram_freq[w])


def correct_realword(sentence, target_idx: int) -> str:
    if isinstance(sentence, str):
        words = sentence.split()
    else:
        words = list(sentence)

    if not 0 <= target_idx < len(words):
        return ""

    original = words[target_idx].lower()
    previous_word = words[target_idx - 1].lower() if target_idx > 0 else ""
    next_word = words[target_idx + 1].lower() if target_idx + 1 < len(words) else ""

    candidates = (gen_candidates_a(original) | gen_candidates_b(original)) & vocab
    candidates.discard(original)

    if not candidates:
        return original

    def score(candidate: str) -> float:
        candidate_word = candidate.lower()
        s = 1.0
        if previous_word:
            s *= bigram_probability(previous_word, candidate_word, bigram_model, unigram_freq, k=1.0)
        if next_word:
            s *= bigram_probability(candidate_word, next_word, bigram_model, unigram_freq, k=1.0)
        return s

    original_score = score(original)
    best_word = original
    best_score = original_score

    for candidate in candidates:
        candidate_score = score(candidate)
        if candidate_score > best_score * 2:
            best_word = candidate
            best_score = candidate_score

    return best_word


__all__ = [
    "vocab",
    "unigram_freq",
    "bigram_model",
    "gen_candidates_a",
    "gen_candidates_b",
    "correct_nonword",
    "correct_realword",
]
