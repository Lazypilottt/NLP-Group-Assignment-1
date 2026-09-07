from __future__ import annotations

from collections import Counter, defaultdict
from math import log

from nltk.corpus import brown

# Shared brown-based LM trained once at import time and reused across Q4 components.
# k = 0.5 is a standard compromise: small enough to retain useful frequencies,
# large enough to smooth sparse bigram/trigram contexts without over-diluting the model.
K = 0.5


def _normalize_tokens(sentence: list[str]) -> list[str]:
    return [token.lower() for token in sentence if token and token.isalpha()]


def _build_counts():
    vocab: set[str] = set()
    bigram_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    trigram_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for sent in brown.sents():
        tokens = ["<s>"] + [w.lower() for w in sent if w.isalpha()] + ["</s>"]
        for token in tokens:
            vocab.add(token)

        for w1, w2 in zip(tokens, tokens[1:]):
            bigram_counts[w1][w2] += 1

        for w1, w2, w3 in zip(tokens, tokens[1:], tokens[2:]):
            trigram_counts[(w1, w2)][w3] += 1

    return vocab, bigram_counts, trigram_counts


VOCAB, BIGRAM_COUNTS, TRIGRAM_COUNTS = _build_counts()
VOCAB_SIZE = len(VOCAB)


def _bigram_prob(w1: str, w2: str) -> float:
    context_count = sum(BIGRAM_COUNTS[w1].values())
    numerator = BIGRAM_COUNTS[w1][w2] + K
    denominator = context_count + K * VOCAB_SIZE
    return numerator / denominator


def _trigram_prob(w1: str, w2: str, w3: str) -> float:
    context = (w1, w2)
    context_count = sum(TRIGRAM_COUNTS[context].values())
    numerator = TRIGRAM_COUNTS[context][w3] + K
    denominator = context_count + K * VOCAB_SIZE
    return numerator / denominator


def bigram_logprob(sentence: list[str]) -> float:
    tokens = ["<s>"] + _normalize_tokens(sentence) + ["</s>"]
    if len(tokens) < 2:
        return 0.0
    total = 0.0
    for prev, curr in zip(tokens, tokens[1:]):
        total += log(_bigram_prob(prev, curr))
    return total


def trigram_logprob(sentence: list[str]) -> float:
    tokens = ["<s>"] + _normalize_tokens(sentence) + ["</s>"]
    if len(tokens) < 3:
        return 0.0
    total = 0.0
    for w1, w2, w3 in zip(tokens, tokens[1:], tokens[2:]):
        total += log(_trigram_prob(w1, w2, w3))
    return total


__all__ = ["K", "bigram_logprob", "trigram_logprob"]
