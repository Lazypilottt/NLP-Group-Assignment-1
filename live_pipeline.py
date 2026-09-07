from __future__ import annotations

import random
import time
from statistics import mean

from nltk.corpus import brown

from q1_module import segment_beam, simulate_typing
from q3_module import correct_nonword, correct_realword, gen_candidates_a, vocab
from shared_lm import bigram_logprob, trigram_logprob

# Choose N=6: wide enough to capture local syntactic plausibility, but short enough
# to avoid over-smoothing and to keep grammar checks responsive in a live stream.
GRAMMAR_WINDOW = 6

# A small Brown-derived reference set for rough anomaly detection.
_REFERENCE_SENTENCES = [
    [w.lower() for w in sent if w.isalpha()]
    for sent in brown.sents()[:250]
    if 3 <= len([w for w in sent if w.isalpha()]) <= 12
]


def _score_phrase(words: list[str]) -> float:
    if not words:
        return float("-inf")
    return bigram_logprob(words) + trigram_logprob(words)


def _expected_window_score(length: int) -> float:
    scores = []
    for sent in _REFERENCE_SENTENCES:
        if len(sent) == length:
            scores.append(_score_phrase(sent))
    if not scores:
        return -length * 12.0
    return sum(scores) / len(scores)


def _is_anomalous(window: list[str]) -> bool:
    if len(window) < GRAMMAR_WINDOW:
        return False
    score = _score_phrase(window)
    expected = _expected_window_score(len(window))
    return score < expected - 6.0


def _segment_if_appropriate(token: str) -> tuple[str, bool]:
    token_l = token.lower()
    if token_l in vocab and len(token_l) <= 14:
        return token, False

    parts = segment_beam(token_l)
    valid_parts = [p for p in parts if p and p.isalpha()]
    if len(valid_parts) >= 2 and all(p in vocab for p in valid_parts):
        actual_score = _score_phrase([token_l])
        candidate_score = _score_phrase(valid_parts)
        if candidate_score > actual_score:
            return " ".join(valid_parts), True

    return token, False


def _coerce_words(sentence):
    if isinstance(sentence, str):
        return sentence.split()
    return list(sentence)


def live_pipeline(passage: str, p: float = 0.08, grammar_window: int = GRAMMAR_WINDOW):
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")

    seg_spell_latencies = []
    grammar_latencies = []
    recent_window = []
    processed = 0

    print(f"Starting live pipeline with p={p}, grammar_window={grammar_window}")
    print("-" * 60)

    for token in simulate_typing(passage, p):
        processed += 1
        token_start = time.perf_counter()
        original = token
        current_tokens = [token]

        if len(token) > 12 or token.lower() not in vocab:
            segmented, did_segment = _segment_if_appropriate(token)
            if did_segment:
                current_tokens = segmented.split()
                print(f"[{processed:02d}] {original!r} -> {current_tokens!r} [SEGMENT ALERT]")

        for current in current_tokens:
            current_l = current.lower()
            if current_l not in vocab:
                suggestion = correct_nonword(current_l)
                if suggestion != current_l and suggestion in vocab:
                    print(f"[{processed:02d}] {current!r} -> {suggestion!r} [SPELL ALERT]")
                    current = suggestion

            seg_spell_latencies.append((time.perf_counter() - token_start) * 1000.0)
            recent_window.append(current)

            if len(recent_window) >= grammar_window:
                grammar_start = time.perf_counter()
                window = recent_window[-grammar_window:]
                score = _score_phrase(window)
                if _is_anomalous(window):
                    print(f"[{processed:02d}] window={window} score={score:.2f} [GRAMMAR ALERT] low-likelihood sequence")

                idx = len(window) - 1
                actual = window[idx]
                realword_suggestion = correct_realword(window, idx)
                if realword_suggestion != actual and realword_suggestion in vocab:
                    print(f"[{processed:02d}] actual={window} suggestion={realword_suggestion} [GRAMMAR ALERT] real-word context likely error")

                candidates = set(gen_candidates_a(actual)) & vocab
                if candidates:
                    candidate_phrase = list(window)
                    best = actual
                    best_score = score
                    for cand in candidates:
                        candidate_phrase[idx] = cand
                        cand_score = _score_phrase(candidate_phrase)
                        if cand_score > best_score + 8.0:
                            best = cand
                            best_score = cand_score
                    if best != actual:
                        print(f"[{processed:02d}] phrase={window} candidate={best} [GRAMMAR ALERT] local candidate phrase scored significantly higher")

                grammar_latencies.append((time.perf_counter() - grammar_start) * 1000.0)

    print("-" * 60)
    print(f"Avg seg+spell latency: {mean(seg_spell_latencies):.3f} ms/token")
    if grammar_latencies:
        print(f"Avg grammar latency: {mean(grammar_latencies):.3f} ms/check")
    else:
        print("Avg grammar latency: 0.000 ms/check")
    print("Pipeline complete.")


if __name__ == "__main__":
    random.seed(42)
    sample = (
        "The Fulton County GrandJury said Friday aninvestigation of the wether in the city "
        "was cold and teh people looked at the sky ."
    )
    live_pipeline(sample, p=0.08, grammar_window=6)
