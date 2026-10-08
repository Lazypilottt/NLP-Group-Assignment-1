from __future__ import annotations

import math
import pickle
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


# ==============================================================================
# MODEL ARTIFACT LOADERS
# ==============================================================================

DIR_PATH = Path(__file__).resolve().parent
EN_MODEL_PATH = DIR_PATH / "q1_models_en.pkl"
ES_MODEL_PATH = DIR_PATH / "q1_models_es.pkl"
LEGACY_MODEL_PATH = DIR_PATH / "q1_trigram_lm.pkl"


def _load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {path}. Run `python q1_train.py` first."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


_EN_MODELS: dict[str, Any] | None = None
_ES_MODELS: dict[str, Any] | None = None


def get_models(lang: str = "en") -> dict[str, Any]:
    global _EN_MODELS, _ES_MODELS
    if lang == "en":
        if _EN_MODELS is None:
            _EN_MODELS = _load_pickle(EN_MODEL_PATH)
        return _EN_MODELS
    elif lang == "es":
        if _ES_MODELS is None:
            _ES_MODELS = _load_pickle(ES_MODEL_PATH)
        return _ES_MODELS
    else:
        raise ValueError(f"Unsupported language {lang!r}. Choose 'en' or 'es'.")


# Backwards compatibility export for Question 4
if LEGACY_MODEL_PATH.exists():
    with LEGACY_MODEL_PATH.open("rb") as _fh:
        trigram_lm = pickle.load(_fh)
else:
    trigram_lm = {"vocab": [], "token_counts": {}, "trigram_counts": {}}


# ==============================================================================
# WORD SEGMENTATION (TRIGRAM LM + DYNAMIC PROGRAMMING / VITERBI)
# ==============================================================================

def _word_log_prob(w: str, w_prev1: str, w_prev2: str, lm_data: dict[str, Any]) -> float:
    """Compute smoothed trigram log probability log P(w | w_prev1, w_prev2) using Jelinek-Mercer / Add-k smoothing."""
    k = lm_data.get("k", 0.5)
    vocab_size = len(lm_data["vocab"])
    trigram_counts = lm_data["trigram_counts"]
    bigram_counts = lm_data["bigram_counts"]
    unigram_counts = lm_data["unigram_counts"]
    total_words = lm_data["total_words"]

    # Unigram MLE
    p_uni = (unigram_counts.get(w, 0) + k) / (total_words + k * vocab_size)

    # Bigram MLE
    if w_prev1 in bigram_counts:
        denom_bi = sum(bigram_counts[w_prev1].values())
        p_bi = (bigram_counts[w_prev1].get(w, 0) + k * p_uni * vocab_size) / (denom_bi + k * vocab_size)
    else:
        p_bi = p_uni

    # Trigram MLE
    context_2 = (w_prev2, w_prev1)
    if context_2 in trigram_counts:
        denom_tri = sum(trigram_counts[context_2].values())
        p_tri = (trigram_counts[context_2].get(w, 0) + k * p_bi * vocab_size) / (denom_tri + k * vocab_size)
    else:
        p_tri = p_bi

    return math.log(max(p_tri, 1e-12))


def segment_dp(text: str, lang: str = "en", max_word_len: int = 25) -> list[str]:
    """Dynamic programming (Viterbi) word segmentation using trained Trigram LM."""
    cleaned = text.lower().strip()
    if not cleaned:
        return []

    models = get_models(lang)
    lm_data = models["trigram_lm"]
    vocab_set = lm_data["vocab_set"]
    n = len(cleaned)

    # dp[i] stores (best_neg_log_score, best_word_list, prev1, prev2)
    dp: list[tuple[float, list[str], str, str] | None] = [None] * (n + 1)
    dp[0] = (0.0, [], "<s>", "<s>")

    for i in range(n):
        if dp[i] is None:
            continue
        curr_score, curr_words, prev1, prev2 = dp[i]

        max_span = min(n - i, max_word_len)
        for span in range(1, max_span + 1):
            sub = cleaned[i : i + span]
            in_vocab = sub in vocab_set

            if in_vocab:
                log_p = _word_log_prob(sub, prev1, prev2, lm_data)
                # Word cost is negative log-prob with a slight length bonus to favor valid words over fragment letters
                step_cost = -log_p - (0.5 * span)
            else:
                # Heavy penalty for out-of-vocabulary splits
                step_cost = 50.0 + (span * 5.0)

            candidate_score = curr_score + step_cost
            next_idx = i + span

            if dp[next_idx] is None or candidate_score < dp[next_idx][0]:
                dp[next_idx] = (candidate_score, curr_words + [sub], sub, prev1)

    if dp[n] is not None:
        return dp[n][1]

    # Fallback to character split if unparseable
    return list(cleaned)


def segment_beam(token: str) -> list[str]:
    """Backwards-compatible wrapper for English DP segmentation."""
    return segment_dp(token, lang="en")


# ==============================================================================
# SEGMENTATION BASELINE: GREEDY LONGEST-MATCH
# ==============================================================================

def baseline_segment_greedy(text: str, lang: str = "en", max_word_len: int = 25) -> list[str]:
    """Greedy longest-match baseline for word segmentation."""
    cleaned = text.lower().strip()
    if not cleaned:
        return []

    models = get_models(lang)
    vocab_set = models["trigram_lm"]["vocab_set"]
    n = len(cleaned)
    words: list[str] = []
    i = 0

    while i < n:
        matched = False
        max_span = min(n - i, max_word_len)
        # Try matching the longest valid word starting from longest to shortest
        for span in range(max_span, 0, -1):
            candidate = cleaned[i : i + span]
            if candidate in vocab_set:
                words.append(candidate)
                i += span
                matched = True
                break

        if not matched:
            # If no vocab word matches, advance single character
            words.append(cleaned[i])
            i += 1

    return words


# ==============================================================================
# POS TAGGING MODEL (HMM EMISSION + TRANSITION + VITERBI DECODING)
# ==============================================================================

def _get_emission_log_prob(word: str, tag: str, hmm_data: dict[str, Any]) -> float:
    """Compute log P(word | tag). Known words only emit observed tags; OOV words use suffix heuristics."""
    w_lower = word.lower()
    emission_counts = hmm_data["emission_counts"].get(tag, {})
    tag_count = hmm_data["tag_counts"].get(tag, 1)
    word_counts = hmm_data["word_counts"]

    # Case 1: Word was observed with THIS tag
    if w_lower in emission_counts:
        prob = emission_counts[w_lower] / tag_count
        return math.log(max(prob, 1e-12))

    # Case 2: Word was observed in training, but NEVER with this tag
    if w_lower in word_counts:
        return -35.0  # Heavy penalty: known word does not take unobserved tag

    # Case 3: Truly OOV word -> apply morphological suffix and capitalisation heuristics
    suffix_score = 0.0
    suffix_counts = hmm_data.get("suffix_counts", {}).get(tag, {})
    if len(w_lower) >= 3 and w_lower[-3:] in suffix_counts:
        suffix_score += suffix_counts[w_lower[-3:]] * 3.0
    if len(w_lower) >= 2 and w_lower[-2:] in suffix_counts:
        suffix_score += suffix_counts[w_lower[-2:]]

    if suffix_score > 0:
        prob = suffix_score / (tag_count * 10.0)
    else:
        # Default prior proportional to tag frequency
        prob = tag_count / sum(hmm_data["tag_counts"].values())

    return math.log(max(prob, 1e-12))


def _get_transition_log_prob(prev_tag: str, curr_tag: str, hmm_data: dict[str, Any]) -> float:
    """Compute log P(curr_tag | prev_tag) with Laplace smoothing."""
    transitions = hmm_data["tag_transitions"].get(prev_tag, {})
    tag_count = hmm_data["tag_counts"].get(prev_tag, sum(transitions.values()))
    num_tags = len(hmm_data["tags"])
    alpha = hmm_data.get("alpha", 1.0)

    num = transitions.get(curr_tag, 0) + alpha
    den = tag_count + alpha * num_tags
    return math.log(num / den)


def pos_tag_hmm(words: list[str], lang: str = "en", morph: bool = False) -> list[tuple[str, str]]:
    """HMM Viterbi POS Tagger supporting standard and morphology-aware tags."""
    if not words:
        return []

    models = get_models(lang)
    hmm_data = models["hmm_morph"] if morph else models["hmm_std"]
    tags = hmm_data["tags"]

    if not tags:
        return [(w, "NOUN") for w in words]

    # Viterbi DP structures
    # V[t][i] = max log probability of tag sequence ending in tag t at word index i
    viterbi: list[dict[str, float]] = []
    backpointer: list[dict[str, str]] = []

    # Step 0: Initialize with start tag <s>
    viterbi.append({})
    backpointer.append({})
    for tag in tags:
        trans_lp = _get_transition_log_prob("<s>", tag, hmm_data)
        emis_lp = _get_emission_log_prob(words[0], tag, hmm_data)
        viterbi[0][tag] = trans_lp + emis_lp
        backpointer[0][tag] = "<s>"

    # Step 1 to N-1: Forward Viterbi recurrence
    for i in range(1, len(words)):
        viterbi.append({})
        backpointer.append({})
        w = words[i]
        for curr_tag in tags:
            emis_lp = _get_emission_log_prob(w, curr_tag, hmm_data)
            best_prev_tag = None
            best_val = float("-inf")
            for prev_tag in tags:
                val = viterbi[i - 1][prev_tag] + _get_transition_log_prob(prev_tag, curr_tag, hmm_data)
                if val > best_val:
                    best_val = val
                    best_prev_tag = prev_tag
            viterbi[i][curr_tag] = best_val + emis_lp
            backpointer[i][curr_tag] = best_prev_tag if best_prev_tag is not None else tags[0]

    # Termination: Transition to </s>
    last_idx = len(words) - 1
    best_last_tag = None
    best_final_val = float("-inf")
    for tag in tags:
        val = viterbi[last_idx][tag] + _get_transition_log_prob(tag, "</s>", hmm_data)
        if val > best_final_val:
            best_final_val = val
            best_last_tag = tag

    if best_last_tag is None:
        best_last_tag = tags[0]

    # Backtrace
    best_tags = [best_last_tag]
    for i in range(last_idx, 0, -1):
        best_tags.append(backpointer[i][best_tags[-1]])
    best_tags.reverse()

    return list(zip(words, best_tags))


def pos_tag(words: list[str]) -> list[tuple[str, str]]:
    """Backwards-compatible English POS tagger returning standard Universal tags."""
    return pos_tag_hmm(words, lang="en", morph=False)


# ==============================================================================
# POS TAGGING BASELINE: MOST-FREQUENT-TAG (MFT)
# ==============================================================================

def baseline_pos_tag_mft(words: list[str], lang: str = "en", morph: bool = False) -> list[tuple[str, str]]:
    """Most-Frequent-Tag (MFT) baseline tagger."""
    if not words:
        return []

    models = get_models(lang)
    hmm_data = models["hmm_morph"] if morph else models["hmm_std"]
    mft_map = hmm_data["mft_map"]
    overall_mft = hmm_data["overall_mft"]

    return [(w, mft_map.get(w.lower(), overall_mft)) for w in words]


# ==============================================================================
# END-TO-END SEGMENTATION + POS TAGGING PIPELINE
# ==============================================================================

def segment_and_tag(
    unspaced_text: str,
    lang: str = "en",
    morph: bool = False,
    use_baseline_seg: bool = False,
    use_baseline_tag: bool = False,
) -> list[tuple[str, str]]:
    """End-to-end pipeline: unspaced text -> word tokens -> POS tags."""
    if use_baseline_seg:
        words = baseline_segment_greedy(unspaced_text, lang=lang)
    else:
        words = segment_dp(unspaced_text, lang=lang)

    if use_baseline_tag:
        return baseline_pos_tag_mft(words, lang=lang, morph=morph)
    else:
        return pos_tag_hmm(words, lang=lang, morph=morph)


# ==============================================================================
# TYPING SIMULATION (PRESERVED FOR Q4)
# ==============================================================================

def simulate_typing(passage: str, p: float = 0.08):
    """Yield tokens from a passage with occasional missed-space merges."""
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


__all__ = [
    "get_models",
    "segment_dp",
    "segment_beam",
    "baseline_segment_greedy",
    "pos_tag_hmm",
    "pos_tag",
    "baseline_pos_tag_mft",
    "segment_and_tag",
    "simulate_typing",
    "trigram_lm",
]
