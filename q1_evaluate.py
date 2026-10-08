from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from typing import Any

from q1_module import (
    baseline_pos_tag_mft,
    baseline_segment_greedy,
    get_models,
    pos_tag_hmm,
    segment_and_tag,
    segment_dp,
)


# ==============================================================================
# SEGMENTATION EVALUATION METRICS
# ==============================================================================

def get_word_boundaries(words: list[str]) -> set[int]:
    """Return the character indices where word boundaries occur."""
    boundaries = set()
    offset = 0
    for w in words[:-1]:
        offset += len(w)
        boundaries.add(offset)
    return boundaries


def evaluate_segmentation(test_sentences: list[list[tuple[str, str]]], lang: str = "en", max_eval: int = 500) -> dict[str, float]:
    """Evaluate DP Trigram Segmentation vs Greedy Longest Match Baseline."""
    eval_sents = test_sentences[:max_eval]

    dp_exact = 0
    greedy_exact = 0
    dp_gold_bounds = 0
    dp_pred_bounds = 0
    dp_true_bounds = 0

    greedy_gold_bounds = 0
    greedy_pred_bounds = 0
    greedy_true_bounds = 0

    for sent in eval_sents:
        gold_words = [w for w, _ in sent if w]
        if not gold_words:
            continue
        unspaced = "".join(gold_words)
        gold_b = get_word_boundaries(gold_words)

        # 1. DP Segmentation
        dp_words = segment_dp(unspaced, lang=lang)
        dp_b = get_word_boundaries(dp_words)
        if dp_words == gold_words:
            dp_exact += 1
        dp_gold_bounds += len(gold_b)
        dp_pred_bounds += len(dp_b)
        dp_true_bounds += len(gold_b & dp_b)

        # 2. Greedy Baseline
        greedy_words = baseline_segment_greedy(unspaced, lang=lang)
        greedy_b = get_word_boundaries(greedy_words)
        if greedy_words == gold_words:
            greedy_exact += 1
        greedy_gold_bounds += len(gold_b)
        greedy_pred_bounds += len(greedy_b)
        greedy_true_bounds += len(gold_b & greedy_b)

    n_sents = len(eval_sents)

    dp_p = dp_true_bounds / dp_pred_bounds if dp_pred_bounds else 0.0
    dp_r = dp_true_bounds / dp_gold_bounds if dp_gold_bounds else 0.0
    dp_f1 = (2 * dp_p * dp_r) / (dp_p + dp_r) if (dp_p + dp_r) else 0.0

    gr_p = greedy_true_bounds / greedy_pred_bounds if greedy_pred_bounds else 0.0
    gr_r = greedy_true_bounds / greedy_gold_bounds if greedy_gold_bounds else 0.0
    gr_f1 = (2 * gr_p * gr_r) / (gr_p + gr_r) if (gr_p + gr_r) else 0.0

    return {
        "dp_exact_acc": (dp_exact / n_sents) * 100.0,
        "dp_precision": dp_p * 100.0,
        "dp_recall": dp_r * 100.0,
        "dp_f1": dp_f1 * 100.0,
        "greedy_exact_acc": (greedy_exact / n_sents) * 100.0,
        "greedy_precision": gr_p * 100.0,
        "greedy_recall": gr_r * 100.0,
        "greedy_f1": gr_f1 * 100.0,
    }


# ==============================================================================
# POS TAGGING EVALUATION & CONFUSION MATRIX
# ==============================================================================

def evaluate_pos_tagging(test_sentences: list[list[tuple[str, str]]], lang: str = "en", morph: bool = False, max_eval: int = 1000) -> tuple[float, float, dict[tuple[str, str], int]]:
    """Evaluate HMM Viterbi POS Tagger vs Most-Frequent-Tag Baseline."""
    eval_sents = test_sentences[:max_eval]

    total_tokens = 0
    hmm_correct = 0
    mft_correct = 0
    confusion_counts: Counter[tuple[str, str]] = Counter()

    for sent in eval_sents:
        words = [w for w, _ in sent if w]
        gold_tags = [t for _, t in sent if _]
        if not words:
            continue

        pred_hmm = pos_tag_hmm(words, lang=lang, morph=morph)
        pred_mft = baseline_pos_tag_mft(words, lang=lang, morph=morph)

        for (_, gold_t), (_, hmm_t), (_, mft_t) in zip(sent, pred_hmm, pred_mft):
            total_tokens += 1
            if hmm_t == gold_t:
                hmm_correct += 1
            if mft_t == gold_t:
                mft_correct += 1

            confusion_counts[(gold_t, hmm_t)] += 1

    hmm_acc = (hmm_correct / total_tokens) * 100.0 if total_tokens else 0.0
    mft_acc = (mft_correct / total_tokens) * 100.0 if total_tokens else 0.0

    return hmm_acc, mft_acc, dict(confusion_counts)


# ==============================================================================
# ERROR SOURCE BREAKDOWN (SEGMENTATION VS GENUINE TAGGING ERRORS)
# ==============================================================================

def evaluate_error_sources(test_sentences: list[list[tuple[str, str]]], lang: str = "en", max_eval: int = 400) -> dict[str, Any]:
    """Quantify error source: Segmentation-induced errors vs. Genuine POS tagging errors."""
    eval_sents = test_sentences[:max_eval]

    total_words = 0
    correct_count = 0
    segmentation_errors = 0
    genuine_tag_errors = 0

    for sent in eval_sents:
        gold_words = [w for w, _ in sent if w]
        gold_tags = [t for _, t in sent if _]
        if not gold_words:
            continue

        unspaced = "".join(gold_words)
        pred_pairs = segment_and_tag(unspaced, lang=lang, morph=False)
        pred_words = [w for w, _ in pred_pairs]
        pred_tags = [t for _, t in pred_pairs]

        # Match tokens along character positions
        gold_spans = []
        c = 0
        for w, t in sent:
            gold_spans.append((c, c + len(w), w, t))
            c += len(w)

        pred_spans = {}
        c = 0
        for w, t in pred_pairs:
            pred_spans[(c, c + len(w))] = (w, t)
            c += len(w)

        for start, end, gw, gt in gold_spans:
            total_words += 1
            if (start, end) in pred_spans:
                pw, pt = pred_spans[(start, end)]
                if pt == gt:
                    correct_count += 1
                else:
                    genuine_tag_errors += 1
            else:
                # Word was segmented with wrong boundary
                segmentation_errors += 1

    return {
        "total_words": total_words,
        "correct_count": correct_count,
        "correct_pct": (correct_count / total_words) * 100.0 if total_words else 0.0,
        "seg_errors": segmentation_errors,
        "seg_error_pct": (segmentation_errors / total_words) * 100.0 if total_words else 0.0,
        "tag_errors": genuine_tag_errors,
        "tag_error_pct": (genuine_tag_errors / total_words) * 100.0 if total_words else 0.0,
    }


# ==============================================================================
# PRETTY PRINTING UTILITIES
# ==============================================================================

def print_confusion_matrix(confusion: dict[tuple[str, str], int], top_n: int = 7) -> None:
    # Find most frequent tags
    tag_counts: Counter[str] = Counter()
    for (gold, pred), count in confusion.items():
        tag_counts[gold] += count
    top_tags = [t for t, _ in tag_counts.most_common(top_n)]

    col_w = 10
    header = "Actual \\ Pred".ljust(15) + "".join(t.rjust(col_w) for t in top_tags)
    print(header)
    print("-" * len(header))

    for gold in top_tags:
        row = gold.ljust(15)
        for pred in top_tags:
            val = confusion.get((gold, pred), 0)
            row += str(val).rjust(col_w)
        print(row)


def run_sample_strings() -> None:
    print("\n" + "=" * 75)
    print("DEMO: SAMPLE TEST STRINGS EVALUATION")
    print("=" * 75)

    samples_en = [
        "thequickbrownfoxjumpsoverthelazydog",
        "thehouseisnearthepark",
        "adogandafoxareinthehouse",
    ]

    samples_es = [
        "mispadrespuedenviajar",
        "elcielodespejadoesazul",
        "lacasarojaesgrande",
    ]

    print("\n--- ENGLISH DEMOS ---")
    for s in samples_en:
        std = segment_and_tag(s, lang="en", morph=False)
        morph = segment_and_tag(s, lang="en", morph=True)
        print(f"\nInput unspaced:  {s}")
        print(f"Standard POS:    {std}")
        print(f"Morphology POS:  {morph}")

    print("\n--- SPANISH DEMOS ---")
    for s in samples_es:
        std = segment_and_tag(s, lang="es", morph=False)
        morph = segment_and_tag(s, lang="es", morph=True)
        print(f"\nInput unspaced:  {s}")
        print(f"Standard POS:    {std}")
        print(f"Morphology POS:  {morph}")


# ==============================================================================
# MAIN BENCHMARK RUNNER
# ==============================================================================

def main() -> None:
    print("=" * 75)
    print("QUESTION 1: COMPREHENSIVE EVALUATION BENCHMARK")
    print("=" * 75)

    en_models = get_models("en")
    es_models = get_models("es")

    # 1. English Evaluation
    print("\n>>> 1. ENGLISH (Brown Corpus Test Split: 11,468 sentences)")
    print("Running Segmentation Evaluation...")
    en_seg = evaluate_segmentation(en_models["test_std"], lang="en", max_eval=400)
    print(f"  [Segmentation] DP Trigram F1: {en_seg['dp_f1']:.2f}% | Exact Sent Acc: {en_seg['dp_exact_acc']:.2f}%")
    print(f"  [Segmentation] Greedy Baseline F1: {en_seg['greedy_f1']:.2f}% | Exact Sent Acc: {en_seg['greedy_exact_acc']:.2f}%")
    print(f"  --> DP Improvement over Greedy Baseline: +{en_seg['dp_f1'] - en_seg['greedy_f1']:.2f}% F1")

    print("\nRunning POS Tagging Evaluation...")
    en_hmm_std_acc, en_mft_std_acc, en_conf_std = evaluate_pos_tagging(en_models["test_std"], lang="en", morph=False, max_eval=1000)
    en_hmm_morph_acc, en_mft_morph_acc, _ = evaluate_pos_tagging(en_models["test_morph"], lang="en", morph=True, max_eval=1000)
    print(f"  [Standard POS] HMM Viterbi Acc: {en_hmm_std_acc:.2f}% | Baseline MFT Acc: {en_mft_std_acc:.2f}%")
    print(f"  [Morphology POS] HMM Viterbi Acc: {en_hmm_morph_acc:.2f}% | Baseline MFT Acc: {en_mft_morph_acc:.2f}%")
    print(f"  --> HMM Improvement over MFT Baseline: +{en_hmm_std_acc - en_mft_std_acc:.2f}%")

    print("\nEnglish Confusion Matrix (Top Standard Tags):")
    print_confusion_matrix(en_conf_std, top_n=6)

    print("\nRunning English Error-Source Decomposition (Joint Segmentation + Tagging)...")
    en_errs = evaluate_error_sources(en_models["test_std"], lang="en", max_eval=400)
    print(f"  Total words evaluated:        {en_errs['total_words']}")
    print(f"  Correct (Word & Tag):         {en_errs['correct_count']} ({en_errs['correct_pct']:.2f}%)")
    print(f"  Segmentation-Induced Errors:  {en_errs['seg_errors']} ({en_errs['seg_error_pct']:.2f}%)")
    print(f"  Genuine POS Tagging Errors:   {en_errs['tag_errors']} ({en_errs['tag_error_pct']:.2f}%)")

    # 2. Spanish Evaluation
    print("\n" + "=" * 75)
    print(">>> 2. SPANISH (UD_Spanish-GSD Test Split: 427 sentences)")
    print("Running Segmentation Evaluation...")
    es_seg = evaluate_segmentation(es_models["test_std"], lang="es", max_eval=300)
    print(f"  [Segmentation] DP Trigram F1: {es_seg['dp_f1']:.2f}% | Exact Sent Acc: {es_seg['dp_exact_acc']:.2f}%")
    print(f"  [Segmentation] Greedy Baseline F1: {es_seg['greedy_f1']:.2f}% | Exact Sent Acc: {es_seg['greedy_exact_acc']:.2f}%")
    print(f"  --> DP Improvement over Greedy Baseline: +{es_seg['dp_f1'] - es_seg['greedy_f1']:.2f}% F1")

    print("\nRunning POS Tagging Evaluation...")
    es_hmm_std_acc, es_mft_std_acc, es_conf_std = evaluate_pos_tagging(es_models["test_std"], lang="es", morph=False, max_eval=400)
    es_hmm_morph_acc, es_mft_morph_acc, _ = evaluate_pos_tagging(es_models["test_morph"], lang="es", morph=True, max_eval=400)
    print(f"  [Standard POS] HMM Viterbi Acc: {es_hmm_std_acc:.2f}% | Baseline MFT Acc: {es_mft_std_acc:.2f}%")
    print(f"  [Morphology POS] HMM Viterbi Acc: {es_hmm_morph_acc:.2f}% | Baseline MFT Acc: {es_mft_morph_acc:.2f}%")
    print(f"  --> HMM Improvement over MFT Baseline: +{es_hmm_std_acc - es_mft_std_acc:.2f}%")

    print("\nSpanish Confusion Matrix (Top Standard Tags):")
    print_confusion_matrix(es_conf_std, top_n=6)

    print("\nRunning Spanish Error-Source Decomposition (Joint Segmentation + Tagging)...")
    es_errs = evaluate_error_sources(es_models["test_std"], lang="es", max_eval=300)
    print(f"  Total words evaluated:        {es_errs['total_words']}")
    print(f"  Correct (Word & Tag):         {es_errs['correct_count']} ({es_errs['correct_pct']:.2f}%)")
    print(f"  Segmentation-Induced Errors:  {es_errs['seg_errors']} ({es_errs['seg_error_pct']:.2f}%)")
    print(f"  Genuine POS Tagging Errors:   {es_errs['tag_errors']} ({es_errs['tag_error_pct']:.2f}%)")

    # 3. Sample string outputs
    run_sample_strings()


if __name__ == "__main__":
    main()
