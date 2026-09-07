from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from statistics import median

from q4_pcfg_parser import parse_sentence
from shared_lm import TRIGRAM_COUNTS, bigram_logprob, trigram_logprob


@dataclass
class SentenceEvaluation:
    sentence: str
    pcfg_result: str
    bigram_score: float
    trigram_score: float
    chosen_method: str
    verdict: str
    seg_merges: int = 0
    spelling_corrections: int = 0
    pcfg_logprob: float | None = None


def _tokenize_sentence(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?]", text)
    return tokens


def _trigram_coverage(words: list[str]) -> float:
    if len(words) < 3:
        return 0.0
    seen = 0
    total = 0
    for w1, w2, w3 in zip(words, words[1:], words[2:]):
        total += 1
        if (w1.lower(), w2.lower()) in TRIGRAM_COUNTS and w3.lower() in TRIGRAM_COUNTS[(w1.lower(), w2.lower())]:
            seen += 1
    return seen / total if total else 0.0


def _pcfg_outlier(logprob: float, parse_probs: list[float]) -> bool:
    if len(parse_probs) < 2:
        return False
    med = median(parse_probs)
    deviations = [abs(p - med) for p in parse_probs]
    mad = median(deviations)
    if mad == 0:
        return logprob < med - 1.5
    return logprob < med - 2.5 * mad


def _decide_method(tokens: list[str], pcfg_logprob: float | None, bigram_score: float, trigram_score: float, all_parse_probs: list[float]) -> str:
    if pcfg_logprob is not None and not _pcfg_outlier(pcfg_logprob, all_parse_probs):
        return "PCFG"
    if len(tokens) >= 3 and _trigram_coverage(tokens) >= 0.15:
        return "Trigram"
    return "Bigram"


def _verdict_for(score_label: str, pcfg_logprob: float | None, bigram_score: float, trigram_score: float) -> str:
    if score_label == "PCFG":
        return "grammatical" if pcfg_logprob is not None else "weak"
    if score_label == "Trigram":
        return "likely grammatical" if trigram_score > -30 else "borderline"
    return "likely ungrammatical" if bigram_score < -20 else "borderline"


def _split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p and p.strip()]


def analyze_passage(corrected_passage: str, seg_merges_resolved: int = 0, spelling_corrections_applied: int = 0) -> list[SentenceEvaluation]:
    sentences = _split_sentences(corrected_passage)
    if not sentences:
        return []

    parsed_probabilities: list[float] = []
    parsed_sentence_indices: list[int] = []

    for i, sentence in enumerate(sentences):
        words = _tokenize_sentence(sentence)
        if not words:
            continue
        tree = parse_sentence(words)
        if tree is not None:
            parsed_probabilities.append(float(tree.logprob()))
            parsed_sentence_indices.append(i)

    evaluations: list[SentenceEvaluation] = []
    for i, sentence in enumerate(sentences):
        words = _tokenize_sentence(sentence)
        if not words:
            continue

        tree = parse_sentence(words)
        pcfg_logprob = float(tree.logprob()) if tree is not None else None
        bigram_score = bigram_logprob(words)
        trigram_score = trigram_logprob(words)
        chosen = _decide_method(words, pcfg_logprob, bigram_score, trigram_score, parsed_probabilities)
        verdict = _verdict_for(chosen, pcfg_logprob, bigram_score, trigram_score)
        pcfg_result = f"parsed ({pcfg_logprob:.3f})" if tree is not None else "fail"

        evaluations.append(
            SentenceEvaluation(
                sentence=sentence,
                pcfg_result=pcfg_result,
                bigram_score=bigram_score,
                trigram_score=trigram_score,
                chosen_method=chosen,
                verdict=verdict,
                seg_merges=seg_merges_resolved,
                spelling_corrections=spelling_corrections_applied,
                pcfg_logprob=pcfg_logprob,
            )
        )

    return evaluations


def _print_table(results: list[SentenceEvaluation]) -> None:
    headers = [
        "Sentence",
        "PCFG result",
        "Bigram score",
        "Trigram score",
        "Chosen method",
        "Verdict",
        "Seg merges",
        "Spell fixes",
    ]
    rows = [
        [
            r.sentence,
            r.pcfg_result,
            f"{r.bigram_score:.3f}",
            f"{r.trigram_score:.3f}",
            r.chosen_method,
            r.verdict,
            str(r.seg_merges),
            str(r.spelling_corrections),
        ]
        for r in results
    ]

    widths = [max(len(str(header)), max(len(str(row[idx])) for row in [headers] + rows)) for idx, header in enumerate(headers)]
    sep = " | ".join("-" * widths[i] for i in range(len(headers)))
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print(sep)
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Final grammar analysis over a corrected passage.")
    parser.add_argument("passage", nargs="?", default="The Fulton County Grand Jury said Friday an investigation of the weather in the city was cold and the people looked at the sky.", help="Corrected text passage to evaluate.")
    parser.add_argument("--seg-merges", type=int, default=3, help="Number of segmentation merges resolved in the pre-processing stage.")
    parser.add_argument("--spell-fixes", type=int, default=3, help="Number of spelling corrections applied before final scoring.")
    args = parser.parse_args()

    results = analyze_passage(args.passage, args.seg_merges, args.spell_fixes)
    _print_table(results)


if __name__ == "__main__":
    main()
