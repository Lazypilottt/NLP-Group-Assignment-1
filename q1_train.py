from __future__ import annotations

import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nltk
from nltk.corpus import brown


# ==============================================================================
# DATA LOADERS & PREPROCESSING
# ==============================================================================

def load_conllu_sentences(file_path: str | Path) -> list[list[dict[str, str]]]:
    """Load sentences from a CoNLL-U format file (e.g., UD Spanish-GSD)."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CoNLL-U dataset not found at {file_path}")

    sentences: list[list[dict[str, str]]] = []
    current_tokens: list[dict[str, str]] = []

    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    current_tokens = []
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            token_id = parts[0]
            # Ignore multi-word tokens (e.g. 1-2) and empty nodes (e.g. 3.1)
            if "-" in token_id or "." in token_id:
                continue

            form = parts[1]
            upos = parts[3]
            feats_str = parts[5]

            # Parse features for morphology-aware tagging
            feats = {}
            if feats_str and feats_str != "_":
                for item in feats_str.split("|"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        feats[k] = v

            current_tokens.append({
                "form": form,
                "upos": upos,
                "feats": feats_str,
                "gender": feats.get("Gender", ""),
                "number": feats.get("Number", ""),
                "tense": feats.get("Tense", ""),
                "person": feats.get("Person", ""),
            })

    if current_tokens:
        sentences.append(current_tokens)

    return sentences


def get_spanish_data(base_dir: str | Path | None = None) -> dict[str, Any]:
    """Load UD_Spanish-GSD train, dev, and test sets."""
    base_path = Path(base_dir) if base_dir else Path(__file__).resolve().parent / "UD_Spanish-GSD"
    train_file = base_path / "es_gsd-ud-train.conllu"
    dev_file = base_path / "es_gsd-ud-dev.conllu"
    test_file = base_path / "es_gsd-ud-test.conllu"

    train_sents = load_conllu_sentences(train_file)
    dev_sents = load_conllu_sentences(dev_file)
    test_sents = load_conllu_sentences(test_file)

    def extract_pairs(sents):
        res_std = []
        res_morph = []
        for s in sents:
            pair_std = []
            pair_morph = []
            for tok in s:
                w = tok["form"]
                upos = tok["upos"]
                # Create morphology-aware tag
                m_parts = [upos]
                if tok["gender"]:
                    m_parts.append(tok["gender"])
                if tok["number"]:
                    m_parts.append(tok["number"])
                morph_tag = "-".join(m_parts)

                pair_std.append((w, upos))
                pair_morph.append((w, morph_tag))
            if pair_std:
                res_std.append(pair_std)
                res_morph.append(pair_morph)
        return res_std, res_morph

    train_std, train_morph = extract_pairs(train_sents)
    dev_std, dev_morph = extract_pairs(dev_sents)
    test_std, test_morph = extract_pairs(test_sents)

    return {
        "train_std": train_std,
        "train_morph": train_morph,
        "dev_std": dev_std,
        "dev_morph": dev_morph,
        "test_std": test_std,
        "test_morph": test_morph,
    }


def get_english_data(split_ratio: float = 0.8) -> dict[str, Any]:
    """Load English Brown corpus with Universal POS tags and split 80/20."""
    try:
        tagged_sents = list(brown.tagged_sents(tagset="universal"))
        raw_tagged_sents = list(brown.tagged_sents())  # Detailed Brown tags for morphology
    except LookupError:
        nltk.download("brown", quiet=True)
        nltk.download("universal_tagset", quiet=True)
        tagged_sents = list(brown.tagged_sents(tagset="universal"))
        raw_tagged_sents = list(brown.tagged_sents())

    n_total = len(tagged_sents)
    split_idx = int(n_total * split_ratio)

    train_raw = tagged_sents[:split_idx]
    test_raw = tagged_sents[split_idx:]

    train_detailed = raw_tagged_sents[:split_idx]
    test_detailed = raw_tagged_sents[split_idx:]

    def build_morph_sents(univ_sents, detail_sents):
        morph_sents = []
        for u_sent, d_sent in zip(univ_sents, detail_sents):
            pair_morph = []
            for (w, u_tag), (_, d_tag) in zip(u_sent, d_sent):
                d_upper = d_tag.upper()
                morph_tag = u_tag
                if u_tag == "NOUN":
                    if "NNS" in d_upper or d_upper.endswith("S"):
                        morph_tag = "NOUN-Pl"
                    else:
                        morph_tag = "NOUN-Sg"
                elif u_tag == "VERB":
                    if "VBD" in d_upper:
                        morph_tag = "VERB-Past"
                    elif "VBG" in d_upper:
                        morph_tag = "VERB-Prog"
                    elif "VBZ" in d_upper:
                        morph_tag = "VERB-Pres-Sg3"
                    elif "VBN" in d_upper:
                        morph_tag = "VERB-Part"
                    else:
                        morph_tag = "VERB-Base"
                elif u_tag == "ADJ":
                    if "JJR" in d_upper:
                        morph_tag = "ADJ-Comp"
                    elif "JJS" in d_upper:
                        morph_tag = "ADJ-Super"
                    else:
                        morph_tag = "ADJ-Pos"
                elif u_tag == "PRON":
                    if d_upper in {"PP3", "PPO", "PPS", "PPSS", "HE", "SHE", "IT"}:
                        morph_tag = "PRON-Sg"
                    elif d_upper in {"THEY", "THEM", "WE", "US"}:
                        morph_tag = "PRON-Pl"
                pair_morph.append((w, morph_tag))
            morph_sents.append(pair_morph)
        return morph_sents

    train_morph = build_morph_sents(train_raw, train_detailed)
    test_morph = build_morph_sents(test_raw, test_detailed)

    return {
        "train_std": train_raw,
        "train_morph": train_morph,
        "test_std": test_raw,
        "test_morph": test_morph,
    }


# ==============================================================================
# MODEL TRAINERS (TRIGRAM LM & HMM POS TAGGER)
# ==============================================================================

def train_trigram_lm(sentences: list[list[tuple[str, str]] | list[str]], k: float = 0.5) -> dict[str, Any]:
    """Train a smoothed trigram language model over word tokens."""
    unigram_counts: Counter[str] = Counter()
    bigram_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    trigram_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    total_words = 0

    for sent in sentences:
        if not sent:
            continue
        if isinstance(sent[0], tuple):
            words = [w.lower() for w, _ in sent if w]
        else:
            words = [w.lower() for w in sent if w]

        if not words:
            continue

        tokens = ["<s>", "<s>"] + words + ["</s>"]
        for w in tokens:
            unigram_counts[w] += 1
            total_words += 1

        for w1, w2 in zip(tokens, tokens[1:]):
            bigram_counts[w1][w2] += 1

        for w1, w2, w3 in zip(tokens, tokens[1:], tokens[2:]):
            trigram_counts[(w1, w2)][w3] += 1

    vocab = set(unigram_counts.keys())

    return {
        "vocab": sorted(vocab),
        "vocab_set": vocab,
        "unigram_counts": dict(unigram_counts),
        "bigram_counts": {k: dict(v) for k, v in bigram_counts.items()},
        "trigram_counts": {k: dict(v) for k, v in trigram_counts.items()},
        "total_words": total_words,
        "k": k,
    }


def train_hmm_pos_tagger(sentences: list[list[tuple[str, str]]], alpha: float = 1.0, beta: float = 1.0) -> dict[str, Any]:
    """Train a Hidden Markov Model (HMM) POS tagger with transition and emission counts."""
    tag_counts: Counter[str] = Counter()
    tag_transitions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    tag_start_counts: Counter[str] = Counter()
    word_counts: Counter[str] = Counter()
    emission_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    word_to_tag_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    suffix_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for sent in sentences:
        if not sent:
            continue
        prev_tag = "<s>"
        for i, (word, tag) in enumerate(sent):
            w_lower = word.lower()
            tag_counts[tag] += 1
            word_counts[w_lower] += 1
            emission_counts[tag][w_lower] += 1
            word_to_tag_counts[w_lower][tag] += 1

            # Morphological suffix heuristic counts for OOV (last 1, 2, 3 chars)
            if len(w_lower) >= 2:
                suffix_counts[tag][w_lower[-2:]] += 1
            if len(w_lower) >= 3:
                suffix_counts[tag][w_lower[-3:]] += 1

            if i == 0:
                tag_start_counts[tag] += 1
            tag_transitions[prev_tag][tag] += 1
            prev_tag = tag

        tag_transitions[prev_tag]["</s>"] += 1

    # Most frequent tag baseline dictionary
    most_frequent_tag_map = {
        w: counts.most_common(1)[0][0] for w, counts in word_to_tag_counts.items()
    }
    overall_most_frequent_tag = tag_counts.most_common(1)[0][0] if tag_counts else "NOUN"

    all_tags = sorted(tag_counts.keys())

    return {
        "tags": all_tags,
        "tag_counts": dict(tag_counts),
        "tag_transitions": {k: dict(v) for k, v in tag_transitions.items()},
        "tag_start_counts": dict(tag_start_counts),
        "emission_counts": {k: dict(v) for k, v in emission_counts.items()},
        "word_counts": dict(word_counts),
        "suffix_counts": {k: dict(v) for k, v in suffix_counts.items()},
        "mft_map": most_frequent_tag_map,
        "overall_mft": overall_most_frequent_tag,
        "alpha": alpha,
        "beta": beta,
    }


# ==============================================================================
# MAIN MODEL COMPILATION & SERIALIZATION
# ==============================================================================

def train_and_save_all(out_dir: str | Path | None = None) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir else Path(__file__).resolve().parent

    print("=" * 70)
    print("STEP 1: Loading English Brown Corpus (80/20 train/test split)...")
    en_data = get_english_data(split_ratio=0.8)
    print(f"  English train sentences: {len(en_data['train_std'])}")
    print(f"  English test sentences:  {len(en_data['test_std'])}")

    print("\nSTEP 2: Training English Trigram LM & HMM Taggers (Standard & Morphology)...")
    en_trigram_lm = train_trigram_lm(en_data["train_std"])
    en_hmm_std = train_hmm_pos_tagger(en_data["train_std"])
    en_hmm_morph = train_hmm_pos_tagger(en_data["train_morph"])

    en_payload = {
        "lang": "en",
        "trigram_lm": en_trigram_lm,
        "hmm_std": en_hmm_std,
        "hmm_morph": en_hmm_morph,
        "test_std": en_data["test_std"],
        "test_morph": en_data["test_morph"],
    }
    with (out_path / "q1_models_en.pkl").open("wb") as fh:
        pickle.dump(en_payload, fh)
    print("  Saved English models to q1_models_en.pkl")

    # Save backwards-compatible q1_trigram_lm.pkl for Q4
    legacy_trigram = {
        "vocab": en_trigram_lm["vocab"],
        "token_counts": en_trigram_lm["unigram_counts"],
        "trigram_counts": en_trigram_lm["trigram_counts"],
        "sentences": [" ".join([w for w, _ in s]) for s in en_data["train_std"][:100]],
    }
    with (out_path / "q1_trigram_lm.pkl").open("wb") as fh:
        pickle.dump(legacy_trigram, fh)
    print("  Saved backwards-compatible q1_trigram_lm.pkl")

    print("\nSTEP 3: Loading Spanish UD_Spanish-GSD Corpus...")
    es_data = get_spanish_data()
    print(f"  Spanish train sentences: {len(es_data['train_std'])}")
    print(f"  Spanish test sentences:  {len(es_data['test_std'])}")

    print("\nSTEP 4: Training Spanish Trigram LM & HMM Taggers (Standard & Morphology)...")
    es_trigram_lm = train_trigram_lm(es_data["train_std"])
    es_hmm_std = train_hmm_pos_tagger(es_data["train_std"])
    es_hmm_morph = train_hmm_pos_tagger(es_data["train_morph"])

    es_payload = {
        "lang": "es",
        "trigram_lm": es_trigram_lm,
        "hmm_std": es_hmm_std,
        "hmm_morph": es_hmm_morph,
        "test_std": es_data["test_std"],
        "test_morph": es_data["test_morph"],
    }
    with (out_path / "q1_models_es.pkl").open("wb") as fh:
        pickle.dump(es_payload, fh)
    print("  Saved Spanish models to q1_models_es.pkl")

    print("=" * 70)
    print("All Question 1 models successfully trained and serialized!")
    return {"en": en_payload, "es": es_payload}


def save_model(path: str | Path | None = None) -> dict:
    """Entry point for legacy script compatibility."""
    return train_and_save_all(Path(path).parent if path else None)


if __name__ == "__main__":
    train_and_save_all()
