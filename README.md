# NLP Group Assignment 1

## Project overview

This repository contains the implementation for the four main assignment components:

- Q1: merged-token segmentation, POS tagging, and trigram-based beam decoding
- Q2: dependency parsing / dependency structure visualization
- Q3: spelling correction using vocabulary and language-model heuristics
- Q4: integrated live typing pipeline, grammar alerts, PCFG parsing, and final sentence scoring

The project is structured to keep training-time code separate from runtime inference code and to reuse trained artifacts instead of retraining during Q4.

---

## Question 1 — Segmentation + POS Tagging + Beam Decoder

- `q1_module.py` exposes:
  - `segment_beam(token: str) -> list[str]`
  - `pos_tag(words: list[str]) -> list[tuple[str, str]]`
  - `trigram_lm` as the loaded trained model artifact
- Training logic is kept in `q1_train.py`.
- Q4 reuses the trained Q1 outputs without retraining.

## Question 2 — Dependency Parser

- The dependency parser work is implemented and included as part of the overall project.
- `q2_dependency_parser2.py` provides the dependency parsing component used for structural analysis.
- Dependency-tree screenshots can be generated using the helper script `render_dependency_trees.py`.

## Question 3 — Spelling Corrector

- `q3_module.py` exposes:
  - `vocab`
  - `unigram_freq`
  - `bigram_model`
  - `gen_candidates_a(word)`
  - `gen_candidates_b(word)`
  - `correct_nonword(word)`
  - `correct_realword(sentence, target_idx)`
- Training logic remains in `q3_train.py`.
- Q4 loads and reuses the trained spelling assets without retraining.

## Question 4 — Live Pipeline + Final Analysis

Q4 integrates the reusable outputs from Q1 and Q3 with a shared Brown-based language model:

- `shared_lm.py` provides the single shared LM for bigram/trigram scoring
- `live_pipeline.py` performs segmentation, spell correction, and grammar checks in sequence
- `q4_pcfg_parser.py` implements a safe PCFG parser with tag reconciliation
- `final_analysis.py` scores sentences and decides the best grammaticality method
- `app.py` wraps the pipeline in a Streamlit interface
- `speed_demon.py` benchmarks latency for different processing layers

Key parameter choices:

- `p = 0.08` for simulated missed-space merges
- `N = 6` words for grammar-window checks
- `k = 0.5` for add-k smoothing

---

## Quick start

### 1) Install dependencies

```bash
pip install nltk pandas streamlit spacy
python -m spacy download en_core_web_sm
```

### 2) Run the Streamlit app

```bash
streamlit run app.py
```

### 3) Run the benchmark

```bash
python speed_demon.py
```

```

---

## Main files

```text
.
├── app.py                     # Streamlit app for live typing + final analysis
├── final_analysis.py          # Sentence scoring and final verdict logic
├── live_pipeline.py           # Live segmentation + spell + grammar check pipeline
├── q1_module.py               # Reusable Q1 runtime module
├── q1_train.py                # Q1 training logic (offline)
├── q2_dependency_parser2.py   # Dependency parser implementation
├── q3_module.py               # Reusable Q3 spelling correction module
├── q3_train.py                # Q3 training logic (offline)
├── shared_lm.py               # Shared Brown bigram/trigram LM
├── q4_pcfg_parser.py          # PCFG parser + tag reconciliation
├── speed_demon.py             # Benchmark script for latency comparisons
├── render_dependency_trees.py # Dependency tree generation
├── README.md                 # Project overview and usage instructions
├── AssignmentReport_NLPA1_G10.pdf #Assignment Report
└── ...
```

---

## Notes

- Training code is kept separate from runtime inference code, as required by the assignment.
- The final analysis prefers PCFG when it is reliable; otherwise it falls back to trigram and then bigram.
- This README is intentionally short and project-focused; it summarizes the implementation rather than reproducing the full assignment narrative which we;ve put together in the report PDF. 
