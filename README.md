# NLP Group Assignment 1 — Group 10

This repository contains the complete implementation and evaluation for all four questions of NLP Group Assignment 1:

- **Question 1**: Word Segmentation (DP Trigram LM), HMM POS Tagging, Morphology-Aware Agreement, Baselines, and Error-Source Decomposition (English & Spanish).
- **Question 2**: Arc-Standard Transition-Based Dependency Parser with Oracle Training, Feature Extraction, and Visualization.
- **Question 3**: Vocabulary & Language Model Spelling Corrector (Method A: Edits-1 vs. Method B: Symmetric Delete) with Speed Demon Benchmark & Terminal CLI.
- **Question 4**: Integrated Streaming Background Editor with Live Typo Injection ($p=0.08$), Live Alerts ($N=6$), PCFG Constituency Parsing, Multi-Criteria Sentence Scoring, and Interactive Streamlit UI.

---

## 👥 Group Information (Group 10)

| # | Name | Roll Number |
| :---: | :--- | :---: |
| 1 | Aditya Jha | 12340090 |
| 2 | Aditya Yadav | 12340100 |
| 3 | Hirannya Mhaisbadwe | 12340950 |
| 4 | Niraj Chandekar | 12341480 |
| 5 | Sanjani Kumari | 12341890 |

---

## 📂 Repository Structure

```text
NLP-Group-Assignment-1/
├── README.md                      # Comprehensive project guide & execution commands
├── AssignmentReport_NLPA1_G10.pdf # Complete 39-page final assignment report
├── Group Assignment 1.pdf         # Assignment problem specifications
│
├── ── Question 1: Word Segmentation & POS Tagging ──
├── q1_train.py                    # Trains English (Brown 80/20) & Spanish (UD_Spanish-GSD) LMs & HMMs
├── q1_module.py                   # DP word segmenter, HMM Viterbi tagger, baselines & Q4 exports
├── q1_evaluate.py                 # Full benchmark suite (F1, Accuracy, Baselines, Confusion Matrix, Error Decomposition)
├── q1_models_en.pkl               # Serialized English trained models
├── q1_models_es.pkl               # Serialized Spanish trained models
├── q1_trigram_lm.pkl              # Reusable trigram LM artifact for Q4
├── q1_terminal_english.png        # English evaluation terminal output
├── q1_terminal_spanish.png        # Spanish evaluation terminal output
├── q1_terminal_samples.png        # Sample test strings output
│
├── ── Question 2: Transition-Based Dependency Parser ──
├── q2_dependency_parser2.py       # CoNLL-U reader, Arc-Standard oracle, feature extractor, classifier & LAS eval
├── render_dependency_trees.py     # Standalone / SpaCy SVG dependency tree generator
├── dependency_tree_screenshots/   # Rendered dependency tree SVGs
├── UD_English-EWT/                # Universal Dependencies English EWT dataset
│
├── ── Question 3: Spelling Corrector & Speed Demon ──
├── q3_train.py                    # Builds Brown unigrams, bigrams, and vocabulary
├── q3_spelling_model.pkl          # Pickled spelling model dictionary
├── q3_module.py                   # Candidate generation (Method A edits1, Method B delete dict), non-word & real-word logic
├── q3_spelling_corrector.py       # Standalone test generator, speed benchmark & interactive terminal CLI
│
├── ── Question 4: Live Editor, PCFG & Web App ──
├── shared_lm.py                   # Shared Add-k smoothed (k=0.5) Brown Bigram/Trigram LM
├── q4_pcfg_parser.py              # Penn Treebank PCFG induction with Brown->Penn tag reconciliation
├── live_pipeline.py               # Streaming typing simulation (p=0.08) with segmentation, spelling & grammar alerts (N=6)
├── final_analysis.py              # Hierarchical sentence analysis (PCFG -> Trigram -> Bigram decision rule)
├── speed_demon.py                 # Pipeline overhead micro-benchmark (1,000 synthetic words)
├── app.py                         # Streamlit interactive live typing web application
└── build_updated_report.py        # Automated report PDF generator & assembler
```

---

## ⚡ Quick Start & Environment Setup

### 1. Create and Activate Virtual Environment

```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Required Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install nltk scikit-learn pandas streamlit matplotlib pillow reportlab pymupdf
```

### 3. Download Required Corpora (Automated or Manual)

```bash
python -c "import nltk; nltk.download('brown'); nltk.download('universal_tagset'); nltk.download('treebank'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 🚀 Execution Guide by Question

### 🔹 Question 1: Word Segmentation & POS Tagging
```bash
# 1. Train English & Spanish models (Trigram LM + Standard & Morphology HMMs)
python q1_train.py

# 2. Run comprehensive evaluation benchmark (Baselines, Confusion Matrices, Error Breakdown, Sample Strings)
python q1_evaluate.py
```
**Key Highlights**:
- English Segmentation: **97.15% F1** (+14.99% over Greedy baseline).
- Spanish Segmentation: **92.63% F1** (+22.76% over Greedy baseline).
- English POS Accuracy: **93.86%** standard, **93.33%** morphology-aware.
- Spanish POS Accuracy: **90.79%** standard, **89.70%** morphology-aware.
- Error Decomposition: **92.98%** English correct, **87.82%** Spanish correct.

---

### 🔹 Question 2: Transition-Based Dependency Parser
```bash
# 1. Train and evaluate Arc-Standard dependency parser on UD English-EWT
python q2_dependency_parser2.py

# 2. Generate dependency tree SVG diagrams for assignment test sentences
python render_dependency_trees.py
```
**Key Highlights**:
- Evaluated on `en_ewt-ud-dev.conllu` (`2,001` sentences, `25,148` tokens).
- Unlabeled Attachment Score (UAS): **64.05%**.
- Labeled Attachment Score (LAS): **54.41%**.

---

### 🔹 Question 3: Efficient Spelling Corrector
```bash
# 1. Train Brown spelling models
python q3_train.py

# 2. Run candidate generation benchmark (Method A vs. Method B) & accuracy tests
python q3_spelling_corrector.py
```
**Key Highlights**:
- **Method A**: Levenshtein Distance 1 candidate generation (`edits1`).
- **Method B**: Symmetric Delete spelling correction via precomputed delete dictionary.
- Contextual real-word error correction using bigram transition probabilities.

---

### 🔹 Question 4: Live Editor, PCFG & Interactive Web App
```bash
# 1. Run live streaming typing simulation with real-time alerts
python live_pipeline.py

# 2. Run final passage decision-matrix scoring table
python final_analysis.py

# 3. Run Speed Demon micro-benchmark
python speed_demon.py

# 4. Launch interactive Streamlit web dashboard
streamlit run app.py
```
**Key Parameters**:
- Merge Probability $p = 0.08$ (simulates missed-space fast typing).
- Trigger Window $N = 6$ words for rolling grammar log-likelihood checks.
- Add-$k$ Smoothing $k = 0.5$ on Brown Bigram/Trigram LM.
- Hierarchical decision rule: Prefer PCFG (with MAD outlier rejection) $\rightarrow$ fallback to Trigram (coverage $\ge 15\%$) $\rightarrow$ fallback to Bigram.

---

## 📄 Final Report

The complete written report with all derivations, tables, confusion matrices, error breakdowns, diagrams, and terminal screenshots is available in:
- [`AssignmentReport_NLPA1_G10.pdf`](AssignmentReport_NLPA1_G10.pdf)
