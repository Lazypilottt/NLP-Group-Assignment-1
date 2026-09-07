from __future__ import annotations

import re
import time
from collections import deque

import pandas as pd
import streamlit as st

from final_analysis import analyze_passage
from q1_module import segment_beam
from q3_module import correct_nonword, vocab
from shared_lm import bigram_logprob, trigram_logprob


GRAMMAR_WINDOW = 6


if "typed_text" not in st.session_state:
    st.session_state.typed_text = ""
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "recent_window" not in st.session_state:
    st.session_state.recent_window = deque(maxlen=GRAMMAR_WINDOW)
if "seg_merges" not in st.session_state:
    st.session_state.seg_merges = 0
if "spell_fixes" not in st.session_state:
    st.session_state.spell_fixes = 0
if "total_latency_ms" not in st.session_state:
    st.session_state.total_latency_ms = 0.0


def _iter_tokens(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?]", text)


def _grammar_anomaly(window: list[str]) -> bool:
    if len(window) < GRAMMAR_WINDOW:
        return False
    score = bigram_logprob(window) + trigram_logprob(window)
    baseline = -len(window) * 12.0
    return score < baseline - 6.0


def _process_one_token(token: str):
    messages: list[str] = []
    seg_merges = 0
    spell_fixes = 0
    token_to_check = token

    if len(token_to_check) > 12 or token_to_check.lower() not in vocab:
        token_l = token_to_check.lower()
        parts = segment_beam(token_l)
        valid_parts = [p for p in parts if p and p.isalpha()]
        if len(valid_parts) >= 2 and all(p in vocab for p in valid_parts):
            actual_score = bigram_logprob([token_l]) + trigram_logprob([token_l])
            candidate_score = bigram_logprob(valid_parts) + trigram_logprob(valid_parts)
            if candidate_score > actual_score:
                token_to_check = " ".join(valid_parts)
                messages.append(f"[SEGMENT ALERT] {token!r} -> {token_to_check!r}")
                seg_merges += 1
                token_to_check = token_to_check

    if token_to_check.lower() not in vocab:
        suggestion = correct_nonword(token_to_check.lower())
        if suggestion != token_to_check.lower() and suggestion in vocab:
            messages.append(f"[SPELL ALERT] {token_to_check!r} -> {suggestion!r}")
            token_to_check = suggestion
            spell_fixes += 1

    return token_to_check, messages, seg_merges, spell_fixes


def _process_incremental_text(raw_text: str):
    st.session_state.alerts = []
    st.session_state.seg_merges = 0
    st.session_state.spell_fixes = 0
    st.session_state.recent_window = deque(maxlen=GRAMMAR_WINDOW)

    for token in _iter_tokens(raw_text):
        processed_token, messages, seg_delta, spell_delta = _process_one_token(token)
        st.session_state.seg_merges += seg_delta
        st.session_state.spell_fixes += spell_delta
        for msg in messages:
            st.session_state.alerts.append(msg)

        token_words = re.findall(r"[A-Za-z]+", processed_token)
        for word in token_words:
            st.session_state.recent_window.append(word.lower())
            if len(st.session_state.recent_window) == GRAMMAR_WINDOW and _grammar_anomaly(list(st.session_state.recent_window)):
                st.session_state.alerts.append(f"[GRAMMAR ALERT] window={list(st.session_state.recent_window)} looks anomalous")


def _finalize_analysis():
    text = st.session_state.typed_text.strip()
    if not text:
        return
    t0 = time.perf_counter()
    results = analyze_passage(text, st.session_state.seg_merges, st.session_state.spell_fixes)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    st.session_state.total_latency_ms = latency_ms
    rows = [
        {
            "Sentence": r.sentence,
            "PCFG": r.pcfg_result,
            "Bigram": round(r.bigram_score, 3),
            "Trigram": round(r.trigram_score, 3),
            "Chosen": r.chosen_method,
            "Verdict": r.verdict,
            "Seg merges": r.seg_merges,
            "Spell fixes": r.spelling_corrections,
        }
        for r in results
    ]
    return pd.DataFrame(rows)


st.title("Live NLP typing monitor")

user_text = st.text_input(
    "Type a passage:",
    value=st.session_state.typed_text,
    key="live_input",
    placeholder="Start typing here...",
)

if user_text != st.session_state.typed_text:
    st.session_state.typed_text = user_text
    if user_text:
        _process_incremental_text(user_text)
    else:
        st.session_state.alerts = []
        st.session_state.seg_merges = 0
        st.session_state.spell_fixes = 0
        st.session_state.recent_window = deque(maxlen=GRAMMAR_WINDOW)

st.write("Current text:", st.session_state.typed_text)

if st.session_state.alerts:
    st.subheader("Live alerts")
    for alert in st.session_state.alerts:
        st.write(alert)
else:
    st.caption("No alerts yet. Keep typing...")

finalize = st.button("Finalize and score passage")
if finalize and st.session_state.typed_text.strip():
    final_table = _finalize_analysis()
    if final_table is not None and not final_table.empty:
        st.subheader("Final sentence analysis")
        st.dataframe(final_table, use_container_width=True)
        st.metric("Total latency", f"{st.session_state.total_latency_ms:.2f} ms")
