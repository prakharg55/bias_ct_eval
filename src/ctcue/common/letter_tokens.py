#!/usr/bin/env python3
"""
Letter-token sets, v6 (30 Sep 2026). The shared loader's get_letter_token_ids (utils.py; NOT modified)
returns, per letter L, the single-token forms "L" and " L". Llama-3.1's tokenizer also has SINGLE tokens "(A", "(B", ...
and Llama frequently answers "(B)", i.e. its first generated token is "(B": with the v5 sets the first-position readout
then compares only the bare-letter tokens and returns the wrong letter on 7-20% of prompts (readout_check6.py: every
disagreement with the generated answer has a "(X" first token; letter-first prompts agree 99.8-100%).
v6 = the v5 forms PLUS every single token whose decoded text, stripped of surrounding whitespace, is "(L" or "(L)".
Drop-in replacement: from letters6 import get_letter_token_ids.
"""
import sys
sys.path.insert(0, "/data/code")
from utils import get_letter_token_ids as _base, LETTERS


def get_letter_token_ids(tokenizer, letters=LETTERS):
    out = {L: list(v) for L, v in _base(tokenizer, letters).items()}
    want = {}
    for L in letters: want[f"({L}"] = L; want[f"({L})"] = L
    for tid in range(len(tokenizer)):
        s = tokenizer.decode([tid]).strip()
        L = want.get(s)
        if L is not None and tid not in out[L]: out[L].append(tid)
    return out
