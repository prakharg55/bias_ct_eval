#!/usr/bin/env python3
"""
v7 first-position answer readout (30 Sep 2026). Token sets = letters6.py (each letter's single-token forms "L", " L", "(L",
"(L)"). Letter score = the MAX logit over the letter's forms (v5/v6 used logsumexp = the letter's probability summed over
forms), so the readout letter is the letter of the highest-scoring letter-form token, i.e. the model's GREEDY answer
whenever its first generated token is a letter form.
Why: the v6 gate failed (ana_gate6.py: v6 agrees with the greedy generated answer on 97.3% of test prompts for untrained
Llama but only 94.9% for BCT s42). All the disagreements are prompts whose top token's letter differs from the letter with
the largest summed probability: Llama splits its mass between formats ("C" vs "(C"), so "most probable letter" and "letter
the model writes" differ on ~5% of prompts. The generated answer is the reference (papers score generated text), hence max.
v7.1: exact bf16 ties between letter-form tokens are broken like greedy decoding (lowest token id), see score().
MODE = "sum" restores logsumexp (v6; on gemma = v5 exactly, since letters6 adds no gemma token): used ONLY by the regression
test that checks the v7 instruments against the v5 outputs.
"""
import torch
from letters6 import get_letter_token_ids   # noqa: F401  (re-exported: the v7 instruments import it from here)
MODE = "max"; EPS = 1e-3; VOCAB = 262144.0   # 2^18 > both vocabularies (Llama 128256, gemma 256000): the tie term is < 1e-3 for every token id


def score(row, ids):
    """Letter score from the final-position logits `row` (1-D, float) and the letter's form token ids `ids` (LongTensor).
    v7 (MODE 'max'): max logit over the forms, MINUS EPS * (lowest tied form id) / VOCAB. The subtracted term (< 1e-3 = below
    the bf16 spacing 2^-10 of logits in [0.125, 0.25), let alone of answer logits ~10-30, so it never reorders distinct logits) breaks EXACT TIES the way greedy decoding
    does (torch.argmax over the vocabulary returns the lowest index): v7.1, added after the first gate runs showed that all
    remaining first-vs-generated disagreements were bf16 logit ties between e.g. "C" and "(A" (the project log).
    MODE 'sum': logsumexp over the forms (v5/v6), no tie term: used only for the gemma regression test."""
    assert MODE in ("max", "sum"), MODE
    x = row[ids.to(row.device)]
    if MODE == "sum": return torch.logsumexp(x, 0)
    m = x.max(); best = int(ids.to(row.device)[x == m].min())
    return m.double() - EPS * best / VOCAB     # float64: the tie term (>= 1.2e-7) is below float32 resolution at logit ~20 (1.9e-6)


def agg(x):
    """v7.0 (kept for reference, no longer called): max over forms without the tie term / logsumexp."""
    return x.max() if MODE == "max" else torch.logsumexp(x, 0)
