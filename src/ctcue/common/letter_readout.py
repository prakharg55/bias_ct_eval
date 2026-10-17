"""First-position answer readout used by the instruments that read the answer from one forward pass.

With MODE = "max" (the default), a letter's score is the highest logit among its token forms
(letter_tokens.py). Exact bf16 ties are broken by the lowest token id, as greedy decoding does, so
this readout equals the generated answer whenever the first generated token is a letter form.
MODE = "sum" (option --agg sum) gives the earlier readout, the log-sum-exp over the forms. Most
Gemma runs use "sum". The Gemma cue battery and all Llama runs use "max".
"""

import torch

# noqa: F401  (re-exported: the v7 instruments import it from here)
from ctcue.common.letter_tokens import get_letter_token_ids

# 2^18 > both vocabularies (Llama 128256, gemma 256000): the tie term is < 1e-3 for every token id
MODE = "max"
EPS = 1e-3
VOCAB = 262144.0


def score(row, ids):
    """Letter score from the final-position logits `row` (1-D, float) and the letter's form token ids `ids` (LongTensor).
    v7 (MODE 'max'): max logit over the forms, MINUS EPS * (lowest tied form id) / VOCAB. The subtracted term (< 1e-3 = below
    the bf16 spacing 2^-10 of logits in [0.125, 0.25), let alone of answer logits ~10-30, so it never reorders distinct logits) breaks EXACT TIES the way greedy decoding
    does (torch.argmax over the vocabulary returns the lowest index): v7.1, added after the first gate runs showed that all
    remaining first-vs-generated disagreements were bf16 logit ties between e.g. "C" and "(A" (the project log).
    MODE 'sum': logsumexp over the forms (v5/v6), no tie term: used only for the gemma regression test."""
    assert MODE in ("max", "sum"), MODE
    x = row[ids.to(row.device)]
    if MODE == "sum":
        return torch.logsumexp(x, 0)
    m = x.max()
    best = int(ids.to(row.device)[x == m].min())
    # float64: the tie term (>= 1.2e-7) is below float32 resolution at logit ~20 (1.9e-6)
    return m.double() - EPS * best / VOCAB


def agg(x):
    """v7.0 (kept for reference, no longer called): max over forms without the tie term / logsumexp."""
    return x.max() if MODE == "max" else torch.logsumexp(x, 0)
