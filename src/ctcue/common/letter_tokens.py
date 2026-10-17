"""Token ids for each answer letter, including the single tokens "(A", "(B", ... of Llama.

Llama-3.1 often answers "(B)", so reading only the tokens "B" and " B" at the first answer position
misreads many of its answers. `get_letter_token_ids` returns, for each letter, the tokens "L", " L"
and every single token "(L" or "(L)".
"""

from ctcue.common.dumps import get_letter_token_ids as _base, LETTERS


def get_letter_token_ids(tokenizer, letters=LETTERS):
    out = {L: list(v) for L, v in _base(tokenizer, letters).items()}
    want = {}
    for L in letters:
        want[f"({L}"] = L
        want[f"({L})"] = L
    for tid in range(len(tokenizer)):
        s = tokenizer.decode([tid]).strip()
        L = want.get(s)
        if L is not None and tid not in out[L]:
            out[L].append(tid)
    return out
