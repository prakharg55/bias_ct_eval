"""Compute the original model's answer distribution on each training question (the Bayes-BCT prior).

For the uncued prompt of every training question, stores the softmax over its options of the
first-position letter scores. A letter's score is the log-sum-exp of the logits of its tokens "L"
and " L" (common/dumps.py), so the prior sums each letter's probability over its forms. It is not
the max readout of common/letter_readout.py. Used for Gemma. Llama uses answer_prior_v2.py.

Usage: python -m ctcue.data.answer_prior --model gemma2b
Output: $CT_ROOT/shared/p0_<model>.jsonl
"""

import argparse, json, re
import torch

from ctcue.common.dumps import get_letter_token_ids
from ctcue.common.load_merged import load_model
from ctcue.paths import CT_ROOT

SH = f"{CT_ROOT}/shared"
CHOICES = "Answer choices:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gemma2b", "llama8b"])
    a = ap.parse_args()
    tok, model = load_model("none", model=a.model)
    li = {L: torch.tensor(v) for L, v in get_letter_token_ids(tok).items() if v}
    recs = [json.loads(l) for l in open(f"{SH}/shared_train.jsonl")]
    out = []
    with torch.inference_mode():
        for r in recs:
            body = r["clean"]
            assert body.count(CHOICES) == 1, r["i"]
            opts = re.findall(r"^\(([A-P])\) ", body.split(CHOICES)[-1], re.M)
            assert (
                len(opts) >= 2
                and len(set(opts)) == len(opts)
                and r["T"] in opts
                and r["X"] in opts
                and all(o in li for o in opts)
            ), (r["i"], opts)
            x = tok.apply_chat_template(
                [{"role": "user", "content": body}], add_generation_prompt=True, return_tensors="pt"
            ).cuda()
            row = model(input_ids=x, use_cache=False).logits[0, -1].float()
            sc = torch.stack([torch.logsumexp(row[li[o].to(row.device)], 0) for o in opts])
            p = torch.softmax(sc, 0).tolist()
            out.append(
                dict(
                    i=r["i"],
                    opts=opts,
                    p0=dict(zip(opts, p)),
                    pred=opts[max(range(len(opts)), key=lambda k: p[k])],
                    T=r["T"],
                    X=r["X"],
                )
            )
    with open(f"{SH}/p0_{a.model}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    acc = sum(o["pred"] == o["T"] for o in out) / len(out)
    print(
        f"[p0 {a.model}] {len(out)} items; argmax accuracy {acc:.3f}; mean max-prob {sum(max(o['p0'].values()) for o in out) / len(out):.3f}; "
        f"options per item {sorted(set(len(o['opts']) for o in out))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
