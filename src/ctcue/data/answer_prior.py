#!/usr/bin/env python3
"""
Phase 5 A1 input (27 Sep 2026): the UNTRAINED model's answer distribution p0(a|q) on each shared training question's
CLEAN prompt, over that question's options, with the evaluation readout (softmax over options of logsumexp over each
letter's token variants at the final position of the chat-templated prompt). GPU, ~1 min (gemma) / ~3 min (llama).
Usage: pmodel_prior.py --model gemma2b|llama8b  ->  ct_compare/shared/p0_<model>.jsonl  (i, opts, p0, pred, T, X)
"""
import argparse, json, re, sys
import torch
sys.path.insert(0, "/data/code"); sys.path.insert(0, "/data/code")
from utils import get_letter_token_ids
from pub_load2 import load_model
SH = "/data/ct_root/shared"; CHOICES = "Answer choices:"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["gemma2b", "llama8b"]); a = ap.parse_args()
    tok, model = load_model("none", model=a.model)
    li = {L: torch.tensor(v) for L, v in get_letter_token_ids(tok).items() if v}
    recs = [json.loads(l) for l in open(f"{SH}/shared_train.jsonl")]; out = []
    with torch.inference_mode():
        for r in recs:
            body = r["clean"]; assert body.count(CHOICES) == 1, r["i"]
            opts = re.findall(r"^\(([A-P])\) ", body.split(CHOICES)[-1], re.M)
            assert len(opts) >= 2 and len(set(opts)) == len(opts) and r["T"] in opts and r["X"] in opts and all(o in li for o in opts), (r["i"], opts)
            x = tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True, return_tensors="pt").cuda()
            row = model(input_ids=x, use_cache=False).logits[0, -1].float()
            sc = torch.stack([torch.logsumexp(row[li[o].to(row.device)], 0) for o in opts]); p = torch.softmax(sc, 0).tolist()
            out.append(dict(i=r["i"], opts=opts, p0=dict(zip(opts, p)), pred=opts[max(range(len(opts)), key=lambda k: p[k])], T=r["T"], X=r["X"]))
    with open(f"{SH}/p0_{a.model}.jsonl", "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    acc = sum(o["pred"] == o["T"] for o in out) / len(out)
    print(f"[p0 {a.model}] {len(out)} items; argmax accuracy {acc:.3f}; mean max-prob {sum(max(o['p0'].values()) for o in out) / len(out):.3f}; "
          f"options per item {sorted(set(len(o['opts']) for o in out))}", flush=True)


if __name__ == "__main__":
    main()
