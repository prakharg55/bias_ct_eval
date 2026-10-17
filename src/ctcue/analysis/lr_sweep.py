#!/usr/bin/env python3
"""
ana_lrsweep_u (9 Oct 2026): EXPLORATORY learning-rate robustness of the gen-battery signature (prereg NOTE 09 Oct 2026, 00:12:39 EDT).
For each seed-42 sweep checkpoint one grid step below / above the selected LR (results/w7u/gen_<model>_lrsweep_<arm>_lr<lr>_s42.json) and the
selected-LR seed-42 final (utags routing), against the untrained model (results/w7/gen_<model>_none.json): paired helpful-uptake and
harmful-caving drops on the absolute, proportional and log-odds scales (ana_scale_u definitions, one seed: no bootstrap) and the relative
reduction of harmful caving by position (ana_w3 cells: each model's own known-uncued items, mean over the 6 wordings). CPU, seconds.
Output: results/w7u/ana_lrsweep_u.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/data/code")
import utags
from ana_scale_u import load, per_item, drops
R = "/data/ct_root/results"; POS = ("front", "pre", "post", "end")
SWEEP = {"gemma2b": {"bct": ("3.16e-5", "3.16e-4"), "act": ("3.16e-6", "3.16e-5"), "attct": ("5e-6", "5e-5"), "rmct": ("1e-5",)},
         "llama8b": {"bct": ("1.58e-5", "1.58e-4"), "act": ("1.58e-6", "1.58e-5"), "attct": ("1.58e-6", "1.58e-5"), "rmct": ("3.16e-6", "3.16e-5")}}


def one(bd, bkey, items, path):
    d, key = load(path); assert sorted(key) == items, path
    assert all(key[k]["T"] == bkey[k]["T"] and key[k]["X"] == bkey[k]["X"] for k in items), path
    P = bd["positions"]; hb, cb, nw = per_item(bd, bkey, items, P); hc, cc, _ = per_item(d, key, items, P)
    r0 = np.array([bkey[k]["pred0"] == bkey[k]["T"] for k in items]); r1 = np.array([key[k]["pred0"] == key[k]["T"] for k in items])
    I, J = (~r0) & (~r1), r0 & r1; n = nw * len(P)
    H = drops(hb[I].mean(), hc[I].mean(), n * I.sum()); C = drops(cb[J].mean(), cc[J].mean(), n * J.sum())
    out = {"n_wrong_both": int(I.sum()), "n_right_both": int(J.sum()), "uncued_acc": float(r1.mean()), "scales": {}}
    for k in H: out["scales"][k] = {"helpful_drop": float(H[k]), "harmful_drop": float(C[k]), "ratio": float(H[k] / C[k]) if C[k] > 0.02 else None}
    W = [w[0] for w in d["wordings"]]
    cave = lambda dd, kk, right, p: np.mean([[kk[k]["cond"][f"{w}|{p}|wrong"][2] == kk[k]["X"] for w in W] for k, ok in zip(items, right) if ok])
    out["relred"] = {p: float(1 - cave(d, key, r1, p) / cave(bd, bkey, r0, p)) for p in POS}
    return out


def main():
    res = {}
    for model, arms in SWEEP.items():
        bd, bkey = load(f"{R}/w7/gen_{model}_none.json"); items = sorted(bkey); res[model] = {}
        for arm, lrs in arms.items():
            row = {"selected": one(bd, bkey, items, f"{R}/w7u/gen_{utags.inst_tag(model, arm, 42)}.json")}
            for lr in lrs:
                f = f"{R}/w7u/gen_{model}_lrsweep_{arm}_lr{lr}_s42.json"
                row[lr] = one(bd, bkey, items, f) if os.path.exists(f) else None
            res[model][arm] = row
            for k, v in row.items():
                if v is None: print(f"{model} {arm:6s} {k:9s} MISSING", flush=True); continue
                s = v["scales"]; fm = lambda x: "  --" if x is None else f"{x:5.2f}"
                print(f"{model} {arm:6s} {k:9s} harmful {s['absolute']['harmful_drop']:+.3f} helpful {s['absolute']['helpful_drop']:+.3f} | ratio abs {fm(s['absolute']['ratio'])} "
                      f"prop {fm(s['proportional']['ratio'])} log {fm(s['log-odds']['ratio'])} | relred " + " ".join(f"{p} {v['relred'][p]:+.2f}" for p in POS), flush=True)
    # authors'-recipe Llama checkpoints (prereg NOTE 09 Oct 2026, 00:28:55 EDT)
    bd, bkey = load(f"{R}/w7/gen_llama8b_none.json"); items = sorted(bkey); res["llama8b_repro"] = {}
    for arm in ("bct", "act", "attct"):
        f = f"{R}/w7u/gen_llama8b_repro_{arm}.json"; v = one(bd, bkey, items, f) if os.path.exists(f) else None; res["llama8b_repro"][arm] = v
        if v is None: print(f"llama8b repro {arm}: MISSING", flush=True); continue
        s_ = v["scales"]; fm = lambda x: "  --" if x is None else f"{x:5.2f}"
        print(f"llama8b repro {arm:6s} harmful {s_['absolute']['harmful_drop']:+.3f} helpful {s_['absolute']['helpful_drop']:+.3f} | ratio abs {fm(s_['absolute']['ratio'])} prop {fm(s_['proportional']['ratio'])} "
              f"log {fm(s_['log-odds']['ratio'])} | relred " + " ".join(f"{p} {v['relred'][p]:+.2f}" for p in POS), flush=True)
    json.dump(res, open(f"{R}/w7u/ana_lrsweep_u.json", "w"), indent=1); print(f"-> {R}/w7u/ana_lrsweep_u.json")


if __name__ == "__main__":
    main()
