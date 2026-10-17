#!/usr/bin/env python3
"""
ana_letterpos_u (11 Oct 2026, pre-submission RMCT audit; EXPLORATORY): does the position pattern survive a letter-preference control?

Cue battery files of T3 / Figure 4 (original model: w7/gen_<model>_none.json; methods: w7u/gen_<utags.inst_tag>.json). Each checkpoint
is scored on its OWN known items (uncued answer correct), wrong-answer cue, six wordings, four positions. Relative reduction at a
position = 1 - (method's following rate) / (original model's following rate), on three bases:
  cued         following = the answer is the cued letter (the paper's basis; T3 / Figure 4)
  wrong        following = any answer other than the correct one (a switch to a third letter does not count as resisting)
  wrong_xmodal as wrong, leaving out the items whose correct answer is the checkpoint's own most frequent uncued letter (the original
               model is scored on the same item subset); every method is treated alike
Per seed, then averaged over seeds (T3's convention). Self-checks before anything new is written: the cued basis reproduces
paper_tables.json T3 (gen relred) and the wrong basis, as a ratio of seed means, reproduces ana_scale_u.json pt_basis.
Also per checkpoint: the modal uncued letter and its share over the 400 battery items. Output: results/w7u/ana_letterpos_u.json
"""
import json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, "/data/code")
import utags

R = "/data/ct_root/results"; POS = ["front", "pre", "post", "end"]; SEEDS = (42, 43, 44)
ARMS = {"gemma2b": ("bct", "act", "attct", "rmct", "rmcta"), "llama8b": ("bct", "act", "attct", "rmct")}
TKEY = {"gemma2b": "gemma", "llama8b": "llama"}


def load(path):
    d = json.load(open(path)); return d["items"], [w[0] for w in d["wordings"]]


def rate(its, wn, follows, keep):
    """per position: share of (known item, wording) pairs on which the wrong-answer cue is followed"""
    out = []
    for p in POS:
        n = k = 0
        for it in its:
            if it["pred0"] != it["T"] or not keep(it): continue
            for w in wn: k += follows(it["cond"][f"{w}|{p}|wrong"][2], it["T"], it["X"]); n += 1
        out.append(k / n if n else float("nan"))
    return np.array(out)


def modal(its):
    c = Counter(it["pred0"] for it in its); L, n = c.most_common(1)[0]; return L, n / len(its)


def main():
    T = json.load(open(f"{R}/paper_u/paper_tables.json")); SC = json.load(open(f"{R}/w7u/ana_scale_u.json"))
    cued = lambda a, t, x: a == x; wrong = lambda a, t, x: a != t; every = lambda it: True
    out = {"doc": __doc__.strip().splitlines()[0], "positions": POS, "seeds": list(SEEDS)}
    for model, arms in ARMS.items():
        base, wn = load(f"{R}/w7/gen_{model}_none.json"); L0, sh0 = modal(base)
        o = {"original": {"modal_letter": L0, "modal_share": sh0}}
        for arm in arms:
            per = []
            for s in SEEDS:
                its, wn_ = load(f"{R}/w7u/gen_{utags.inst_tag(model, arm, s)}.json"); assert wn_ == wn
                assert [(i["ds"], i["hash"], i["T"], i["X"]) for i in its] == [(i["ds"], i["hash"], i["T"], i["X"]) for i in base], f"item set differs: {model} {arm} s{s}"
                L, sh = modal(its); xm = lambda it, L=L: it["T"] != L
                r = {"modal_letter": L, "modal_share": sh}
                for name, fol, keep in (("cued", cued, every), ("wrong", wrong, every), ("wrong_xmodal", wrong, xm)):
                    r[f"rate_{name}"] = rate(its, wn, fol, keep).tolist(); r[f"base_{name}"] = rate(base, wn, fol, keep).tolist()
                    r[name] = (1 - np.array(r[f"rate_{name}"]) / np.array(r[f"base_{name}"])).tolist()
                r["n_known_xmodal"] = sum(it["pred0"] == it["T"] and xm(it) for it in its); per.append(r)
            a = {"per_seed": per}
            for name in ("cued", "wrong", "wrong_xmodal"): a[name] = {p: float(np.mean([r[name][i] for r in per])) for i, p in enumerate(POS)}
            a["wrong_ratio_of_means"] = {p: float(1 - np.mean([r["rate_wrong"][i] for r in per]) / per[0]["base_wrong"][i]) for i, p in enumerate(POS)}
            # self-checks against the paper's existing numbers
            if arm in T[TKEY[model]]["gen"]["arms"]:
                ref = T[TKEY[model]]["gen"]["arms"][arm]["relred"]
                for p in POS: assert abs(a["cued"][p] - ref[p]["mean"]) < 1e-9, (model, arm, p, a["cued"][p], ref[p]["mean"])
            if arm in SC.get(model, {}):
                for p in POS: assert abs(a["wrong_ratio_of_means"][p] - SC[model][arm]["pt_basis"][p]["relred_not_T"]) < 1e-9, (model, arm, p)
            o[arm] = a
            print(f"{model} {arm:6s} cued " + "/".join(f"{a['cued'][p]:.2f}" for p in POS) + "  wrong " + "/".join(f"{a['wrong'][p]:.2f}" for p in POS)
                  + "  wrong_xmodal " + "/".join(f"{a['wrong_xmodal'][p]:.2f}" for p in POS) + "  modal " + " ".join(f"{r['modal_letter']}{r['modal_share']:.2f}" for r in per), flush=True)
        out[model] = o
    p = f"{R}/w7u/ana_letterpos_u.json"; json.dump(out, open(p, "w"), indent=1); print("self-checks passed; saved ->", p)


if __name__ == "__main__":
    main()
