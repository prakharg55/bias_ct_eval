#!/usr/bin/env python3
"""
ana_extra_u (8 Oct 2026): two EXPLORATORY appendix analyses of the gen battery, on the UNMERGED primary checkpoints (utags routing;
gemma ACT / AttCT at their re-selected LRs). CPU, seconds. Inputs: results/w7u/gen_<inst_tag>.json; untrained = results/w7/gen_<model>_none.json
(no adapter, merge-independent; the untrained file every unmerged table uses).
 1. Wording generalisation: caving (ana_w3.ana_gen, each model's own known-uncued items) averaged over the 3 TRAINING-family and
    the 3 NOVEL wordings at each position; relative reduction = 1 - seed-mean arm caving / untrained caving, per family and position.
 2. Stronger hints (= ana_stronger7.py on these inputs): over the 24 wording x position cells, strength = untrained caving;
    Pearson r(strength, relative reduction), OLS slope of relative reduction on strength with position fixed effects, and the
    strongest cell ("new_phd" at "end").
Output: results/w7u/ana_extra_u.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/data/code")
import ana_w3, utags
R = "/data/ct_root/results"; SEEDS = (42, 43, 44); POS = ("front", "pre", "post", "end")
ARMS = {"gemma2b": ("bct", "act", "attct", "rmct", "rmcta", "fix_posdiv"), "llama8b": ("bct", "act", "attct", "rmct", "fix_posdiv")}


def per_checkpoint(model):
    files = {"none": f"{R}/w7/gen_{model}_none.json"} | {f"{a}_s{s}": f"{R}/w7u/gen_{utags.inst_tag(model, a, s)}.json" for a in ARMS[model] for s in SEEDS}
    view = f"{R}/w7u/extraview_{model}"; os.makedirs(view, exist_ok=True)
    for k, src in files.items():
        assert os.path.exists(src), src; dst = f"{view}/gen_ev_{k}.json"
        if os.path.islink(dst): assert os.readlink(dst) == src, dst
        else: os.symlink(src, dst)
    for f in os.listdir(view): assert f in {f"gen_ev_{k}.json" for k in files}, f"stale {f}"
    ana_w3.R = view; per = ana_w3.ana_gen("ev")["per_checkpoint"]
    return {k[len("ev_"):]: v for k, v in per.items()}, files


def main():
    out = {}
    for model in ARMS:
        per, files = per_checkpoint(model); base = per["none"]; o = {"inputs": files, "untrained": {}, "arms": {}}
        for fam in ("train", "novel"):
            for p in POS: o["untrained"][f"{fam}|{p}"] = base[f"{fam}|{p}"]["cave"]
        cells = sorted(base["cells"]); assert len(cells) == 24, len(cells); strength = np.array([base["cells"][c]["cave"] for c in cells])
        posd = np.array([[c.split("|")[1] == p for p in POS] for c in cells], float); phd = cells.index("new_phd|end")
        for a in ARMS[model]:
            fam_cave = {f"{fam}|{p}": float(np.mean([per[f"{a}_s{s}"][f"{fam}|{p}"]["cave"] for s in SEEDS])) for fam in ("train", "novel") for p in POS}
            relred = {k: 1 - v / o["untrained"][k] for k, v in fam_cave.items()}
            cav = np.mean([[per[f"{a}_s{s}"]["cells"][c]["cave"] for c in cells] for s in SEEDS], 0); rr = 1 - cav / strength
            coef = np.linalg.lstsq(np.column_stack([strength, posd]), rr, rcond=None)[0]
            o["arms"][a] = {"family_cave": fam_cave, "family_relred": relred, "stronger": {"r": float(np.corrcoef(strength, rr)[0, 1]), "slope_pos_fe": float(coef[0]),
                            "phd_end_cave": float(cav[phd]), "phd_end_untrained": float(strength[phd])}}
            print(f"{model} {a:10s} relred train/novel  " + "  ".join(f"{p} {relred['train|' + p]:+.2f}/{relred['novel|' + p]:+.2f}" for p in POS)
                  + f" | r {o['arms'][a]['stronger']['r']:+.2f} slope {coef[0]:+.2f} phd-end {cav[phd]:.2f} (untrained {strength[phd]:.2f})", flush=True)
        out[model] = o
    json.dump(out, open(f"{R}/w7u/ana_extra_u.json", "w"), indent=1); print(f"-> {R}/w7u/ana_extra_u.json")


if __name__ == "__main__":
    main()
