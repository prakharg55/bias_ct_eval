"""Two exploratory cue-battery analyses: training-family against novel wordings, and cue strength.

  1. The share of harmful-cue following removed for the three training-family and the three novel
     wordings, at each position.
  2. Over the 24 combinations of wording and position, how this share relates to the original
     model's following (the cue's strength).

Usage: python -m ctcue.analysis.wording_and_strength
Output: $CT_ROOT/results/w7u/ana_extra_u.json
"""

import json, os
import numpy as np

from ctcue.analysis import measures
from ctcue.common import checkpoints
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
SEEDS = (42, 43, 44)
POS = ("front", "pre", "post", "end")
ARMS = {
    "gemma2b": ("bct", "act", "attct", "rmct", "rmcta", "fix_posdiv"),
    "llama8b": ("bct", "act", "attct", "rmct", "fix_posdiv"),
}


def per_checkpoint(model):
    files = {"none": f"{R}/w7/gen_{model}_none.json"} | {
        f"{a}_s{s}": f"{R}/w7u/gen_{checkpoints.inst_tag(model, a, s)}.json"
        for a in ARMS[model]
        for s in SEEDS
    }
    view = f"{R}/w7u/extraview_{model}"
    os.makedirs(view, exist_ok=True)
    for k, src in files.items():
        assert os.path.exists(src), src
        dst = f"{view}/gen_ev_{k}.json"
        if os.path.islink(dst):
            assert os.readlink(dst) == src, dst
        else:
            os.symlink(src, dst)
    for f in os.listdir(view):
        assert f in {f"gen_ev_{k}.json" for k in files}, f"stale {f}"
    measures.R = view
    per = measures.ana_gen("ev")["per_checkpoint"]
    return {k[len("ev_") :]: v for k, v in per.items()}, files


def main():
    out = {}
    for model in ARMS:
        per, files = per_checkpoint(model)
        base = per["none"]
        o = {"inputs": files, "untrained": {}, "arms": {}}
        for fam in ("train", "novel"):
            for p in POS:
                o["untrained"][f"{fam}|{p}"] = base[f"{fam}|{p}"]["cave"]
        cells = sorted(base["cells"])
        assert len(cells) == 24, len(cells)
        strength = np.array([base["cells"][c]["cave"] for c in cells])
        posd = np.array([[c.split("|")[1] == p for p in POS] for c in cells], float)
        phd = cells.index("new_phd|end")
        for a in ARMS[model]:
            fam_cave = {
                f"{fam}|{p}": float(np.mean([per[f"{a}_s{s}"][f"{fam}|{p}"]["cave"] for s in SEEDS]))
                for fam in ("train", "novel")
                for p in POS
            }
            relred = {k: 1 - v / o["untrained"][k] for k, v in fam_cave.items()}
            cav = np.mean([[per[f"{a}_s{s}"]["cells"][c]["cave"] for c in cells] for s in SEEDS], 0)
            rr = 1 - cav / strength
            coef = np.linalg.lstsq(np.column_stack([strength, posd]), rr, rcond=None)[0]
            o["arms"][a] = {
                "family_cave": fam_cave,
                "family_relred": relred,
                "stronger": {
                    "r": float(np.corrcoef(strength, rr)[0, 1]),
                    "slope_pos_fe": float(coef[0]),
                    "phd_end_cave": float(cav[phd]),
                    "phd_end_untrained": float(strength[phd]),
                },
            }
            print(
                f"{model} {a:10s} relred train/novel  "
                + "  ".join(f"{p} {relred['train|' + p]:+.2f}/{relred['novel|' + p]:+.2f}" for p in POS)
                + f" | r {o['arms'][a]['stronger']['r']:+.2f} slope {coef[0]:+.2f} phd-end {cav[phd]:.2f} (untrained {strength[phd]:.2f})",
                flush=True,
            )
        out[model] = o
    json.dump(out, open(f"{R}/w7u/ana_extra_u.json", "w"), indent=1)
    print(f"-> {R}/w7u/ana_extra_u.json")


if __name__ == "__main__":
    main()
