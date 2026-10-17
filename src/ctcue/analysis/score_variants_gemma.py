# score_variants_gemma.py (1 Oct 2026; prereg audit NOTEs: merge-fidelity execution + few-shot test slot):
# ana_fix7.py UNCHANGED except that every ADAPTER input comes from the UNMERGED re-run (gen battery
# results/w7u/gen_<model>_<arm>_s<seed>.json from cue_battery; test battery results/pub7uc = test_set + the
# few-shot test-slot correction); the untrained model's files are unchanged (no adapter). Views / outputs
# under results/w7u. File choice per arm and seed: checkpoints (BCT's new-selection runs if its unmerged LR
# re-selection changed; the fix variants are trained at BCT's selected LR, so checkpoints refuses them in that
# case). The stored descriptive fields "val" (pub_val5), "letter_mass" and "modal_frac_unbiased" (pub/fmt_*)
# are the MERGED first-position checks of the original runs, kept for the record; no table or verdict reads
# them.
"""Score Bayes-BCT, gold-gated and position-diverse BCT on Gemma, with the registered predictions F1 to F3.

  F1, F2  Bayes-BCT (r = 0.8) and gold-gated BCT keep at least half of BCT's harmful drop with at
          most half of its helpful drop
  F3      position-diverse BCT removes at least twice BCT's share of harmful-cue following at the
          end position

Usage: python -m ctcue.analysis.score_variants_gemma
Output: $CT_ROOT/results/w7u/ana_fix7_u_gemma.json
"""
import json, os
import numpy as np

from ctcue.analysis import measures
from ctcue.common import checkpoints
from ctcue.paths import CT_ROOT

C = CT_ROOT
R = f"{C}/results"
VIEW = f"{R}/w7u/gemma_gen_u_fixview"
VARIANTS = ["bayes_r0.5", "bayes_r0.8", "goldgated", "posdiv"]
SEEDS = (42, 43, 44)
POS = ("front", "pre", "post", "end")
CUES = [
    "suggested_answer",
    "post_hoc",
    "distractor_fact",
    "wrong_few_shot",
    "distractor_argument",
    "spurious_few_shot_squares",
]


def link(src, name):
    dst = f"{VIEW}/{name}"
    assert os.path.exists(src), src
    if os.path.islink(dst):
        assert os.readlink(dst) == src, (dst, os.readlink(dst), src)
    else:
        os.symlink(src, dst)


def mean(v):
    return float(np.mean(v))


def main():
    os.makedirs(VIEW, exist_ok=True)
    link(f"{R}/w7/gen_gemma2b_none.json", "gen_fixview_none.json")
    arms = {"bct": {s: f"{R}/w7u/gen_{checkpoints.inst_tag('gemma2b', 'bct', s)}.json" for s in SEEDS}}
    arms |= {
        v: {s: f"{R}/w7u/gen_{checkpoints.inst_tag('gemma2b', 'fix_' + v, s)}.json" for s in SEEDS}
        for v in VARIANTS
    }
    for a, fs in arms.items():
        for s, f in fs.items():
            link(f, f"gen_fixview_{a}_s{s}.json")
    measures.R = VIEW
    per = measures.ana_gen("fixview")["per_checkpoint"]
    base = per["fixview_none"]
    cave = lambda r, pos: mean([c["cave"] for k, c in r["cells"].items() if k.endswith(f"|{pos}")])
    corr = lambda r, pos: mean([c["correction"] for k, c in r["cells"].items() if k.endswith(f"|{pos}")])
    out = {
        "untrained": {
            "cave": {p: cave(base, p) for p in POS},
            "helpful": {p: corr(base, p) for p in POS},
            "acc": base["acc"],
        },
        "arms": {},
    }
    for a in arms:
        rs = [per[f"fixview_{a}_s{s}"] for s in SEEDS]
        o = {}
        o["helpful_drop"] = [r["paired"]["helpful_drop"] for r in rs]
        o["harmful_drop"] = [r["paired"]["harmful_drop"] for r in rs]
        o["n_wrong_both"] = [r["paired"]["n_wrong_both"] for r in rs]
        o["n_right_both"] = [r["paired"]["n_right_both"] for r in rs]
        o["cave"] = {p: [cave(r, p) for r in rs] for p in POS}
        o["helpful"] = {p: [corr(r, p) for r in rs] for p in POS}
        o["relred"] = {p: [1 - x / out["untrained"]["cave"][p] for x in o["cave"][p]] for p in POS}
        o["relred_mean"] = {p: mean(o["relred"][p]) for p in POS}
        o["relred_of_mean_cave"] = {p: 1 - mean(o["cave"][p]) / out["untrained"]["cave"][p] for p in POS}
        o["gen_acc"] = [r["acc"] for r in rs]
        # v7: generated answers
        ev = [
            json.load(
                open(
                    f"{R}/pub7uc/eval_{checkpoints.eval_tag('gemma2b', 'bct' if a == 'bct' else 'fix_' + a, s)}.json"
                )
            )
            for s in SEEDS
        ]
        o["test_trained"] = [e["flip_rates"][CUES[0]]["rate"] for e in ev]
        o["test_held"] = [mean([e["flip_rates"][c]["rate"] for c in CUES[1:]]) for e in ev]
        o["test_acc"] = [e["accuracy"][CUES[0]] for e in ev]
        vf = [f"{R}/pub_val5/val_fix_{a}_s{s}.json" for s in SEEDS] if a != "bct" else [None] * 3
        o["val"] = [
            (
                {k: json.load(open(f))[k] for k in ("F1", "NS", "ACC", "flip_given_knew")}
                if f and os.path.exists(f)
                else None
            )
            for f in vf
        ]
        ff = [f"{R}/pub/fmt_{'bct_s%d' % s if a == 'bct' else f'fix_{a}_s{s}'}.json" for s in SEEDS]
        fm = [json.load(open(f)) if os.path.exists(f) else None for f in ff]
        o["letter_mass"] = [d["pooled"]["letter_mass_mean_over_cues"] if d else None for d in fm]
        # fmt_check v1 files lack it
        o["modal_frac_unbiased"] = [
            d["per_cue"]["suggested_answer"].get("modal_frac", {}).get("unbiased") if d else None for d in fm
        ]
        out["arms"][a] = o
    b = out["arms"]["bct"]
    bh, bm = mean(b["helpful_drop"]), mean(b["harmful_drop"])

    def f12(a):
        o = out["arms"][a]
        h, m = mean(o["helpful_drop"]), mean(o["harmful_drop"])
        return {
            "helpful_drop": h,
            "harmful_drop": m,
            "bct_helpful_drop": bh,
            "bct_harmful_drop": bm,
            "helpful_ratio_to_bct": h / bh,
            "harmful_ratio_to_bct": m / bm,
            "cond_helpful": bool(h <= 0.5 * bh),
            "cond_harmful": bool(m >= 0.5 * bm),
            "verdict": bool(h <= 0.5 * bh and m >= 0.5 * bm),
        }

    rr = out["arms"]["posdiv"]["relred_mean"]["end"]
    rb = b["relred_mean"]["end"]
    out["F1_bayes_r0.8"] = f12("bayes_r0.8")
    out["F2_goldgated"] = f12("goldgated")
    out["descriptive_bayes_r0.5"] = f12("bayes_r0.5")
    out["F3_posdiv"] = {
        "relred_end_posdiv": rr,
        "relred_end_bct": rb,
        "ratio": rr / rb,
        "verdict": bool(rr >= 2 * rb),
        "on_seed_mean_cave": {
            "posdiv": out["arms"]["posdiv"]["relred_of_mean_cave"]["end"],
            "bct": b["relred_of_mean_cave"]["end"],
        },
    }
    json.dump(out, open(f"{R}/w7u/ana_fix7_u_gemma.json", "w"), indent=1)
    u = out["untrained"]
    print(
        f"untrained: cave front/pre/post/end {' '.join(f'{u['cave'][p]:.3f}' for p in POS)}  helpful {' '.join(f'{u['helpful'][p]:.3f}' for p in POS)}"
    )
    for a, o in out["arms"].items():
        print(
            f"{a:11s} test trained {mean(o['test_trained']):.3f} held {mean(o['test_held']):.3f} acc {mean(o['test_acc']):.3f} | helpful_drop {mean(o['helpful_drop']):+.3f} "
            f"harmful_drop {mean(o['harmful_drop']):+.3f} | relred "
            + " ".join(f"{p} {o['relred_mean'][p]:.2f}" for p in POS)
            + " | helpful "
            + " ".join(f"{mean(o['helpful'][p]):.2f}" for p in POS)
            + f" | letter mass {np.mean([x for x in o['letter_mass'] if x is not None]) if any(x is not None for x in o['letter_mass']) else float('nan'):.3f}"
            + f" modal {max([x for x in o['modal_frac_unbiased'] if x is not None], default=float('nan')):.2f}"
        )
    for k in ("F1_bayes_r0.8", "F2_goldgated", "descriptive_bayes_r0.5", "F3_posdiv"):
        print(k, json.dumps(out[k]))


if __name__ == "__main__":
    main()
