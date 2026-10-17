"""Report the letter-prior control, patch recovery and layer swap with and without the two few-shot cues.

It re-aggregates the stored per-cue and per-item results and refits nothing. For the letter-prior
control, the held-out mean is taken over three held-out cues instead of five, and the letter offsets
stay as fitted on all cues.

Usage: python -m ctcue.analysis.fewshot_sensitivity
Output: $CT_ROOT/results/w7u/ana_fs_sens.json
"""

import json, os
import numpy as np

from ctcue.analysis import measures
from ctcue.common import checkpoints
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
SEEDS = (42, 43, 44)
CUES = [
    "suggested_answer",
    "post_hoc",
    "distractor_fact",
    "wrong_few_shot",
    "distractor_argument",
    "spurious_few_shot_squares",
]
FS = ("wrong_few_shot", "spurious_few_shot_squares")
HELD5 = CUES[1:]
HELD3 = [c for c in HELD5 if c not in FS]
NONFS = [c for c in CUES if c not in FS]
SCOPES = ("EARLY", "CARRIER", "CIRCUIT", "LATE")
msd = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


def lp_section(view, tags_by_arm, gmax_view=None):
    st = json.load(open(f"{view}/ana_letterprior2.json"))
    gm = (
        json.load(open(f"{gmax_view}/ana_letterprior2.json"))
        if gmax_view and os.path.exists(f"{gmax_view}/ana_letterprior2.json")
        else {}
    )
    held = lambda sc, cues: float(np.mean([sc["per_cue"][c]["flip_given_knew"] for c in cues]))
    out = {
        "untrained": {"held5": held(st["untrained"], HELD5), "held3": held(st["untrained"], HELD3)},
        "arms": {},
    }
    assert abs(out["untrained"]["held5"] - st["untrained"]["held"]) < 1e-12
    for arm, tags in tags_by_arm.items():
        rows = []
        for t in tags:
            r = gm[t] if t in gm else st[t]
            a5, a3, c5, c3 = (
                held(r["arm"], HELD5),
                held(r["arm"], HELD3),
                held(r["letter_prior_control"], HELD5),
                held(r["letter_prior_control"], HELD3),
            )
            assert (
                abs(a5 - r["arm"]["held"]) < 1e-12 and abs(c5 - r["letter_prior_control"]["held"]) < 1e-12
            ), t
            rows.append(
                {
                    "tag": t,
                    "v71_readout": t in gm,
                    "arm_held5": a5,
                    "arm_held3": a3,
                    "control_held5": c5,
                    "control_held3": c3,
                    "bpd_held5": c5 - a5,
                    "bpd_held3": c3 - a3,
                }
            )
        out["arms"][arm] = {
            "per_seed": rows,
            **{
                k: msd([r[k] for r in rows])
                for k in (
                    "arm_held5",
                    "arm_held3",
                    "control_held5",
                    "control_held3",
                    "bpd_held5",
                    "bpd_held3",
                )
            },
        }
    return out


def causal_section(files):
    out = {}
    for name, f in files.items():
        d = json.load(open(f))
        P = d["patch"]

        def rec(s, cues):
            k = sum(P[s]["per_cue"].get(c, {}).get("recovered", 0) for c in cues)
            n = sum(P[s]["per_cue"].get(c, {}).get("n", 0) for c in cues)
            return k, n

        k, n = rec("circuit", CUES)
        assert (k, n) == (P["circuit"]["recovered"], P["circuit"]["n_flips"]), (name, k, n)
        rs = [s for s in P if s.startswith("random")]
        o = {}
        for lab, cues in (("all", CUES), ("no_fewshot", NONFS)):
            k, n = rec("circuit", cues)
            br = max(rs, key=lambda s: (rec(s, cues)[0] / max(rec(s, cues)[1], 1)))
            o[lab] = {
                "recovered": k,
                "n_flips": n,
                "circuit": k / n if n else float("nan"),
                "wilson95": list(measures.wilson(k, n)) if n else None,
                "best_random": rec(br, cues)[0] / max(rec(br, cues)[1], 1),
                "best_random_set": br,
            }
        out[name] = o
    return out


def med_fwd(sub, arms):
    """measures.ana_route summ()'s med_fwd, verbatim arithmetic."""
    if not sub:
        return None
    c1 = np.mean([it["pred1b"] == it["X"] for it in sub])
    fixed = 1 - c1
    n_fixed = sum(it["pred1b"] != it["X"] for it in sub)
    ok = n_fixed >= 20
    return {
        "n": len(sub),
        "n_fixed": int(n_fixed),
        "fracs": {
            a.split("@")[1]: (
                float((1 - np.mean([it["arms"][a]["fwd"][2] == it["X"] for it in sub])) / fixed)
                if ok
                else float("nan")
            )
            for a in arms
        },
    }


def d1(fr):
    if any(fr[s] != fr[s] for s in SCOPES):
        return None
    best = max(SCOPES, key=lambda s: fr[s])
    return {"largest_scope": best, "holds": bool(best in ("EARLY", "CARRIER") and fr["LATE"] < 0.15)}


def route_section(files, prefix_view):
    out = {}
    arms = ["W_all@ALL"] + [f"W_all@{s}" for s in SCOPES]
    for name, f in files.items():
        d = json.load(open(f))
        its = d["items"]
        allr, nofs = med_fwd(its, arms), med_fwd([it for it in its if it["cue"] not in FS], arms)
        out[name] = {
            "all": {**allr, "D1_style": d1(allr["fracs"])},
            "no_fewshot": {**nofs, "D1_style": d1(nofs["fracs"])},
        }
    if prefix_view:  # regression of the copied formula against ana_route on the same files
        view, prefix, names = prefix_view
        measures.R = view
        ar = measures.ana_route(prefix)["per_checkpoint"]
        for name, tag in names.items():
            for a in arms:
                x, y = ar[tag]["pooled"]["arms"][a]["med_fwd"], out[name]["all"]["fracs"][a.split("@")[1]]
                assert (x != x and y != y) or abs(x - y) < 1e-12, (name, a, x, y)
    return out


def main():
    res = {"definition": __doc__.strip().split("\n")[0], "cues_removed": list(FS)}
    W = f"{R}/w7u"
    # letter-prior
    res["letterprior_gemma"] = lp_section(
        f"{W}/lpview_gemma_u",
        {m: [f"pub_{m}_s{s}_step1" for s in SEEDS] for m in ("bct", "act", "attct", "rmct", "rmcta")},
        f"{W}/lpview_gemma_max_u",
    )
    res["letterprior_gemma_fix"] = lp_section(
        f"{W}/lpview_gemma_fix_u",
        {"bct": [f"pub_bct_s{s}_step1" for s in SEEDS]}
        | {v: [f"fix_{v}_s{s}" for s in SEEDS] for v in ("bayes_r0.5", "bayes_r0.8", "goldgated", "posdiv")},
        f"{W}/lpview_gemma_max_u",
    )
    res["letterprior_llama"] = lp_section(
        f"{W}/lpview_llama_u",
        {m: [f"llama8b_{m}_s{s}" for s in SEEDS] for m in ("bct", "act", "attct", "rmct")},
    )
    # patch recovery (gate-failing gemma pre-registered arms: their v7.1-readout files, as the main table,
    # when present)
    gates = json.load(open(f"{W}/gates_u.json"))
    gx = set(gates["gmax_needed"]["causal"])
    cf = {
        "gemma2b_none": f"{R}/w3/causal3_pub_none.json",
        "llama8b_none": f"{R}/w7/causal3_llama8b_none.json",
    }
    for m in ("bct", "act", "attct", "rmct"):
        for s in SEEDS:
            it = checkpoints.inst_tag("gemma2b", m, s)
            et = checkpoints.eval_tag("gemma2b", m, s)
            cf[it] = (
                f"{W}/causal3_gemma2b_max_{it[len('gemma2b_'):]}.json"
                if et in gx
                else f"{W}/causal3_{it}.json"
            )
        it = checkpoints.inst_tag("llama8b", m, 42)
        cf[it] = f"{W}/causal3_{it}.json"
    if gx:
        cf["gemma2b_max_none"] = f"{R}/w7/causal3_gemma2b_max_none.json"
    res["patch_recovery"] = causal_section({k: v for k, v in cf.items() if os.path.exists(v)})
    res["patch_recovery_missing"] = [k for k, v in cf.items() if not os.path.exists(v)]
    # route
    rf = {
        checkpoints.inst_tag("llama8b", m, 42): f"{W}/route_{checkpoints.inst_tag('llama8b', m, 42)}.json"
        for m in ("bct", "act", "attct")
    }
    rf |= {
        checkpoints.inst_tag("gemma2b", m, s): f"{W}/route_{checkpoints.inst_tag('gemma2b', m, s)}.json"
        for m in ("bct", "act", "attct")
        for s in SEEDS
    }
    have = {k: v for k, v in rf.items() if os.path.exists(v)}
    lv = {
        k: f"llview_{k.split('_')[1].replace('7u', '')}_s42_step1" for k in have if k.startswith("llama8b_")
    }
    res["route"] = route_section(
        have,
        (f"{W}/routeview_llama_u", "llview", lv) if lv and os.path.isdir(f"{W}/routeview_llama_u") else None,
    )
    res["route_missing"] = [k for k in rf if k not in have]
    json.dump(res, open(f"{W}/ana_fs_sens.json", "w"), indent=1)
    for key in ("letterprior_gemma", "letterprior_gemma_fix", "letterprior_llama"):
        o = res[key]
        print(
            f"[{key}] untrained held-out 5 cues {o['untrained']['held5']:.3f} | 3 cues {o['untrained']['held3']:.3f}"
        )
        for a, v in o["arms"].items():
            print(
                f"   {a:11s} beyond-prior drop held-out: 5 cues {v['bpd_held5'][0]:+.3f} ± {v['bpd_held5'][1]:.3f} | 3 cues {v['bpd_held3'][0]:+.3f} ± {v['bpd_held3'][1]:.3f}"
            )
    for k, v in res["patch_recovery"].items():
        print(
            f"[patch] {k:28s} all {v['all']['circuit']:.3f} ({v['all']['recovered']}/{v['all']['n_flips']}) | no few-shot {v['no_fewshot']['circuit']:.3f} ({v['no_fewshot']['recovered']}/{v['no_fewshot']['n_flips']})"
        )
    for k, v in res["route"].items():
        print(
            f"[route] {k:22s} all "
            + " ".join(f"{s} {v['all']['fracs'][s]:.2f}" for s in SCOPES)
            + f" D1 {v['all']['D1_style']} | no few-shot (n {v['no_fewshot']['n']}, fixed {v['no_fewshot']['n_fixed']}) "
            + " ".join(f"{s} {v['no_fewshot']['fracs'][s]:.2f}" for s in SCOPES)
            + f" D1 {v['no_fewshot']['D1_style']}"
        )
    print("missing:", res["patch_recovery_missing"], res["route_missing"])
    print("saved ->", f"{W}/ana_fs_sens.json")


if __name__ == "__main__":
    main()
