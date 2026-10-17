"""Score the layer swap on the final runs, including the registered Llama prediction D1.

D1 (Llama, seed 42): for BCT, ACT and AttCT, the block with the largest share of reproduced fixes
(pooled over the six cues) lies before the circuit (layers 0-8 or 9-14), and the block after it
(layers 19-31) stays below 0.15. D1 is undefined if the trained model fixes fewer than 20 items.
Gemma is scored without a prediction. The merged values, from score_layer_swap_merged.py and
$CT_ROOT/results/w3/ana_route_pub.json, are reported next to the unmerged ones.

Usage: python -m ctcue.analysis.score_layer_swap
Output: $CT_ROOT/results/w7u/score_route_u.json
"""

import json, os

from ctcue.analysis import measures
from ctcue.common import checkpoints
from ctcue.analysis.score_layer_swap_merged import fracs, d1, ARMS, SCOPES, LLAMA_SCOPES, LLAMA_R_LAYERS
from ctcue.paths import CT_ROOT

C = CT_ROOT
R = f"{C}/results"
SEEDS = (42, 43, 44)
GEMMA_SCOPES = {
    "ALL": list(range(26)),
    "EARLY": list(range(0, 7)),
    "CARRIER": list(range(7, 13)),
    "CIRCUIT": list(range(13, 19)),
    "LATE": list(range(19, 26)),
}
GEMMA_R_LAYERS = [8, 12, 16, 18, 20, 22]


def build(view, links):
    os.makedirs(view, exist_ok=True)
    for name, src in links.items():
        dst = f"{view}/{name}"
        if os.path.islink(dst):
            assert os.readlink(dst) == src, (dst, os.readlink(dst), src)
        else:
            os.symlink(src, dst)
    # ana_route globs the whole directory
    for f in os.listdir(view):
        assert f in links, f"stale file in the view: {f}"


def check_file(d, model, agg):
    sc, rl = (LLAMA_SCOPES, LLAMA_R_LAYERS) if model == "llama8b" else (GEMMA_SCOPES, GEMMA_R_LAYERS)
    assert (
        d["model"] == model
        and d["scopes"] == sc
        and d["r_layers"] == rl
        and d["rows"] == "1200:1300"
        and d["max_per_cell"] == 10
        and d["n_checked"] >= 1
    ), (d["tag"], d["rows"], d["n_checked"])


def main():
    out = {
        "definition": "med_fwd of W_all@scope, pooled over S; inclusion: >= 20 items fixed by M1; M1 UNMERGED (ct_route7u)",
        "llama": {"arms": {}},
        "gemma": {"arms": {}},
    }
    # ---- Llama (D1) ----
    lf = {
        m: f"{R}/w7u/route_{checkpoints.inst_tag('llama8b', m, 42)}.json" for m in ARMS
    }  # file choice: checkpoints
    want = {f"route_llview_{m}_s42_step1.json": lf[m] for m in ARMS if os.path.exists(lf[m])}
    build(f"{R}/w7u/routeview_llama_u", want)
    merged = json.load(open(f"{R}/w7/score_route_llama7.json"))
    res = {}
    if want:
        measures.R = f"{R}/w7u/routeview_llama_u"
        res = measures.ana_route("llview")["per_checkpoint"]
    for m in ARMS:
        tag = f"llview_{m}_s42_step1"
        if f"route_{tag}.json" not in want:
            out["llama"]["arms"][m] = None
            print(f"llama {m}: no unmerged route file yet")
            continue
        d = json.load(open(want[f"route_{tag}.json"]))
        check_file(d, "llama8b", "max")
        r = res[tag]
        p = r["pooled"]
        fr = fracs(p)
        v = d1(fr)
        out["llama"]["arms"][m] = {
            "n": p["n"],
            "n_fixed": p["n_fixed"],
            "cave_M1": p["cave_M1"],
            "fracs": fr,
            "D1": v,
            "checks": r["checks"],
            "counts": r["counts"],
            "trained_cue": fracs(r["trained"]) if r["trained"] else None,
            "held_out_cues": fracs(r["held"]) if r["held"] else None,
            "merged": (
                {k: merged["arms"][m][k] for k in ("n", "n_fixed", "cave_M1", "fracs", "D1")}
                if merged["arms"].get(m)
                else None
            ),
        }
        mf = merged["arms"][m]["fracs"] if merged["arms"].get(m) else None
        print(
            f"llama {m:5s} s42: |S| {p['n']} fixed {p['n_fixed']} cave_M1 {p['cave_M1']:.3f} | "
            + "  ".join(
                f"{sc} {fr[sc]:.2f}" + (f" (merged {mf[sc]:.2f})" if mf else "") for sc in ("ALL",) + SCOPES
            )
            + f" | D1 {('HOLDS' if v['holds'] else 'FAILS') + ' (largest ' + v['largest_scope'] + ')' if v else 'undefined (< 20 fixed)'}"
        )
        print(
            "      exactness checks: "
            + "; ".join(" ".join(f"{k} {x:.1e}" for k, x in c.items()) for c in r["checks"].values())
        )
    per = {m: (o["D1"]["holds"] if o and o["D1"] else None) for m, o in out["llama"]["arms"].items()}
    out["llama"]["D1"] = {
        "per_arm": per,
        "verdict": all(per.values()) if all(v is not None for v in per.values()) else None,
    }
    out["llama"]["D1_merged"] = merged["D1"]
    print("D1 (unmerged):", json.dumps(out["llama"]["D1"]), "| merged:", json.dumps(merged["D1"]))
    # ---- gemma (descriptive) ----
    gf = {(m, s): f"{R}/w7u/route_{checkpoints.inst_tag('gemma2b', m, s)}.json" for m in ARMS for s in SEEDS}
    want = {
        f"route_pub_{m}_s{s}_step1.json": gf[(m, s)]
        for m in ARMS
        for s in SEEDS
        if os.path.exists(gf[(m, s)])
    }
    build(f"{R}/w7u/routeview_gemma_u", want)
    gm = json.load(open(f"{R}/w3/ana_route_pub.json"))["per_checkpoint"]
    if want:
        measures.R = f"{R}/w7u/routeview_gemma_u"
        res = measures.ana_route("pub")["per_checkpoint"]
    for m in ARMS:
        rows = []
        for s in SEEDS:
            tag = f"pub_{m}_s{s}_step1"
            if f"route_{tag}.json" not in want:
                rows.append(None)
                continue
            d = json.load(open(want[f"route_{tag}.json"]))
            check_file(d, "gemma2b", "sum")
            p = res[tag]["pooled"]
            fr = fracs(p)
            fm = fracs(gm[tag]["pooled"])
            rows.append(
                {
                    "seed": s,
                    "n": p["n"],
                    "n_fixed": p["n_fixed"],
                    "cave_M1": p["cave_M1"],
                    "fracs": fr,
                    "D1_style": d1(fr),
                    "merged_fracs": fm,
                    "checks": res[tag]["checks"],
                }
            )
            print(
                f"gemma {m:5s} s{s}: |S| {p['n']} fixed {p['n_fixed']} | "
                + "  ".join(f"{sc} {fr[sc]:.2f} (merged {fm[sc]:.2f})" for sc in ("ALL",) + SCOPES)
            )
        out["gemma"]["arms"][m] = rows
    json.dump(out, open(f"{R}/w7u/score_route_u.json", "w"), indent=1)
    print("saved ->", f"{R}/w7u/score_route_u.json")


if __name__ == "__main__":
    main()
