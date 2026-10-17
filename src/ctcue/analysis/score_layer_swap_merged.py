"""score_layer_swap.py on the merged Llama runs (earlier version). score_layer_swap.py reuses its code
and reads its output.

Before scoring, it checks that its extraction reproduces the logged Gemma values in the stored merged
results ($CT_ROOT/results/w3/ana_route_pub.json).

Usage: python -m ctcue.analysis.score_layer_swap_merged
Output: $CT_ROOT/results/w7/score_route_llama7.json
"""

import argparse, json, os

from ctcue.analysis import measures
from ctcue.paths import CT_ROOT

C = CT_ROOT
R = f"{C}/results"
ARMS = ("bct", "act", "attct")
SCOPES = ("EARLY", "CARRIER", "CIRCUIT", "LATE")
LLAMA_SCOPES = {
    "ALL": list(range(32)),
    "EARLY": list(range(0, 9)),
    "CARRIER": list(range(9, 15)),
    "CIRCUIT": list(range(15, 19)),
    "LATE": list(range(19, 32)),
}
LLAMA_R_LAYERS = [10, 14, 16, 18, 20, 22]
# CARRIER, seeds 42/43/44 (the project log erratum)
GEMMA_LOGGED = {"bct": (0.64, 0.80, 0.68), "act": (0.53, 0.57, 0.53), "attct": (0.65, 0.63, 0.66)}


def fracs(pooled):
    return {sc: pooled["arms"][f"W_all@{sc}"]["med_fwd"] for sc in ("ALL",) + SCOPES}


def d1(fr):
    # NaN: fewer than 20 items fixed by M1
    if any(fr[sc] != fr[sc] for sc in SCOPES):
        return None
    best = max(SCOPES, key=lambda sc: fr[sc])
    return {
        "largest_scope": best,
        "largest": fr[best],
        "late": fr["LATE"],
        "holds": bool(best in ("EARLY", "CARRIER") and fr["LATE"] < 0.15),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    g = json.load(open(f"{R}/w3/ana_route_pub.json"))["per_checkpoint"]
    print("gemma regression of the extraction (stored ana_route_pub.json):")
    for m in ARMS:
        got = []
        for s in (42, 43, 44):
            fr = fracs(g[f"pub_{m}_s{s}_step1"]["pooled"])
            got.append(fr["CARRIER"])
            print(
                f"  gemma {m:5s} s{s}: "
                + "  ".join(f"{sc} {fr[sc]:.2f}" for sc in ("ALL",) + SCOPES)
                + f"  -> D1-style: {d1(fr)}"
            )
            assert fr["LATE"] <= 0.035, (m, s, fr["LATE"])
        assert tuple(round(x, 2) for x in got) == GEMMA_LOGGED[m], (m, got, GEMMA_LOGGED[m])
    print("  gemma extraction reproduces the logged CARRIER fractions and LATE <= .03: OK\n")
    view = f"{R}/w7/routeview_llama{'_smoke' if a.smoke else ''}"
    os.makedirs(view, exist_ok=True)
    want = {}
    for m in ARMS:
        src = f"{R}/w7/route_{'smoke7_' if a.smoke else ''}llama8b_{m}_s42.json"
        dst = f"{view}/route_llview_{m}_s42_step1.json"
        if not os.path.exists(src):
            continue
        want[os.path.basename(dst)] = src
        if os.path.islink(dst):
            assert os.readlink(dst) == src, (dst, os.readlink(dst), src)
        else:
            os.symlink(src, dst)
    # ana_route globs the whole directory
    for f in os.listdir(view):
        assert f in want, f"stale file in the view: {f}"
    out = {
        "smoke": a.smoke,
        "arms": {},
        "definition": "med_fwd of W_all@scope, pooled over S; inclusion: >= 20 items fixed by M1",
    }
    if want:
        measures.R = view
        res = measures.ana_route("llview")["per_checkpoint"]
    for m in ARMS:
        tag = f"llview_{m}_s42_step1"
        if f"route_{tag}.json" not in want:
            out["arms"][m] = None
            print(f"llama {m}: no route file yet")
            continue
        d = json.load(open(want[f"route_{tag}.json"]))
        assert (
            d["model"] == "llama8b"
            and d["scopes"] == LLAMA_SCOPES
            and d["r_layers"] == LLAMA_R_LAYERS
            and d["rows"] == ("1200:1212" if a.smoke else "1200:1300")
        ), (m, d["rows"], d["r_layers"])
        assert d["max_per_cell"] == (1 if a.smoke else 10) and d["n_checked"] >= 1, (
            m,
            d["max_per_cell"],
            d["n_checked"],
        )
        r = res[tag]
        p = r["pooled"]
        fr = fracs(p)
        v = d1(fr)
        out["arms"][m] = {
            "n": p["n"],
            "n_fixed": p["n_fixed"],
            "cave_M1": p["cave_M1"],
            "fracs": fr,
            "D1": v,
            "checks": r["checks"],
            "counts": r["counts"],
            "trained_cue": fracs(r["trained"]) if r["trained"] else None,
            "held_out_cues": fracs(r["held"]) if r["held"] else None,
        }
        print(
            f"llama {m:5s} s42: |S| {p['n']}  fixed by M1 {p['n_fixed']}  cave_M1 {p['cave_M1']:.3f} | "
            + "  ".join(f"{sc} {fr[sc]:.2f}" for sc in ("ALL",) + SCOPES)
            + f" | D1 {('HOLDS' if v['holds'] else 'FAILS') + ' (largest ' + v['largest_scope'] + ')' if v else 'undefined (< 20 fixed)'}"
        )
        print(
            f"      exactness checks: "
            + "; ".join(" ".join(f"{k} {x:.1e}" for k, x in c.items()) for c in r["checks"].values())
        )
    per = {m: (o["D1"]["holds"] if o and o["D1"] else None) for m, o in out["arms"].items()}
    out["D1"] = {
        "per_arm": per,
        "verdict": all(per.values()) if all(v is not None for v in per.values()) else None,
    }
    json.dump(out, open(f"{R}/w7/score_route_llama7{'_smoke' if a.smoke else ''}.json", "w"), indent=1)
    print("D1:", json.dumps(out["D1"]))


if __name__ == "__main__":
    main()
