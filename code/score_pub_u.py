#!/usr/bin/env python3
"""
score_pub_u (1 Oct 2026; prereg audit NOTEs: merge-fidelity execution + few-shot test slot). gemma P1-P5 with score_pub.py UNCHANGED,
every ADAPTER input taken from the UNMERGED re-run through symlink views:
  eval  (P1, P2, test rates)  results/pub7uc (ct_eval7u + few-shot test-slot correction; untrained file unchanged)  -> view pub7uc/gemma_eval7uc_view
  gen   (P5)                  results/w7u/gen_gemma2b_<arm>_s<seed>.json (ct_gen7u; untrained w7/gen_gemma2b_none)  -> view w7u/gemma_gen_u_pubview
  probe (P3)                  results/w7u/probe_gemma2b_<arm>_s<seed> (ct_probe7u --agg sum; untrained w3/probe_pub_none) -> view w7u/gemma_probe_u_view
  weights (P4)                computed from the adapters themselves (no merge involved): w3/weights_pub*.json; for an arm whose unmerged LR
                              re-selection CHANGED, its entries come from ct_weights6 on the new-selection adapters (w7u/weights_pub_u.json /
                              weights_pub_rmcta_u.json, built by weights_u.py), linked into the probe view under the names score_pub reads.
File choice per arm and seed: utags (new-selection runs where the re-selection changed an arm). If no arm changed, the P4 fields must equal
score_pub.json exactly (asserted); otherwise the unchanged arms' centroids must. Output results/w7u/score_pub_u_gemma.json. CPU, seconds.
"""
import json, os, sys
sys.path.insert(0, "/data/code")
import ana_w3, ana_probe3, score_pub, utags
R = "/data/ct_root/results"; OUT = f"{R}/w7u/score_pub_u_gemma.json"
EVIEW, GVIEW, PVIEW = f"{R}/pub7uc/gemma_eval7uc_view", f"{R}/w7u/gemma_gen_u_pubview", f"{R}/w7u/gemma_probe_u_view"
ARMS = ("bct", "act", "attct", "rmct", "rmcta"); SEEDS = (42, 43, 44)


def build(view, links):
    os.makedirs(view, exist_ok=True)
    for name, src in links.items():
        assert os.path.exists(src), src; dst = f"{view}/{name}"
        if os.path.islink(dst): assert os.readlink(dst) == src, (dst, os.readlink(dst))
        else: os.symlink(src, dst)
    for f in os.listdir(view): assert f in links, f"stale file in view: {view}/{f}"


def diffs(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k}".lstrip(".") for k in set(a) ^ set(b)]
        for k in set(a) & set(b): out += diffs(a[k], b[k], f"{path}.{k}".lstrip("."))
        return out
    return [] if a == b else [path]


def main():
    ev = {"eval_pub_none.json": f"{R}/pub7uc/eval_gemma2b_final_none.json", "eval_final_none.json": f"{R}/pub7uc/eval_gemma2b_final_none.json"}
    ev |= {f"eval_final_{m}_s{s}.json": f"{R}/pub7uc/eval_{utags.eval_tag('gemma2b', m, s)}.json" for m in ARMS for s in SEEDS}
    gv = {"gen_pub_none.json": f"{R}/w7/gen_gemma2b_none.json"} | {f"gen_pub_{m}_s{s}_step1.json": f"{R}/w7u/gen_{utags.inst_tag('gemma2b', m, s)}.json" for m in ARMS for s in SEEDS}
    pv = {f"probe_pub_none.{x}": f"{R}/w3/probe_pub_none.{x}" for x in ("json", "npz")}
    pv |= {f"probe_pub_{m}_s{s}_step1.{x}": f"{R}/w7u/probe_{utags.inst_tag('gemma2b', m, s)}.{x}" for m in ARMS for s in SEEDS for x in ("json", "npz")}
    chg = [m for m in ARMS if utags.changed("gemma2b", m)]
    wp = f"{R}/w7u/weights_pub_u.json" if any(m in chg for m in ("bct", "act", "attct", "rmct")) else f"{R}/w3/weights_pub.json"
    wa = f"{R}/w7u/weights_pub_rmcta_u.json" if "rmcta" in chg else f"{R}/w3/weights_pub_rmcta.json"
    pv |= {"weights_pub.json": wp, "weights_pub_rmcta.json": wa, "gen_pub_none.json": f"{R}/w7/gen_gemma2b_none.json"}
    build(EVIEW, ev); build(GVIEW, gv); build(PVIEW, pv)
    orig = score_pub.ana_gen
    def ana_gen_u(prefix):
        ana_w3.R = GVIEW; return orig(prefix)
    score_pub.ana_gen = ana_gen_u; score_pub.RP = EVIEW; score_pub.RW = PVIEW; ana_probe3.R = PVIEW
    os.makedirs(f"{R}/w7u", exist_ok=True)
    sys.argv = ["score_pub.py", "--out", OUT, "--weights", f"{PVIEW}/weights_pub.json"]; score_pub.main()
    new, old = json.load(open(OUT)), json.load(open(f"{R}/pub/score_pub.json"))
    if not chg:
        d = diffs(new.get("weights"), old.get("weights")); assert not d, f"P4 weight fields differ from score_pub.json: {d}"
    else:   # unchanged arms: identical centroids; changed arms: recomputed from the new-selection adapters
        for m in ("bct", "act", "attct", "rmct"):
            if m not in chg:
                for k in ("ov_centroid", "qk_centroid"): assert new["weights"][k][m] == old["weights"][k][m], (m, k)
        print("P4: recomputed for the re-selected arms", chg)
    print("P1", new["verdicts"]["P1"], "P2", new["verdicts"]["P2"], "P3", new["verdicts"]["P3"], "P4", new["verdicts"]["P4"], "P5", new["verdicts"]["P5"])


if __name__ == "__main__":
    main()
