#!/usr/bin/env python3
"""
Validation of the gemma gen-battery re-score (30 Sep 2026; amendment "GEN-BATTERY GATE RESULT"). For each of the 31 checkpoints:
  (a) the re-run file (results/w7/gen_gemma2b_<tag>.json, ct_gen7 --agg max = v7.1 readout) has exactly the old file's items
      (ds, hash, T, X, n_opts, same order); reported: the share of prompts whose answer letter changed vs the old (logsumexp) readout,
      per hint position;
  (b) for every checkpoint that has a gen_gate7 file: the NEW readout vs the gate's greedy generated answer, prompt by prompt on the
      gate subsample, per position (unparsed generation = disagreement); required >= .97 at every position and uncued (amendment).
CPU, seconds. Output results/w7/check_gemma_gen7.json.
"""
import json, os
R = "/data/ct_root/results"; POS = ("front", "pre", "post", "end"); SEEDS = (42, 43, 44)
CK = {"none": f"{R}/w3/gen_pub_none.json"}
CK |= {f"{m}_s{s}": f"{R}/w3/gen_pub_{m}_s{s}_step1.json" for m in ("bct", "act", "attct", "rmct", "rmcta") for s in SEEDS}
CK |= {f"fix_{v}_s{s}": f"{R}/w5/gen_fix_{v}_s{s}.json" for v in ("bayes_r0.5", "bayes_r0.8", "goldgated", "posdiv") for s in SEEDS}
CK |= {f"bctro_s{s}": f"{R}/w5/gen_bctro_s{s}.json" for s in SEEDS}
GATE = {"none": "none", "bct_s42": "bct_s42", "act_s42": "act_s42", "attct_s42": "attct_s42", "rmct_s42": "rmct_s42", "fix_posdiv_s42": "posdiv_s42", "bctro_s42": "bctro_s42"}


def main():
    out = {"checkpoints": {}, "missing": []}; ok = True
    for t, oldf in CK.items():
        newf = f"{R}/w7/gen_gemma2b_{t}.json"
        if not os.path.exists(newf): out["missing"].append(t); continue
        o, n = json.load(open(oldf)), json.load(open(newf))
        key = lambda it: (it["ds"], it["hash"], it["T"], it["X"], it["n_opts"])
        same_items = [key(i) for i in o["items"]] == [key(i) for i in n["items"]] and o["wordings"] == n["wordings"] and o["positions"] == n["positions"]
        assert n["model"] == "gemma2b" and n["rows"] == o["rows"], t
        ch = {p: [0, 0] for p in POS}; ch0 = [0, 0]
        for a, b in zip(o["items"], n["items"]):
            ch0[0] += 1; ch0[1] += a["pred0"] != b["pred0"]
            for k, v in a["cond"].items():
                p = k.split("|")[1]; ch[p][0] += 1; ch[p][1] += v[2] != b["cond"][k][2]
        rec = {"same_items": same_items, "changed_uncued": ch0[1] / ch0[0], "changed": {p: ch[p][1] / ch[p][0] for p in POS}}
        g = GATE.get(t); gf = f"{R}/w7/gengate_gemma2b_{g}.json" if g else None
        if gf and os.path.exists(gf):
            G = json.load(open(gf)); agree = {p: [0, 0] for p in POS + ("uncued",)}; agree_old = {p: [0, 0] for p in POS + ("uncued",)}
            for r in G["records"]:
                it = n["items"][r["k"]]; assert (it["ds"], it["hash"], it["T"], it["X"]) == (r["ds"], r["hash"], r["T"], r["X"]), (t, r["k"])
                agree["uncued"][0] += 1; agree["uncued"][1] += it["pred0"] == r["uncued"][1]
                agree_old["uncued"][0] += 1; agree_old["uncued"][1] += r["uncued"][0] == r["uncued"][1]
                for k, a in r["cond"].items():
                    p = k.split("|")[1]; agree[p][0] += 1; agree[p][1] += it["cond"][k][2] == a[1]; agree_old[p][0] += 1; agree_old[p][1] += a[0] == a[1]
            rec["gate_new"] = {p: v[1] / v[0] for p, v in agree.items()}; rec["gate_old"] = {p: v[1] / v[0] for p, v in agree_old.items()}
            rec["gate_n_items"] = len(G["records"]); rec["gate_pass_new"] = bool(min(rec["gate_new"].values()) >= 0.97); ok &= rec["gate_pass_new"]
        ok &= same_items; out["checkpoints"][t] = rec
        print(f"{t:22s} items identical {same_items} | answer changed vs old readout: uncued {rec['changed_uncued']:.3f} " + " ".join(f"{p} {rec['changed'][p]:.3f}" for p in POS)
              + (f" | gate (new vs generated) " + " ".join(f"{p} {v:.4f}" for p, v in rec["gate_new"].items()) + f" [old: min {min(rec['gate_old'].values()):.4f}] -> {'PASS' if rec['gate_pass_new'] else 'FAIL'}"
                 if "gate_new" in rec else ""))
    out["all_ok"] = bool(ok and not out["missing"]); json.dump(out, open(f"{R}/w7/check_gemma_gen7.json", "w"), indent=1)
    print("missing:", out["missing"]); print("ALL OK" if out["all_ok"] else "!!! NOT ALL OK (see above)")


if __name__ == "__main__":
    main()
