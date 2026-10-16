#!/usr/bin/env python3
"""
weights_u (1 Oct 2026; prereg NOTEs 11:19 item 3 + 19:43): the P4 weight profile (ct_weights6: per-head relative weight change of the
LoRA update; scored by score_pub.score_weights) for the UNMERGED primary results. Adapter weights do not depend on merging, so every arm
whose selection is unchanged keeps the stored profile (results/w3/weights_pub.json; RMCT-anchor results/w3/weights_pub_rmcta.json); for an
arm whose unmerged LR re-selection CHANGED (utags), its entries "<arm>_s<seed>_step1" are replaced by ct_weights6's profile of the
new-selection adapters (results/w7u/weights_final7u_gemma2b.json = ct_weights6 --model gemma2b --root models/published/final7u, run by
a CPU batch job). The new file's meta lists the replaced entries. Output results/w7u/weights_pub_u.json (and weights_pub_rmcta_u.json if
RMCT-anchor changed). CPU, seconds.
"""
import copy, json, sys
sys.path.insert(0, "/data/code")
import utags
R = "/data/ct_root/results"; SEEDS = (42, 43, 44)


def main():
    new = json.load(open(f"{R}/w7u/weights_final7u_gemma2b.json")); out_files = {}
    for src, dst, arms in ((f"{R}/w3/weights_pub.json", f"{R}/w7u/weights_pub_u.json", ("bct", "act", "attct", "rmct")),
                           (f"{R}/w3/weights_pub_rmcta.json", f"{R}/w7u/weights_pub_rmcta_u.json", ("rmcta",))):
        chg = [m for m in arms if utags.changed("gemma2b", m)]
        if not chg: continue
        d = json.load(open(src)); o = copy.deepcopy(d); rep = []
        for m in chg:
            for s in SEEDS:
                k = f"{m}_s{s}_step1"; assert k in d["profile"] and k in new["profile"], k
                o["profile"][k] = new["profile"][k]; rep.append(k)
        assert new["meta"]["NL"] == d["meta"]["NL"] and new["meta"]["NH"] == d["meta"]["NH"] and new["meta"]["HD"] == d["meta"]["HD"]
        o["meta"]["unmerged_reselection"] = {"replaced_entries": rep, "from": f"{R}/w7u/weights_final7u_gemma2b.json", "new_adapters": {k: v for k, v in new["meta"]["adapters"].items()}}
        json.dump(o, open(dst, "w")); out_files[dst] = rep
    for f, rep in out_files.items(): print(f"{f}: replaced {rep}")
    if not out_files: print("no gemma arm changed: the stored profiles stand (nothing written)")


if __name__ == "__main__":
    main()
