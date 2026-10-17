"""Collect the weight profiles of the final Gemma adapters for the registered depth prediction (P4).

Keeps the stored profile of every method whose learning rate did not change, and takes the
profiles of the new adapters for the others. Those come from evaluation/weight_profile.py, stored as
$CT_ROOT/results/w7u/weights_final7u_gemma2b.json. If no method changed, nothing is written.

Usage: python -m ctcue.analysis.weight_depth
Output: $CT_ROOT/results/w7u/weights_pub_u.json (the four main methods) and
        $CT_ROOT/results/w7u/weights_pub_rmcta_u.json (RMCT+anchor), each only if one of its methods changed
"""

import copy, json

from ctcue.common import checkpoints
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
SEEDS = (42, 43, 44)


def main():
    new = json.load(open(f"{R}/w7u/weights_final7u_gemma2b.json"))
    out_files = {}
    for src, dst, arms in (
        (f"{R}/w3/weights_pub.json", f"{R}/w7u/weights_pub_u.json", ("bct", "act", "attct", "rmct")),
        (f"{R}/w3/weights_pub_rmcta.json", f"{R}/w7u/weights_pub_rmcta_u.json", ("rmcta",)),
    ):
        chg = [m for m in arms if checkpoints.changed("gemma2b", m)]
        if not chg:
            continue
        d = json.load(open(src))
        o = copy.deepcopy(d)
        rep = []
        for m in chg:
            for s in SEEDS:
                k = f"{m}_s{s}_step1"
                assert k in d["profile"] and k in new["profile"], k
                o["profile"][k] = new["profile"][k]
                rep.append(k)
        assert (
            new["meta"]["NL"] == d["meta"]["NL"]
            and new["meta"]["NH"] == d["meta"]["NH"]
            and new["meta"]["HD"] == d["meta"]["HD"]
        )
        o["meta"]["unmerged_reselection"] = {
            "replaced_entries": rep,
            "from": f"{R}/w7u/weights_final7u_gemma2b.json",
            "new_adapters": {k: v for k, v in new["meta"]["adapters"].items()},
        }
        json.dump(o, open(dst, "w"))
        out_files[dst] = rep
    for f, rep in out_files.items():
        print(f"{f}: replaced {rep}")
    if not out_files:
        print("no gemma arm changed: the stored profiles stand (nothing written)")


if __name__ == "__main__":
    main()
