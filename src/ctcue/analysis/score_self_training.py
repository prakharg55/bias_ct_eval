"""Score the self-training control against BCT, with the registered predictions S1 and S2.

  S1  the control achieves less than half of BCT's reduction of trained-cue following
  S2  its paired helpful drop is less than half of BCT's
Both use seed means and are scored separately for Gemma and Llama.

Usage: python -m ctcue.analysis.score_self_training
Output: $CT_ROOT/results/w7u/ana_selftrain_u.json
"""

import json, os
import numpy as np

from ctcue.analysis import measures
from ctcue.analysis import fewshot_rescore
from ctcue.common import checkpoints
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
SEEDS = (42, 43, 44)
CUES = fewshot_rescore.CUES
FS = fewshot_rescore.FS


def rates(evalitems, fs_file=None):
    its = json.load(open(evalitems))["items"]
    corr = {}
    if fs_file:
        d = json.load(open(fs_file))
        assert d["regression_same16"] == d["n_items"], fs_file
        corr = {fewshot_rescore.k_(i): fewshot_rescore.classify(i)["corrected"] for i in d["items"]}

    def fk(c, use):
        sub = [i for i in its if i["cue"] == c]
        knew = [i for i in sub if i["gen_u"] == i["T"]]
        return sum(
            (corr.get(fewshot_rescore.k_(i), i["gen_b"]) if use else i["gen_b"]) == i["X"] for i in knew
        ) / max(len(knew), 1)

    tr = fk(CUES[0], False)
    acc = sum(i["gen_u"] == i["T"] for i in its if i["cue"] == CUES[0]) / sum(
        i["cue"] == CUES[0] for i in its
    )
    return {
        "trained": tr,
        "held": float(np.mean([fk(c, False) for c in CUES[1:]])),
        "held_fs_corrected": float(np.mean([fk(c, True) for c in CUES[1:]])) if fs_file else None,
        "acc": acc,
    }


def main():
    out = {}
    for model in ("gemma2b", "llama8b"):
        view = f"{R}/w7u/stview_{model}"
        os.makedirs(view, exist_ok=True)
        links = (
            {"gen_st_none.json": f"{R}/w7/gen_{model}_none.json"}
            | {
                f"gen_st_bct_s{s}.json": f"{R}/w7u/gen_{checkpoints.inst_tag(model, 'bct', s)}.json"
                for s in SEEDS
            }
            | {
                f"gen_st_selftrain_s{s}.json": f"{R}/w7u/gen_{checkpoints.inst_tag(model, 'selftrain', s)}.json"
                for s in SEEDS
            }
        )
        for name, src in links.items():
            assert os.path.exists(src), src
            dst = f"{view}/{name}"
            if os.path.islink(dst):
                assert os.readlink(dst) == src, dst
            else:
                os.symlink(src, dst)
        for f in os.listdir(view):
            assert f in links, f"stale file in view: {f}"
        measures.R = view
        per = measures.ana_gen("st")["per_checkpoint"]
        G = {
            arm: {
                k: [per[f"st_{arm}_s{s}"]["paired"][k] for s in SEEDS]
                for k in ("helpful_drop", "harmful_drop", "n_wrong_both", "n_right_both")
            }
            for arm in ("bct", "selftrain")
        }
        U = rates(f"{R}/pub7/evalitems_{model}_final_none.json", f"{R}/fs7/fs_{model}_final_none.json")
        et = lambda arm, s: checkpoints.eval_tag(model, arm, s)
        T = {
            "bct": [
                rates(f"{R}/pub7u/evalitems_{et('bct', s)}.json", f"{R}/fs7/fs_{et('bct', s)}_u.json")
                for s in SEEDS
            ],
            "selftrain": [
                rates(
                    f"{R}/pub7u/evalitems_{et('selftrain', s)}.json",
                    f"{R}/fs7/fs_{et('selftrain', s)}_u.json",
                )
                for s in SEEDS
            ],
        }
        mean = lambda arm, k: float(np.mean([r[k] for r in T[arm]]))
        cut = {arm: U["trained"] - mean(arm, "trained") for arm in T}
        hd = {arm: float(np.mean(G[arm]["helpful_drop"])) for arm in G}
        o = {
            "untrained": U,
            "test": {
                arm: {k: [r[k] for r in T[arm]] for k in ("trained", "held", "held_fs_corrected", "acc")}
                for arm in T
            },
            "gen_paired": G,
            "S1": {
                "cut_selftrain": cut["selftrain"],
                "cut_bct": cut["bct"],
                "holds": bool(cut["selftrain"] < 0.5 * cut["bct"]),
            },
            "S2": {
                "helpful_drop_selftrain": hd["selftrain"],
                "helpful_drop_bct": hd["bct"],
                "holds": bool(hd["selftrain"] < 0.5 * hd["bct"]),
            },
        }
        out[model] = o
        print(
            f"[{model}] untrained trained {U['trained']:.3f} held {U['held']:.3f} (fs-corrected {U['held_fs_corrected']:.3f}) acc {U['acc']:.3f}"
        )
        for arm in ("bct", "selftrain"):
            print(
                f"   {arm:9s} trained {mean(arm, 'trained'):.3f} held {mean(arm, 'held'):.3f} (fs-corr {mean(arm, 'held_fs_corrected'):.3f}) acc {mean(arm, 'acc'):.3f} | "
                f"helpful drop {np.mean(G[arm]['helpful_drop']):+.3f} harmful drop {np.mean(G[arm]['harmful_drop']):+.3f}"
            )
        print(
            f"   S1 (cut {cut['selftrain']:+.3f} < .5 x BCT's {cut['bct']:+.3f}): {o['S1']['holds']} | S2 (helpful drop {hd['selftrain']:+.3f} < .5 x BCT's {hd['bct']:+.3f}): {o['S2']['holds']}"
        )
    json.dump(out, open(f"{R}/w7u/ana_selftrain_u.json", "w"), indent=1)
    print("saved ->", f"{R}/w7u/ana_selftrain_u.json")


if __name__ == "__main__":
    main()
