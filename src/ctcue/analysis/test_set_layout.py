#!/usr/bin/env python3
"""
ana_layout_u (9 Oct 2026): POST HOC, EXPLORATORY split of the test battery's trained-cue (suggested_answer) flip|knew by the cue layout
each dump row already has. Found during the final audit: training moved every cue to a prefix (build_shared_data.py), but the test
battery (ct_eval7u.py) keeps the dumps' own layout - the cue before the question ("front", the trained layout), between the question and
the options ("pre") or after the options ("post"); rows whose cue sits after the answer instruction lose it to the loader and are dropped.
Not pre-registered; results were seen before this script was written (the project log). CPU, seconds.
Per model: untrained (pub7/evalitems_<model>_final_none.json) and each arm x seed (pub7u, unmerged): flip|knew per layout (knew = the
checkpoint's own generated uncued answer is T), relative reduction 1 - arm/untrained per seed (seed mean), and a 95% percentile interval
from a question bootstrap within dataset (B = 2000; the same resampled questions for the untrained model and every seed).
Output: results/w7u/ana_layout_u.json.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/data/code")
from utils import load_bct_source
DUMPS = "/data/dataset_dumps/test"; C = "/data/ct_root/results"
DS = ["mmlu", "hellaswag", "logiqa", "truthfulqa"]; CUE = "suggested_answer"; LAY = ("front", "pre", "post"); B = 2000
ARMS = {"gemma2b": ("bct", "act", "attct", "rmct", "rmcta"), "llama8b": ("bct", "act", "attct", "rmct")}


def layouts():
    """(ds, question hash) -> layout, for exactly the trained-cue items ct_eval7u scores (same rows, exclusions and filters)."""
    def txt(m): return "\n".join(x["content"] for x in m if x["role"] == "user")
    excl = set()
    for ds in DS:
        pairs, _ = load_bct_source(f"{DUMPS}/{CUE}/{ds}_{CUE}.jsonl"); excl |= {p["original_question_hash"] for p in pairs[100:500]}
    lay = {}
    for ds in DS:
        pairs, _ = load_bct_source(f"{DUMPS}/{CUE}/{ds}_{CUE}.jsonl"); seg = pairs[1200:1300] if len(pairs) > 1250 else pairs[-100:]
        for p in seg:
            if p["original_question_hash"] in excl or p["bias_target"] == p["correct_letter"] or p["biased_messages"] == p["unbiased_messages"]: continue
            b, u = txt(p["biased_messages"]), txt(p["unbiased_messages"])
            lay[(ds, p["original_question_hash"])] = "front" if b.endswith(u) else ("pre" if b.index("Answer choices") > u.index("Answer choices") else "post")
    return lay


def table(path, keys):
    """per item in keys: (knew, caved) arrays for the trained cue."""
    it = {(i["ds"], i["hash"]): i for i in json.load(open(path))["items"] if i["cue"] == CUE}
    assert set(it) == set(keys), path
    knew = np.array([it[k]["gen_u"] == it[k]["T"] for k in keys]); cave = np.array([it[k]["gen_b"] == it[k]["X"] for k in keys])
    return knew, cave


def rate(knew, cave, sel, w=None):
    w = np.ones(len(knew)) if w is None else w; m = knew & sel
    return float((cave * w)[m].sum() / w[m].sum()) if w[m].sum() > 0 else float("nan")


def main():
    lay = layouts(); keys = sorted(lay); L = np.array([lay[k] for k in keys]); ds = np.array([k[0] for k in keys])
    rng = np.random.default_rng(0); boots = []
    for _ in range(B):   # multiplicity weights from resampling questions within dataset
        w = np.zeros(len(keys))
        for d in DS:
            idx = np.flatnonzero(ds == d); np.add.at(w, rng.choice(idx, len(idx)), 1)   # (10 Oct) add.at: repeated draws count repeatedly (a fancy-index += counted each question once)
        boots.append(w)
    out = {"n_items": {l: int((L == l).sum()) for l in LAY}, "B": B, "models": {}}
    for model, arms in ARMS.items():
        bk, bc = table(f"{C}/pub7/evalitems_{model}_final_none.json", keys)
        res = {"untrained": {l: {"rate": rate(bk, bc, L == l), "n_knew": int((bk & (L == l)).sum())} for l in LAY}, "arms": {}}
        for arm in arms:
            nm = f"final7u_{arm}" if (model == "gemma2b" and arm in ("act", "attct")) else f"final_{arm}"
            seeds = [table(f"{C}/pub7u/evalitems_{model}_{nm}_s{s}.json", keys) for s in (42, 43, 44)]
            def stat(w):
                rr = {l: float(np.mean([1 - rate(k, c, L == l, w) / rate(bk, bc, L == l, w) for k, c in seeds])) for l in LAY}
                return rr, rr["front"] - rr["post"]
            rr, gap = stat(None); bs = [stat(w) for w in boots]
            ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            res["arms"][arm] = {"rate": {l: float(np.mean([rate(k, c, L == l) for k, c in seeds])) for l in LAY}, "relred": rr,
                                "relred_ci95": {l: ci([b[0][l] for b in bs]) for l in LAY}, "front_minus_post": gap, "front_minus_post_ci95": ci([b[1] for b in bs])}
            print(model, arm, {l: round(rr[l], 2) for l in LAY}, "front-post", round(gap, 2), [round(x, 2) for x in res["arms"][arm]["front_minus_post_ci95"]], flush=True)
        out["models"][model] = res
    print("items", out["n_items"], {m: {l: v["n_knew"] for l, v in r["untrained"].items()} for m, r in out["models"].items()})
    json.dump(out, open(f"{C}/w7u/ana_layout_u.json", "w"), indent=1); print(f"-> {C}/w7u/ana_layout_u.json")


if __name__ == "__main__":
    main()
