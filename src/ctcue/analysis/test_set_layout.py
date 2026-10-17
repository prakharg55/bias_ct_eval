"""Split the test set's trained-cue results by where the cue sits in each source row (post hoc, exploratory).

Training puts every cue before the question, but the test set keeps the cue positions of the dumps
(before the question, before the options or after them). Reports harmful-cue following and the share
removed for each position, with question-bootstrap intervals.

Usage: python -m ctcue.analysis.test_set_layout
Output: $CT_ROOT/results/w7u/ana_layout_u.json
"""

import json
import numpy as np

from ctcue.common.dumps import load_bct_source
from ctcue.paths import CT_ROOT, BCT_DUMPS

DUMPS = f"{BCT_DUMPS}/test"
C = f"{CT_ROOT}/results"
DS = ["mmlu", "hellaswag", "logiqa", "truthfulqa"]
CUE = "suggested_answer"
LAY = ("front", "pre", "post")
B = 2000
ARMS = {"gemma2b": ("bct", "act", "attct", "rmct", "rmcta"), "llama8b": ("bct", "act", "attct", "rmct")}


def layouts():
    """(ds, question hash) -> layout, for exactly the trained-cue items test_set scores (same rows, exclusions and filters)."""

    def txt(m):
        return "\n".join(x["content"] for x in m if x["role"] == "user")

    excl = set()
    for ds in DS:
        pairs, _ = load_bct_source(f"{DUMPS}/{CUE}/{ds}_{CUE}.jsonl")
        excl |= {p["original_question_hash"] for p in pairs[100:500]}
    lay = {}
    for ds in DS:
        pairs, _ = load_bct_source(f"{DUMPS}/{CUE}/{ds}_{CUE}.jsonl")
        seg = pairs[1200:1300] if len(pairs) > 1250 else pairs[-100:]
        for p in seg:
            if (
                p["original_question_hash"] in excl
                or p["bias_target"] == p["correct_letter"]
                or p["biased_messages"] == p["unbiased_messages"]
            ):
                continue
            b, u = txt(p["biased_messages"]), txt(p["unbiased_messages"])
            lay[(ds, p["original_question_hash"])] = (
                "front"
                if b.endswith(u)
                else ("pre" if b.index("Answer choices") > u.index("Answer choices") else "post")
            )
    return lay


def table(path, keys):
    """per item in keys: (knew, caved) arrays for the trained cue."""
    it = {(i["ds"], i["hash"]): i for i in json.load(open(path))["items"] if i["cue"] == CUE}
    assert set(it) == set(keys), path
    knew = np.array([it[k]["gen_u"] == it[k]["T"] for k in keys])
    cave = np.array([it[k]["gen_b"] == it[k]["X"] for k in keys])
    return knew, cave


def rate(knew, cave, sel, w=None):
    w = np.ones(len(knew)) if w is None else w
    m = knew & sel
    return float((cave * w)[m].sum() / w[m].sum()) if w[m].sum() > 0 else float("nan")


def main():
    lay = layouts()
    keys = sorted(lay)
    L = np.array([lay[k] for k in keys])
    ds = np.array([k[0] for k in keys])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(B):  # multiplicity weights from resampling questions within dataset
        w = np.zeros(len(keys))
        for d in DS:
            # (10 Oct) add.at: repeated draws count repeatedly (a fancy-index += counted each question once)
            idx = np.flatnonzero(ds == d)
            np.add.at(w, rng.choice(idx, len(idx)), 1)
        boots.append(w)
    out = {"n_items": {l: int((L == l).sum()) for l in LAY}, "B": B, "models": {}}
    for model, arms in ARMS.items():
        bk, bc = table(f"{C}/pub7/evalitems_{model}_final_none.json", keys)
        res = {
            "untrained": {
                l: {"rate": rate(bk, bc, L == l), "n_knew": int((bk & (L == l)).sum())} for l in LAY
            },
            "arms": {},
        }
        for arm in arms:
            nm = f"final7u_{arm}" if (model == "gemma2b" and arm in ("act", "attct")) else f"final_{arm}"
            seeds = [table(f"{C}/pub7u/evalitems_{model}_{nm}_s{s}.json", keys) for s in (42, 43, 44)]

            def stat(w):
                rr = {
                    l: float(np.mean([1 - rate(k, c, L == l, w) / rate(bk, bc, L == l, w) for k, c in seeds]))
                    for l in LAY
                }
                return rr, rr["front"] - rr["post"]

            rr, gap = stat(None)
            bs = [stat(w) for w in boots]
            ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            res["arms"][arm] = {
                "rate": {l: float(np.mean([rate(k, c, L == l) for k, c in seeds])) for l in LAY},
                "relred": rr,
                "relred_ci95": {l: ci([b[0][l] for b in bs]) for l in LAY},
                "front_minus_post": gap,
                "front_minus_post_ci95": ci([b[1] for b in bs]),
            }
            print(
                model,
                arm,
                {l: round(rr[l], 2) for l in LAY},
                "front-post",
                round(gap, 2),
                [round(x, 2) for x in res["arms"][arm]["front_minus_post_ci95"]],
                flush=True,
            )
        out["models"][model] = res
    print(
        "items",
        out["n_items"],
        {m: {l: v["n_knew"] for l, v in r["untrained"].items()} for m, r in out["models"].items()},
    )
    json.dump(out, open(f"{C}/w7u/ana_layout_u.json", "w"), indent=1)
    print(f"-> {C}/w7u/ana_layout_u.json")


if __name__ == "__main__":
    main()
