"""Assemble the appendix tables on merging, learning-rate selection, the layer swap and the few-shot cues.

The layer-swap table sets the unmerged Llama results next to the merged ones. The script only reads
stored outputs, such as those of merge_fidelity.py, score_layer_swap.py and fewshot_sensitivity.py,
and computes only means and standard deviations of stored values.

Usage: python -m ctcue.analysis.appendix_tables
Output: $CT_ROOT/results/paper_u/appendix_u.json and .md
"""

import json, os
import numpy as np

from ctcue.common import checkpoints
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
OUT = f"{R}/paper_u"
SEEDS = (42, 43, 44)
CUES = [
    "suggested_answer",
    "post_hoc",
    "distractor_fact",
    "wrong_few_shot",
    "distractor_argument",
    "spurious_few_shot_squares",
]
J = lambda f: json.load(open(f)) if os.path.exists(f) else None
f3 = lambda x: "n/a" if x is None or x != x else f"{x:.3f}"


def tab(head, rows):
    return (
        ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
        + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
        + [""]
    )


def _unresolved(model, arm):
    try:
        checkpoints.changed(model, arm)
        return False
    except RuntimeError:
        return True


def t1(ev):
    return (
        ev["flip_rates"][CUES[0]]["rate"],
        float(np.mean([ev["flip_rates"][c]["rate"] for c in CUES[1:]])),
        ev["accuracy"][CUES[0]],
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    L = ["# Appendix tables (unmerged re-run; pre-write-up audit)", ""]
    m7 = J(f"{R}/pub7u/ana_merge7.json") or {}
    res["A1"] = m7
    L += [
        "## A1. Merge fidelity (bf16 merge_and_unload vs the adapter unmerged, as trained)",
        "",
        "Declared rule: merged results stand iff the generated answers agree on >= .99 of the 3406 test prompts AND each T1 rate differs by <= .02.",
        "",
    ]
    L += tab(
        [
            "checkpoint",
            "agreement (generated)",
            "merged trained / held / acc",
            "unmerged trained / held / acc",
            "difference",
            "merged results stand",
        ],
        [
            [
                t,
                f"{r['agree_generated']:.4f}",
                " / ".join(f3(r["merged"][k]) for k in ("trained", "held", "acc")),
                " / ".join(f3(r["unmerged"][k]) for k in ("trained", "held", "acc")),
                " / ".join(f"{r['diff'][k]:+.3f}" for k in ("trained", "held", "acc")),
                "yes" if r["merged_results_stand"] else "no",
            ]
            for t, r in sorted(m7.items())
        ],
    )
    sel = J(f"{R}/pub_val7g/selection_unmerged.json") or {}
    res["A2"] = sel
    rows = []
    for model in ("gemma2b", "llama8b"):
        for arm, r in (sel.get(model) or {}).items():
            if arm == "untrained":
                continue
            grid = (
                " ".join(
                    f"{lr}:{v['F1']:.3f}{'D' if v.get('degenerate') else ''}"
                    for lr, v in (r.get("rows") or {}).items()
                )
                if r.get("rows")
                else " ".join(
                    f"{lr}:{p['F1']:.3f}{'D' if p['modal_frac'] >= .95 else ''}"
                    for lr, p in sorted((r.get("points") or {}).items(), key=lambda kv: float(kv[0]))
                )
            )
            rows.append(
                [
                    model,
                    arm,
                    r.get("stored_selection"),
                    r.get("selected"),
                    r.get("status"),
                    "YES" if r.get("changed") else ("no" if r.get("changed") is False else "n/a"),
                    grid,
                ]
            )
    L += [
        "## A2. LR selection: stored vs unmerged generated-answer re-selection",
        "",
        "Stored: gemma = first-position readout on merged checkpoints (select_lr.py), Llama = generated answers on merged checkpoints (select_lr7.py); re-selection = the same rule "
        "(pre-registered + Amendments 1-2) on generated-answer F1 of the UNMERGED sweep checkpoints (select_lr_gen7u.py). Grid: LR:F1 (D = degenerate, excluded).",
        "",
    ]
    L += tab(
        ["model", "arm", "stored", "re-selected", "status", "changed", "F1 grid (unmerged, generated)"], rows
    )
    rows = []
    res["A3"] = {}
    for model in ("gemma2b", "llama8b"):
        for arm in checkpoints.SWEPT[model]:
            try:
                if not checkpoints.changed(model, arm):
                    continue
            except RuntimeError:
                continue
            old = [J(f"{R}/pub7uc/eval_{model}_final_{arm}_s{s}.json") for s in SEEDS]
            new = [J(f"{R}/pub7uc/eval_{checkpoints.eval_tag(model, arm, s)}.json") for s in SEEDS]

            def ms(evs):
                if any(e is None for e in evs):
                    return None
                v = np.array([t1(e) for e in evs])
                return [(float(v[:, i].mean()), float(v[:, i].std(ddof=1))) for i in range(3)]

            o, n = ms(old), ms(new)
            res["A3"][f"{model}:{arm}"] = {
                "old_lr": sel[model][arm]["stored_selection"],
                "new_lr": sel[model][arm]["selected"],
                "old": o,
                "new": n,
            }
            fmt = lambda x: " / ".join(f"{a:.3f} ± {b:.3f}" for a, b in x) if x else "pending"
            rows.append(
                [model, arm, sel[model][arm]["stored_selection"], fmt(o), sel[model][arm]["selected"], fmt(n)]
            )
    L += [
        "## A3. Re-selected arms: unmerged test rates at the old and the new LR (trained / held-out / accuracy, seed mean ± SD)",
        "",
    ] + (
        tab(["model", "arm", "old LR", "old LR rates", "new LR", "new LR rates (primary)"], rows)
        if rows
        else ["No arm changed.", ""]
    )
    ru = J(f"{R}/w7u/score_route_u.json")
    res["A4"] = ru
    if ru:
        rows = [
            [
                m,
                " ".join(f"{sc} {f3(o['fracs'][sc])}" for sc in ("EARLY", "CARRIER", "CIRCUIT", "LATE")),
                (
                    " ".join(
                        f"{sc} {f3(o['merged']['fracs'][sc])}"
                        for sc in ("EARLY", "CARRIER", "CIRCUIT", "LATE")
                    )
                    if o.get("merged")
                    else "n/a"
                ),
                (o["D1"] or {}).get("holds"),
                ((o.get("merged") or {}).get("D1") or {}).get("holds"),
            ]
            for m, o in ru["llama"]["arms"].items()
            if o
        ]
        L += ["## A4. Route (Llama, seed 42): unmerged (primary) vs bf16-merged weights", ""] + tab(
            ["arm", "unmerged med_fwd", "merged med_fwd", "D1 unmerged", "D1 merged"], rows
        )
        L += [
            f"D1 verdict: unmerged {ru['llama']['D1']['verdict']}, merged {ru['llama']['D1_merged']['verdict']}.",
            "",
        ]
    fs = J(f"{R}/fs7/ana_fs7_unmerged.json")
    res["A5"] = {}
    if fs:
        FSC = ("wrong_few_shot", "spurious_few_shot_squares")
        VAR = ("stored", "corrected", "U", "CC")
        rows = []
        ck = fs["checkpoints"]

        def row(model, arm, tags):
            if not all(t in ck for t in tags):
                return None
            evs = [J(f"{R}/{'pub7' if arm is None else 'pub7u'}/eval_{t}.json") for t in tags]
            other = [[e["flip_rates"][c]["rate"] for c in CUES[1:] if c not in FSC] for e in evs]
            o = {}
            for v in VAR:
                for c in FSC:
                    o[f"{c}:{v}"] = float(np.mean([ck[t][c][v]["rate"] for t in tags]))
                o[f"held:{v}"] = float(
                    np.mean(
                        [
                            (sum(oth) + sum(ck[t][c][v]["rate"] for c in FSC)) / 5
                            for t, oth in zip(tags, other)
                        ]
                    )
                )
            o["n_list"] = {c: [ck[t][c]["n_list"] for t in tags] for c in FSC}
            return o

        for model in ("gemma2b", "llama8b"):
            arms = [(None, [f"{model}_final_none"])] + [
                (m, [checkpoints.eval_tag(model, m, s) for s in SEEDS])
                for m in checkpoints.SWEPT[model]
                if not _unresolved(model, m)
            ]
            for arm, tags in arms:
                o = row(model, arm, tags)
                if o is None:
                    continue
                res["A5"][f"{model}:{arm or 'untrained'}"] = o
                rows.append(
                    [model, arm or "untrained"]
                    + [" / ".join(f3(o[f"{c}:{v}"]) for v in VAR) for c in FSC]
                    + [
                        " / ".join(f3(o[f"held:{v}"]) for v in VAR),
                        " | ".join("/".join(map(str, o["n_list"][c])) for c in FSC),
                    ]
                )
        L += [
            "## A5. Few-shot cues at the test slot (unmerged; NOTE 11:19 item 2)",
            "",
            "flip|knew, seed mean: stored 16-token answer / test-slot corrected (reported) / bound U (list output = no answer) / bound CC (list without a test slot = item "
            "dropped); held-out mean over the 5 held-out cues under each; list-output counts per seed (wrong_few_shot | squares).",
            "",
        ]
        L += tab(
            [
                "model",
                "arm",
                "wrong_few_shot",
                "spurious_few_shot_squares",
                "held-out mean",
                "list outputs per seed",
            ],
            rows,
        )
    se = J(f"{R}/w7u/ana_fs_sens.json")
    res["A6"] = se
    if se:
        rows = []
        for key in ("letterprior_gemma", "letterprior_gemma_fix", "letterprior_llama"):
            for a, v in se[key]["arms"].items():
                rows.append(
                    [
                        key.replace("letterprior_", ""),
                        a,
                        f"{v['bpd_held5'][0]:+.3f} ± {f3(v['bpd_held5'][1])}",
                        f"{v['bpd_held3'][0]:+.3f} ± {f3(v['bpd_held3'][1])}",
                    ]
                )
        L += [
            "## A6. Sensitivity without the two few-shot cues (exploratory; NOTE 11:19 item 2)",
            "",
            "Letter-prior control: beyond-prior drop of held-out caving, 5 held-out cues vs the 3 non-few-shot ones (seed mean ± SD):",
            "",
        ]
        L += tab(["model", "arm", "5 cues", "3 cues"], rows)
        L += tab(
            ["checkpoint", "patch recovery, all cues", "without few-shot cues"],
            [
                [
                    k,
                    f"{f3(v['all']['circuit'])} ({v['all']['recovered']}/{v['all']['n_flips']})",
                    f"{f3(v['no_fewshot']['circuit'])} ({v['no_fewshot']['recovered']}/{v['no_fewshot']['n_flips']})",
                ]
                for k, v in se["patch_recovery"].items()
            ],
        )
        L += tab(
            [
                "route checkpoint",
                "all cues EARLY / CARRIER / CIRCUIT / LATE",
                "without few-shot cues",
                "D1-style all / without",
            ],
            [
                [
                    k,
                    " / ".join(f3(v["all"]["fracs"][s]) for s in ("EARLY", "CARRIER", "CIRCUIT", "LATE")),
                    " / ".join(
                        f3(v["no_fewshot"]["fracs"][s]) for s in ("EARLY", "CARRIER", "CIRCUIT", "LATE")
                    ),
                    f"{(v['all']['D1_style'] or {}).get('holds')} / {(v['no_fewshot']['D1_style'] or {}).get('holds')}",
                ]
                for k, v in se["route"].items()
            ],
        )
    json.dump(res, open(f"{OUT}/appendix_u.json", "w"), indent=1)
    open(f"{OUT}/appendix_u.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("saved ->", f"{OUT}/appendix_u.md")


if __name__ == "__main__":
    main()
