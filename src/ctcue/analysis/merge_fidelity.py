#!/usr/bin/env python3
"""
ana_merge7 (1 Oct 2026; prereg NOTE "PRE-WRITE-UP AUDIT", item 1 MERGE FIDELITY). For every checkpoint evaluated UNMERGED by
ct_eval7u (results/pub7u/eval[items]_<tag>.json) compare with the MERGED ct_eval7 result of the same tag (results/pub7):
item-by-item agreement of the generated answers over all 3406 unbiased + biased test prompts, and the T1 rates (trained-cue
flip|knew, held-out mean of 5 cues, trained-cue accuracy) on the generated answers (and, for reference, the first-position
readout). DECLARED RULE: merged results stand iff agreement >= .99 AND each of the three rates differs by <= .02; otherwise
the arm's functional instruments are re-run unmerged. Also: the unmerged model's own first-vs-generated gate. CPU, seconds.
Output results/pub7u/ana_merge7.json.
"""
import glob, json, os
R = "/data/ct_root/results"
CUES = ["suggested_answer", "post_hoc", "distractor_fact", "wrong_few_shot", "distractor_argument", "spurious_few_shot_squares"]


def t1(ev):
    fr, acc = ev["flip_rates"], ev["accuracy"]
    return {"trained": fr[CUES[0]]["rate"], "held": sum(fr[c]["rate"] for c in CUES[1:]) / 5, "acc": acc[CUES[0]]}


def main():
    out = {}
    for f in sorted(glob.glob(f"{R}/pub7u/eval_*.json")):
        if f.endswith("_smoke.json"): continue
        u = json.load(open(f)); tag = u["tag"]
        if not os.path.exists(f"{R}/pub7/eval_{tag}.json"): continue   # no merged counterpart (e.g. the self-training control, evaluated unmerged only)
        m = json.load(open(f"{R}/pub7/eval_{tag}.json"))
        assert u["merge"] == "unmerged" and u["adapter"] == m["adapter"] and u["agg"] == m["agg"] and u["max_new"] == m["max_new"] == 16 and u["eval_rows"] == m["eval_rows"], tag
        Iu = json.load(open(f"{R}/pub7u/evalitems_{tag}.json"))["items"]; Im = json.load(open(f"{R}/pub7/evalitems_{tag}.json"))["items"]
        key = lambda i: (i["cue"], i["ds"], i["hash"], i["X"], i["T"]); assert [key(i) for i in Iu] == [key(i) for i in Im], tag
        n = 2 * len(Iu); agree = sum(a["gen_" + s] == b["gen_" + s] for a, b in zip(Iu, Im) for s in "ub") / n
        agree_first = sum(a["first_" + s] == b["first_" + s] for a, b in zip(Iu, Im) for s in "ub") / n
        tu, tm = t1(u), t1(m); tuf, tmf = t1(u["first"]), t1(m["first"])
        diff = {k: tu[k] - tm[k] for k in tu}; ok = bool(agree >= 0.99 and all(abs(v) <= 0.02 for v in diff.values()))
        out[tag] = {"agree_generated": agree, "agree_first": agree_first, "merged": tm, "unmerged": tu, "diff": diff, "merged_first": tmf, "unmerged_first": tuf,
                    "unmerged_gate_agree_first_gen": u["agree_first_gen"], "unmerged_parsed": u["parsed"], "merged_results_stand": ok}
        print(f"{tag:30s} agree gen {agree:.4f} first {agree_first:.4f} | merged tr/held/acc {tm['trained']:.3f}/{tm['held']:.3f}/{tm['acc']:.3f} -> unmerged "
              f"{tu['trained']:.3f}/{tu['held']:.3f}/{tu['acc']:.3f} (diff {diff['trained']:+.3f}/{diff['held']:+.3f}/{diff['acc']:+.3f}) | unmerged gate {u['agree_first_gen']:.4f} | "
              f"{'MERGED STANDS' if ok else '!!! RE-RUN UNMERGED'}", flush=True)
    json.dump(out, open(f"{R}/pub7u/ana_merge7.json", "w"), indent=1)
    bad = [t for t, r in out.items() if not r["merged_results_stand"]]; print(f"{len(out)} checkpoints; merged results stand for {len(out) - len(bad)}; re-run unmerged: {bad}")


if __name__ == "__main__":
    main()
