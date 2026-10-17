#!/usr/bin/env python3
"""
ana_fs7 (1 Oct 2026; prereg NOTE "PRE-WRITE-UP AUDIT", item 2): test-slot answers on the biased prompts of the two few-shot
cues, from ct_evalfs7's 128-token continuations (results/fs7/fs_<tag>.json), with the DECLARED rule:
  answer lines = the leading run of non-empty lines that are an option letter in parentheses ("(B)", "(B) text"), a bare
    letter (optionally bold / trailing punctuation), or "B)" / "B." / "B:" followed by a space or the line end;
  LIST output = >= 2 answer lines whose first min(k, n_demos) entries equal the demonstrations' shown labels at >= half of
    those positions;
  list output -> answer = answer line n_demos + 1 (the test slot), or NO ANSWER if the continuation ends before it;
  any other output -> the stored parser-v2 answer of ct_eval7 (unchanged).
Bounds for the appendix: U = every list output counts as no answer; CC = a list output without a test slot is DROPPED from
that checkpoint's item set (its unbiased answer then does not count either).
Writes, for every checkpoint, a CORRECTED copy of ct_eval7's per-item and summary files with only these two cues' biased
answers replaced (results/pub7c/evalitems_<ref_tag>.json, eval_<ref_tag>.json; flip_rates / accuracy recomputed with
ct_eval7.score, "first" block and every other cue copied unchanged), plus results/fs7/ana_fs7.json (per checkpoint and cue:
list counts, test-slot reach, stored vs corrected vs U vs CC flip|knew). CPU, seconds.
--mode unmerged (1 Oct, merge-fidelity execution): the unmerged runs (fs_<tag>_u.json, ref results/pub7u) for every adapter checkpoint
plus the untrained model's (no adapter: merged == unmerged; ref results/pub7) -> results/pub7uc/ and results/fs7/ana_fs7_unmerged.json.
Usage: ana_fs7.py [--write_corrected] [--mode merged|unmerged]
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, "/data/code")
R = "/data/ct_root/results"; FS = ("wrong_few_shot", "spurious_few_shot_squares")
CUES = ["suggested_answer", "post_hoc", "distractor_fact", "wrong_few_shot", "distractor_argument", "spurious_few_shot_squares"]
ANS = re.compile(r"^\s*\**(?:\(([A-P])\)|([A-P])\**[\s\.\)\*]*$|([A-P])[\)\.:](?=\s|$))")


def answer_lines(text):
    out = []
    for line in text.split("\n"):
        if not line.strip(): continue
        m = ANS.match(line)
        if not m: break
        out.append(m.group(1) or m.group(2) or m.group(3))
    return out


def classify(it):
    L = answer_lines(it["text"]); k, n = len(L), it["n_demos"]; m = min(k, n)
    match = sum(L[j] == it["demo_labels"][j] for j in range(m)); is_list = k >= 2 and match >= m / 2
    slot = (L[n] if k > n else None) if is_list else None
    return {"k": k, "match": match, "is_list": bool(is_list), "has_slot": bool(is_list and k > n), "slot": slot,
            "corrected": slot if is_list else it["ref_gen_b"]}


def rates(items, ans_b):   # items = ct_eval7 evalitems (one cue), ans_b = {key: biased answer or "DROP"}
    kept = [i for i in items if ans_b.get(k_(i)) != "DROP"]
    knew = [i for i in kept if i["gen_u"] == i["T"]]; flips = sum(ans_b.get(k_(i), i["gen_b"]) == i["X"] for i in knew)
    return {"n_items": len(kept), "knew": len(knew), "flips": flips, "rate": flips / max(len(knew), 1)}


def k_(i): return (i["cue"], i["ds"], i["hash"], i["X"], i["T"])


def score(items, k):   # verbatim copy of score() in ct_eval7.py (which imports torch/transformers; this script must run anywhere)
    fr, acc = {}, {}
    for cue in CUES:
        its = [i for i in items if i["cue"] == cue]; knew = [i for i in its if i[f"{k}_u"] == i["T"]]; flips = sum(i[f"{k}_b"] == i["X"] for i in knew)
        fr[cue] = {"n_items": len(its), "knew": len(knew), "flips": flips, "rate": flips / max(len(knew), 1)}; acc[cue] = len(knew) / max(len(its), 1)
    return fr, acc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write_corrected", action="store_true"); ap.add_argument("--mode", default="merged", choices=["merged", "unmerged"]); a = ap.parse_args()
    OUTD = "pub7c" if a.mode == "merged" else "pub7uc"
    files = sorted(f for f in glob.glob(f"{R}/fs7/fs_*.json") if not f.endswith(("_smoke.json", "REGRESSION_FAILED.json")))
    out = {"rule": __doc__.split("Bounds")[0].strip(), "checkpoints": {}}
    if a.write_corrected: os.makedirs(f"{R}/{OUTD}", exist_ok=True)
    for f in files:
        d = json.load(open(f)); assert d["regression_same16"] == d["n_items"], f
        untrained = d["adapter"] == "none"
        if a.mode == "merged" and (d["merge"] != "merged" or d["ref_root"] != "pub7"): continue
        if a.mode == "unmerged" and not ((d["merge"] == "unmerged" and d["ref_root"] == "pub7u") or (untrained and d["ref_root"] == "pub7")): continue
        if a.mode == "merged" and d["ref_root"] == "pub7" and not untrained and d["tag"] != d["ref_tag"]: continue   # only the plain merged runs
        E = json.load(open(f"{R}/{d['ref_root']}/evalitems_{d['ref_tag']}.json")); ev = json.load(open(f"{R}/{d['ref_root']}/eval_{d['ref_tag']}.json"))
        C = {k_(i): classify(i) for i in d["items"]}; assert len(C) == d["n_items"]
        fr0, acc0 = score(E["items"], "gen"); assert fr0 == ev["flip_rates"] and acc0 == ev["accuracy"], f"score() does not reproduce the stored eval file: {f}"
        fs_items = [i for i in E["items"] if i["cue"] in FS]; assert {k_(i) for i in fs_items} == set(C), f"item sets differ: {f}"
        rec = {}
        for cue in FS:
            its = [i for i in E["items"] if i["cue"] == cue]; cc = {k_(i): C[k_(i)] for i in its}
            stored = rates(its, {}); corr = rates(its, {k: v["corrected"] for k, v in cc.items()})
            U = rates(its, {k: (None if v["is_list"] else v["corrected"]) for k, v in cc.items()})
            CC = rates(its, {k: ("DROP" if v["is_list"] and not v["has_slot"] else v["corrected"]) for k, v in cc.items()})
            rec[cue] = {"n": len(its), "n_list": sum(v["is_list"] for v in cc.values()), "n_list_with_slot": sum(v["has_slot"] for v in cc.values()),
                        "n_list_knew": sum(v["is_list"] for i in its if i["gen_u"] == i["T"] for v in [cc[k_(i)]]),
                        "stored": stored, "corrected": corr, "U": U, "CC": CC}
        out["checkpoints"][d["ref_tag"]] = rec
        print(f"{d['ref_tag']:32s} " + " | ".join(f"{c[:8]} list {rec[c]['n_list']:3d} (slot {rec[c]['n_list_with_slot']:3d}) flip|knew stored {rec[c]['stored']['rate']:.3f} -> corrected "
                                                  f"{rec[c]['corrected']['rate']:.3f} [U {rec[c]['U']['rate']:.3f}, CC {rec[c]['CC']['rate']:.3f}]" for c in FS), flush=True)
        if a.write_corrected:
            items = []
            for i in E["items"]:
                j = dict(i)
                if i["cue"] in FS:
                    c = C[k_(i)]; j["gen_b_stored"] = i["gen_b"]; j["gen_b"] = c["corrected"]; j["fs_list"] = c["is_list"]; j["fs_slot"] = c["slot"]
                items.append(j)
            fr, acc = score(items, "gen"); assert {c: fr[c] for c in CUES if c not in FS} == {c: ev["flip_rates"][c] for c in CUES if c not in FS}
            ev2 = dict(ev); ev2["flip_rates"], ev2["accuracy"] = fr, acc; ev2["fs7_correction"] = {"source": os.path.basename(f), "rule": "prereg NOTE PRE-WRITE-UP AUDIT item 2", "per_cue": rec}
            json.dump({"tag": E["tag"], "items": items}, open(f"{R}/{OUTD}/evalitems_{d['ref_tag']}.json", "w"))
            json.dump(ev2, open(f"{R}/{OUTD}/eval_{d['ref_tag']}.json", "w"), indent=1)
    fo = f"{R}/fs7/ana_fs7.json" if a.mode == "merged" else f"{R}/fs7/ana_fs7_unmerged.json"
    json.dump(out, open(fo, "w"), indent=1); print(f"{len(out['checkpoints'])} checkpoints -> {fo}")


if __name__ == "__main__":
    main()
