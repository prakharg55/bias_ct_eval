#!/usr/bin/env python3
"""
Phase 5 A data (27 Sep 2026; pre-registered 14:21 EDT). Builds BCT data directories (bct_non_cot.jsonl + the SAME
instruct_samples.jsonl as standard BCT) for the fix variants, from the model's own BCT data (shared/bct_<model>/):
  bayes_r{r}: target letter ~ p(a|q,h) ~ p0(a|q) L(h|a), L(h=X|a) = r if a == X else (1-r)/(K-1); p0 from p0_<model>.jsonl;
              the own target text with its letter replaced by the sampled one (own targets with no parseable letter kept).
  goldgated : own target correct -> own target; hint correct (X == T) & own target wrong -> own text with T (or "T \\n"
              if the own target has no letter); hint wrong & own target wrong -> dropped.
  posdiv    : own targets; the hint placed at a seeded uniform position in {front, pre, post, end} (ct_gen3.insert).
Every variant keeps the BCT prompt order; sampling uses random.Random(0) in item order. CPU, seconds.
bayescal_r{r}: bayes_r{r} with the calibrated prior p0^(1/T_cal) (T_cal from calib_<model>.json; Phase-5 amendment 3).
Usage: build_fix_data.py --model gemma2b [--variants bayes_r0.5,bayes_r0.8,goldgated,posdiv]
"""
import argparse, json, os, random, re, shutil, sys
sys.path.insert(0, "/data/code")
SH = "/data/ct_root/shared"
BARE = re.compile(r"^\s*\**\(?([A-P])\)?\**[\s\.\)\*]*$")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--variants", default="bayes_r0.5,bayes_r0.8,goldgated,posdiv")
    a = ap.parse_args()
    sh = [json.loads(l) for l in open(f"{SH}/shared_train.jsonl")]
    bct = [json.loads(l) for l in open(f"{SH}/bct_{a.model}/bct_non_cot.jsonl")]
    assert len(sh) == len(bct) and all(b["messages"][0]["content"] == s["biased"] for b, s in zip(bct, sh)), "BCT pairs out of order"
    own = []
    for b in bct:
        t = b["messages"][-1]["content"]; m = BARE.match(t); own.append((t, m.group(1) if m else None, m.span(1) if m else None))
    replace = lambda t, span, L: t[:span[0]] + L + t[span[1]:]
    for v in a.variants.split(","):
        out, stats = [], {"n": 0, "dropped": 0, "kept_unparsed": 0, "target_eq_X": 0, "target_eq_T": 0, "changed_vs_own": 0}
        rng = random.Random(0)
        if v.startswith("bayes_r") or v.startswith("bayescal_r"):
            cal = v.startswith("bayescal_r"); r = float(v.split("_r")[1]); p0 = [json.loads(l) for l in open(f"{SH}/p0_{a.model}.jsonl")]
            Tc = json.load(open(f"{SH}/calib_{a.model}.json"))["T_cal"] if cal else 1.0; stats["T_prior"] = Tc
            if cal:   # AMENDMENT 3 (Phase 5): calibrated prior p0^(1/T_cal), T_cal fitted on the validation slice (calib_temp.py)
                p0 = [dict(p, p0={o: q ** (1.0 / Tc) / sum(x ** (1.0 / Tc) for x in p["p0"].values()) for o, q in p["p0"].items()}) for p in p0]
            assert [p["i"] for p in p0] == [s["i"] for s in sh]
            for s, b, (t, L, span), p in zip(sh, bct, own, p0):
                opts = p["opts"]; K = len(opts); X = s["X"]
                w = [p["p0"][o] * (r if o == X else (1 - r) / (K - 1)) for o in opts]; Z = sum(w)
                L_new = rng.choices(opts, weights=[x / Z for x in w])[0]
                if L is None: tgt = t; stats["kept_unparsed"] += 1
                else: tgt = replace(t, span, L_new); stats["changed_vs_own"] += L_new != L
                out.append({"messages": [b["messages"][0], {"role": "assistant", "content": tgt}]})
                lt = BARE.match(tgt).group(1) if BARE.match(tgt) else None
                stats["target_eq_X"] += lt == X; stats["target_eq_T"] += lt == s["T"]
        elif v == "goldgated":
            for s, b, (t, L, span) in zip(sh, bct, own):
                if L == s["T"]: tgt = t
                elif s["X"] == s["T"]: tgt = replace(t, span, s["T"]) if L is not None else f"{s['T']} \n"; stats["changed_vs_own"] += 1
                else: stats["dropped"] += 1; continue
                out.append({"messages": [b["messages"][0], {"role": "assistant", "content": tgt}]})
                stats["target_eq_X"] += s["X"] == s["T"]; stats["target_eq_T"] += 1
        elif v == "posdiv":
            from ct_gen3 import insert
            for s, b, (t, L, span) in zip(sh, bct, own):
                assert insert(s["clean"], s["cue"], "front") == s["biased"], s["i"]
                pos = rng.choice(["front", "pre", "post", "end"]); stats[pos] = stats.get(pos, 0) + 1
                out.append({"messages": [{"role": "user", "content": insert(s["clean"], s["cue"], pos)}, b["messages"][1]]})
                stats["target_eq_X"] += L == s["X"]; stats["target_eq_T"] += L == s["T"]
        else:
            raise ValueError(v)
        stats["n"] = len(out)
        d = f"{SH}/bct_{v}_{a.model}"; os.makedirs(d, exist_ok=True)
        assert not os.path.exists(f"{d}/bct_non_cot.jsonl"), f"{d} exists: refusing to overwrite"
        with open(f"{d}/bct_non_cot.jsonl", "w") as f:
            for o in out: f.write(json.dumps(o) + "\n")
        shutil.copy2(f"{SH}/bct_{a.model}/instruct_samples.jsonl", f"{d}/instruct_samples.jsonl")
        json.dump(stats, open(f"{d}/build_stats.json", "w"), indent=1)
        print(f"{v:12s} -> {d}: {stats}")


if __name__ == "__main__":
    main()
