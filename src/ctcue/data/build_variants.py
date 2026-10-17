"""Build the training data of the BCT variants from a model's own BCT data.

  bayes_r{r}  resamples each target letter from p0(a|q) L(h|a), treating the cue as right with
              probability r (p0 from answer_prior.py)
  goldgated   keeps correct targets, corrects a wrong target when the cue is right, drops the rest
  posdiv      keeps the targets and moves each cue to one of the four positions at random
Every variant keeps the instruction mix of standard BCT. Used for all Gemma variants and for the
Llama gold-gated and position-diverse data. The Llama Bayes-BCT data come from build_variants_v2.py.

Usage: python -m ctcue.data.build_variants --model gemma2b [--variants bayes_r0.5,bayes_r0.8,goldgated,posdiv]
       python -m ctcue.data.build_variants --model llama8b --variants goldgated,posdiv
Output: $CT_ROOT/shared/bct_<variant>_<model>/
"""

import argparse, json, os, random, re, shutil
from ctcue.paths import CT_ROOT

SH = f"{CT_ROOT}/shared"
BARE = re.compile(r"^\s*\**\(?([A-P])\)?\**[\s\.\)\*]*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default="bayes_r0.5,bayes_r0.8,goldgated,posdiv")
    a = ap.parse_args()
    sh = [json.loads(l) for l in open(f"{SH}/shared_train.jsonl")]
    bct = [json.loads(l) for l in open(f"{SH}/bct_{a.model}/bct_non_cot.jsonl")]
    assert len(sh) == len(bct) and all(
        b["messages"][0]["content"] == s["biased"] for b, s in zip(bct, sh)
    ), "BCT pairs out of order"
    own = []
    for b in bct:
        t = b["messages"][-1]["content"]
        m = BARE.match(t)
        own.append((t, m.group(1) if m else None, m.span(1) if m else None))
    replace = lambda t, span, L: t[: span[0]] + L + t[span[1] :]
    for v in a.variants.split(","):
        out, stats = [], {
            "n": 0,
            "dropped": 0,
            "kept_unparsed": 0,
            "target_eq_X": 0,
            "target_eq_T": 0,
            "changed_vs_own": 0,
        }
        rng = random.Random(0)
        if v.startswith("bayes_r") or v.startswith("bayescal_r"):
            cal = v.startswith("bayescal_r")
            r = float(v.split("_r")[1])
            p0 = [json.loads(l) for l in open(f"{SH}/p0_{a.model}.jsonl")]
            Tc = json.load(open(f"{SH}/calib_{a.model}.json"))["T_cal"] if cal else 1.0
            stats["T_prior"] = Tc
            # AMENDMENT 3 (Phase 5): calibrated prior p0^(1/T_cal), T_cal fitted on the validation slice
            # (calib_temp.py)
            if cal:
                p0 = [
                    dict(
                        p,
                        p0={
                            o: q ** (1.0 / Tc) / sum(x ** (1.0 / Tc) for x in p["p0"].values())
                            for o, q in p["p0"].items()
                        },
                    )
                    for p in p0
                ]
            assert [p["i"] for p in p0] == [s["i"] for s in sh]
            for s, b, (t, L, span), p in zip(sh, bct, own, p0):
                opts = p["opts"]
                K = len(opts)
                X = s["X"]
                w = [p["p0"][o] * (r if o == X else (1 - r) / (K - 1)) for o in opts]
                Z = sum(w)
                L_new = rng.choices(opts, weights=[x / Z for x in w])[0]
                if L is None:
                    tgt = t
                    stats["kept_unparsed"] += 1
                else:
                    tgt = replace(t, span, L_new)
                    stats["changed_vs_own"] += L_new != L
                out.append({"messages": [b["messages"][0], {"role": "assistant", "content": tgt}]})
                lt = BARE.match(tgt).group(1) if BARE.match(tgt) else None
                stats["target_eq_X"] += lt == X
                stats["target_eq_T"] += lt == s["T"]
        elif v == "goldgated":
            for s, b, (t, L, span) in zip(sh, bct, own):
                if L == s["T"]:
                    tgt = t
                elif s["X"] == s["T"]:
                    tgt = replace(t, span, s["T"]) if L is not None else f"{s['T']} \n"
                    stats["changed_vs_own"] += 1
                else:
                    stats["dropped"] += 1
                    continue
                out.append({"messages": [b["messages"][0], {"role": "assistant", "content": tgt}]})
                stats["target_eq_X"] += s["X"] == s["T"]
                stats["target_eq_T"] += 1
        elif v == "posdiv":
            from ctcue.evaluation.cue_battery_v1 import insert

            for s, b, (t, L, span) in zip(sh, bct, own):
                assert insert(s["clean"], s["cue"], "front") == s["biased"], s["i"]
                pos = rng.choice(["front", "pre", "post", "end"])
                stats[pos] = stats.get(pos, 0) + 1
                out.append(
                    {
                        "messages": [
                            {"role": "user", "content": insert(s["clean"], s["cue"], pos)},
                            b["messages"][1],
                        ]
                    }
                )
                stats["target_eq_X"] += L == s["X"]
                stats["target_eq_T"] += L == s["T"]
        else:
            raise ValueError(v)
        stats["n"] = len(out)
        d = f"{SH}/bct_{v}_{a.model}"
        os.makedirs(d, exist_ok=True)
        assert not os.path.exists(f"{d}/bct_non_cot.jsonl"), f"{d} exists: refusing to overwrite"
        with open(f"{d}/bct_non_cot.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        shutil.copy2(f"{SH}/bct_{a.model}/instruct_samples.jsonl", f"{d}/instruct_samples.jsonl")
        json.dump(stats, open(f"{d}/build_stats.json", "w"), indent=1)
        print(f"{v:12s} -> {d}: {stats}")


if __name__ == "__main__":
    main()
