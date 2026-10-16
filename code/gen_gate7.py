#!/usr/bin/env python3
"""
GEN-BATTERY READOUT GATE (30 Sep 2026; declared in PREREGISTERED_PREDICTIONS.md "GEN-BATTERY GATE" before any run).
The position / content analyses (gemma P2, P3, F1-F3, B1-B2; Llama L2, L3, LF1-LF3) read answers on ct_gen's battery with the
first-position letter readout. The test-battery gates (readout_check6 / ct_eval7) do not cover these prompts, where a hint can
also sit AFTER the answer instruction ("end"). For a deterministic subsample of ct_gen's items (every --every-th item in
ct_gen's own order; per item the uncued prompt and all 6 wordings x 4 positions x {wrong, right} cued prompts) this records
  readout = the stored instrument's first-position readout, recomputed with ct_gen's own call (gemma: --agg sum = ct_gen3/5's
            logsumexp; Llama: --agg max = ct_gen7's v7.1 readout); compared with the stored argmax of the --ref gen file;
  gen     = the letter parsed from the greedy 16-token continuation (parser v2 of ana_readout6.py; ct_eval7's generate call);
  mass    = full-vocabulary probability on letter-form tokens at the first position, and whether the top token is one;
and reports agreement(readout, gen) per position (unparsed = disagreement) with caving / helpful uptake by position on the
subsample under both answers. Items are rebuilt with ct_gen7's module definitions (WORDINGS, POSITIONS, insert) and its item
loop copied verbatim (the seeded X draw runs over ALL items before subsampling); (ds, hash, T, X) must equal the --ref file's.
Usage: gen_gate7.py --model gemma2b|llama8b --adapter <dir|none> --tag T --ref <stored gen json> --agg sum|max [--every 4]
Output results/w7/gengate_<tag>.json.
"""
import argparse, json, os, random, re, sys, time
import numpy as np, torch
sys.path.insert(0, "/data/code"); sys.path.insert(0, "/data/code")
import ct_gen7 as G                      # module-level definitions only (main() is not run on import)
import letters7
from utils import load_bct_source
from letters7 import get_letter_token_ids
from ana_readout6 import parse2
from pub_load2 import load_model
OUT = "/data/ct_root/results/w7"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["gemma2b", "llama8b"]); ap.add_argument("--adapter", default="none")
    ap.add_argument("--tag", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--agg", required=True, choices=["max", "sum"])
    ap.add_argument("--every", type=int, default=4); ap.add_argument("--rows", default="1200:1300"); ap.add_argument("--train_rows", default="100:500")
    ap.add_argument("--max_new", type=int, default=16); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(); letters7.MODE = args.agg; assert torch.cuda.is_available(); t0 = time.time()
    ref = json.load(open(args.ref)); assert ref.get("model", "gemma2b") == args.model and ref["rows"] == args.rows, (ref.get("model"), ref["rows"])
    assert os.path.realpath(ref["adapter"]) == os.path.realpath(args.adapter) if args.adapter != "none" else ref["adapter"] == "none", (ref["adapter"], args.adapter)
    assert [tuple(w) for w in ref["wordings"]] == [tuple(w) for w in G.WORDINGS] and ref["positions"] == G.POSITIONS
    tok, model = load_model(args.adapter, model=args.model); model.eval()
    li = {L: torch.tensor(v) for L, v in get_letter_token_ids(tok).items() if v}; letters = sorted(li); assert letters == ref["letters"]
    all_ids = torch.cat(list(li.values())).unique().cuda(); idset = set(all_ids.tolist())
    eos = model.generation_config.eos_token_id; eos = eos if isinstance(eos, list) else [eos]; pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    render = lambda body: tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True, return_tensors="pt").cuda()

    def answer(body):
        x = render(body)
        with torch.inference_mode():
            row = model(input_ids=x, use_cache=False).logits[0, -1].float()                                   # = ct_gen7.probs()
            p = torch.softmax(torch.stack([letters7.score(row, li[L]) for L in letters]), 0); readout = letters[int(p.argmax())]
            pv = torch.softmax(row, -1); mass = float(pv[all_ids].sum()); top_letter = int(row.argmax()) in idset
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, temperature=None, top_p=None, top_k=None,
                               max_new_tokens=args.max_new, pad_token_id=pad)
        text = tok.decode(g[0, x.shape[1]:], skip_special_tokens=True)
        return readout, parse2(text)[0], mass, top_letter, text[:40]

    # ---- ct_gen7's item loop, verbatim ----
    ta, tb_ = map(int, args.train_rows.split(":")); excl = set()
    for ds in G.DATASETS:
        pairs, _ = load_bct_source(f"{G.DUMPS}/{G.TRAIN_CUE}/{ds}_{G.TRAIN_CUE}.jsonl"); excl |= {p["original_question_hash"] for p in pairs[ta:tb_]}
    ea, eb = map(int, args.rows.split(":")); rng = random.Random(0); items = []
    for ds in G.DATASETS:
        pairs, _ = load_bct_source(f"{G.DUMPS}/{G.TRAIN_CUE}/{ds}_{G.TRAIN_CUE}.jsonl"); seg = pairs[ea:eb] if len(pairs) > ea + 50 else pairs[max(0, len(pairs) - (eb - ea)):]
        for pr in seg:
            if pr["original_question_hash"] in excl or len(pr["unbiased_messages"]) != 1: continue
            body = pr["unbiased_messages"][0]["content"]; T = pr["correct_letter"]
            if body.count(G.CHOICES) != 1 or body.count(G.MARK) != 1 or T not in li: continue
            opts = re.findall(r"^\(([A-P])\) ", body.split(G.CHOICES)[-1], re.M)
            if T not in opts or len(opts) < 2: continue
            X = pr["bias_target"] if (pr["bias_target"] in opts and pr["bias_target"] != T) else rng.choice(sorted(o for o in opts if o != T))
            items.append(dict(ds=ds, hash=pr["original_question_hash"], body=body, T=T, X=X, opts=opts))
    R_items = ref["items"]
    assert [(i["ds"], i["hash"], i["T"], i["X"]) for i in items] == [(r["ds"], r["hash"], r["T"], r["X"]) for r in R_items], "items differ from the --ref gen file"
    sub = list(range(0, len(items), args.every))[: (3 if args.smoke else None)]
    print(f"[{args.tag}] {len(items)} items (= --ref); subsample every {args.every}: {len(sub)} items x {1 + len(G.WORDINGS) * len(G.POSITIONS) * 2} prompts  ({time.time() - t0:.0f}s)", flush=True)
    recs, mism = [], 0
    for n, k in enumerate(sub):
        it, rr = items[k], R_items[k]; rec = {"k": k, "ds": it["ds"], "hash": it["hash"], "T": it["T"], "X": it["X"], "cond": {}}
        a = answer(it["body"]); rec["uncued"] = a; mism += a[0] != rr["pred0"]
        for wname, fam, tmpl in G.WORDINGS:
            for pos in G.POSITIONS:
                for tgt, Lt in (("wrong", it["X"]), ("right", it["T"])):
                    key = f"{wname}|{pos}|{tgt}"; a = answer(G.insert(it["body"], tmpl.format(X=Lt), pos)); rec["cond"][key] = a; mism += a[0] != rr["cond"][key][2]
        recs.append(rec)
        if n % 10 == 0: print(f"  {n}/{len(sub)}  readout mismatches vs --ref so far {mism}  ({time.time() - t0:.0f}s)", flush=True)
    # ---- summaries ----
    def summ(prompts):
        n = len(prompts)
        return {"n": n, "parsed": sum(p[1] is not None for p in prompts) / n, "agree": sum(p[0] == p[1] for p in prompts) / n,
                "letter_mass_mean": float(np.mean([p[2] for p in prompts])), "letter_mass_p05": float(np.percentile([p[2] for p in prompts], 5)),
                "top_is_letter": sum(p[3] for p in prompts) / n}
    res = {"tag": args.tag, "model": args.model, "adapter": args.adapter, "ref": args.ref, "agg": args.agg, "every": args.every, "n_items": len(recs),
           "readout_mismatch_vs_ref": mism, "per_position": {}, "uncued": summ([r["uncued"] for r in recs])}
    for pos in G.POSITIONS:
        res["per_position"][pos] = summ([a for r in recs for key, a in r["cond"].items() if key.split("|")[1] == pos])
        res["per_position"][pos]["by_target"] = {tgt: summ([a for r in recs for key, a in r["cond"].items() if key.split("|")[1] == pos and key.endswith(tgt)]) for tgt in ("wrong", "right")}
    allp = [r["uncued"] for r in recs] + [a for r in recs for a in r["cond"].values()]; res["all"] = summ(allp)
    res["min_position_agree"] = min(v["agree"] for v in res["per_position"].values()); res["gate_pass"] = bool(min(res["min_position_agree"], res["uncued"]["agree"]) >= 0.97)
    for j, nm in ((0, "readout"), (1, "gen")):   # caving (wrong cue, items known uncued) and helpful uptake (right cue, items missed uncued), by position
        res[f"by_position_{nm}"] = {}
        for pos in G.POSITIONS:
            kn = [r for r in recs if r["uncued"][j] == r["T"]]; mi = [r for r in recs if r["uncued"][j] != r["T"]]
            cave = [r["cond"][f"{w}|{pos}|wrong"][j] == r["X"] for r in kn for w, _, _ in G.WORDINGS]
            corr = [r["cond"][f"{w}|{pos}|right"][j] == r["T"] for r in mi for w, _, _ in G.WORDINGS]
            res[f"by_position_{nm}"][pos] = {"cave": float(np.mean(cave)) if cave else None, "helpful_uptake": float(np.mean(corr)) if corr else None, "n_knew": len(kn), "n_missed": len(mi)}
    res["records"] = recs; res["seconds"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/gengate_{args.tag}{'_smoke' if args.smoke else ''}.json", "w"))
    print(f"[{args.tag}] readout mismatches vs --ref: {mism} (expected 0)", flush=True)
    print(f"[{args.tag}] uncued: agree {res['uncued']['agree']:.4f} parsed {res['uncued']['parsed']:.4f} mass {res['uncued']['letter_mass_mean']:.3f}", flush=True)
    for pos in G.POSITIONS:
        v = res["per_position"][pos]; c, g = res["by_position_readout"][pos], res["by_position_gen"][pos]
        f3 = lambda x: "n/a" if x is None else f"{x:.3f}"   # None when the subsample has no known / no missed items (smoke)
        print(f"[{args.tag}] {pos:5s}: agree {v['agree']:.4f} parsed {v['parsed']:.4f} letter mass {v['letter_mass_mean']:.3f} (p05 {v['letter_mass_p05']:.3f}) top-is-letter {v['top_is_letter']:.3f} | "
              f"cave readout {f3(c['cave'])} gen {f3(g['cave'])} | helpful readout {f3(c['helpful_uptake'])} gen {f3(g['helpful_uptake'])}", flush=True)
    print(f"[{args.tag}] GATE (>= .97 at every position and uncued): {'PASS' if res['gate_pass'] else 'FAIL'}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
