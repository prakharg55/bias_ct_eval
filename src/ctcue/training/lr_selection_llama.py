#!/usr/bin/env python3
"""
LR selection v7 (30 Sep 2026) = select_lr5.py reading the v7 outputs for Llama (val_eval7 generated answers, results/pub_val7; fmt_check7 results/pub7;
untrained file val_llama8b_none.json); gemma branch unchanged.
LR selection v5 (27 Sep 2026): select_lr.py's rule (pre-registered + Amendments 1-2) with its select() UNCHANGED, for
--model gemma2b (select_lr.py's own grids and files; must reproduce results/pub_val/selection.json exactly) or llama8b
(Phase 5 C, pre-registered 14:21 EDT): val_eval5 -> results/pub_val5/val_llama8b_<m>_lr<lr>_s42.json, fmt_check5 --val ->
results/pub5/fmt_val_llama8b_<m>_lr<lr>_s42.json; grids BCT/ACT/AttCT {1.58e-6 .. 1.58e-4}, RMCT {3.16e-6 .. 2.86e-4};
published LRs (tie-break) = the repo's native Llama configs (5e-6) and RMCT's paper LR 2.86e-4.
Writes selection_<model>.json next to the val files (never touches select_lr.py's selection.json). CPU, instant.
"""
import argparse, json, os, sys
sys.path.insert(0, "/data/code")
import select_lr as S
C = "/data/ct_root/results"
LLAMA = {"V": f"{C}/pub_val7", "FMT": f"{C}/pub7", "PUBLISHED": {"bct": "5e-6", "act": "5e-6", "attct": "5e-6", "rmct": "2.86e-4"},
         "BASE": {m: ["1.58e-6", "5e-6", "1.58e-5", "5e-5", "1.58e-4"] for m in ("bct", "act", "attct")} | {"rmct": ["3.16e-6", "1e-5", "3.16e-5", "1e-4", "2.86e-4"]}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["gemma2b", "llama8b"]); a = ap.parse_args()
    if a.model == "llama8b":
        S.V, S.FMT, S.PUBLISHED, S.BASE = LLAMA["V"], LLAMA["FMT"], LLAMA["PUBLISHED"], LLAMA["BASE"]
        def load(m, lr):
            f = f"{S.V}/val_llama8b_{m}_lr{lr}_s42.json"
            return json.load(open(f)) if os.path.exists(f) else None
        def degenerate(m, lr):
            f = f"{S.FMT}/fmt_val_llama8b_{m}_lr{lr}_s42.json"
            if not os.path.exists(f): return None
            d = json.load(open(f)); assert d.get("items") == "val" and d.get("model") == "llama8b", f
            return bool(d["per_cue"]["suggested_answer"]["modal_frac"]["unbiased"] >= 0.95)
        S.load, S.degenerate = load, degenerate
        untrained = f"{S.V}/val_llama8b_none.json"
    else:
        untrained = f"{S.V}/val_none.json"
    out = {}
    if os.path.exists(untrained):
        b = json.load(open(untrained)); out["untrained"] = b
        print(f"untrained: F1 {b['F1']:.4f} (SE {S.f1_se(b):.4f})  NS {b['NS']:.4f}  ACC {b['ACC']:.4f}  flip|knew {b['flip_given_knew']:.3f}")
    for m in S.BASE:
        s = S.select(m); out[m] = s
        if s["status"] not in ("final",) and "rows" not in s:
            print(f"=> {m}: {s['status']} {s.get('missing') or s.get('next')}"); continue
        if s["status"] == "no_valid_lr":
            print(f"=> {m}: NO VALID LR - every point degenerate: {s['degenerate']}"); continue
        for lr in s["grid"]:
            r = s["rows"][lr]
            print(f"  {m:5s} lr {lr:>8s}: F1 {r['F1']:.4f} (SE {r['F1_se']:.4f})  NS {r['NS']:.4f}  ACC {r['ACC']:.4f}  flip|knew {r['flip_given_knew']:.3f}"
                  + f"  degenerate {r['degenerate']}" + ("  <- best" if lr == s["best"] else "") + ("  <- SELECTED" if lr == s["selected"] else ""))
        print(f"=> {m}: {'selected' if s['status'] == 'final' else 'PROVISIONAL (' + str(s['missing']) + ')'} {s['selected']} (best {s['best']} F1 {s['best_F1']:.4f}; "
              f"tied {s['tied']}; extensions {s['extensions']})" + (f"; draft rule would give {s['draft_rule_alternative']}" if "draft_rule_alternative" in s else ""))
    json.dump(out, open(f"{S.V}/selection_{a.model}.json", "w"), indent=1)
    if a.model == "gemma2b":   # regression: identical to select_lr.py's own output
        ref = json.load(open(f"{S.V}/selection.json")); same = json.dumps(ref, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("REGRESSION vs selection.json:", "IDENTICAL" if same else "DIFFERENT"); assert same


if __name__ == "__main__":
    main()
