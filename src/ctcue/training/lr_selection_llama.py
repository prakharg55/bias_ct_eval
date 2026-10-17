"""The earlier Llama learning-rate selection, with lr_rule.py's rule and the Llama grids and reference rates.

It reads the stored validation results of the merged Llama sweep checkpoints, scored on generated
answers. The Gemma branch is lr_rule.py's and must give the same result. select_learning_rates.py,
which makes the selection used in the paper, reuses the Llama grids and reference rates.

Usage: python -m ctcue.training.lr_selection_llama --model llama8b
Output: $CT_ROOT/results/pub_val7/selection_llama8b.json
"""

import argparse, json, os

from ctcue.training import lr_rule as S
from ctcue.paths import CT_ROOT

C = f"{CT_ROOT}/results"
LLAMA = {
    "V": f"{C}/pub_val7",
    "FMT": f"{C}/pub7",
    "PUBLISHED": {"bct": "5e-6", "act": "5e-6", "attct": "5e-6", "rmct": "2.86e-4"},
    "BASE": {m: ["1.58e-6", "5e-6", "1.58e-5", "5e-5", "1.58e-4"] for m in ("bct", "act", "attct")}
    | {"rmct": ["3.16e-6", "1e-5", "3.16e-5", "1e-4", "2.86e-4"]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gemma2b", "llama8b"])
    a = ap.parse_args()
    if a.model == "llama8b":
        S.V, S.FMT, S.PUBLISHED, S.BASE = LLAMA["V"], LLAMA["FMT"], LLAMA["PUBLISHED"], LLAMA["BASE"]

        def load(m, lr):
            f = f"{S.V}/val_llama8b_{m}_lr{lr}_s42.json"
            return json.load(open(f)) if os.path.exists(f) else None

        def degenerate(m, lr):
            f = f"{S.FMT}/fmt_val_llama8b_{m}_lr{lr}_s42.json"
            if not os.path.exists(f):
                return None
            d = json.load(open(f))
            assert d.get("items") == "val" and d.get("model") == "llama8b", f
            return bool(d["per_cue"]["suggested_answer"]["modal_frac"]["unbiased"] >= 0.95)

        S.load, S.degenerate = load, degenerate
        untrained = f"{S.V}/val_llama8b_none.json"
    else:
        untrained = f"{S.V}/val_none.json"
    out = {}
    if os.path.exists(untrained):
        b = json.load(open(untrained))
        out["untrained"] = b
        print(
            f"untrained: F1 {b['F1']:.4f} (SE {S.f1_se(b):.4f})  NS {b['NS']:.4f}  ACC {b['ACC']:.4f}  flip|knew {b['flip_given_knew']:.3f}"
        )
    for m in S.BASE:
        s = S.select(m)
        out[m] = s
        if s["status"] not in ("final",) and "rows" not in s:
            print(f"=> {m}: {s['status']} {s.get('missing') or s.get('next')}")
            continue
        if s["status"] == "no_valid_lr":
            print(f"=> {m}: NO VALID LR - every point degenerate: {s['degenerate']}")
            continue
        for lr in s["grid"]:
            r = s["rows"][lr]
            print(
                f"  {m:5s} lr {lr:>8s}: F1 {r['F1']:.4f} (SE {r['F1_se']:.4f})  NS {r['NS']:.4f}  ACC {r['ACC']:.4f}  flip|knew {r['flip_given_knew']:.3f}"
                + f"  degenerate {r['degenerate']}"
                + ("  <- best" if lr == s["best"] else "")
                + ("  <- SELECTED" if lr == s["selected"] else "")
            )
        print(
            f"=> {m}: {'selected' if s['status'] == 'final' else 'PROVISIONAL (' + str(s['missing']) + ')'} {s['selected']} (best {s['best']} F1 {s['best_F1']:.4f}; "
            f"tied {s['tied']}; extensions {s['extensions']})"
            + (
                f"; draft rule would give {s['draft_rule_alternative']}"
                if "draft_rule_alternative" in s
                else ""
            )
        )
    json.dump(out, open(f"{S.V}/selection_{a.model}.json", "w"), indent=1)
    if a.model == "gemma2b":  # regression: identical to lr_rule.py's own output
        ref = json.load(open(f"{S.V}/selection.json"))
        same = json.dumps(ref, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("REGRESSION vs selection.json:", "IDENTICAL" if same else "DIFFERENT")
        assert same


if __name__ == "__main__":
    main()
