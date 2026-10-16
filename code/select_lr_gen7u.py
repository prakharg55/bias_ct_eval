#!/usr/bin/env python3
"""
select_lr_gen7u (1 Oct 2026; prereg audit NOTEs 11:19 item 3 + 19:43 MERGE-FIDELITY EXECUTION). The LR selection re-applied to the
UNMERGED validation evaluations of every swept arm of both models (val_eval7g --merge unmerged:
results/pub_val7g/val_<model>_<m>_lr<lr>_s42_unmerged.json; untrained model: val_<model>_none_merged.json = no adapter), F1 on the
GENERATED answers:
  gemma BCT, ACT, AttCT, RMCT, RMCT-anchor: select_lr.select UNCHANGED (pre-registered rule + Amendments 1-2; select_lr's own grids and
    published LRs); degeneracy = the generated answer is one letter on >= 95% of the unbiased prompts of the wrong-cue validation
    items (11:19 note item 3);
  gemma response-only BCT: make_bctro_finals.py's rule (grid 1e-5 .. 3.16e-4 plus any evaluated extension point, best F1 among
    non-degenerate points, one half-decade extension from a grid end; no published-LR tie-break), same degeneracy;
  Llama BCT, ACT, AttCT, RMCT: select_lr7.py's configuration (grids, published LRs) with select_lr.select UNCHANGED; degeneracy =
    Amendment 2's statistic as select_lr7 measured it (fmt_check7 --val: the v7.1 first-position letter is one letter on >= 95% of
    those prompts), here from the per-item first_u of the same unmerged files; the generated-answer version is reported alongside.
A selection that needs an unevaluated extension point reports it (status needs_extension). Compared with the stored selections
(gemma: results/pub_val/selection.json, first-position merged; response-only BCT: results/pub_val5/selection_bctro.json; Llama:
results/pub_val7/selection_llama8b.json, generated merged). CPU, instant. Output results/pub_val7g/selection_unmerged.json.
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, "/data/code")
import select_lr as S, select_lr7
R = "/data/ct_root/results"; V = f"{R}/pub_val7g"
BCTRO_GRID = ["1e-5", "3.16e-5", "1e-4", "3.16e-4"]; BCTRO_EXT = {"1e-5": "3.16e-6", "3.16e-4": "1e-3"}   # make_bctro_finals.py
GEMMA = {"BASE": dict(S.BASE), "PUBLISHED": dict(S.PUBLISHED)}; LLAMA = {"BASE": select_lr7.LLAMA["BASE"], "PUBLISHED": select_lr7.LLAMA["PUBLISHED"]}


def vfile(model, m, lr): return f"{V}/val_{model}_{m}_lr{lr}_s42_unmerged.json"


def modal_share(d, key):
    a = [i[key] for i in d["items"] if i["wrong"]]; c = Counter(a); return max(c.values()) / len(a)


def load_checked(f, model):
    d = json.load(open(f)); assert d["model"] == model and d["merge"] == "unmerged" and d["agg"] == ("sum" if model == "gemma2b" else "max"), f
    return d


def select_arm(model, m):
    deg_key = "gen_u" if model == "gemma2b" else "first_u"
    S.load = lambda mm, lr: load_checked(vfile(model, mm, lr), model) if os.path.exists(vfile(model, mm, lr)) else None
    S.degenerate = lambda mm, lr: bool(modal_share(load_checked(vfile(model, mm, lr), model), deg_key) >= 0.95) if os.path.exists(vfile(model, mm, lr)) else None
    cfg = GEMMA if model == "gemma2b" else LLAMA; S.BASE, S.PUBLISHED = cfg["BASE"], cfg["PUBLISHED"]
    r = S.select(m)
    r["degeneracy_statistic"] = f"modal share of {deg_key} over the unbiased prompts of the wrong-cue validation items >= .95"
    pts = sorted({*r.get("grid", []), *S.BASE[m]}, key=float)
    r["modal_share"] = {lr: {k: modal_share(load_checked(vfile(model, m, lr), model), k) for k in ("gen_u", "first_u")} for lr in pts if os.path.exists(vfile(model, m, lr))}
    r["parsed"] = {lr: load_checked(vfile(model, m, lr), model)["parsed"] for lr in pts if os.path.exists(vfile(model, m, lr))}
    return r


def select_bctro():
    pts = {}
    for lr in BCTRO_GRID + list(BCTRO_EXT.values()):
        f = vfile("gemma2b", "bctro", lr)
        if os.path.exists(f): d = load_checked(f, "gemma2b"); pts[lr] = {"F1": d["F1"], "NS": d["NS"], "ACC": d["ACC"], "flip_given_knew": d["flip_given_knew"], "modal_frac": modal_share(d, "gen_u"),
                                                                       "modal_frac_first": modal_share(d, "first_u"), "parsed": d["parsed"]}
    missing = [lr for lr in BCTRO_GRID if lr not in pts]
    if missing: return {"status": "incomplete", "missing": missing, "points": pts}
    order = sorted(BCTRO_GRID + [lr for lr in BCTRO_EXT.values() if lr in pts], key=float)
    cand = [lr for lr in order if pts[lr]["modal_frac"] < 0.95]
    if not cand: return {"status": "no_valid_lr", "points": pts, "grid": order}
    best = max(cand, key=lambda lr: pts[lr]["F1"])
    need = best in (order[0], order[-1]) and best in BCTRO_EXT and BCTRO_EXT[best] not in pts
    return {"points": pts, "grid": order, "best": best, "status": "needs_extension" if need else "final", "next": BCTRO_EXT[best] if need else None, "selected": None if need else best}


def main():
    out = {"rule": "see docstring", "gemma2b": {}, "llama8b": {}}
    stored = {"gemma2b": json.load(open(f"{R}/pub_val/selection.json")), "llama8b": json.load(open(f"{R}/pub_val7/selection_llama8b.json"))}
    stored_ro = json.load(open(f"{R}/pub_val5/selection_bctro.json"))
    for model in ("gemma2b", "llama8b"):
        uf = f"{V}/val_{model}_none_merged.json"
        if os.path.exists(uf):
            u = json.load(open(uf)); out[model]["untrained"] = {k: u[k] for k in ("F1", "NS", "ACC", "flip_given_knew", "parsed")}
            print(f"[{model}] untrained (generated): F1 {u['F1']:.4f} NS {u['NS']:.4f} ACC {u['ACC']:.4f}")
        for m in (GEMMA if model == "gemma2b" else LLAMA)["BASE"]:
            r = select_arm(model, m); old = stored[model].get(m, {}).get("selected") if isinstance(stored[model].get(m), dict) else None
            r["stored_selection"] = old; r["changed"] = (r.get("selected") != old) if r.get("status") == "final" else None
            out[model][m] = r
            rows = " ".join(f"{lr}:{v['F1']:.3f}{'D' if v['degenerate'] else ''}" for lr, v in r.get("rows", {}).items())
            print(f"[{model}] {m:6s}: status {r['status']}, selected {r.get('selected')} (best {r.get('best')}; tied {r.get('tied')}; extensions {r.get('extensions')}); stored {old}; "
                  f"changed {r.get('changed')} | {rows}" + (f" | missing {r.get('missing')}" if r.get("missing") else "") + (f" | NEXT {r.get('next')}" if r.get("next") else ""))
        if model == "gemma2b":
            r = select_bctro(); r["stored_selection"] = stored_ro.get("selected"); r["changed"] = (r["selected"] != r["stored_selection"]) if r.get("status") == "final" else None
            out[model]["bctro"] = r
            print(f"[gemma2b] bctro : status {r['status']}, selected {r.get('selected')} (best {r.get('best')}); stored {r['stored_selection']}; changed {r['changed']} | "
                  + " ".join(f"{lr}:{p['F1']:.3f}{'D' if p['modal_frac'] >= .95 else ''}" for lr, p in sorted(r["points"].items(), key=lambda kv: float(kv[0])))
                  + (f" | missing {r.get('missing')}" if r.get("missing") else "") + (f" | NEXT {r.get('next')}" if r.get("next") else ""))
    json.dump(out, open(f"{V}/selection_unmerged.json", "w"), indent=1); print("saved ->", f"{V}/selection_unmerged.json")


if __name__ == "__main__":
    main()
