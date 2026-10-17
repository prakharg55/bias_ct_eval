#!/usr/bin/env python3
"""
LR selection for the published-code arms. Rule as pre-registered (PREREGISTERED_PREDICTIONS.md, 26 Sep 2026 23:47 EDT):
  "Best F1 at a grid end -> extend one half-decade that way (at most twice); |dF1| < 0.005 ties go to the published LR."
AMENDMENT 2 (02:31:51, before any RMCT extension result): degenerate checkpoints (one letter on >= 95% of unbiased val
prompts) are excluded; an all-degenerate grid extends downward (select() docstring).
Implementation (27 Sep 00:27, rewritten before any RMCT result and before BCT's extension result; the 00:06 draft applied the
edge test to the tie-broken pick instead of to the best F1, contrary to the text):
  1. Start from the pre-registered base grid. While the BEST validation F1 sits at a grid end (at most twice), the next
     half-decade point that way joins the grid; if it has not been evaluated yet, selection stops and reports it.
     Extension points evaluated but never triggered are ignored.
  2. On the final grid: tied = {LR : best F1 - F1 < 0.005}. The published LR if tied, else the best-F1 LR.
     (The draft's alternative for an untied published LR, the tied LR closest to it in log10, is reported when it differs.)
F1 standard errors (delta method) are printed so the noise in the choice is visible. CPU, instant. Writes selection.json.
"""
import json, math, os
V = "/data/ct_root/results/pub_val"
PUBLISHED = {"bct": "1e-6", "act": "1e-6", "attct": "5e-6", "rmct": "2.86e-4", "rmcta": "2.86e-4"}   # rmcta = EXPLORATORY RMCT-anchor (02:07 addendum)
BASE = {"bct": ["1e-6", "3.16e-6", "1e-5", "3.16e-5", "1e-4"], "act": ["1e-6", "3.16e-6", "1e-5", "3.16e-5", "1e-4"],
        "attct": ["5e-6", "1.58e-5", "5e-5", "1.58e-4", "5e-4"], "rmct": ["3.16e-5", "1e-4", "2.86e-4", "5e-4", "1e-3"],
        "rmcta": ["3.16e-5", "1e-4", "2.86e-4"]}
LADDERS = [["1e-8", "3.16e-8", "1e-7", "3.16e-7", "1e-6", "3.16e-6", "1e-5", "3.16e-5", "1e-4", "3.16e-4", "1e-3", "3.16e-3", "1e-2"],
           ["5e-8", "1.58e-7", "5e-7", "1.58e-6", "5e-6", "1.58e-5", "5e-5", "1.58e-4", "5e-4", "1.58e-3", "5e-3"],
           ["2.86e-5", "9.04e-5", "2.86e-4", "9.04e-4", "2.86e-3"]]   # exact half-decades through RMCT's published 2.86e-4


def half_decade(lr, up):
    for lad in LADDERS:
        if lr in lad:
            i = lad.index(lr) + (1 if up else -1); assert 0 <= i < len(lad); return lad[i]
    raise ValueError(lr)


def f1_se(d):
    """Delta-method SE of F1 = 2 NS ACC / (NS + ACC), NS and ACC treated as independent binomial proportions."""
    ns, acc, nw, n = d["NS"], d["ACC"], d["n_wrong_cue"], d["n_items"]
    if ns + acc == 0: return float("nan")
    dns = 2 * acc ** 2 / (ns + acc) ** 2; dacc = 2 * ns ** 2 / (ns + acc) ** 2
    return math.sqrt(dns ** 2 * ns * (1 - ns) / nw + dacc ** 2 * acc * (1 - acc) / n)


def load(m, lr):
    f = f"{V}/val_{m}_lr{lr}_s42.json"
    return json.load(open(f)) if os.path.exists(f) else None


FMT = "/data/ct_root/results/pub"


def degenerate(m, lr):
    """AMENDMENT 2 (27 Sep 02:31:51 EDT): True if the letter readout is one letter on >= 95% of the unbiased validation prompts
    (fmt_check2.py --val, modal_frac), False if measured below, None if not measured yet."""
    f = f"{FMT}/fmt_val_{m}_lr{lr}_s42.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); assert d.get("items") == "val", f
    return bool(d["per_cue"]["suggested_answer"]["modal_frac"]["unbiased"] >= 0.95)


def select(m):
    """Pre-registered rule + Amendment 2: degenerate checkpoints are excluded; once any point of an arm is degenerate, every
    point of that arm must be measured; an all-degenerate grid extends DOWNWARD; at most two extensions in total; the final
    pick (and the best, if different) must be measured non-degenerate."""
    grid = list(BASE[m]); res = {lr: load(m, lr) for lr in grid}; ext = []; pub = PUBLISHED[m]
    missing = [lr for lr in grid if res[lr] is None]
    if missing: return {"status": "incomplete", "missing": missing}
    while True:
        order = sorted(grid, key=float); deg = {lr: degenerate(m, lr) for lr in order}
        if any(v is True for v in deg.values()) and any(v is None for v in deg.values()):
            return {"status": "needs_fmt_check", "missing": [lr for lr in order if deg[lr] is None], "grid": order, "extensions": ext}
        cand = [lr for lr in order if deg[lr] is not True]
        if cand:
            best = max(cand, key=lambda lr: res[lr]["F1"])
            if best not in (order[0], order[-1]) or len(ext) == 2: break
            up, frm = best == order[-1], best
        else:
            if len(ext) == 2:
                return {"status": "no_valid_lr", "grid": order, "extensions": ext, "degenerate": deg,
                        "rows": {lr: {k: res[lr][k] for k in ("F1", "NS", "ACC", "flip_given_knew")} for lr in order}}
            up, frm = False, order[0]
        nxt = half_decade(frm, up=up); r = load(m, nxt)
        if r is None: return {"status": "needs_extension", "next": nxt, "grid": order, "extensions": ext, "all_degenerate": not cand}
        grid.append(nxt); res[nxt] = r; ext.append(nxt)
    bF1 = res[best]["F1"]; tied = [lr for lr in cand if bF1 - res[lr]["F1"] < 0.005]
    pick = pub if pub in tied else best
    unverified = sorted({lr for lr in (pick, best) if deg[lr] is None}, key=float)
    alt = min(tied, key=lambda lr: abs(math.log10(float(lr)) - math.log10(float(pub))))
    out = {"status": "final" if not unverified else "needs_fmt_check", "missing": unverified, "selected": pick, "selected_float": float(pick),
           "best": best, "best_F1": bF1, "tied": tied, "grid": order, "extensions": ext, "degenerate": deg,
           "edge_after_extensions": best in (order[0], order[-1]),
           "rows": {lr: {k: res[lr][k] for k in ("F1", "NS", "ACC", "flip_given_knew")} | {"F1_se": f1_se(res[lr]), "degenerate": deg[lr]} for lr in order}}
    if alt != pick: out["draft_rule_alternative"] = alt
    return out


def main():
    out = {}
    b = json.load(open(f"{V}/val_none.json")) if os.path.exists(f"{V}/val_none.json") else None
    if b:
        out["untrained"] = b; print(f"untrained: F1 {b['F1']:.4f} (SE {f1_se(b):.4f})  NS {b['NS']:.4f}  ACC {b['ACC']:.4f}  flip|knew {b['flip_given_knew']:.3f}")
    for m in BASE:
        s = select(m); out[m] = s
        if s["status"] not in ("final",) and "rows" not in s:
            print(f"=> {m}: {s['status']} {s.get('missing') or s.get('next')}"); continue
        if s["status"] == "no_valid_lr":
            print(f"=> {m}: NO VALID LR - every point degenerate: {s['degenerate']}"); continue
        for lr in s["grid"]:
            r = s["rows"][lr]
            print(f"  {m:5s} lr {lr:>8s}: F1 {r['F1']:.4f} (SE {r['F1_se']:.4f})  NS {r['NS']:.4f}  ACC {r['ACC']:.4f}  flip|knew {r['flip_given_knew']:.3f}"
                  + f"  degenerate {r['degenerate']}" + ("  <- best" if lr == s["best"] else "") + ("  <- SELECTED" if lr == s["selected"] else ""))
        print(f"=> {m}: {'selected' if s['status'] == 'final' else 'PROVISIONAL (degeneracy not yet measured for ' + str(s['missing']) + ')'} {s['selected']} (best {s['best']} F1 {s['best_F1']:.4f}; tied {s['tied']}; extensions {s['extensions']})"
              + (f"; draft rule would give {s['draft_rule_alternative']}" if "draft_rule_alternative" in s else ""))
    json.dump(out, open(f"{V}/selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
