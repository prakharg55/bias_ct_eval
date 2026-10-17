#!/usr/bin/env python3
"""
utags (1 Oct 2026; prereg audit NOTEs 11:19 item 3 + 19:43 MERGE-FIDELITY EXECUTION): the checkpoint naming of the UNMERGED results, in
one place, so that every unmerged scorer / view / table picks the same files.
For each (model, arm): if the unmerged LR re-selection (select_lr_gen7u.py -> results/pub_val7g/selection_unmerged.json) CHANGED the arm's
selected LR, the arm's primary results are the NEW-SELECTION runs (seed 42 = the sweep run at the new LR; seeds 43/44 trained at it), whose
instrument outputs carry the infix "7u"; otherwise the original finals and their original tags. An arm whose re-selection is not final
(incomplete / needs an extension) raises, so that nothing is scored on an unresolved selection.
  eval tag   (results/pub7u, pub7uc, fs7/fs_<tag>_u.json):  <model>_final_<arm>_s<seed>   | <model>_final7u_<arm>_s<seed>
  instrument (results/w7u gen_ / probe_ / causal3_ / route_): <model>_<arm>_s<seed>       | <model>_<arm>7u_s<seed>
  relapse    (results/relapse7u/relapse_<tag>.json):          <pre>_<arm>_s<seed>          | <pre>_<arm>7u_s<seed>   (pre: gemma | llama8b)
  adapter:   models/published[_llama]/final[7]/<arm>_s<seed>/step1 | models/published[_llama]/final7u/<arm>_s<seed>/step1
Arms outside the LR sweeps keep their own names: gemma fix variants (eval gemma2b_fix_<v>_s<s>, instruments gemma2b_fix_<v>_s<s>), gemma
response-only BCT (gemma2b_bctro_s<s>), Llama fix variants (llama8b_fix_<v>_s<s>; Bayes = fix_bayes6_r0.8), self-training control
(<model>_selftrain_s<s>). They are defined at BCT's selected LR: if BCT's selection changes, they are NOT re-mapped here (raises).
"""
import json
R = "/data/ct_root/results"; M = "/data/ct_root/models"
SELU = f"{R}/pub_val7g/selection_unmerged.json"
SWEPT = {"gemma2b": ("bct", "act", "attct", "rmct", "rmcta"), "llama8b": ("bct", "act", "attct", "rmct")}
_SEL = {}


def selection():
    if not _SEL: _SEL.update(json.load(open(SELU)))
    return _SEL


def changed(model, arm):
    """True iff the unmerged re-selection of a swept arm is final and differs from the stored selection; raises if not final."""
    if arm not in SWEPT[model]: return False
    r = selection()[model][arm]
    if r.get("status") != "final": raise RuntimeError(f"unmerged LR re-selection of {model} {arm} is not final: {r.get('status')} {r.get('missing') or r.get('next')}")
    return bool(r["changed"])


def _guard_bct_derived(model, arm):
    """Fix variants and the self-training control are trained at BCT's selected LR; response-only BCT has its own selection (bctro)."""
    if (arm.startswith("fix_") or arm == "selftrain") and changed(model, "bct"):
        raise RuntimeError(f"{model} {arm} is defined at BCT's selected LR, which changed: its new-LR runs are not mapped yet")
    if arm == "bctro":
        r = selection()[model]["bctro"]
        if r.get("status") != "final": raise RuntimeError(f"unmerged LR re-selection of {model} bctro is not final: {r.get('status')}")
        if r["changed"]: raise RuntimeError(f"{model} bctro: the unmerged re-selection changed its LR ({r['stored_selection']} -> {r['selected']}): new runs not mapped yet")


def eval_tag(model, arm, seed):
    if arm is None: return f"{model}_final_none"
    _guard_bct_derived(model, arm)
    if arm in SWEPT[model]: return f"{model}_final{'7u' if changed(model, arm) else ''}_{arm}_s{seed}"
    return f"{model}_{arm}_s{seed}"   # fix_<v>, bctro, selftrain


def inst_tag(model, arm, seed):
    if arm is None: return f"{model}_none"
    _guard_bct_derived(model, arm)
    if arm in SWEPT[model] and changed(model, arm): return f"{model}_{arm}7u_s{seed}"
    return f"{model}_{arm}_s{seed}"


def relapse_tag(model, arm, seed):
    pre = "gemma" if model == "gemma2b" else "llama8b"
    if arm is None: return f"{pre}_none"
    return f"{pre}_{arm}{'7u' if arm in SWEPT[model] and changed(model, arm) else ''}_s{seed}"


def adapter(model, arm, seed):
    root = f"{M}/published" if model == "gemma2b" else f"{M}/published_llama"
    if arm in SWEPT[model]:
        return f"{root}/final7u/{arm}_s{seed}/step1" if changed(model, arm) else f"{root}/{'final' if model == 'gemma2b' else 'final7'}/{arm}_s{seed}/step1"
    raise KeyError(arm)


def summary():
    out = {}
    for model, arms in SWEPT.items():
        for a in arms:
            try: out[f"{model}:{a}"] = {"changed": changed(model, a), "selected": selection()[model][a].get("selected"), "stored": selection()[model][a].get("stored_selection")}
            except RuntimeError as e: out[f"{model}:{a}"] = {"error": str(e)}
    return out


if __name__ == "__main__":
    for k, v in summary().items(): print(k, v)
