"""Map a model, method and seed to the names of its result files and, for swept methods, its adapter path.

If the final learning-rate selection (training/select_learning_rates.py) changed the learning rate
of a swept method, its results come from the runs at the new rate, whose names contain "7u".
Otherwise they come from the original final runs. The BCT variants, response-only BCT and the
self-training control keep their own names. They have no runs at a new rate, so the module raises
an error if their rate changed. It also raises an error if a selection it needs is not final. Most
scripts that read the final results get their file names from here. Run as a script, it prints
whether each swept method's rate changed.

Usage: python -m ctcue.common.checkpoints
"""

import json
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
M = f"{CT_ROOT}/models"
SELU = f"{R}/pub_val7g/selection_unmerged.json"
SWEPT = {"gemma2b": ("bct", "act", "attct", "rmct", "rmcta"), "llama8b": ("bct", "act", "attct", "rmct")}
_SEL = {}


def selection():
    if not _SEL:
        _SEL.update(json.load(open(SELU)))
    return _SEL


def changed(model, arm):
    """True iff the unmerged re-selection of a swept arm is final and differs from the stored selection; raises if not final."""
    if arm not in SWEPT[model]:
        return False
    r = selection()[model][arm]
    if r.get("status") != "final":
        raise RuntimeError(
            f"unmerged LR re-selection of {model} {arm} is not final: {r.get('status')} {r.get('missing') or r.get('next')}"
        )
    return bool(r["changed"])


def _guard_bct_derived(model, arm):
    """Fix variants and the self-training control are trained at BCT's selected LR; response-only BCT has its own selection (bctro)."""
    if (arm.startswith("fix_") or arm == "selftrain") and changed(model, "bct"):
        raise RuntimeError(
            f"{model} {arm} is defined at BCT's selected LR, which changed: its new-LR runs are not mapped yet"
        )
    if arm == "bctro":
        r = selection()[model]["bctro"]
        if r.get("status") != "final":
            raise RuntimeError(f"unmerged LR re-selection of {model} bctro is not final: {r.get('status')}")
        if r["changed"]:
            raise RuntimeError(
                f"{model} bctro: the unmerged re-selection changed its LR ({r['stored_selection']} -> {r['selected']}): new runs not mapped yet"
            )


def eval_tag(model, arm, seed):
    if arm is None:
        return f"{model}_final_none"
    _guard_bct_derived(model, arm)
    if arm in SWEPT[model]:
        return f"{model}_final{'7u' if changed(model, arm) else ''}_{arm}_s{seed}"
    return f"{model}_{arm}_s{seed}"  # fix_<v>, bctro, selftrain


def inst_tag(model, arm, seed):
    if arm is None:
        return f"{model}_none"
    _guard_bct_derived(model, arm)
    if arm in SWEPT[model] and changed(model, arm):
        return f"{model}_{arm}7u_s{seed}"
    return f"{model}_{arm}_s{seed}"


def relapse_tag(model, arm, seed):
    pre = "gemma" if model == "gemma2b" else "llama8b"
    if arm is None:
        return f"{pre}_none"
    return f"{pre}_{arm}{'7u' if arm in SWEPT[model] and changed(model, arm) else ''}_s{seed}"


def adapter(model, arm, seed):
    root = f"{M}/published" if model == "gemma2b" else f"{M}/published_llama"
    if arm in SWEPT[model]:
        return (
            f"{root}/final7u/{arm}_s{seed}/step1"
            if changed(model, arm)
            else f"{root}/{'final' if model == 'gemma2b' else 'final7'}/{arm}_s{seed}/step1"
        )
    raise KeyError(arm)


def summary():
    out = {}
    for model, arms in SWEPT.items():
        for a in arms:
            try:
                out[f"{model}:{a}"] = {
                    "changed": changed(model, a),
                    "selected": selection()[model][a].get("selected"),
                    "stored": selection()[model][a].get("stored_selection"),
                }
            except RuntimeError as e:
                out[f"{model}:{a}"] = {"error": str(e)}
    return out


if __name__ == "__main__":
    for k, v in summary().items():
        print(k, v)
