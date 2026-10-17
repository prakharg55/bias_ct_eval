"""Run the letter-prior control on the final attention-probe files.

Builds views of symbolic links that give letter_prior.py the files it expects, one view per model and
set of checkpoints, and runs it on each view. Gemma checkpoints that fail the readout gate (from
readout_gates.py) get a further view on the max-logit readout. The script also builds the view of the
Gemma patching files that paper/tables.py reads.

Usage: python -m ctcue.analysis.letter_prior_views
Output: $CT_ROOT/results/w7u/lpview_*/ana_letterprior2.json
        $CT_ROOT/results/w7u/gemma_causal_u_view/
"""

import json, os, subprocess, sys

from ctcue.common import checkpoints
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
SEEDS = (42, 43, 44)
FIX = ("bayes_r0.5", "bayes_r0.8", "goldgated", "posdiv")


def build(view, links):
    os.makedirs(view, exist_ok=True)
    for name, src in links.items():
        assert os.path.exists(src), src
        dst = f"{view}/{name}"
        if os.path.islink(dst):
            if os.readlink(dst) != src:
                raise RuntimeError(
                    f"view link changed: {dst} -> {os.readlink(dst)} (wanted {src}); remove the view and rebuild"
                )
        else:
            os.symlink(src, dst)
    extra = [f for f in os.listdir(view) if f not in links and f != "ana_letterprior2.json"]
    assert not extra, f"stale files in view {view}: {extra[:5]}"


def probe_links(name_tag, src_prefix):
    return {f"probe_{name_tag}.{x}": f"{src_prefix}.{x}" for x in ("json", "npz")}


def run(view, tags):
    subprocess.run([sys.executable, "-m", "ctcue.analysis.letter_prior", "--root", view] + tags, check=True)


def main():
    views = {}
    # gemma main arms
    L = probe_links("pub_none", f"{R}/w3/probe_pub_none")
    for m in ("bct", "act", "attct", "rmct", "rmcta"):
        for s in SEEDS:
            L |= probe_links(f"pub_{m}_s{s}_step1", f"{R}/w7u/probe_{checkpoints.inst_tag('gemma2b', m, s)}")
    views["lpview_gemma_u"] = (
        L,
        [f"pub_{m}_s{s}_step1" for m in ("bct", "act", "attct", "rmct", "rmcta") for s in SEEDS],
    )
    # gemma fix variants
    L = probe_links("pub_none", f"{R}/w3/probe_pub_none")
    for s in SEEDS:
        L |= probe_links(f"pub_bct_s{s}_step1", f"{R}/w7u/probe_{checkpoints.inst_tag('gemma2b', 'bct', s)}")
    for v in FIX:
        for s in SEEDS:
            L |= probe_links(
                f"fix_{v}_s{s}", f"{R}/w7u/probe_{checkpoints.inst_tag('gemma2b', 'fix_' + v, s)}"
            )
    views["lpview_gemma_fix_u"] = (
        L,
        [f"pub_bct_s{s}_step1" for s in SEEDS] + [f"fix_{v}_s{s}" for v in FIX for s in SEEDS],
    )
    # Llama
    L = probe_links("pub_none", f"{R}/w7/probe_llama8b_none")
    for m in ("bct", "act", "attct", "rmct"):
        for s in SEEDS:
            L |= probe_links(f"llama8b_{m}_s{s}", f"{R}/w7u/probe_{checkpoints.inst_tag('llama8b', m, s)}")
    views["lpview_llama_u"] = (
        L,
        [f"llama8b_{m}_s{s}" for m in ("bct", "act", "attct", "rmct") for s in SEEDS],
    )
    # gemma gate failures on the v7.1 readout
    g = json.load(open(f"{R}/w7u/gates_u.json"))["gmax_needed"]["probe"]
    if g:
        L = probe_links("pub_none", f"{R}/w7/probe_gemma2b_max_none")
        names = []
        for t in g:  # t = an eval tag of a gate-failing gemma checkpoint
            arm, seed, name = parse_eval_tag(t)
            it = checkpoints.inst_tag("gemma2b", arm, seed)
            L |= probe_links(name, f"{R}/w7u/probe_gemma2b_max_{it[len('gemma2b_'):]}")
            names.append(name)
        views["lpview_gemma_max_u"] = (L, names)
    for v, (links, tags) in views.items():
        build(f"{R}/w7u/{v}", links)
        run(f"{R}/w7u/{v}", tags)
        print(f"[{v}] {len(tags)} checkpoints", flush=True)
    # gemma patch recovery for tables --unmerged (measures.ana_causal reads causal3_pub_*.json): untrained w3
    # (patching_v1 = patching_merged --agg sum on gemma, check_v7)
    L = {"causal3_pub_none.json": f"{R}/w3/causal3_pub_none.json"}
    L |= {
        f"causal3_pub_{m}_s{s}_step1.json": f"{R}/w7u/causal3_{checkpoints.inst_tag('gemma2b', m, s)}.json"
        for m in ("bct", "act", "attct", "rmct")
        for s in SEEDS
    }
    build(f"{R}/w7u/gemma_causal_u_view", L)
    print(f"[gemma_causal_u_view] {len(L) - 1} checkpoints", flush=True)


def parse_eval_tag(t):
    """gemma eval tag -> (arm, seed, the name the letter-prior views use: pub_<arm>_s<seed>_step1 or fix_<v>_s<seed>)."""
    import re

    m = re.match(r"gemma2b_final(?:7u)?_([\w.]+?)_s(\d+)$", t)
    if m:
        return m.group(1), int(m.group(2)), f"pub_{m.group(1)}_s{m.group(2)}_step1"
    m = re.match(r"gemma2b_fix_([\w.]+?)_s(\d+)$", t)
    if m:
        return "fix_" + m.group(1), int(m.group(2)), f"fix_{m.group(1)}_s{m.group(2)}"
    raise ValueError(t)


if __name__ == "__main__":
    main()
