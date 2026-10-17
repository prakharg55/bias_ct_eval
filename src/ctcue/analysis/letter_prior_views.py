#!/usr/bin/env python3
"""
build_lpviews_u (1 Oct 2026; prereg audit NOTE 19:43 MERGE-FIDELITY EXECUTION): the letter-prior control (ana_letterprior2.py UNCHANGED,
exploratory) on the UNMERGED probes (ct_probe7u), through symlink views that give ana_letterprior2 the file names it reads
(probe_<tag>.json/.npz, untrained = probe_pub_none). File choice per arm and seed: utags.inst_tag (new-selection runs where the unmerged LR
re-selection changed an arm). Views (results/w7u):
  lpview_gemma_u      gemma 4 arms + RMCT-anchor (v5 readout, --agg sum); untrained = w3/probe_pub_none (ct_probe3 = ct_probe7 --agg sum on
                      gemma, check_v7); tags pub_<arm>_s<seed>_step1 (as w3). (score_pub_u's P3 view w7u/gemma_probe_u_view has the same probe links.)
  lpview_gemma_fix_u  gemma BCT + 4 fix variants (v5 readout); tags pub_bct_s<seed>_step1, fix_<v>_s<seed> (as w5/lpview).
  lpview_llama_u      Llama 4 arms (v7.1 readout, --agg max); untrained = w7/probe_llama8b_none; tags llama8b_<arm>_s<seed> (as w7/lpview_llama).
  lpview_gemma_max_u  gemma checkpoints whose UNMERGED test-battery gate fails (results/w7u/gates_u.json gmax_needed.probe), on the v7.1
                      readout (ct_probe7u --agg max, file w7u/probe_gemma2b_max_<instrument tag without "gemma2b_">, as the merged
                      w7/probe_gemma2b_max_bct_s44); untrained = w7/probe_gemma2b_max_none; view names as the two views above. Built only if any.
Then runs ana_letterprior2.py --root <view> <tags> for each view (CPU, seconds). Writes <view>/ana_letterprior2.json. Also builds
gemma_causal_u_view (causal3_pub_<arm>_s<seed>_step1.json -> the unmerged ct_causal7u files; untrained w3/causal3_pub_none) for paper_tables.
"""
import json, os, subprocess, sys
sys.path.insert(0, "/data/code")
import utags
R = "/data/ct_root/results"; SC = "/data/code"; SEEDS = (42, 43, 44)
FIX = ("bayes_r0.5", "bayes_r0.8", "goldgated", "posdiv")


def build(view, links):
    os.makedirs(view, exist_ok=True)
    for name, src in links.items():
        assert os.path.exists(src), src; dst = f"{view}/{name}"
        if os.path.islink(dst):
            if os.readlink(dst) != src: raise RuntimeError(f"view link changed: {dst} -> {os.readlink(dst)} (wanted {src}); remove the view and rebuild")
        else: os.symlink(src, dst)
    extra = [f for f in os.listdir(view) if f not in links and f != "ana_letterprior2.json"]
    assert not extra, f"stale files in view {view}: {extra[:5]}"


def probe_links(name_tag, src_prefix):
    return {f"probe_{name_tag}.{x}": f"{src_prefix}.{x}" for x in ("json", "npz")}


def run(view, tags):
    subprocess.run([sys.executable, f"{SC}/ana_letterprior2.py", "--root", view] + tags, check=True)


def main():
    views = {}
    # gemma main arms
    L = probe_links("pub_none", f"{R}/w3/probe_pub_none")
    for m in ("bct", "act", "attct", "rmct", "rmcta"):
        for s in SEEDS: L |= probe_links(f"pub_{m}_s{s}_step1", f"{R}/w7u/probe_{utags.inst_tag('gemma2b', m, s)}")
    views["lpview_gemma_u"] = (L, [f"pub_{m}_s{s}_step1" for m in ("bct", "act", "attct", "rmct", "rmcta") for s in SEEDS])
    # gemma fix variants
    L = probe_links("pub_none", f"{R}/w3/probe_pub_none")
    for s in SEEDS: L |= probe_links(f"pub_bct_s{s}_step1", f"{R}/w7u/probe_{utags.inst_tag('gemma2b', 'bct', s)}")
    for v in FIX:
        for s in SEEDS: L |= probe_links(f"fix_{v}_s{s}", f"{R}/w7u/probe_{utags.inst_tag('gemma2b', 'fix_' + v, s)}")
    views["lpview_gemma_fix_u"] = (L, [f"pub_bct_s{s}_step1" for s in SEEDS] + [f"fix_{v}_s{s}" for v in FIX for s in SEEDS])
    # Llama
    L = probe_links("pub_none", f"{R}/w7/probe_llama8b_none")
    for m in ("bct", "act", "attct", "rmct"):
        for s in SEEDS: L |= probe_links(f"llama8b_{m}_s{s}", f"{R}/w7u/probe_{utags.inst_tag('llama8b', m, s)}")
    views["lpview_llama_u"] = (L, [f"llama8b_{m}_s{s}" for m in ("bct", "act", "attct", "rmct") for s in SEEDS])
    # gemma gate failures on the v7.1 readout
    g = json.load(open(f"{R}/w7u/gates_u.json"))["gmax_needed"]["probe"]
    if g:
        L = probe_links("pub_none", f"{R}/w7/probe_gemma2b_max_none"); names = []
        for t in g:   # t = an eval tag of a gate-failing gemma checkpoint
            arm, seed, name = parse_eval_tag(t); it = utags.inst_tag("gemma2b", arm, seed)
            L |= probe_links(name, f"{R}/w7u/probe_gemma2b_max_{it[len('gemma2b_'):]}"); names.append(name)
        views["lpview_gemma_max_u"] = (L, names)
    for v, (links, tags) in views.items():
        build(f"{R}/w7u/{v}", links); run(f"{R}/w7u/{v}", tags); print(f"[{v}] {len(tags)} checkpoints", flush=True)
    # gemma patch recovery for paper_tables --unmerged (ana_w3.ana_causal reads causal3_pub_*.json): untrained w3 (ct_causal3 = ct_causal7 --agg sum on gemma, check_v7)
    L = {"causal3_pub_none.json": f"{R}/w3/causal3_pub_none.json"}
    L |= {f"causal3_pub_{m}_s{s}_step1.json": f"{R}/w7u/causal3_{utags.inst_tag('gemma2b', m, s)}.json" for m in ("bct", "act", "attct", "rmct") for s in SEEDS}
    build(f"{R}/w7u/gemma_causal_u_view", L); print(f"[gemma_causal_u_view] {len(L) - 1} checkpoints", flush=True)


def parse_eval_tag(t):
    """gemma eval tag -> (arm, seed, the name the letter-prior views use: pub_<arm>_s<seed>_step1 or fix_<v>_s<seed>)."""
    import re
    m = re.match(r"gemma2b_final(?:7u)?_([\w.]+?)_s(\d+)$", t)
    if m: return m.group(1), int(m.group(2)), f"pub_{m.group(1)}_s{m.group(2)}_step1"
    m = re.match(r"gemma2b_fix_([\w.]+?)_s(\d+)$", t)
    if m: return "fix_" + m.group(1), int(m.group(2)), f"fix_{m.group(1)}_s{m.group(2)}"
    raise ValueError(t)


if __name__ == "__main__":
    main()
