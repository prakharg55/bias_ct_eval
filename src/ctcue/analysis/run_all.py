"""Run the analysis steps in order, from the stored instrument outputs to the paper tables (CPU).

The first step is the learning-rate selection (training/select_learning_rates.py). Each step runs as
its own process, and the pipeline stops at the first failure. Without --partial it also stops if a
learning-rate selection is not final, or if a max-logit probe or patching file of a Gemma checkpoint
that fails the readout gate is missing. With --partial the steps run on whatever exists, the tables
go to $CT_ROOT/results/paper_u_partial, and the appendix tables are skipped. test_set_layout.py,
paper/latex_tables.py, paper/latex_numbers.py and paper/figures.py are run separately afterwards.

Usage: python -m ctcue.analysis.run_all [--partial]
Output: $CT_ROOT/results/paper_u/
"""

import subprocess, sys, time
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"


def step(args):
    t0 = time.time()
    print(f"\n===== {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-m"] + args)
    print(f"===== exit {r.returncode} ({time.time() - t0:.0f}s)", flush=True)
    if r.returncode != 0:
        sys.exit(f"STEP FAILED: {' '.join(args)}")


def main():
    partial = "--partial" in sys.argv
    out = f"{R}/paper_u_partial" if partial else f"{R}/paper_u"
    from ctcue.common import checkpoints

    step(["ctcue.training.select_learning_rates"])
    checkpoints._SEL.clear()
    s = checkpoints.summary()
    bad = {k: v for k, v in s.items() if "error" in v}
    if bad and not partial:
        sys.exit(f"selection not final: {bad}")
    # (released copy) the cluster job-status preflight (status_u.py over the job files) is omitted
    step(["ctcue.analysis.merge_fidelity"])
    step(["ctcue.analysis.fewshot_rescore", "--mode", "unmerged", "--write_corrected"])
    step(["ctcue.analysis.readout_gates"])
    # gate-failing gemma checkpoints need their v7.1-readout recomputation (w4cmds/u_gmax_p.txt ->
    # w7u/probe_gemma2b_max_*, causal3_gemma2b_max_*)
    import json, os
    from ctcue.analysis import letter_prior_views

    g = json.load(open(f"{R}/w7u/gates_u.json"))["gmax_needed"]
    short = lambda t: checkpoints.inst_tag("gemma2b", *letter_prior_views.parse_eval_tag(t)[:2])[
        len("gemma2b_") :
    ]
    need = [f"{R}/w7u/probe_gemma2b_max_{short(t)}.json" for t in g["probe"]] + [
        f"{R}/w7u/causal3_gemma2b_max_{short(t)}.json" for t in g["causal"]
    ]
    missing = [f for f in need if not os.path.exists(f)]
    if missing and not partial:
        sys.exit(f"v7.1 recomputation missing (chain b_u_gmax, w4cmds/u_gmax_p.txt): {missing}")
    print(f"gmax files: {len(need) - len(missing)}/{len(need)} present", flush=True)
    if any(k.startswith("gemma2b:") and v.get("changed") for k, v in s.items()):
        step(["ctcue.analysis.weight_depth"])
    for a in (
        ["ctcue.analysis.score_gemma"],
        ["ctcue.analysis.score_llama"],
        ["ctcue.analysis.score_variants_gemma"],
        ["ctcue.analysis.score_variants", "--model", "gemma2b_v7"],
        ["ctcue.analysis.score_variants", "--model", "llama8b"],
        ["ctcue.analysis.letter_prior_views"],
        ["ctcue.analysis.relapse"],
        ["ctcue.analysis.score_layer_swap"],
        ["ctcue.analysis.score_self_training"],
        ["ctcue.analysis.fewshot_sensitivity"],
        ["ctcue.analysis.wording_and_strength"],
        ["ctcue.analysis.selectivity_scales"],
        ["ctcue.analysis.lr_sweep"],
    ):
        step(a)
    step(["ctcue.paper.tables", "--unmerged", "--B", "10000", "--seed", "0", "--out_dir", out])
    step(
        [
            "ctcue.paper.tables_extra",
            "--unmerged",
            "--B",
            "10000",
            "--seed",
            "0",
            "--tables",
            f"{out}/paper_tables.json",
            "--out_dir",
            out,
        ]
    )
    # reads the relapse results in paper_tables.json, so it runs after the tables
    step(["ctcue.analysis.mechanism_checks"])
    # (11 Oct) letter-preference control of the position pattern (exploratory); checks itself against
    # paper_tables.json
    step(["ctcue.analysis.letter_position_control"])
    step(["ctcue.paper.figures_v1", "--tables", f"{out}/paper_tables.json", "--out_dir", f"{out}/figs"])
    if not partial:
        step(["ctcue.analysis.appendix_tables"])
    print(f"\nALL STEPS DONE -> {out}")


if __name__ == "__main__":
    main()
