#!/usr/bin/env python3
"""
run_final_u (1 Oct 2026): the CPU pipeline of the UNMERGED primary results, in dependency order (run as one CPU batch job; every step is a
subprocess and the pipeline stops at the first failure). Pre-flight: every arm's unmerged LR re-selection is final and every chain output
exists (checked on the cluster by a job-status script that is not part of this release), unless --partial (then the steps run on what exists, for a dry run; tables are written to
results/paper_u_partial and are NOT the final tables).
  1 ana_merge7 (A1)            2 ana_fs7 --mode unmerged --write_corrected (pub7uc)      3 check_gates_u        4 select_lr_gen7u
  5 weights_u (if a gemma arm was re-selected; needs w7u/weights_final7u_gemma2b.json)  6 score_pub_u   7 score_llama_u
  8 ana_fix7_u                 9 ana_fix6_u --model gemma2b_v7 / llama8b               10 build_lpviews_u (+ gemma causal view)
 11 ana_relapse_u             12 score_route_u        13 ana_selftrain        14 ana_fs_sens   14b ana_extra_u (wording / stronger hints)
 15 paper_tables --unmerged --B 10000   16 paper_tables_extra --unmerged --B 10000   17 paper_figs   18 appendix_u
"""
import subprocess, sys, time
SC = "/data/code"; R = "/data/ct_root/results"
sys.path.insert(0, SC)


def step(args):
    t0 = time.time(); print(f"\n===== {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable] + args, cwd=SC)
    print(f"===== exit {r.returncode} ({time.time() - t0:.0f}s)", flush=True)
    if r.returncode != 0: sys.exit(f"STEP FAILED: {' '.join(args)}")


def main():
    partial = "--partial" in sys.argv; out = f"{R}/paper_u_partial" if partial else f"{R}/paper_u"
    import utags
    step(["select_lr_gen7u.py"]); utags._SEL.clear()
    s = utags.summary(); bad = {k: v for k, v in s.items() if "error" in v}
    if bad and not partial: sys.exit(f"selection not final: {bad}")
    # (released copy) the cluster job-status preflight (status_u.py over the job files) is omitted
    step(["ana_merge7.py"]); step(["ana_fs7.py", "--mode", "unmerged", "--write_corrected"]); step(["check_gates_u.py"])
    # gate-failing gemma checkpoints need their v7.1-readout recomputation (w4cmds/u_gmax_p.txt -> w7u/probe_gemma2b_max_*, causal3_gemma2b_max_*)
    import json, os, build_lpviews_u
    g = json.load(open(f"{R}/w7u/gates_u.json"))["gmax_needed"]; short = lambda t: utags.inst_tag("gemma2b", *build_lpviews_u.parse_eval_tag(t)[:2])[len("gemma2b_"):]
    need = [f"{R}/w7u/probe_gemma2b_max_{short(t)}.json" for t in g["probe"]] + [f"{R}/w7u/causal3_gemma2b_max_{short(t)}.json" for t in g["causal"]]
    missing = [f for f in need if not os.path.exists(f)]
    if missing and not partial: sys.exit(f"v7.1 recomputation missing (chain b_u_gmax, w4cmds/u_gmax_p.txt): {missing}")
    print(f"gmax files: {len(need) - len(missing)}/{len(need)} present", flush=True)
    if any(k.startswith("gemma2b:") and v.get("changed") for k, v in s.items()): step(["weights_u.py"])
    for a in (["score_pub_u.py"], ["score_llama_u.py"], ["ana_fix7_u.py"], ["ana_fix6_u.py", "--model", "gemma2b_v7"], ["ana_fix6_u.py", "--model", "llama8b"],
              ["build_lpviews_u.py"], ["ana_relapse_u.py"], ["score_route_u.py"], ["ana_selftrain.py"], ["ana_fs_sens.py"], ["ana_extra_u.py"], ["ana_scale_u.py"], ["ana_mech_u.py"], ["ana_lrsweep_u.py"]):
        step(a)
    step(["paper_tables.py", "--unmerged", "--B", "10000", "--seed", "0", "--out_dir", out])
    step(["paper_tables_extra.py", "--unmerged", "--B", "10000", "--seed", "0", "--tables", f"{out}/paper_tables.json", "--out_dir", out])
    step(["ana_letterpos_u.py"])   # (11 Oct) letter-preference control of the position pattern (exploratory); checks itself against paper_tables.json
    step(["paper_figs.py", "--tables", f"{out}/paper_tables.json", "--out_dir", f"{out}/figs"])
    if not partial: step(["appendix_u.py"])
    print(f"\nALL STEPS DONE -> {out}")


if __name__ == "__main__":
    main()
