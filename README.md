# Code for "Consistency Training Suppresses Cue Following Without Teaching Cue Selectivity"

This folder holds the code behind every number, table and figure in the paper.

- `code/` has the scripts (data building, training launchers, evaluation instruments, analysis, tables and figures).
- `configs/` has our training configurations for the AttCT repository as overlays, so that no file of that repository is redistributed. Each overlay names a configuration file of the repository at commit `b3c1896` (`base`) and the keys we changed (`set`). `python3 code/materialize_configs.py --attct /data/AttCT --configs configs --out /data/AttCT/configs` writes the full files into that repository's `configs/` folder as `ours_gemma2b/` and `ours_llama8b/`.

Running the analysis step (step 5 below) on the saved per-item results reproduces every number macro, table and figure in the paper exactly. The per-item results and the trained adapters are not part of this archive because of their size. We will release them with the code.

## Paths

The scripts use fixed absolute paths. Create these locations, or symlink them, or edit the constants at the top of each script.

| Path | Contents |
|---|---|
| `/data/code` | the `code/` folder of this release |
| `/data/ct_root` | working root with `shared/` (training data), `models/` (adapters) and `results/` |
| `/data/dataset_dumps` | the cued multiple-choice dumps released with Chua et al. (2024), `test/<cue>/<dataset>_<cue>.jsonl` |
| `/data/hf_home` | Hugging Face cache (`HF_HOME`) with `google/gemma-2-2b-it` and `meta-llama/Llama-3.1-8B-Instruct` |
| `/data/AttCT` | the AttCT repository at commit `b3c1896` |
| `/data/rmct_src`, `/data/rmct_upstream_4392d15` | the RMCT authors' fork of cot-transparency at commit `4392d15` |
| `/data/pylibs/rmct` | `tinker` 0.19.0 and `tinker-cookbook` 0.4.1 |

## Third-party code

We do not redistribute other authors' code. BCT, ACT and AttCT run through the AttCT repository, unmodified at commit `b3c1896`, with our configurations. BCT uses its `run.py` directly. ACT and AttCT use `code/pub_run.py`, which calls the same `run.py` and replaces only the dataloader so that it feeds the shared training prompts. Run both from the repository root.

RMCT runs the authors' trainer (`RLTrainer` and `common.py` from their fork at commit `4392d15`) without changes. The scripts `tinker_local2.py`, `tinker_local3.py` and `tinker_local4.py` replace only the remote Tinker service with a local implementation, because the service does not serve Gemma-2-2B.

`code/utils.py` is a small loader for the Chua et al. dumps. Note that it removes the cue from about 24% of suggested-answer pairs, and every script drops those pairs.

## Pipeline

1. **Data.**
   - `build_shared_data.py` builds the shared training set (1,213 questions, cue placed before the question) and the BCT source files.
   - BCT targets come from the AttCT repository's `scripts/generate_fresh_bct_data.py --use-vllm`.
   - `pmodel_prior.py` (Gemma) and `pmodel_prior6.py` (Llama) compute the prior for Bayes-BCT.
   - `build_fix_data.py` (Gemma) and `build_fix_data6.py` (Llama) build the Bayes-BCT, gold-gated and position-diverse data.
   - `build_selftrain_data.py` builds the self-training control.
2. **Training.**
   - BCT, ACT and AttCT use the configurations in `configs/`. The `sweep/` configurations are the seed-42 learning-rate sweeps and `final/` the seeds 43 and 44 at the selected rate. `phase5/` and `phase6/` are the BCT variants (response-only BCT in `ours_gemma2b/phase6/bctro_*`), and `phase7/` is the self-training control.
   - RMCT uses `rmct_local2.py` (Gemma) and `rmct_local4.py` (Llama). `rmct_local3.py` is the exploratory RMCT+anchor variant.
3. **Learning-rate selection.** `val_eval7g.py` scores the validation set with generated answers. `select_lr_gen7u.py` applies the pre-registered rule, using `select_lr.py` and `select_lr7.py`.
4. **Instruments** (one checkpoint per call; adapters are evaluated unmerged through `pub_load3u.py`):

   | Script | Measures |
   |---|---|
   | `ct_eval7u.py` | test set with generated answers |
   | `ct_evalfs7.py` | few-shot cues scored at the test slot |
   | `ct_gen7u.py` | cue battery |
   | `gen_gate7u.py` | readout gate on the cue battery |
   | `ct_probe7u.py` | attention of the reader heads to the cue |
   | `ct_causal7u.py` | patch recovery |
   | `ct_route7u.py` | layer swap |
   | `relapse_ft7u.py` | relapse under benign fine-tuning |
   | `ct_weights6.py`, `weights_u.py` | weight-edit depth |
   | `attct_eval_ckpt.py` | the AttCT authors' evaluator |
   | `rmct_paper.py`, `tinker_local4.py`, `hle_eval_rmct.py` | RMCT in the authors' own regime |

   Some baselines and checks come from earlier versions of the same instruments, which these scripts produced:
   - original-model baselines: `ct_probe3.py`, `ct_causal3.py`, `ct_causal7.py`, `ct_gen7.py`, `gen_gate7.py`, `check_gemma_gen7.py` and `relapse_ft7.py`;
   - the merged-adapter comparison: `ct_eval7.py`, plus `ct_route7.py` for the merged-weight layer swap;
   - the readout checks of Appendix D.1: `readout_check6.py` and `readout_check6b.py`.
5. **Analysis** (CPU, minutes).
   - `run_final_u.py` runs every analysis step in order and writes `results/paper_u/paper_tables.json`, `paper_tables_extra.json` and `appendix_u.json`.
   - `run_final_u.py` also runs `ana_letterpos_u.py` (the letter-preference control of the position pattern, Appendix E.3).
   - Then run `ana_layout_u.py`, `paper_numbers.py --tables_dir <paper_u> --extra_dir <paper_u> --out <paper dir>` (number macros), `paper_tables_tex.py` with the same arguments (tables) and `paper_figs_story.py` (figures).
   - `utags.py` maps a model, method and seed to its checkpoint and result names.

The other modules in `code/` are helpers that these scripts import, including older versions such as `pub_load2.py`, `letters6.py` and `score_pub.py`.

## Where each part of the paper comes from

| Paper element | Computed by | Rendered by |
|---|---|---|
| Figure 1 (overview of the three main findings) | `ana_scale_u.py`, `paper_tables.py`, `ana_mech_u.py` | `paper_figs_story.py` |
| Figure 2, Tables 6 and 7 (efficacy, transfer, per cue) | `paper_tables.py` | `paper_figs_story.py`, `paper_tables_tex.py` |
| Table 8 (letter-prior control) | `build_lpviews_u.py`, `ana_letterprior2.py`, `paper_tables.py` | `paper_tables_tex.py` |
| Figure 3, Tables 9 to 12 (selectivity) | `paper_tables.py`, `ana_scale_u.py` | `paper_figs_story.py`, `paper_tables_tex.py` |
| Figure 4, Tables 13 to 15 (position) | `paper_tables.py`, `ana_extra_u.py`, `ana_scale_u.py`, `ana_letterpos_u.py`, `ana_layout_u.py`, `ana_lrsweep_u.py` | `paper_figs_story.py`, `paper_tables_tex.py` |
| Figure 5, Tables 16 to 18 (circuit) | `paper_tables.py`, `ana_mech_u.py`, `score_route_u.py` | `paper_figs_story.py`, `paper_tables_tex.py` |
| Figure 6, Table 19 (BCT variants) | `ana_fix7_u.py`, `ana_fix6_u.py`, `ana_selftrain.py`, `paper_tables.py` | `paper_figs_story.py`, `paper_tables_tex.py` |
| Figure 7, Table 20 (relapse) | `ana_relapse_u.py`, `ana_mech_u.py`, `paper_tables_extra.py` | `paper_figs_story.py`, `paper_tables_tex.py` |
| Table 3 (verdicts with intervals) | `paper_tables.py`, `paper_tables_extra.py` | `paper_tables_tex.py` |
| Table 4 (learning-rate selection) | `select_lr_gen7u.py`, `appendix_u.py` | `paper_tables_tex.py` |
| Table 5 (merged and unmerged adapters) | `ana_merge7.py` | `paper_tables_tex.py` |
| Tables 21 and 22 (checks against the source papers) | `attct_eval_ckpt.py`, `rmct_paper.py`, `hle_eval_rmct.py`, `paper_tables.py` | `paper_tables_tex.py` |
| Every number in the text | all of the above | `paper_numbers.py` writes the LaTeX macros |

Tables 1 and 2 quote the pre-registration file, included here as `preregistration_redacted.md` (only the cluster and queue names are redacted). Figure 8 is a drawn example prompt.

## Environment

Python 3.12. Main packages: torch 2.9.0, transformers 4.57.6, peft 0.19.1, accelerate 1.7.0, vllm 0.11.2, datasets 4.4.2, safetensors 0.8.0, tokenizers 0.22.1, numpy 1.26.4, scipy 1.15.2, scikit-learn 1.6.1, pandas 2.2.3, matplotlib 3.8.4, PyYAML 6.0.1, wandb 0.20.1 (imported by the AttCT authors' evaluator, run with WANDB_MODE=disabled), and tinker 0.19.0 with tinker-cookbook 0.4.1 (interfaces used by the RMCT authors' trainer, whose remote service is replaced locally). All training and evaluation ran on single NVIDIA A100 GPUs.
