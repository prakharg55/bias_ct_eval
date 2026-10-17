# Where each figure and table comes from

Module names are relative to `ctcue`. The numbering follows the current version of the paper.

| Paper element | Computed by | Drawn or typeset by |
|---|---|---|
| Figure 1 (overview of the three findings) | `analysis.selectivity_scales`, `paper.tables`, `analysis.mechanism_checks` | `paper.figures` |
| Figure 2, Tables 6 and 7 (efficacy and transfer, per cue) | `paper.tables` | `paper.figures`, `paper.latex_tables` |
| Table 8 (letter-prior control) | `analysis.letter_prior_views`, `analysis.letter_prior`, `paper.tables` | `paper.latex_tables` |
| Figure 3, Tables 9 to 12 (correct and wrong cues) | `paper.tables`, `analysis.selectivity_scales` | `paper.figures`, `paper.latex_tables` |
| Figure 4, Tables 13 to 15 (cue position) | `paper.tables`, `analysis.wording_and_strength`, `analysis.selectivity_scales`, `analysis.letter_position_control`, `analysis.test_set_layout`, `analysis.lr_sweep` | `paper.figures`, `paper.latex_tables` |
| Figure 5, Tables 16 to 18 (circuit) | `paper.tables`, `analysis.mechanism_checks`, `analysis.score_layer_swap` | `paper.figures`, `paper.latex_tables` |
| Figure 6, Table 19 (BCT variants) | `analysis.score_variants_gemma`, `analysis.score_variants`, `analysis.score_self_training`, `paper.tables` | `paper.figures`, `paper.latex_tables` |
| Figure 7, Table 20 (relapse) | `analysis.relapse`, `analysis.mechanism_checks`, `paper.tables_extra` | `paper.figures`, `paper.latex_tables` |
| Table 3 (verdicts on the registered predictions) | `paper.tables`, `paper.tables_extra` | `paper.latex_tables` |
| Table 4 (learning-rate selection) | `training.select_learning_rates`, `analysis.appendix_tables` | `paper.latex_tables` |
| Table 5 (merged and unmerged adapters) | `analysis.merge_fidelity` | `paper.latex_tables` |
| Tables 21 and 22 (checks against the source papers) | `evaluation.attct_evaluator`, `training.rmct.train_paper_regime`, `evaluation.rmct_paper_eval`, `paper.tables` | `paper.latex_tables` |
| Result numbers in the text (LaTeX macros) | all of the above | `paper.latex_numbers` |

The analysis modules read the outputs of the GPU instruments in `ctcue.evaluation`. Tables 1 and 2 quote the
[pre-registration](preregistration.md), and Figure 8 is a drawn example prompt.
