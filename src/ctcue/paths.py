"""Where the code looks for data, checkpoints and other repositories.

Set an environment variable of the same name to change a location. The defaults are placeholders.
"""

import os

# Working root with shared/ (training data), models/ (adapters) and results/ (all outputs)
CT_ROOT = os.environ.get("CT_ROOT", "/data/ct_root")
# The cued multiple-choice dumps released by Chua et al. (2024), with test/<cue>/<dataset>_<cue>.jsonl
BCT_DUMPS = os.environ.get("BCT_DUMPS", "/data/dataset_dumps")
# Hugging Face cache with google/gemma-2-2b-it and meta-llama/Llama-3.1-8B-Instruct
HF_HOME = os.environ.get("HF_HOME", "/data/hf_home")
# The AttCT repository (github.com/c-wei/AttCT) at commit b3c1896
ATTCT_REPO = os.environ.get("ATTCT_REPO", "/data/AttCT")
# The RMCT authors' fork of cot-transparency at commit 4392d15. One checkout can serve as both of these.
RMCT_SRC = os.environ.get("RMCT_SRC", "/data/rmct_src")
RMCT_UPSTREAM = os.environ.get("RMCT_UPSTREAM", "/data/rmct_upstream_4392d15")
# A pip --target folder with tinker 0.19.0 and tinker-cookbook 0.4.1, used by the RMCT trainer
RMCT_PYLIBS = os.environ.get("RMCT_PYLIBS", "/data/pylibs/rmct")
