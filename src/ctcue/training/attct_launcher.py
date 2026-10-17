"""Train ACT or AttCT with the AttCT repository's code (c-wei/AttCT at commit b3c1896) on our training set.

Everything is the repository's run.main(): configuration, model and LoRA loading, losses, trainer and
checkpoints. Only the dataloader is replaced, so that it feeds our cued prompts from
$CT_ROOT/shared/shared_train.jsonl. BCT needs no replacement and runs with the repository's own run.py.
Run from the root of the AttCT repository. Our runs passed --skip-eval and --no-checkpoint, which skip
the repository's own evaluations.

Usage: python -m ctcue.training.attct_launcher --config configs/ours_gemma2b/attct_seed42.yaml
           --data-mode sycophancy --skip-eval --no-checkpoint
"""

import json, os, sys
from functools import partial
from ctcue.paths import CT_ROOT

REPO = os.getcwd()
assert os.path.exists(os.path.join(REPO, "run.py")) and os.path.exists(
    os.path.join(REPO, "losses/losses.py")
), "cwd must be the AttCT repo"
sys.path.insert(0, REPO)
SHARED = f"{CT_ROOT}/shared/shared_train.jsonl"


class FixedCueWrapper:
    """Duck-types their AdversarialWrapper for AttCTDataset: .templates (checked in __init__) and .wrap(prompt)."""

    # falsy -> their 'all templates need MCQ choices' filter is skipped
    templates = []

    def __init__(self, table):
        self.table = table

    def wrap(self, prompt):
        return self.table[prompt], None, None  # their __getitem__ uses only the first element


def get_dataloader_shared(config, split="train"):
    """Line-for-line mirror of their data.attct_datasets.get_dataloader (non-persona branch), shared prompts + fixed cues."""
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from data.attct_datasets import AttCTDataset, collate_fn_batch1

    assert split == "train", "only the training loader is swapped (evals are skipped with --skip-eval)"
    data_cfg = config.get("data", {})
    mode = data_cfg.get("mode", "jailbreak")
    assert mode == "sycophancy"
    batch_size = data_cfg.get("batch_size", 1)
    max_length = data_cfg.get("max_length", None)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    recs = [json.loads(l) for l in open(SHARED)]
    table = {r["clean"]: r["biased"] for r in recs}
    assert len(table) == len(recs)
    prompts = [r["clean"] for r in recs]
    dataset = AttCTDataset(
        prompts, tokenizer, wrapper=FixedCueWrapper(table), mode=mode, max_length=max_length
    )
    assert len(dataset) == len(recs), "their dataset dropped prompts"
    print(f"    [shared] {len(dataset)} prompts from {SHARED} (fixed prefix cues)", flush=True)
    collate = partial(collate_fn_batch1, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate)


if __name__ == "__main__":
    import run as their_run  # their module; run.py did `from data import get_dataloader`

    their_run.get_dataloader = get_dataloader_shared
    sys.argv = ["run.py"] + sys.argv[1:]
    their_run.main()
