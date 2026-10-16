#!/usr/bin/env python3
"""
Launcher for the AttCT authors' UNMODIFIED training code (c-wei/AttCT @ b3c1896) on the SHARED training set.
Run with cwd = the AttCT repo root. Everything (config merge, model/LoRA loading, seed, loss, Trainer, checkpointing) is
their run.main(); the ONLY substitution is the dataloader for the ACT/AttCT path: their data.attct_datasets.get_dataloader
builds AttCTDataset(prompts, tokenizer, mode, max_length) with their random prefix templates; the mirror below is that
function line for line except (1) prompts = the shared clean questions and (2) wrapper = FixedCueWrapper, whose wrap()
returns each question's shared prefix-cued prompt. Their AttCTDataset.__getitem__ then does all chat templating, content
window, truncation, suffix window (match_len) and wrapper mask exactly as for their own data.
BCT needs no substitution (their get_bct_dataloader reads bct_root files; see build_shared_data.py + their generator).
Usage: python pub_run.py <their run.py args...>   (e.g. --config configs/ours_gemma2b/attct_seed42.yaml --data-mode sycophancy)
"""
import json, os, sys
from functools import partial
REPO = os.getcwd(); assert os.path.exists(os.path.join(REPO, "run.py")) and os.path.exists(os.path.join(REPO, "losses/losses.py")), "cwd must be the AttCT repo"
sys.path.insert(0, REPO)
SHARED = "/data/ct_root/shared/shared_train.jsonl"


class FixedCueWrapper:
    """Duck-types their AdversarialWrapper for AttCTDataset: .templates (checked in __init__) and .wrap(prompt)."""
    templates = []                               # falsy -> their 'all templates need MCQ choices' filter is skipped

    def __init__(self, table):
        self.table = table

    def wrap(self, prompt):
        return self.table[prompt], None, None     # their __getitem__ uses only the first element


def get_dataloader_shared(config, split="train"):
    """Line-for-line mirror of their data.attct_datasets.get_dataloader (non-persona branch), shared prompts + fixed cues."""
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from data.attct_datasets import AttCTDataset, collate_fn_batch1
    assert split == "train", "only the training loader is swapped (evals are skipped with --skip-eval)"
    data_cfg = config.get("data", {})
    mode = data_cfg.get("mode", "jailbreak"); assert mode == "sycophancy"
    batch_size = data_cfg.get("batch_size", 1)
    max_length = data_cfg.get("max_length", None)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    recs = [json.loads(l) for l in open(SHARED)]
    table = {r["clean"]: r["biased"] for r in recs}; assert len(table) == len(recs)
    prompts = [r["clean"] for r in recs]
    dataset = AttCTDataset(prompts, tokenizer, wrapper=FixedCueWrapper(table), mode=mode, max_length=max_length)
    assert len(dataset) == len(recs), "their dataset dropped prompts"
    print(f"    [shared] {len(dataset)} prompts from {SHARED} (fixed prefix cues)", flush=True)
    collate = partial(collate_fn_batch1, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate)


if __name__ == "__main__":
    import run as their_run                      # their module; run.py did `from data import get_dataloader`
    their_run.get_dataloader = get_dataloader_shared
    sys.argv = ["run.py"] + sys.argv[1:]
    their_run.main()
