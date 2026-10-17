#!/usr/bin/env python3
"""
Shared loader v3 (1 Oct 2026; prereg NOTE "PRE-WRITE-UP AUDIT", item 1 MERGE FIDELITY) = pub_load2 + a merge switch.
merge=True  -> pub_load2.load_model UNCHANGED (bf16 base, eager attention, adapter merged into the bf16 weights; every
               reported result so far was computed this way).
merge=False -> the same base model, the same untie logic for an adapter that trains lm_head (gemma's tied embeddings; RMCT),
               and the adapter kept UNMERGED as a PEFT model (PEFT's default autocast: adapter weights in fp32), i.e. the
               computation used during training and by the authors' evaluator (T4e). Nothing else differs.
"""
import json, os
import torch
import pub_load2
from pub_load2 import MODELS
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


def load_model(adapter="none", model="gemma2b", verify=True, merge=True):
    if merge or adapter == "none": return pub_load2.load_model(adapter, model=model, verify=verify)
    path = snapshot_download(MODELS[model], local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").cuda()
    from peft import PeftModel
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json"))); tm = cfg.get("target_modules")
    targets = set(tm) if isinstance(tm, (list, tuple)) else {str(tm)}
    untie = any("lm_head" in t for t in targets) and m.config.tie_word_embeddings
    emb0 = m.get_input_embeddings().weight.detach().clone()
    if untie:   # as pub_load2: give lm_head its own copy of the embedding so the adapter's lm_head LoRA cannot touch the input embedding
        m.lm_head.weight = torch.nn.Parameter(m.get_input_embeddings().weight.detach().clone()); m.config.tie_word_embeddings = False
    pm = PeftModel.from_pretrained(m, adapter); pm.eval()
    assert torch.equal(pm.get_input_embeddings().weight, emb0), "input embeddings changed by loading the adapter"
    dt = sorted({str(p.dtype) for n, p in pm.named_parameters() if "lora_" in n})
    print(f"[pub_load3:{model}] UNMERGED {adapter} (targets {sorted(targets)}; untied lm_head: {untie}; lora dtypes {dt})", flush=True)
    return tok, pm
