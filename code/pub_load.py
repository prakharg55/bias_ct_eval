#!/usr/bin/env python3
"""
Shared loader for evaluating the published-code checkpoints (26 Sep 2026). gemma-2-2b-it + optional PEFT adapter, merged.
Tied-embedding safety: RMCT (Tinker defaults: train_unembed=True) puts LoRA on lm_head, whose weight gemma TIES to the
input embedding. During training the LoRA delta only changes the output projection; merge_and_unload() would write it
into the shared tensor and silently change the input embeddings too. So when lm_head is a LoRA target the output
projection is UNTIED (its own copy of the weight) before merging. Verified on every load (verify=True): the merged model's
logits match the unmerged PEFT model's on a fixed prompt, and (when untied) the input embeddings equal the base model's.
"""
import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
GEMMA = "google/gemma-2-2b-it"


def load_model(adapter="none", verify=True):
    path = snapshot_download(GEMMA, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").cuda()
    if adapter != "none":
        from peft import PeftModel
        cfg = json.load(open(os.path.join(adapter, "adapter_config.json"))); tm = cfg.get("target_modules")
        targets = set(tm) if isinstance(tm, (list, tuple)) else {str(tm)}
        untie = any("lm_head" in t for t in targets) and model.config.tie_word_embeddings
        emb0 = model.get_input_embeddings().weight.detach().clone()
        if untie:
            model.lm_head.weight = torch.nn.Parameter(model.get_input_embeddings().weight.detach().clone()); model.config.tie_word_embeddings = False
        pm = PeftModel.from_pretrained(model, adapter)
        x = tok.apply_chat_template([{"role": "user", "content": "Which option is correct?\n\nAnswer choices:\n(A) one\n(B) two\n\nAnswer with just the single letter (A, B, C, D, ...), no explanation or reasoning."}],
                                    add_generation_prompt=True, return_tensors="pt").cuda()
        if verify:
            with torch.no_grad(): ref = pm(input_ids=x).logits[0, -1].float()
        model = pm.merge_and_unload()
        if verify:
            with torch.no_grad(): got = model(input_ids=x).logits[0, -1].float()
            # gross-error gate only: bf16 merge rounding moves logits by up to ~1 (softcap 30); the real risk (a delta written
            # into the tied input embedding) is caught exactly by the embedding check below, for EVERY adapter
            d = float((got - ref).abs().max()); assert d < 2.0, f"merge changed the model: max |dlogit| {d}"
            assert torch.equal(model.get_input_embeddings().weight, emb0), "input embeddings changed by the merge"
            print(f"[pub_load] merged {adapter} (targets {sorted(targets)}; untied lm_head: {untie}; merged-vs-unmerged max |dlogit| {d:.4f})", flush=True)
    model.eval()
    assert "Peft" not in type(model).__name__
    return tok, model
