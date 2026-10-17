"""Load Gemma-2-2B-it or Llama-3.1-8B-Instruct with an optional LoRA adapter merged into the weights.

A copy of load_gemma_merged.py with a model choice. For Gemma it behaves exactly like
load_gemma_merged.load_model. Llama has untied embeddings, so only the merge check applies. The
merged versions of the instruments use this loader.
"""

import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

MODELS = {"gemma2b": "google/gemma-2-2b-it", "llama8b": "meta-llama/Llama-3.1-8B-Instruct"}


def load_model(adapter="none", model="gemma2b", verify=True):
    path = snapshot_download(MODELS[model], local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").cuda()
    if adapter != "none":
        from peft import PeftModel

        cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
        tm = cfg.get("target_modules")
        targets = set(tm) if isinstance(tm, (list, tuple)) else {str(tm)}
        untie = any("lm_head" in t for t in targets) and m.config.tie_word_embeddings
        emb0 = m.get_input_embeddings().weight.detach().clone()
        if untie:
            m.lm_head.weight = torch.nn.Parameter(m.get_input_embeddings().weight.detach().clone())
            m.config.tie_word_embeddings = False
        pm = PeftModel.from_pretrained(m, adapter)
        x = tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Which option is correct?\n\nAnswer choices:\n(A) one\n(B) two\n\nAnswer with just the single letter (A, B, C, D, ...), no explanation or reasoning.",
                }
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        ).cuda()
        if verify:
            with torch.no_grad():
                ref = pm(input_ids=x).logits[0, -1].float()
        m = pm.merge_and_unload()
        if verify:
            with torch.no_grad():
                got = m(input_ids=x).logits[0, -1].float()
            d = float((got - ref).abs().max())
            assert d < 2.0, f"merge changed the model: max |dlogit| {d}"
            assert torch.equal(m.get_input_embeddings().weight, emb0), "input embeddings changed by the merge"
            print(
                f"[pub_load2:{model}] merged {adapter} (targets {sorted(targets)}; untied lm_head: {untie}; merged-vs-unmerged max |dlogit| {d:.4f})",
                flush=True,
            )
    m.eval()
    assert "Peft" not in type(m).__name__
    return tok, m
