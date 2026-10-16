#!/usr/bin/env python3
"""
Loader v3u (1 Oct 2026; prereg audit NOTE item 1, MERGE FIDELITY): the UNMERGED model for hook-based instruments. pub_load3.load_model
(merge=False) returns a PeftModel; this returns the HF model inside it (pm.base_model.model), whose target Linear layers are PEFT
lora.Linear modules carrying the adapter (fp32 adapter weights, unmerged): the same forward computation, with the standard HF module
paths (model.model.layers[...]) that the hook-based instruments address. merge=True / adapter "none" -> pub_load2 exactly.
"""
import pub_load3
from pub_load2 import MODELS


def load_model(adapter="none", model="gemma2b", verify=True):
    tok, m = pub_load3.load_model(adapter, model=model, verify=verify, merge=False)
    if adapter == "none": return tok, m
    hf = m.base_model.model; hf.eval(); return tok, hf
