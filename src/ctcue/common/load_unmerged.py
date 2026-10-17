"""Load a model with its LoRA adapter unmerged, for the instruments that use forward hooks.

Returns the Hugging Face model inside the PEFT wrapper. Its target layers carry the adapter (fp32,
unmerged), and the usual module paths (model.model.layers[...]) stay valid for hooks. Without an
adapter this is load_merged.load_model.
"""

from ctcue.common import load_peft
from ctcue.common.load_merged import MODELS


def load_model(adapter="none", model="gemma2b", verify=True):
    tok, m = load_peft.load_model(adapter, model=model, verify=verify, merge=False)
    if adapter == "none":
        return tok, m
    hf = m.base_model.model
    hf.eval()
    return tok, hf
