#!/usr/bin/env python3
"""
Standalone runner for the AttCT repo's OWN evaluator (experiments/sycophancy/evaluate_sycophancy.py::SycophancyEvaluator,
c-wei/AttCT @ b3c1896, unmodified) on a saved checkpoint (29 Sep 2026; method sanity check). Mirrors run.py:404-448 /
519+: model loaded in bf16 with the same attention implementation run.py would use (sdpa unless --attn eager), LoRA adapter
attached UNMERGED (PeftModel, adapters enabled) exactly as run.py evaluates after training, tokenizer pad = eos, then
SycophancyEvaluator(prefix, results_csv, max_samples=--n [DEFAULT 1000 = the paper's n], held_out_path=
datasets/sycophancy_bct/control_cot_eval.jsonl, anthropic_eval=True).evaluate(). Substrates (their code): (1) MMLU
"on-the-fly" BRR + clean accuracy, (2) held-out BRR (12 templates on control_cot_eval), (3) Anthropic model-written-evals
sycophancy rate. Output: <out>/attct_eval_<tag>.json (+ their CSV). Usage: attct_eval_ckpt.py --model gemma2b|llama8b
--adapter <dir|none> --tag T [--attn sdpa|eager] [--n 1000] [--substrate all|anthropic]
The Anthropic substrate: their load_dataset("Anthropic/model-written-evals", data_files=...) cannot resolve its cache offline
(compute nodes have no internet), so the wrapper redirects THAT call to the identical jsonl files in the HF hub cache
(datasets--Anthropic--model-written-evals/snapshots/*/sycophancy/); --substrate anthropic runs only _evaluate_anthropic and merges
the result into an existing attct_eval_<tag>.json.
"""
import argparse, json, os, sys, time
REPO = "/data/AttCT"; OUT = "/data/ct_root/results/attct_eval"
MODEL_IDS = {"gemma2b": "google/gemma-2-2b-it", "llama8b": "meta-llama/Llama-3.1-8B-Instruct"}
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=sorted(MODEL_IDS)); ap.add_argument("--adapter", default="none")
    ap.add_argument("--tag", required=True); ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"]); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--substrate", default="all", choices=["all", "anthropic"])
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); t0 = time.time()
    import torch, wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    import glob, datasets
    from experiments.sycophancy import evaluate_sycophancy as ES
    from experiments.sycophancy.evaluate_sycophancy import SycophancyEvaluator
    snap = glob.glob(f"{os.environ['HF_HOME']}/hub/datasets--Anthropic--model-written-evals/snapshots/*")[0]
    _ld = datasets.load_dataset
    def _load(name, *args, data_files=None, split=None, **kw):                        # same files, local path (offline cache-key issue)
        if name == "Anthropic/model-written-evals": return _ld("json", data_files=os.path.join(snap, data_files), split=split)
        return _ld(name, *args, data_files=data_files, split=split, **kw) if data_files is not None else _ld(name, *args, split=split, **kw)
    ES.load_dataset = _load
    wandb.init(mode="disabled")                                                        # evaluator calls wandb.log
    path = snapshot_download(MODEL_IDS[a.model], local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token                            # run.py:400-402
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=a.attn).cuda()
    if a.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter); model.enable_adapter_layers()   # unmerged, as in run.py post-train eval
        print(f"[attct_eval] adapter {a.adapter} (unmerged PEFT)", flush=True)
    model.eval()
    os.chdir(REPO)                                                                     # their relative dataset paths
    ev = SycophancyEvaluator(model, tok, torch.device("cuda"), max_samples=a.n, prefix=a.tag, results_csv=f"{OUT}/syco_{a.tag}.csv",
                             held_out_path="datasets/sycophancy_bct/control_cot_eval.jsonl", anthropic_eval=True)
    if a.substrate == "anthropic":
        ar = ev._evaluate_anthropic(); ev._report_anthropic(ar)
        f = f"{OUT}/attct_eval_{a.tag}.json"; res = json.load(open(f)) if os.path.exists(f) else {}
        res["anthropic_sycophancy_rate"] = ar["sycophancy_rate"]; res["anthropic_n_questions"] = ar["n_questions"]
        res.update({f"anthropic_{k}": v for k, v in ar.items() if k.startswith("rate_")})
        json.dump(res, open(f, "w"), indent=1, default=str); print(f"[attct_eval] {a.tag}: Anthropic sycophancy rate {ar['sycophancy_rate']:.3f} (n={ar['n_questions']})", flush=True); return
    res = ev.evaluate()
    res.update(tag=a.tag, model=MODEL_IDS[a.model], adapter=a.adapter, attn=a.attn, n=a.n, seconds=time.time() - t0)
    json.dump(res, open(f"{OUT}/attct_eval_{a.tag}.json", "w"), indent=1, default=str)
    print(f"[attct_eval] {a.tag}: MMLU BRR {res.get('brr'):.3f} clean_acc {res.get('clean_acc'):.3f} NS {res.get('not_sycophantic_pct'):.3f} F1 {res.get('f1'):.3f} | "
          f"held-out BRR {res.get('held_out_brr'):.3f} | Anthropic {res.get('anthropic_sycophancy_rate'):.3f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
