"""Run the AttCT authors' own evaluator on a checkpoint.

Calls their SycophancyEvaluator (c-wei/AttCT at commit b3c1896, unchanged) with the adapter unmerged:
the MMLU bias rate and accuracy, the held-out bias rate and the Anthropic sycophancy rate. The script
makes the evaluator load the Anthropic data from the Hugging Face cache instead of downloading it, so
the environment variable HF_HOME must be set and the cache must hold Anthropic/model-written-evals.

Usage: python -m ctcue.evaluation.attct_evaluator --model gemma2b --adapter <dir|none> --tag <name>
Output: $CT_ROOT/results/attct_eval/attct_eval_<tag>.json and syco_<tag>.csv
"""

import argparse, json, os, sys, time
from ctcue.paths import CT_ROOT, ATTCT_REPO

REPO = ATTCT_REPO
OUT = f"{CT_ROOT}/results/attct_eval"
MODEL_IDS = {"gemma2b": "google/gemma-2-2b-it", "llama8b": "meta-llama/Llama-3.1-8B-Instruct"}
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--substrate", default="all", choices=["all", "anthropic"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    import torch, wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    import glob, datasets
    from experiments.sycophancy import evaluate_sycophancy as ES
    from experiments.sycophancy.evaluate_sycophancy import SycophancyEvaluator

    snap = glob.glob(f"{os.environ['HF_HOME']}/hub/datasets--Anthropic--model-written-evals/snapshots/*")[0]
    _ld = datasets.load_dataset

    # same files, local path (offline cache-key issue)
    def _load(name, *args, data_files=None, split=None, **kw):
        if name == "Anthropic/model-written-evals":
            return _ld("json", data_files=os.path.join(snap, data_files), split=split)
        return (
            _ld(name, *args, data_files=data_files, split=split, **kw)
            if data_files is not None
            else _ld(name, *args, split=split, **kw)
        )

    ES.load_dataset = _load
    # evaluator calls wandb.log
    wandb.init(mode="disabled")
    path = snapshot_download(MODEL_IDS[a.model], local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # run.py:400-402
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=a.attn
    ).cuda()
    if a.adapter != "none":
        from peft import PeftModel

        # unmerged, as in run.py post-train eval
        model = PeftModel.from_pretrained(model, a.adapter)
        model.enable_adapter_layers()
        print(f"[attct_eval] adapter {a.adapter} (unmerged PEFT)", flush=True)
    model.eval()
    # their relative dataset paths
    os.chdir(REPO)
    ev = SycophancyEvaluator(
        model,
        tok,
        torch.device("cuda"),
        max_samples=a.n,
        prefix=a.tag,
        results_csv=f"{OUT}/syco_{a.tag}.csv",
        held_out_path="datasets/sycophancy_bct/control_cot_eval.jsonl",
        anthropic_eval=True,
    )
    if a.substrate == "anthropic":
        ar = ev._evaluate_anthropic()
        ev._report_anthropic(ar)
        f = f"{OUT}/attct_eval_{a.tag}.json"
        res = json.load(open(f)) if os.path.exists(f) else {}
        res["anthropic_sycophancy_rate"] = ar["sycophancy_rate"]
        res["anthropic_n_questions"] = ar["n_questions"]
        res.update({f"anthropic_{k}": v for k, v in ar.items() if k.startswith("rate_")})
        json.dump(res, open(f, "w"), indent=1, default=str)
        print(
            f"[attct_eval] {a.tag}: Anthropic sycophancy rate {ar['sycophancy_rate']:.3f} (n={ar['n_questions']})",
            flush=True,
        )
        return
    res = ev.evaluate()
    res.update(
        tag=a.tag, model=MODEL_IDS[a.model], adapter=a.adapter, attn=a.attn, n=a.n, seconds=time.time() - t0
    )
    json.dump(res, open(f"{OUT}/attct_eval_{a.tag}.json", "w"), indent=1, default=str)
    print(
        f"[attct_eval] {a.tag}: MMLU BRR {res.get('brr'):.3f} clean_acc {res.get('clean_acc'):.3f} NS {res.get('not_sycophantic_pct'):.3f} F1 {res.get('f1'):.3f} | "
        f"held-out BRR {res.get('held_out_brr'):.3f} | Anthropic {res.get('anthropic_sycophancy_rate'):.3f}  ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
