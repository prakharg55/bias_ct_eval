#!/usr/bin/env python3
"""
ct_evalfs7 (1 Oct 2026; prereg NOTE "PRE-WRITE-UP AUDIT", item 2 FEW-SHOT CUES: TEST-SLOT ANSWERS).
The BIASED prompts of the two few-shot cues (wrong_few_shot, spurious_few_shot_squares) hold n labelled demonstrations
("Label: (L)" / "The best answer is: (L)") before the test question. Models often answer EVERY demonstration in order
("(A)\n(B)\n(D)..."), so parser v2's first-line rule (and the first-position readout) return the answer to demonstration 1,
and ct_eval7's 16 new tokens often stop before the test question's slot. This instrument regenerates exactly ct_eval7's
biased prompts of these two cues (same item set and filters, same chat template, greedy decoding, batch 1) with up to
--max_new (128) new tokens and stores the full continuation plus the demonstrations' labels read from the prompt.
REGRESSION: the first 16 new tokens must decode to exactly the text stored by ct_eval7 for the same checkpoint
(results/<ref_root>/evalitems_<ref_tag>.json); the run aborts if any item differs. Scoring is offline (ana_fs7.py) with
the declared rule. Output results/fs7/fs_<tag>.json.
Usage: ct_evalfs7.py --model gemma2b|llama8b --adapter <dir|none> --tag T --ref_tag <ct_eval7 tag> [--ref_root pub7] [--merge merged]
"""
import argparse, json, os, re, sys, time
import torch
sys.path.insert(0, "/data/code"); sys.path.insert(0, "/data/code")
from utils import load_bct_source
from pub_load3 import load_model, MODELS
DUMPS = "/data/dataset_dumps/test"; SCR = "/data/ct_root"
DATASETS = ["mmlu", "hellaswag", "logiqa", "truthfulqa"]; TRAIN_CUE = "suggested_answer"
FS = ["wrong_few_shot", "spurious_few_shot_squares"]   # ct_eval7's order of these two cues
LABEL = {"spurious_few_shot_squares": re.compile(r"\nLabel: \(([A-P])\)"), "wrong_few_shot": re.compile(r"The best answer is: \(([A-P])\)")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", default="none"); ap.add_argument("--tag", required=True); ap.add_argument("--ref_tag", required=True)
    ap.add_argument("--ref_root", default="pub7"); ap.add_argument("--merge", default="merged", choices=["merged", "unmerged"]); ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--eval_rows", default="1200:1300"); ap.add_argument("--train_rows", default="100:500"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="gemma2b", choices=sorted(MODELS)); args = ap.parse_args(); assert torch.cuda.is_available()
    OUT = f"{SCR}/results/fs7"; os.makedirs(OUT, exist_ok=True); t0 = time.time()
    ref = {(i["cue"], i["ds"], i["hash"], i["X"], i["T"]): i for i in json.load(open(f"{SCR}/results/{args.ref_root}/evalitems_{args.ref_tag}.json"))["items"] if i["cue"] in FS}
    tok, model = load_model(args.adapter, model=args.model, merge=(args.merge == "merged")); model.eval()
    eos = model.generation_config.eos_token_id; eos = eos if isinstance(eos, list) else [eos]; pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    ta, tb_ = map(int, args.train_rows.split(":")); excl = set()
    for ds in DATASETS:
        pairs, _ = load_bct_source(f"{DUMPS}/{TRAIN_CUE}/{ds}_{TRAIN_CUE}.jsonl"); excl |= {p["original_question_hash"] for p in pairs[ta:tb_]}
    ea, eb = map(int, args.eval_rows.split(":")); items = []; n_lim = 3 if args.smoke else None
    for cue in FS:
        for ds in DATASETS:
            pairs, _ = load_bct_source(f"{DUMPS}/{cue}/{ds}_{cue}.jsonl"); seg = pairs[ea:eb] if len(pairs) > ea + 50 else pairs[max(0, len(pairs) - (eb - ea)):]
            sel = [p for p in seg if p["original_question_hash"] not in excl and p["bias_target"] != p["correct_letter"] and p["biased_messages"] != p["unbiased_messages"]]   # ct_eval7's filter
            for p in sel[:n_lim]:
                key = (cue, ds, p["original_question_hash"], p["bias_target"], p["correct_letter"]); r = ref[key]
                x = tok.apply_chat_template(p["biased_messages"], add_generation_prompt=True, return_tensors="pt").cuda()
                with torch.no_grad():
                    g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, temperature=None, top_p=None, top_k=None,
                                       max_new_tokens=args.max_new, pad_token_id=pad)
                new = g[0, x.shape[1]:]; text = tok.decode(new, skip_special_tokens=True); text16 = tok.decode(new[:16], skip_special_tokens=True)
                labels = LABEL[cue].findall(p["biased_messages"][-1]["content"]); assert labels, key
                items.append(dict(cue=cue, ds=ds, hash=key[2], X=key[3], T=key[4], n_demos=len(labels), demo_labels=labels, text=text, text16=text16,
                                  same16=bool(text16 == r["text_b"]), ref_gen_b=r["gen_b"], ref_gen_u=r["gen_u"], n_new=int(new.shape[0]),
                                  ended=bool(any(int(t) in eos for t in new.tolist()))))
        print(f"  [{args.tag}] {cue}: {len(items)} items so far ({time.time() - t0:.0f}s)", flush=True)
    n_same = sum(i["same16"] for i in items); n_end = sum(i["ended"] for i in items)
    res = {"tag": args.tag, "ref_tag": args.ref_tag, "ref_root": args.ref_root, "adapter": args.adapter, "model": MODELS[args.model], "merge": args.merge, "max_new": args.max_new,
           "eval_rows": args.eval_rows, "n_items": len(items), "regression_same16": n_same, "ended_with_eos": n_end, "seconds": time.time() - t0, "items": items}
    print(f"  [{args.tag}] items {len(items)} | first 16 tokens == stored ct_eval7 text: {n_same}/{len(items)} | ended with EOS {n_end}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    sfx = "_smoke" if args.smoke else ""
    if n_same != len(items):   # keep the evidence under a name the runner's SKIPIF does not treat as done
        json.dump(res, open(f"{OUT}/fs_{args.tag}{sfx}.REGRESSION_FAILED.json", "w"), indent=1)
        raise SystemExit(f"REGRESSION FAILED: {len(items) - n_same} items' first 16 tokens differ from ct_eval7's stored text")
    json.dump(res, open(f"{OUT}/fs_{args.tag}{sfx}.json", "w"), indent=1)
    print("saved ->", f"{OUT}/fs_{args.tag}{sfx}.json", flush=True)


if __name__ == "__main__":
    main()
