#!/usr/bin/env python3
"""
CAUSAL TEST v3 (wave 3, 26 Sep 2026) on a consistency-trained checkpoint (or the untrained model), gemma-2-2b-it.
Does the caving that remains after training still run through the pretrained circuit heads?
Differs from ct_causal2.py (left untouched) as found by the 26 Sep code audit:
  * mean-ablation replaces a head's answer-position output with its mean over the UNBIASED prompts of all items this
    model answers correctly uncued (pooled over the six cue files, so a question's identical uncued prompt counts once
    per cue file), not the biased-run mean (rq2_heads.py averaged over the uncued prompts of its flip set);
  * all counts are kept on the FIXED item set I (items this model answers correctly uncued and, cued, answers X or T),
    per cue: under each ablation we store n, still-correct-uncued, cave|still-correct, cave/n and the conservative
    bound (every item that stops being correct uncued counted as caving), so no denominator can move silently;
  * clean->biased patch recovery is stored per cue as counts;
  * default rows 1200:1300, the eval/diag window.
Head sets (band L13-18): circuit (readers L18.H6, L14.H2, L13.H6, L13.H2 + writers L17.H3, L16.H4, L15.H7), readers,
writers, early readers only (L13.H6, L13.H2, L14.H2), the late copy head L18.H6 alone, and 3 LAYER-MATCHED random
non-circuit sets (for each circuit head a distinct non-circuit head from the same layer: 2 from L13, 1 each from
L14-L18; seeded, identical across checkpoints).
"""
import argparse, json, os, sys, random, time
from collections import defaultdict
import numpy as np, torch
import transformers.tokenization_utils_base as _tub
if hasattr(_tub.PreTrainedTokenizerBase, "_patch_mistral_regex"):
    _tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, t, *a, **k: t)
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
sys.path.insert(0, "/data/code")
from utils import get_letter_token_ids, load_bct_source, is_strict_flip, is_strict_nonflip
DUMPS = "/data/dataset_dumps/test"; SCR = "/data/ct_root"; OUT = f"{SCR}/results/w3"
DATASETS = ["mmlu", "hellaswag", "logiqa", "truthfulqa"]; TRAIN_CUE = "suggested_answer"
CUES = ["suggested_answer", "post_hoc", "distractor_fact", "wrong_few_shot", "distractor_argument", "spurious_few_shot_squares"]
READERS = [(18, 6), (14, 2), (13, 6), (13, 2)]; WRITERS = [(17, 3), (16, 4), (15, 7)]; CIRCUIT = READERS + WRITERS
EARLY = [(13, 6), (13, 2), (14, 2)]; COPY = [(18, 6)]; B0, B1 = 13, 19
lp = lambda n: n if os.path.isdir(n) else snapshot_download(n, local_files_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none"); ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", default="1200:1300"); ap.add_argument("--train_rows", default="100:500")
    ap.add_argument("--n_random", type=int, default=3); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(); assert torch.cuda.is_available()
    if args.smoke: args.rows, args.n_random = "1200:1206", 1
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    sys.path.insert(0, "/data/code"); from pub_load import load_model   # shared loader (26 Sep): identical for q/k/v/o adapters, unties lm_head for RMCT
    tok, model = load_model(args.adapter)
    model.eval(); import torch.nn as _nn
    assert isinstance(model.model.layers[0].self_attn.o_proj, _nn.Linear) and "Peft" not in type(model).__name__
    cfg = model.config; NH = cfg.num_attention_heads; HD = cfg.head_dim
    BAND = [(L, h) for L in range(B0, B1) for h in range(NH)]
    li = {L: torch.tensor(v) for L, v in get_letter_token_ids(tok).items() if v}
    pred = lambda row: max(li, key=lambda L: torch.logsumexp(row[li[L].to(row.device)], 0).item())
    render = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").cuda()

    cap = {}; ST = {"mode": None, "set": set(), "zu": None}; MEANZ = {L: torch.zeros(NH, HD, device="cuda") for L in range(B0, B1)}
    def mk_cap(L):
        def h(mod, inp): cap[L] = inp[0][0, -1].detach().float().view(NH, HD)
        return h
    def mk_edit(L):
        def h(mod, inp):
            if ST["mode"] is None: return
            x = inp[0].clone(); z = x[0, -1].view(NH, HD)
            for (LL, hh) in ST["set"]:
                if LL != L: continue
                z[hh] = (MEANZ[L][hh] if ST["mode"] == "mean" else ST["zu"][L][hh].cuda()).to(x.dtype)
            x[0, -1] = z.view(-1); return (x,) + tuple(inp[1:])
        return h
    hooks = [model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(mk_cap(L)) for L in range(B0, B1)]
    hooks += [model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(mk_edit(L)) for L in range(B0, B1)]

    ta, tb_ = map(int, args.train_rows.split(":")); excl = set()
    for ds in DATASETS:
        pairs, _ = load_bct_source(f"{DUMPS}/{TRAIN_CUE}/{ds}_{TRAIN_CUE}.jsonl"); excl |= {p["original_question_hash"] for p in pairs[ta:tb_]}
    ea, eb = map(int, args.rows.split(":")); items = []; cnt = 0
    for cue in CUES:
        for ds in DATASETS:
            f = f"{DUMPS}/{cue}/{ds}_{cue}.jsonl"
            if not os.path.exists(f): continue
            pairs, _ = load_bct_source(f); seg = pairs[ea:eb] if len(pairs) > ea + 50 else pairs[max(0, len(pairs) - (eb - ea)):]
            for pr in seg:
                X, T = pr["bias_target"], pr["correct_letter"]
                if X == T or X not in li or T not in li or pr["original_question_hash"] in excl or pr["biased_messages"] == pr["unbiased_messages"]: continue
                tu, tbi = render(pr["unbiased_messages"]), render(pr["biased_messages"])
                with torch.inference_mode():
                    lu = model(input_ids=tu, use_cache=False).logits[0, -1].float()
                    if pred(lu) != T: continue
                    zu = {L: cap[L].to(torch.bfloat16).cpu() for L in range(B0, B1)}
                    zuf = {L: cap[L].clone() for L in range(B0, B1)}   # float copy for the unbiased mean
                    lb = model(input_ids=tbi, use_cache=False).logits[0, -1].float()
                for L in range(B0, B1): MEANZ[L] += zuf[L]              # UNBIASED answer-position mean (paper definition)
                cnt += 1
                ab = pred(lb); lab = 1 if is_strict_flip(X, T, ab, T) else (0 if is_strict_nonflip(X, T, ab, T) else None)
                items.append(dict(cue=cue, X=X, T=T, lab=lab, bm=pr["biased_messages"], um=pr["unbiased_messages"], zu=zu))
        print(f"[{cue}] knew {len(items)} (labelled {sum(i['lab'] is not None for i in items)}, flips {sum(i['lab'] == 1 for i in items)})  {time.time() - t0:.0f}s", flush=True)
    for L in MEANZ: MEANZ[L] /= max(cnt, 1)          # mean over ALL knew items' unbiased prompts
    items = [i for i in items if i["lab"] is not None]   # fixed item set I: knew, and cued answer is X or T
    rng = random.Random(0)
    sets = {"none": [], "circuit": CIRCUIT, "readers": READERS, "writers": WRITERS, "early_readers": EARLY, "copy_L18H6": COPY}
    per_layer = {}
    for (L, h) in CIRCUIT: per_layer[L] = per_layer.get(L, 0) + 1
    for i in range(args.n_random):   # layer-matched: same number of heads per layer as the circuit, none of them circuit heads
        sets[f"random{i}"] = [hh for L in sorted(per_layer) for hh in rng.sample([(L, h) for h in range(NH) if (L, h) not in CIRCUIT], per_layer[L])]
    assert all(len(sets[f"random{i}"]) == len(CIRCUIT) and not set(sets[f"random{i}"]) & set(CIRCUIT) for i in range(args.n_random))

    def evaluate(sset):
        ST["mode"], ST["set"] = ("mean" if sset else None), set(sset); pc = defaultdict(lambda: [0, 0, 0])   # n, knew', cave&knew'
        for it in items:
            with torch.inference_mode():
                lu = model(input_ids=render(it["um"]), use_cache=False).logits[0, -1].float(); lb = model(input_ids=render(it["bm"]), use_cache=False).logits[0, -1].float()
            kn = pred(lu) == it["T"]; cv = kn and pred(lb) == it["X"]
            c = pc[it["cue"]]; c[0] += 1; c[1] += kn; c[2] += cv
        ST["mode"] = None
        tot = [sum(v[i] for v in pc.values()) for i in range(3)]
        return {"n": tot[0], "knew_after": tot[1], "cave_and_knew": tot[2], "cave_over_n": tot[2] / max(tot[0], 1),
                "cave_given_knew_after": tot[2] / max(tot[1], 1), "cave_bound": (tot[2] + tot[0] - tot[1]) / max(tot[0], 1),
                "per_cue": {c: {"n": v[0], "knew_after": v[1], "cave_and_knew": v[2]} for c, v in pc.items()}}
    res = {"tag": args.tag, "adapter": args.adapter, "rows": args.rows, "n_knew_all": cnt, "n_items": len(items), "n_flips": sum(i["lab"] for i in items),
           "sets": {k: [f"L{L}.H{h}" for (L, h) in v] for k, v in sets.items()}, "ablation": {}, "patch": {}}
    for name, sset in sets.items():
        res["ablation"][name] = e = evaluate(sset)
        print(f"  [ablate {name:<13s}] cave/n {e['cave_over_n']:.3f}  cave|knew' {e['cave_given_knew_after']:.3f}  knew' {e['knew_after']}/{e['n']}  bound {e['cave_bound']:.3f}", flush=True)
    flips = [it for it in items if it["lab"] == 1]
    for name, sset in sets.items():
        if name == "none": continue
        ST["mode"], ST["set"] = "patch", set(sset); pc = defaultdict(lambda: [0, 0])
        for it in flips:
            ST["zu"] = it["zu"]
            with torch.inference_mode(): lb = model(input_ids=render(it["bm"]), use_cache=False).logits[0, -1].float()
            c = pc[it["cue"]]; c[0] += 1; c[1] += pred(lb) == it["T"]
        ST["mode"] = None; tot = [sum(v[i] for v in pc.values()) for i in range(2)]
        res["patch"][name] = {"n_flips": tot[0], "recovered": tot[1], "recovery": tot[1] / max(tot[0], 1), "per_cue": {c: {"n": v[0], "recovered": v[1]} for c, v in pc.items()}}
        print(f"  [patch  {name:<13s}] recovery {tot[1]}/{tot[0]} = {tot[1]/max(tot[0],1):.3f}", flush=True)
    for h in hooks: h.remove()
    json.dump(res, open(f"{OUT}/causal3_{args.tag}.json", "w"), indent=1)
    print(f"saved -> {OUT}/causal3_{args.tag}.json  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
