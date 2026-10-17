# patching.py (1 Oct 2026; prereg audit NOTE item 1, MERGE FIDELITY): patching_merged.py UNCHANGED except (i)
# the model is loaded UNMERGED by load_unmerged (PEFT lora.Linear layers in place of the merged Linear; same
# forward computation as training), (ii) the o_proj type guard also accepts peft's lora.Linear (the hooks read
# / edit the o_proj INPUT, which the LoRA wrapper passes to both base and adapter paths),
# (iii) output dir results/w7u.
"""Patch the circuit heads and count how many of the remaining cue-induced errors this corrects.

Uses the items that the checkpoint answers correctly without the cue and with the cued letter when
cued. For each head set, the heads' answer-position outputs are replaced by their values on the
uncued prompt. Head sets: the circuit (set per model in HEADS), its readers, its writers, a few named
subsets and three random sets matched to the circuit's layers. The script also records cue following
when each head set is mean-ablated. The adapter stays unmerged. Gemma runs use --agg sum, the readout
of the Gemma baseline from patching_v1.py. Gemma checkpoints that fail the test-set readout gate are
rerun with --agg max under tags that start with gemma2b_max_. Llama runs use --agg max.

Usage: python -m ctcue.evaluation.patching --model gemma2b --adapter <dir|none> --tag <name> --agg sum
Output: $CT_ROOT/results/w7u/causal3_<tag>.json
"""
import argparse, json, os, random, time
from collections import defaultdict
import numpy as np, torch
import transformers.tokenization_utils_base as _tub

if hasattr(_tub.PreTrainedTokenizerBase, "_patch_mistral_regex"):
    _tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, t, *a, **k: t)
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from ctcue.common.dumps import load_bct_source, is_strict_flip, is_strict_nonflip

# v7: token sets of letter_tokens, letter score = max over forms (letter_readout.agg)
from ctcue.common import letter_readout
from ctcue.common.letter_readout import get_letter_token_ids
from ctcue.paths import CT_ROOT, BCT_DUMPS

DUMPS = f"{BCT_DUMPS}/test"
SCR = CT_ROOT
OUT = f"{SCR}/results/w7u"
DATASETS = ["mmlu", "hellaswag", "logiqa", "truthfulqa"]
TRAIN_CUE = "suggested_answer"
CUES = [
    "suggested_answer",
    "post_hoc",
    "distractor_fact",
    "wrong_few_shot",
    "distractor_argument",
    "spurious_few_shot_squares",
]
HEADS = {
    "gemma2b": dict(
        READERS=[(18, 6), (14, 2), (13, 6), (13, 2)],
        WRITERS=[(17, 3), (16, 4), (15, 7)],
        B0=13,
        B1=19,
        EXTRA={"early_readers": [(13, 6), (13, 2), (14, 2)], "copy_L18H6": [(18, 6)]},
    ),
    "llama8b": dict(
        READERS=[(18, 28), (15, 8)],
        WRITERS=[(17, 24)],
        B0=15,
        B1=19,
        EXTRA={"reader_L15H8": [(15, 8)], "reader_L18H28": [(18, 28)]},
    ),
}
lp = lambda n: n if os.path.isdir(n) else snapshot_download(n, local_files_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", default="1200:1300")
    ap.add_argument("--train_rows", default="100:500")
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="gemma2b", choices=["gemma2b", "llama8b"])
    ap.add_argument("--agg", default="max", choices=["max", "sum"])
    args = ap.parse_args()
    letter_readout.MODE = args.agg
    assert torch.cuda.is_available()
    Hs = HEADS[args.model]
    READERS, WRITERS, B0, B1 = Hs["READERS"], Hs["WRITERS"], Hs["B0"], Hs["B1"]
    CIRCUIT = READERS + WRITERS
    if args.smoke:
        args.rows, args.n_random = "1200:1206", 1
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    # UNMERGED (prereg audit NOTE item 1)
    from ctcue.common.load_unmerged import load_model

    tok, model = load_model(args.adapter, model=args.model)
    model.eval()
    import torch.nn as _nn
    import peft.tuners.lora as _pl

    assert (
        isinstance(model.model.layers[0].self_attn.o_proj, (_nn.Linear, _pl.Linear))
        and "Peft" not in type(model).__name__
    )
    cfg = model.config
    NH = cfg.num_attention_heads
    HD = cfg.head_dim
    BAND = [(L, h) for L in range(B0, B1) for h in range(NH)]
    li = {L: torch.tensor(v) for L, v in get_letter_token_ids(tok).items() if v}
    pred = lambda row: max(li, key=lambda L: letter_readout.score(row, li[L]).item())
    render = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").cuda()

    cap = {}
    ST = {"mode": None, "set": set(), "zu": None}
    MEANZ = {L: torch.zeros(NH, HD, device="cuda") for L in range(B0, B1)}

    def mk_cap(L):
        def h(mod, inp):
            cap[L] = inp[0][0, -1].detach().float().view(NH, HD)

        return h

    def mk_edit(L):
        def h(mod, inp):
            if ST["mode"] is None:
                return
            x = inp[0].clone()
            z = x[0, -1].view(NH, HD)
            for LL, hh in ST["set"]:
                if LL != L:
                    continue
                z[hh] = (MEANZ[L][hh] if ST["mode"] == "mean" else ST["zu"][L][hh].cuda()).to(x.dtype)
            x[0, -1] = z.view(-1)
            return (x,) + tuple(inp[1:])

        return h

    hooks = [
        model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(mk_cap(L)) for L in range(B0, B1)
    ]
    hooks += [
        model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(mk_edit(L)) for L in range(B0, B1)
    ]

    ta, tb_ = map(int, args.train_rows.split(":"))
    excl = set()
    for ds in DATASETS:
        pairs, _ = load_bct_source(f"{DUMPS}/{TRAIN_CUE}/{ds}_{TRAIN_CUE}.jsonl")
        excl |= {p["original_question_hash"] for p in pairs[ta:tb_]}
    ea, eb = map(int, args.rows.split(":"))
    items = []
    cnt = 0
    for cue in CUES:
        for ds in DATASETS:
            f = f"{DUMPS}/{cue}/{ds}_{cue}.jsonl"
            if not os.path.exists(f):
                continue
            pairs, _ = load_bct_source(f)
            seg = pairs[ea:eb] if len(pairs) > ea + 50 else pairs[max(0, len(pairs) - (eb - ea)) :]
            for pr in seg:
                X, T = pr["bias_target"], pr["correct_letter"]
                if (
                    X == T
                    or X not in li
                    or T not in li
                    or pr["original_question_hash"] in excl
                    or pr["biased_messages"] == pr["unbiased_messages"]
                ):
                    continue
                tu, tbi = render(pr["unbiased_messages"]), render(pr["biased_messages"])
                with torch.inference_mode():
                    lu = model(input_ids=tu, use_cache=False).logits[0, -1].float()
                    if pred(lu) != T:
                        continue
                    zu = {L: cap[L].to(torch.bfloat16).cpu() for L in range(B0, B1)}
                    zuf = {L: cap[L].clone() for L in range(B0, B1)}  # float copy for the unbiased mean
                    lb = model(input_ids=tbi, use_cache=False).logits[0, -1].float()
                # UNBIASED answer-position mean (paper definition)
                for L in range(B0, B1):
                    MEANZ[L] += zuf[L]
                cnt += 1
                ab = pred(lb)
                lab = 1 if is_strict_flip(X, T, ab, T) else (0 if is_strict_nonflip(X, T, ab, T) else None)
                items.append(
                    dict(
                        cue=cue,
                        X=X,
                        T=T,
                        lab=lab,
                        bm=pr["biased_messages"],
                        um=pr["unbiased_messages"],
                        zu=zu,
                    )
                )
        print(
            f"[{cue}] knew {len(items)} (labelled {sum(i['lab'] is not None for i in items)}, flips {sum(i['lab'] == 1 for i in items)})  {time.time() - t0:.0f}s",
            flush=True,
        )
    for L in MEANZ:
        MEANZ[L] /= max(cnt, 1)  # mean over ALL knew items' unbiased prompts
    items = [i for i in items if i["lab"] is not None]  # fixed item set I: knew, and cued answer is X or T
    rng = random.Random(0)
    sets = {"none": [], "circuit": CIRCUIT, "readers": READERS, "writers": WRITERS, **Hs["EXTRA"]}
    per_layer = {}
    for L, h in CIRCUIT:
        per_layer[L] = per_layer.get(L, 0) + 1
    # layer-matched: same number of heads per layer as the circuit, none of them circuit heads
    for i in range(args.n_random):
        sets[f"random{i}"] = [
            hh
            for L in sorted(per_layer)
            for hh in rng.sample([(L, h) for h in range(NH) if (L, h) not in CIRCUIT], per_layer[L])
        ]
    assert all(
        len(sets[f"random{i}"]) == len(CIRCUIT) and not set(sets[f"random{i}"]) & set(CIRCUIT)
        for i in range(args.n_random)
    )

    def evaluate(sset):
        # n, knew', cave&knew'
        ST["mode"], ST["set"] = ("mean" if sset else None), set(sset)
        pc = defaultdict(lambda: [0, 0, 0])
        for it in items:
            with torch.inference_mode():
                lu = model(input_ids=render(it["um"]), use_cache=False).logits[0, -1].float()
                lb = model(input_ids=render(it["bm"]), use_cache=False).logits[0, -1].float()
            kn = pred(lu) == it["T"]
            cv = kn and pred(lb) == it["X"]
            c = pc[it["cue"]]
            c[0] += 1
            c[1] += kn
            c[2] += cv
        ST["mode"] = None
        tot = [sum(v[i] for v in pc.values()) for i in range(3)]
        return {
            "n": tot[0],
            "knew_after": tot[1],
            "cave_and_knew": tot[2],
            "cave_over_n": tot[2] / max(tot[0], 1),
            "cave_given_knew_after": tot[2] / max(tot[1], 1),
            "cave_bound": (tot[2] + tot[0] - tot[1]) / max(tot[0], 1),
            "per_cue": {c: {"n": v[0], "knew_after": v[1], "cave_and_knew": v[2]} for c, v in pc.items()},
        }

    res = {
        "tag": args.tag,
        "adapter": args.adapter,
        "model": args.model,
        "rows": args.rows,
        "n_knew_all": cnt,
        "n_items": len(items),
        "n_flips": sum(i["lab"] for i in items),
        "sets": {k: [f"L{L}.H{h}" for (L, h) in v] for k, v in sets.items()},
        "ablation": {},
        "patch": {},
    }
    for name, sset in sets.items():
        res["ablation"][name] = e = evaluate(sset)
        print(
            f"  [ablate {name:<13s}] cave/n {e['cave_over_n']:.3f}  cave|knew' {e['cave_given_knew_after']:.3f}  knew' {e['knew_after']}/{e['n']}  bound {e['cave_bound']:.3f}",
            flush=True,
        )
    flips = [it for it in items if it["lab"] == 1]
    for name, sset in sets.items():
        if name == "none":
            continue
        ST["mode"], ST["set"] = "patch", set(sset)
        pc = defaultdict(lambda: [0, 0])
        for it in flips:
            ST["zu"] = it["zu"]
            with torch.inference_mode():
                lb = model(input_ids=render(it["bm"]), use_cache=False).logits[0, -1].float()
            c = pc[it["cue"]]
            c[0] += 1
            c[1] += pred(lb) == it["T"]
        ST["mode"] = None
        tot = [sum(v[i] for v in pc.values()) for i in range(2)]
        res["patch"][name] = {
            "n_flips": tot[0],
            "recovered": tot[1],
            "recovery": tot[1] / max(tot[0], 1),
            "per_cue": {c: {"n": v[0], "recovered": v[1]} for c, v in pc.items()},
        }
        print(f"  [patch  {name:<13s}] recovery {tot[1]}/{tot[0]} = {tot[1]/max(tot[0],1):.3f}", flush=True)
    for h in hooks:
        h.remove()
    json.dump(res, open(f"{OUT}/causal3_{args.tag}.json", "w"), indent=1)
    print(f"saved -> {OUT}/causal3_{args.tag}.json  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
