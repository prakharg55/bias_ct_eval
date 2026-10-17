# attention_probe.py (1 Oct 2026; prereg audit NOTE item 1, MERGE FIDELITY): ct_probe7.py UNCHANGED except (i)
# the model is loaded UNMERGED by load_unmerged (PEFT lora.Linear layers in place of the merged Linear; same
# forward computation as training), (ii) the o_proj type guard also accepts peft's lora.Linear (the hooks read
# / edit the o_proj INPUT, which the LoRA wrapper passes to both base and adapter paths),
# (iii) output dir results/w7u.
"""Record letter probabilities and attention from the answer position for every test item.

For the uncued and the cued prompt of each test item: the probabilities of the answer letters and,
for every head, the attention from the final position to the cued and correct letters (split by where
they occur) and, in the cued prompt, to the cue span. analysis/score_gemma.py,
analysis/mechanism_checks.py, the letter-prior control and the paper tables read these files. The
adapter stays unmerged. Gemma runs use --agg sum, the readout of the Gemma baseline from
attention_probe_v1.py. Gemma checkpoints that fail the test-set readout gate are rerun with --agg max
under tags that start with gemma2b_max_. Llama runs use --agg max. The original models' files in
$CT_ROOT/results/w7 (probe_llama8b_none, probe_gemma2b_max_none) came from an earlier copy of this
script that merged adapters and wrote to w7. Without an adapter it computes the same thing.

Usage: python -m ctcue.evaluation.attention_probe --model gemma2b --adapter <dir|none> --tag <name> --agg sum
Output: $CT_ROOT/results/w7u/probe_<tag>.json and .npz
"""
import argparse, difflib, json, os, time
import numpy as np, torch
import transformers.tokenization_utils_base as _tub

if hasattr(_tub.PreTrainedTokenizerBase, "_patch_mistral_regex"):
    _tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, t, *a, **k: t)
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from ctcue.common.dumps import load_bct_source, get_letter_token_ids as _ids5

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
PROBE_LAYERS = [8, 12, 14, 16, 18, 22]
lp = lambda n: n if os.path.isdir(n) else snapshot_download(n, local_files_only=True)


def cue_positions(ut, bt):
    """Biased-prompt positions inserted/replaced relative to the unbiased prompt (difflib on token ids), with every
    INSERT block canonicalised to its leftmost equivalent alignment (shifted left while the token before the block
    equals the block's last token, without crossing into the preceding changed block). Without this, difflib places
    ambiguous insertions one token late (e.g. wrong_few_shot's cue ends "...\n\n" before "Answer with just ...", and
    the block would swallow the final instruction's first token)."""
    pos, prev_eq_start = set(), None
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ut, bt, autojunk=False).get_opcodes():
        if tag == "equal":
            prev_eq_start = j1
            continue
        if tag == "insert" and prev_eq_start is not None:
            while j1 - 1 >= prev_eq_start and bt[j1 - 1] == bt[j2 - 1]:
                j1 -= 1
                j2 -= 1
        if tag in ("insert", "replace"):
            pos.update(range(j1, j2))
        prev_eq_start = None
    return sorted(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", default="1200:1300")
    ap.add_argument("--train_rows", default="100:500")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="gemma2b", choices=["gemma2b", "llama8b"])
    ap.add_argument("--agg", default="max", choices=["max", "sum"])
    args = ap.parse_args()
    letter_readout.MODE = args.agg
    assert torch.cuda.is_available()
    if args.smoke:
        args.rows = "1200:1206"
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
    NL, NH = cfg.num_hidden_layers, cfg.num_attention_heads
    HD = cfg.head_dim
    li = {L: torch.tensor(v) for L, v in get_letter_token_ids(tok).items() if v}
    letters = sorted(li)
    render = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").cuda()
    cap = {}
    hooks = []
    PAREN = {
        int(t)
        for t in (tok.encode("(", add_special_tokens=False) + tok.encode(" (", add_special_tokens=False))
    }
    # v6: "(L" / "(L)" single tokens (empty for gemma)
    _b5 = _ids5(tok)
    PARENLETTER = {int(t) for L in li for t in li[L].tolist() if t not in _b5[L]}
    print(f"[{args.tag}] parenthesised letter tokens: {len(PARENLETTER)}", flush=True)

    def layout(msgs, ids):
        """instruction start token index and per-position 'previous token is an opening parenthesis' flags."""
        s_ = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(s_, add_special_tokens=False, return_offsets_mapping=True)
        assert enc["input_ids"] == ids, "string-then-tokenize differs from apply_chat_template ids"
        ci = s_.rfind("Answer with just the single letter")
        assert ci >= 0
        it = next(k for k, (a, b) in enumerate(enc["offset_mapping"]) if a >= ci)
        # v6: "(L" counts as its own "("
        prevp = torch.tensor(
            [False] + [ids[k - 1] in PAREN or ids[k] in PARENLETTER for k in range(1, len(ids))],
            device="cuda",
        )
        return it, prevp

    def forward(ids):
        cap.clear()
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False, output_attentions=True)
        row = out.logits[0, -1].float()
        sc = torch.stack([letter_readout.score(row, li[L]) for L in letters])
        att = torch.stack([a[0, :, -1, :].float() for a in out.attentions])  # [NL, NH, n]
        return row, sc, att

    ta, tb_ = map(int, args.train_rows.split(":"))
    excl = set()
    for ds in DATASETS:
        pairs, _ = load_bct_source(f"{DUMPS}/{TRAIN_CUE}/{ds}_{TRAIN_CUE}.jsonl")
        excl |= {p["original_question_hash"] for p in pairs[ta:tb_]}
    ea, eb = map(int, args.rows.split(":"))
    meta, PU, PB, AB, AU, CB, CU = [], [], [], [], [], [], []
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
                iu, ib = render(pr["unbiased_messages"]), render(pr["biased_messages"])
                bt, ut = ib[0].tolist(), iu[0].tolist()
                nb, nu = len(bt), len(ut)
                cue_pos = cue_positions(ut, bt)
                assert cue_pos and cue_pos[-1] < nb - 1
                xs, ts = set(li[X].tolist()), set(li[T].tolist())
                # unbiased forward
                row_u, sc_u, att_u = forward(iu)
                xm_u = torch.tensor([t in xs for t in ut], device="cuda")
                tm_u = torch.tensor([t in ts for t in ut], device="cuda")
                it_u, pp_u = layout(pr["unbiased_messages"], ut)
                ins_u = torch.arange(nu, device="cuda") >= it_u
                xl_u, xi_u, xo_u = xm_u & ~ins_u & pp_u, xm_u & ins_u, xm_u & ~ins_u & ~pp_u
                tl_u, ti_u, to_u = tm_u & ~ins_u & pp_u, tm_u & ins_u, tm_u & ~ins_u & ~pp_u
                au = torch.stack(
                    [
                        att_u[..., xl_u].sum(-1),
                        att_u[..., xi_u].sum(-1),
                        att_u[..., xo_u].sum(-1),
                        # [NL, NH, 7]
                        att_u[..., tl_u].sum(-1),
                        att_u[..., ti_u].sum(-1),
                        att_u[..., to_u].sum(-1),
                        att_u[..., 0],
                    ],
                    -1,
                )
                cls_u = torch.zeros(nu, dtype=torch.long, device="cuda")
                cls_u[xl_u] = 3
                cls_u[xi_u] = 4
                cls_u[xo_u] = 5
                cls_u[tl_u] = 7
                cls_u[ti_u] = 8
                cls_u[to_u] = 9
                cls_u[0] = 10
                am_u = cls_u[att_u.argmax(-1)]
                # biased forward
                row_b, sc_b, att_b = forward(ib)
                cm = torch.zeros(nb, dtype=torch.bool, device="cuda")
                cm[cue_pos] = True
                xm = torch.tensor([t in xs for t in bt], device="cuda")
                tm = torch.tensor([t in ts for t in bt], device="cuda")
                it_b, pp_b = layout(pr["biased_messages"], bt)
                ins_b = torch.arange(nb, device="cuda") >= it_b
                # safety net; expected 0 after canonicalisation (recorded per item)
                n_trim = int((cm & ins_b).sum())
                cm &= ~ins_b
                oc = ~cm
                lab_b = oc & ~ins_b & pp_b
                xin, xl, xi_, xo = cm & xm, oc & xm & lab_b, oc & xm & ins_b, oc & xm & ~lab_b & ~ins_b
                tin, tl, ti_, to_ = cm & tm, oc & tm & lab_b, oc & tm & ins_b, oc & tm & ~lab_b & ~ins_b
                ab = torch.stack(
                    [
                        att_b[..., cm].sum(-1),
                        att_b[..., xin].sum(-1),
                        att_b[..., xl].sum(-1),
                        att_b[..., xi_].sum(-1),
                        att_b[..., xo].sum(-1),
                        att_b[..., tin].sum(-1),
                        att_b[..., tl].sum(-1),
                        att_b[..., ti_].sum(-1),
                        # [NL, NH, 10]
                        att_b[..., to_].sum(-1),
                        att_b[..., 0],
                    ],
                    -1,
                )
                cls_b = torch.zeros(nb, dtype=torch.long, device="cuda")
                cls_b[cm] = 1
                cls_b[xin] = 2
                cls_b[xl] = 3
                cls_b[xi_] = 4
                cls_b[xo] = 5
                cls_b[tin] = 6
                cls_b[tl] = 7
                cls_b[ti_] = 8
                cls_b[to_] = 9
                cls_b[0] = 10
                am_b = cls_b[att_b.argmax(-1)]
                pu, pb = torch.softmax(sc_u, 0), torch.softmax(sc_b, 0)
                meta.append(
                    dict(
                        cue=cue,
                        ds=ds,
                        hash=pr["original_question_hash"],
                        X=X,
                        T=T,
                        nb=nb,
                        nu=nu,
                        ncue=len(cue_pos),
                        instr_b=it_b,
                        nx_in=int(xin.sum()),
                        nx_label=int(xl.sum()),
                        nx_instr=int(xi_.sum()),
                        nx_other=int(xo.sum()),
                        nt_in=int(tin.sum()),
                        nt_label=int(tl.sum()),
                        nt_instr=int(ti_.sum()),
                        nt_other=int(to_.sum()),
                        n_trim=n_trim,
                        knew=letters[int(pu.argmax())] == T,
                        cave=letters[int(pb.argmax())] == X,
                        pred_b=letters[int(pb.argmax())],
                    )
                )
                PU.append(pu.cpu().numpy())
                PB.append(pb.cpu().numpy())
                AB.append(ab.half().cpu().numpy())
                AU.append(au.half().cpu().numpy())
                CB.append(am_b.to(torch.int8).cpu().numpy())
                CU.append(am_u.to(torch.int8).cpu().numpy())
        print(
            f"[{args.tag}] {cue:<26s} items so far {len(meta)} (knew {sum(m['knew'] for m in meta)}, knew&cave {sum(m['knew'] and m['cave'] for m in meta)})  {time.time() - t0:.0f}s",
            flush=True,
        )
    for h in hooks:
        h.remove()
    np.savez_compressed(
        f"{OUT}/probe_{args.tag}.npz",
        pu=np.stack(PU),
        pb=np.stack(PB),
        att_b=np.stack(AB),
        att_u=np.stack(AU),
        am_b=np.stack(CB),
        am_u=np.stack(CU),
    )
    json.dump(
        dict(
            tag=args.tag,
            adapter=args.adapter,
            model=args.model,
            rows=args.rows,
            letters=letters,
            att_b_classes=[
                "cue_span",
                "X_in_cue",
                "X_label",
                "X_instr",
                "X_other_out",
                "T_in_cue",
                "T_label",
                "T_instr",
                "T_other_out",
                "BOS",
            ],
            att_u_classes=["X_label", "X_instr", "X_other", "T_label", "T_instr", "T_other", "BOS"],
            argmax_codes={
                0: "other",
                1: "cue_nonletter",
                2: "X_in_cue",
                3: "X_label",
                4: "X_instr",
                5: "X_other_out",
                6: "T_in_cue",
                7: "T_label",
                8: "T_instr",
                9: "T_other_out",
                10: "BOS",
            },
            items=meta,
        ),
        open(f"{OUT}/probe_{args.tag}.json", "w"),
    )
    print(f"saved -> {OUT}/probe_{args.tag}.npz/.json  N={len(meta)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
