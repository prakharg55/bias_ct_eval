"""The first version of the attention probe (Gemma only).

It produced the original Gemma model's reference file. attention_probe.py keeps its measurements of
letter probabilities and attention.

Usage: python -m ctcue.evaluation.attention_probe_v1 --adapter <dir|none> --tag <name>
Output: $CT_ROOT/results/w3/probe_<tag>.json and .npz
"""

import argparse, difflib, json, os, time
import numpy as np, torch
import transformers.tokenization_utils_base as _tub

if hasattr(_tub.PreTrainedTokenizerBase, "_patch_mistral_regex"):
    _tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, t, *a, **k: t)
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from ctcue.common.dumps import get_letter_token_ids, load_bct_source
from ctcue.paths import CT_ROOT, BCT_DUMPS

DUMPS = f"{BCT_DUMPS}/test"
SCR = CT_ROOT
OUT = f"{SCR}/results/w3"
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
    args = ap.parse_args()
    assert torch.cuda.is_available()
    if args.smoke:
        args.rows = "1200:1206"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    # shared loader (26 Sep): identical for q/k/v/o adapters, unties lm_head for RMCT
    from ctcue.common.load_gemma_merged import load_model

    tok, model = load_model(args.adapter)
    model.eval()
    import torch.nn as _nn

    assert (
        isinstance(model.model.layers[0].self_attn.o_proj, _nn.Linear) and "Peft" not in type(model).__name__
    )
    cfg = model.config
    NL, NH = cfg.num_hidden_layers, cfg.num_attention_heads
    HD = cfg.head_dim
    li = {L: torch.tensor(v) for L, v in get_letter_token_ids(tok).items() if v}
    letters = sorted(li)
    render = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").cuda()
    # [V, D] (tied embeddings; unchanged by LoRA)
    WU = model.lm_head.weight.detach()
    # [D, H, hd]
    WO = [
        model.model.layers[L].self_attn.o_proj.weight.detach().float().view(cfg.hidden_size, NH, HD)
        for L in range(NL)
    ]
    WPOST = [
        (1.0 + model.model.layers[L].post_attention_layernorm.weight.detach().float()) for L in range(NL)
    ]
    WFIN = 1.0 + model.model.norm.weight.detach().float()
    EPS = cfg.rms_norm_eps

    cap = {}

    def pre_o(L):
        def h(mod, inp):
            cap[("z", L)] = inp[0][0, -1].detach().float()

        return h

    def post_o(L):
        def h(mod, inp, out):
            cap[("a", L)] = out[0, -1].detach().float()

        return h

    def pre_layer(L):
        def h(mod, args_, kwargs):
            hs = args_[0] if args_ else kwargs["hidden_states"]
            cap[("r", L)] = hs[0, -1].detach().float()

        return h

    def pre_norm(mod, inp):
        cap["fin"] = inp[0][0, -1].detach().float()

    hooks = []
    for L in range(NL):
        at = model.model.layers[L].self_attn
        hooks += [at.o_proj.register_forward_pre_hook(pre_o(L)), at.o_proj.register_forward_hook(post_o(L))]
    for L in PROBE_LAYERS:
        hooks.append(model.model.layers[L].register_forward_pre_hook(pre_layer(L), with_kwargs=True))
    hooks.append(model.model.norm.register_forward_pre_hook(pre_norm))
    rms = lambda v: torch.sqrt(v.pow(2).mean() + EPS)
    PAREN = {
        int(t)
        for t in (tok.encode("(", add_special_tokens=False) + tok.encode(" (", add_special_tokens=False))
    }

    def layout(msgs, ids):
        """instruction start token index and per-position 'previous token is an opening parenthesis' flags."""
        s_ = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(s_, add_special_tokens=False, return_offsets_mapping=True)
        assert enc["input_ids"] == ids, "string-then-tokenize differs from apply_chat_template ids"
        ci = s_.rfind("Answer with just the single letter")
        assert ci >= 0
        it = next(k for k, (a, b) in enumerate(enc["offset_mapping"]) if a >= ci)
        prevp = torch.tensor([False] + [ids[k - 1] in PAREN for k in range(1, len(ids))], device="cuda")
        return it, prevp

    def forward(ids):
        cap.clear()
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False, output_attentions=True)
        row = out.logits[0, -1].float()
        sc = torch.stack([torch.logsumexp(row[li[L].to(row.device)], 0) for L in letters])
        att = torch.stack([a[0, :, -1, :].float() for a in out.attentions])  # [NL, NH, n]
        return row, sc, att

    CHK = {"n": 0}

    def dla(row, X, T):
        # u_L: softmax-over-variants weighted W_U rows (first-order change of the logsumexp letter score)
        def u(Lt):
            ids = li[Lt].to(row.device)
            w = torch.softmax(row[ids], 0)
            return (w[:, None] * WU[ids].float()).sum(0)

        ud = u(X) - u(T)
        fin = WFIN / rms(cap["fin"])
        out = torch.zeros(NL, NH)
        for L in range(NL):
            z = cap[("z", L)].view(NH, HD)  # [H, hd]
            # [H, D] per-head o_proj contribution
            c = torch.einsum("hd,mhd->hm", z, WO[L])
            # the per-head contributions must sum to the observed o_proj output (bf16 tolerance)
            if CHK["n"] < 4:
                err = float((c.sum(0) - cap[("a", L)]).abs().max())
                scale = float(cap[("a", L)].abs().max())
                assert err <= 0.02 * scale + 1e-2, f"head decomposition mismatch L{L}: {err} vs scale {scale}"
            # frozen post-attention norm
            c = c * (WPOST[L] / rms(cap[("a", L)]))[None]
            # frozen final norm, onto u_X - u_T
            out[L] = ((c * fin[None]) @ ud).cpu()
        CHK["n"] += 1
        return out

    ta, tb_ = map(int, args.train_rows.split(":"))
    excl = set()
    for ds in DATASETS:
        pairs, _ = load_bct_source(f"{DUMPS}/{TRAIN_CUE}/{ds}_{TRAIN_CUE}.jsonl")
        excl |= {p["original_question_hash"] for p in pairs[ta:tb_]}
    ea, eb = map(int, args.rows.split(":"))
    meta, PU, PB, AB, AU, CB, CU, DB, DU, RB, RU, AO, AO_IDX = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
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
                ru = {L: cap[("r", L)] for L in PROBE_LAYERS}
                du = dla(row_u, X, T)
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
                rb = {L: cap[("r", L)] for L in PROBE_LAYERS}
                db = dla(row_b, X, T)
                # function-space record: each layer's attention contribution to the residual (after the
                # post-attention norm)
                if len(meta) % 4 == 0:
                    AO.append(
                        np.stack(
                            [
                                (cap[("a", L)] * WPOST[L] / rms(cap[("a", L)])).half().cpu().numpy()
                                for L in range(NL)
                            ]
                        )
                    )
                    AO_IDX.append(len(meta))
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
                DB.append(db.numpy())
                DU.append(du.numpy())
                RB.append(np.stack([rb[L].half().cpu().numpy() for L in PROBE_LAYERS]))
                RU.append(np.stack([ru[L].half().cpu().numpy() for L in PROBE_LAYERS]))
        print(
            f"[{args.tag}] {cue:<26s} items so far {len(meta)} (knew {sum(m['knew'] for m in meta)}, knew&cave {sum(m['knew'] and m['cave'] for m in meta)})  {time.time() - t0:.0f}s",
            flush=True,
        )
    for h in hooks:
        h.remove()
    extra = {}
    if args.adapter == "none":
        ids = sorted({int(i) for L in letters for i in li[L].tolist()})
        extra = dict(
            wu_ids=np.array(ids),
            wu_rows=WU[ids].float().cpu().numpy(),
            final_norm_w=model.model.norm.weight.detach().float().cpu().numpy(),
        )
    np.savez_compressed(
        f"{OUT}/probe_{args.tag}.npz",
        pu=np.stack(PU),
        pb=np.stack(PB),
        att_b=np.stack(AB),
        att_u=np.stack(AU),
        am_b=np.stack(CB),
        am_u=np.stack(CU),
        dla_b=np.stack(DB).astype(np.float32),
        dla_u=np.stack(DU).astype(np.float32),
        resid_b=np.stack(RB),
        resid_u=np.stack(RU),
        probe_layers=np.array(PROBE_LAYERS),
        ao_b=np.stack(AO),
        ao_idx=np.array(AO_IDX),
        **extra,
    )
    json.dump(
        dict(
            tag=args.tag,
            adapter=args.adapter,
            rows=args.rows,
            letters=letters,
            probe_layers=PROBE_LAYERS,
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
