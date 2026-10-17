"""layer_swap.py with the trained model's adapter merged (earlier version).

layer_swap.py is a modified copy of this script. Its Llama outputs, scored by
analysis/score_layer_swap_merged.py, are the merged side of the comparison of merged and unmerged
adapters in analysis/score_layer_swap.py. The merged Gemma values in that comparison come from the
stored first-version results (results/w3/ana_route_pub.json).

Usage: python -m ctcue.evaluation.layer_swap_merged --model llama8b --adapter <dir> --tag <name>
Output: $CT_ROOT/results/w7/route_<tag>.json
"""

import argparse, difflib, json, os, time
import numpy as np, torch
import torch.nn.functional as Fnn
import transformers.tokenization_utils_base as _tub

if hasattr(_tub.PreTrainedTokenizerBase, "_patch_mistral_regex"):
    _tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, t, *a, **k: t)
from ctcue.common.dumps import load_bct_source

# v7: token sets of letter_tokens, letter score = max over forms (letter_readout.agg)
from ctcue.common import letter_readout
from ctcue.common.letter_readout import get_letter_token_ids
from ctcue.paths import CT_ROOT, BCT_DUMPS

DUMPS = f"{BCT_DUMPS}/test"
SCR = CT_ROOT
OUT = f"{SCR}/results/w7"
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
# Per-model circuit band (B0, B1 as HEADS in ct_causal5.py) and resid_pre splice layers. R_LAYERS follows
# ct_route3's gemma list read as offsets from the band edges: [B0-5, B0-1, B1-3, B1-1, B1+1, B1+3] (two below
# the band, two inside, two above).
LAYOUT = {
    "gemma2b": dict(NL=26, B0=13, B1=19, R_LAYERS=[8, 12, 16, 18, 20, 22]),
    "llama8b": dict(NL=32, B0=15, B1=19, R_LAYERS=[10, 14, 16, 18, 20, 22]),
}
N_CARRIER = 6  # CARRIER = the six layers below the lowest reader layer


def layout(model):
    """(NL, SCOPES, R_LAYERS, CK_L): scopes tile 0..NL-1; CK_L = B0-1 is the layer of the r-hook exactness check (gemma: 12)."""
    lay = LAYOUT[model]
    NL, B0, B1 = lay["NL"], lay["B0"], lay["B1"]
    C0 = B0 - N_CARRIER
    scopes = {
        "ALL": list(range(NL)),
        "EARLY": list(range(0, C0)),
        "CARRIER": list(range(C0, B0)),
        "CIRCUIT": list(range(B0, B1)),
        "LATE": list(range(B1, NL)),
    }
    assert (
        0 < C0 < B0 < B1 < NL
        and sorted(scopes["EARLY"] + scopes["CARRIER"] + scopes["CIRCUIT"] + scopes["LATE"]) == scopes["ALL"]
    )
    assert all(0 <= L < NL for L in lay["R_LAYERS"]) and B0 - 1 in lay["R_LAYERS"]
    return NL, scopes, list(lay["R_LAYERS"]), B0 - 1


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


def build_arms(R_LAYERS):
    """(name, kind, components, position key, scope)."""
    A = [
        ("full_o", "act", ("o",), "all", "ALL"),
        ("qkv_all", "act", ("q", "k", "v"), "all", "ALL"),
        ("qk_all", "act", ("q", "k"), "all", "ALL"),
        ("q_all", "act", ("q",), "all", "ALL"),
        ("k_all", "act", ("k",), "all", "ALL"),
        ("v_all", "act", ("v",), "all", "ALL"),
        ("q_ans", "act", ("q",), "ans", "ALL"),
        ("o_ans", "act", ("o",), "ans", "ALL"),
        ("kv_cue", "act", ("k", "v"), "cue", "ALL"),
        ("k_cue", "act", ("k",), "cue", "ALL"),
        ("v_cue", "act", ("v",), "cue", "ALL"),
    ]
    for sc in ("EARLY", "CARRIER", "CIRCUIT", "LATE"):
        A += [
            (f"qk_all@{sc}", "act", ("q", "k"), "all", sc),
            (f"q_ans@{sc}", "act", ("q",), "ans", sc),
            (f"kv_cue@{sc}", "act", ("k", "v"), "cue", sc),
        ]
    for L in R_LAYERS:
        A += [(f"r_ans@L{L}", "act", ("r",), "ans", [L]), (f"r_all@L{L}", "act", ("r",), "all", [L])]
    A += [
        ("W_all@ALL", "w", ("q", "k", "v", "o"), "all", "ALL"),
        ("W_q@ALL", "w", ("q",), "all", "ALL"),
        ("W_k@ALL", "w", ("k",), "all", "ALL"),
        ("W_v@ALL", "w", ("v",), "all", "ALL"),
        ("W_o@ALL", "w", ("o",), "all", "ALL"),
        ("W_qk@ALL", "w", ("q", "k"), "all", "ALL"),
        ("W_vo@ALL", "w", ("v", "o"), "all", "ALL"),
        ("W_q_ans@ALL", "w", ("q",), "ans", "ALL"),
    ]
    for sc in ("EARLY", "CARRIER", "CIRCUIT", "LATE"):
        A += [(f"W_all@{sc}", "w", ("q", "k", "v", "o"), "all", sc)]
    return A


class Splicer:
    """Hooks on q/k/v/o projections (output hooks; o also a pre-hook capturing its input z) and on resid_pre.
    Works on whole projection outputs (no per-head reshaping), so it is independent of head count / head_dim / kv grouping.
    """

    def __init__(self, model):
        self.mode = None
        self.store = {}
        self.spec = {}
        self.hooks = []
        self.capz = False
        self.zstore = {}
        for L, layer in enumerate(model.model.layers):
            at = layer.self_attn
            for comp, mod in (("q", at.q_proj), ("k", at.k_proj), ("v", at.v_proj), ("o", at.o_proj)):
                assert mod.bias is None
                self.hooks.append(mod.register_forward_hook(self._out_hook(comp, L)))
            self.hooks.append(at.o_proj.register_forward_pre_hook(self._z_hook(L)))
            self.hooks.append(layer.register_forward_pre_hook(self._pre_hook(L), with_kwargs=True))

    def _z_hook(self, L):
        def h(mod, inp):
            if self.capz:
                self.zstore[L] = inp[0][0].detach().clone()

        return h

    def _out_hook(self, comp, L):
        def h(mod, inp, out):
            if self.mode == "capture":
                self.store[(comp, L)] = out[0].detach().clone()
            elif self.mode == "patch":
                if ("act", comp, L) in self.spec:
                    pos, src = self.spec[("act", comp, L)]
                    out = out.clone()
                    out[0, pos] = src[pos].to(out.dtype)
                    return out
                if ("w", comp, L) in self.spec:
                    pos, W = self.spec[("w", comp, L)]
                    out = out.clone()
                    out[0, pos] = Fnn.linear(inp[0][0, pos], W).to(out.dtype)
                    return out

        return h

    def _pre_hook(self, L):
        def h(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            if self.mode == "capture":
                self.store[("r", L)] = hs[0].detach().clone()
            elif self.mode == "patch" and ("act", "r", L) in self.spec:
                pos, src = self.spec[("act", "r", L)]
                hs = hs.clone()
                hs[0, pos] = src[pos].to(hs.dtype)
                if args:
                    return (hs,) + tuple(args[1:]), kwargs
                kw = dict(kwargs)
                kw["hidden_states"] = hs
                return args, kw

        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", default="1200:1300")
    ap.add_argument("--train_rows", default="100:500")
    ap.add_argument(
        "--max_per_cell",
        type=int,
        default=10,
        help="cap |S| per (cue, dataset) cell, first items in file order (0 = no cap)",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="gemma2b", choices=["gemma2b", "llama8b"])
    ap.add_argument("--agg", default="max", choices=["max", "sum"])
    args = ap.parse_args()
    letter_readout.MODE = args.agg
    assert torch.cuda.is_available()
    NL, SCOPES, R_LAYERS, CK_L = layout(args.model)
    if args.smoke:
        args.rows, args.max_per_cell = "1200:1212", 1
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    # shared loader v2 (= ct_route3's own loading for gemma2b q/k/v/o adapters)
    from ctcue.common.load_merged import load_model

    tok, m0 = load_model("none", model=args.model)  # M0: untrained, bf16, eager, eval
    # M1: a second copy + adapter, merged (merge verified by the loader), eval
    _, m1 = load_model(args.adapter, model=args.model)
    import torch.nn as _nn

    for m in (m0, m1):
        assert (
            isinstance(m.model.layers[0].self_attn.o_proj, _nn.Linear)
            and "Peft" not in type(m).__name__
            and m.config.num_hidden_layers == NL == len(m.model.layers)
        )
    nd = [n for (n, a), (_, b) in zip(m0.named_parameters(), m1.named_parameters()) if not torch.equal(a, b)]
    bad = [n for n in nd if not any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj"))]
    assert (
        nd and not bad
    ), f"M0 and M1 must differ in q/k/v/o weights only ({len(nd)} tensors differ; outside q/k/v/o: {bad[:5]})"
    print(
        f"[{args.tag}] {args.model}: merged {args.adapter}: {len(nd)} tensors differ, all in q/k/v/o",
        flush=True,
    )
    S0, S1 = Splicer(m0), Splicer(m1)
    WT = {}  # (model_id, comp, L) -> weight
    for mid, m in ((0, m0), (1, m1)):
        for L, layer in enumerate(m.model.layers):
            at = layer.self_attn
            for comp, mod in (("q", at.q_proj), ("k", at.k_proj), ("v", at.v_proj), ("o", at.o_proj)):
                WT[(mid, comp, L)] = mod.weight
    li = {L: torch.tensor(v) for L, v in get_letter_token_ids(tok).items() if v}
    letters = sorted(li)
    render = lambda msgs: tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).cuda()

    def lscores(row):
        return torch.stack([letter_readout.score(row.float(), li[L]) for L in letters])

    def run(model, spl, ids, mode=None, spec=None, capz=False):
        spl.mode, spl.spec, spl.capz = mode, (spec or {}), capz
        if mode == "capture":
            spl.store = {}
        if capz:
            spl.zstore = {}
        with torch.inference_mode():
            row = model(input_ids=ids, use_cache=False).logits[0, -1]
        spl.mode, spl.spec, spl.capz = None, {}, False
        return lscores(row)

    ta, tb_ = map(int, args.train_rows.split(":"))
    excl = set()
    for ds in DATASETS:
        pairs, _ = load_bct_source(f"{DUMPS}/{TRAIN_CUE}/{ds}_{TRAIN_CUE}.jsonl")
        excl |= {p["original_question_hash"] for p in pairs[ta:tb_]}
    ARMS = build_arms(R_LAYERS)
    ea, eb = map(int, args.rows.split(":"))
    res = {
        "tag": args.tag,
        "adapter": args.adapter,
        "model": args.model,
        "rows": args.rows,
        "max_per_cell": args.max_per_cell,
        "arms": [a[0] for a in ARMS],
        "arm_kinds": {a[0]: a[1] for a in ARMS},
        "scopes": SCOPES,
        "r_layers": R_LAYERS,
        "letters": letters,
        "counts": {},
        "items": [],
        "checks": {},
    }
    n_checked = 0
    for cue in CUES:
        cnt = dict(cand=0, knew0=0, cave0=0, in_S=0)
        for ds in DATASETS:
            n_cell = 0
            pairs, _ = load_bct_source(f"{DUMPS}/{cue}/{ds}_{cue}.jsonl")
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
                if args.max_per_cell and n_cell >= args.max_per_cell:
                    break
                cnt["cand"] += 1
                iu, ib = render(pr["unbiased_messages"]), render(pr["biased_messages"])
                xi, ti = letters.index(X), letters.index(T)
                s0u = run(m0, S0, iu)
                if int(s0u.argmax()) != ti:
                    continue
                cnt["knew0"] += 1
                s0b = run(m0, S0, ib)
                if int(s0b.argmax()) != xi:
                    continue
                cnt["cave0"] += 1
                s1u = run(m1, S1, iu)
                if int(s1u.argmax()) != ti:
                    continue
                cnt["in_S"] += 1
                n_cell += 1
                bt, ut = ib[0].tolist(), iu[0].tolist()
                n = len(bt)
                cue_pos = cue_positions(ut, bt)
                assert cue_pos and cue_pos[-1] < n - 1, (cue, len(cue_pos))
                POS = {
                    "ans": torch.tensor([n - 1], device="cuda"),
                    "cue": torch.tensor(cue_pos, device="cuda"),
                    "all": torch.arange(n, device="cuda"),
                }
                sc0 = run(m0, S0, ib, "capture")
                C0 = S0.store
                sc1 = run(m1, S1, ib, "capture", capz=True)
                C1 = S1.store
                Z1 = S1.zstore
                p0u, p0b, p1u, p1b = (torch.softmax(v, 0) for v in (s0u, sc0, s1u, sc1))
                rec = dict(
                    cue=cue,
                    ds=ds,
                    hash=pr["original_question_hash"],
                    X=X,
                    T=T,
                    n_tok=n,
                    n_cue=len(cue_pos),
                    p0u=[float(p0u[xi]), float(p0u[ti])],
                    p0b=[float(p0b[xi]), float(p0b[ti])],
                    p1u=[float(p1u[xi]), float(p1u[ti])],
                    p1b=[float(p1b[xi]), float(p1b[ti])],
                    pred1b=letters[int(p1b.argmax())],
                    sd0b=float(sc0[xi] - sc0[ti]),
                    sd1b=float(sc1[xi] - sc1[ti]),
                    arms={},
                )
                d_m = float((sc1 - sc0).abs().max())
                if n_checked < 3 and d_m > 0.5:
                    ALLP = POS["all"]
                    d_id = float(
                        (
                            run(
                                m0,
                                S0,
                                ib,
                                "patch",
                                {("act", c, L): (ALLP, C0[(c, L)]) for c in "qkv" for L in range(NL)},
                            )
                            - sc0
                        )
                        .abs()
                        .max()
                    )
                    d_of = float(
                        (
                            run(
                                m0,
                                S0,
                                ib,
                                "patch",
                                {("act", "o", L): (ALLP, C1[("o", L)]) for L in range(NL)},
                            )
                            - sc1
                        )
                        .abs()
                        .max()
                    )
                    d_or = float(
                        (
                            run(
                                m1,
                                S1,
                                ib,
                                "patch",
                                {("act", "o", L): (ALLP, C0[("o", L)]) for L in range(NL)},
                            )
                            - sc0
                        )
                        .abs()
                        .max()
                    )
                    run(
                        m0,
                        S0,
                        ib,
                        "patch",
                        {("act", c, L): (ALLP, C1[(c, L)]) for c in "qkv" for L in range(NL)},
                        capz=True,
                    )
                    d_z = max(float((S0.zstore[L] - Z1[L]).abs().max()) for L in range(NL))
                    spec_r = {("act", "r", CK_L): (ALLP, C1[("r", CK_L)])}
                    spec_r.update({("act", "o", L): (ALLP, C1[("o", L)]) for L in range(CK_L, NL)})
                    d_r = float((run(m0, S0, ib, "patch", spec_r) - sc1).abs().max())
                    d_wf = float(
                        (
                            run(
                                m0,
                                S0,
                                ib,
                                "patch",
                                {("w", c, L): (ALLP, WT[(1, c, L)]) for c in "qkvo" for L in range(NL)},
                            )
                            - sc1
                        )
                        .abs()
                        .max()
                    )
                    d_wr = float(
                        (
                            run(
                                m1,
                                S1,
                                ib,
                                "patch",
                                {("w", c, L): (ALLP, WT[(0, c, L)]) for c in "qkvo" for L in range(NL)},
                            )
                            - sc0
                        )
                        .abs()
                        .max()
                    )
                    ch = dict(
                        m0_vs_m1=d_m,
                        identity=d_id,
                        full_o_fwd=d_of,
                        full_o_rev=d_or,
                        qkv_to_z=d_z,
                        r_hook=d_r,
                        weights_fwd=d_wf,
                        weights_rev=d_wr,
                    )
                    res["checks"][f"item{n_checked}"] = ch
                    print("  check " + " ".join(f"{k} {v:.2e}" for k, v in ch.items()), flush=True)
                    assert (
                        d_id < 1e-3
                        and d_of < 5e-2
                        and d_or < 5e-2
                        and d_z < 1e-2
                        and d_r < 5e-2
                        and d_wf < 5e-2
                        and d_wr < 5e-2
                    ), ch
                    n_checked += 1
                for name, kind, comps, pkey, scope in ARMS:
                    layers = SCOPES[scope] if isinstance(scope, str) else scope
                    out = {}
                    for dirn, (mod, spl, src, oth) in (("fwd", (m0, S0, C1, 1)), ("rev", (m1, S1, C0, 0))):
                        if kind == "act":
                            spec = {("act", c, L): (POS[pkey], src[(c, L)]) for c in comps for L in layers}
                        else:
                            spec = {("w", c, L): (POS[pkey], WT[(oth, c, L)]) for c in comps for L in layers}
                        s = run(mod, spl, ib, "patch", spec)
                        p = torch.softmax(s, 0)
                        out[dirn] = [
                            round(float(p[xi]), 6),
                            round(float(p[ti]), 6),
                            letters[int(s.argmax())],
                            round(float(s[xi] - s[ti]), 4),
                        ]
                    rec["arms"][name] = out
                del C0, C1, Z1
                S0.store = {}
                S1.store = {}
                S1.zstore = {}
                S0.zstore = {}
                res["items"].append(rec)
        res["counts"][cue] = cnt
        print(f"[{args.tag}] {cue:<26s} {cnt}  ({time.time() - t0:.0f}s)", flush=True)
    # the analysis refuses files with n_checked == 0 (no item had M0 and M1 differing by > 0.5)
    res["n_checked"] = n_checked
    if n_checked == 0:
        print("!!! WARNING: no item qualified for the exactness checks", flush=True)
    for h in S0.hooks + S1.hooks:
        h.remove()
    json.dump(res, open(f"{OUT}/route_{args.tag}.json", "w"))
    print(
        f"saved -> {OUT}/route_{args.tag}.json  |S|={len(res['items'])}  checks={n_checked}  {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
