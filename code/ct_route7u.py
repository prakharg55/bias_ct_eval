#!/usr/bin/env python3
"""
ct_route7u (1 Oct 2026; prereg audit NOTE "ROUTE UNMERGED") = ct_route7.py with the trained model M1 UNMERGED: M1 = pub_load3u.load_model(adapter)
(the HF model with PEFT lora.Linear modules injected; bf16 base weights untouched, fp32 LoRA, eval), M0 = the untrained model as before.
Changes: (1) loader; (2) the "M0 and M1 differ in q/k/v/o only" check becomes: M1's base weights equal M0's exactly and every LoRA module
sits on a q/k/v/o projection; (3) a WEIGHT splice applies the donor's projection MODULE to the recipient's input (M1: base + LoRA, the
computation used in training; M0: its plain linear), instead of F.linear with a merged weight tensor; (4) output dir results/w7u.
Activation splices, items, arms, exactness checks (asserted) and every other line are ct_route7's. ct_route7's docstring:
v7.1 (30 Sep 2026): letter score via letters7.score = max over forms with the greedy tie-break (lowest token id on exact bf16 ties).
v7 (30 Sep 2026) = ct_route6.py with the letter score = MAX logit over the letter's token forms (letters7.agg; v6: logsumexp),
output dir results/*7 and --agg {max (default), sum = v6, regression only}. See letters7.py / erratum "V7".
v6 (30 Sep 2026) = ct_route5.py with ONE change: letter-token sets from letters6.py (the v5 forms "L", " L" plus single tokens
"(L" / "(L)"; identical for gemma, +1 token per letter for Llama) and output dir results/w7. the project log / readout_check6.py.
v5 (30 Sep 2026) = ct_route3.py + --model {gemma2b, llama8b} (loader pub_load2, called twice: M0 = load_model("none"),
M1 = load_model(adapter)), per-model layer scopes / resid layers (LAYOUT below), output dir results/w5 and a "model" key
in the JSON; nothing else changed (with --model gemma2b it must reproduce ct_route3 exactly: check_route5_regression.py).
Llama-3.1-8B-Instruct circuit (as in ct_causal5.py): writer L17.H24, readers L18.H28 and L15.H8, band L15-18, 32 layers.
NOTE: like ct_route3, this instrument REQUIRES that M0 and M1 differ only in q/k/v/o projection weights (asserted); an
adapter that also trains MLPs / lm_head (the published RMCT arms) is rejected, for either model.

ROUTE DECOMPOSITION (wave 3, 26 Sep 2026; v2 of this file after the wave-3 audit): where is a consistency-trained model's
fix CARRIED (activation splices) and which weight changes CREATE it (weight splices)?

Two models on one GPU: M0 = the untrained model, M1 = M0 + a merged CT adapter (LoRA on attention projections only; MLPs,
norms and embeddings untouched, so M0 and M1 differ ONLY in the four attention projections).

Item set S: eval-window items (rows --rows of every cue file, question-disjoint from the training window, X != T,
cue-less loader pairs dropped) that M0 answers correctly uncued (knew0), caves on when cued (biased argmax == X), and
that M1 also answers correctly uncued (knew1); at most --max_per_cell items per (cue, dataset), first in file order.
Two kinds of arm, each run in both directions (fwd: into M0 from M1 -> does the flip disappear, i.e. sufficiency;
rev: into M1 from M0 -> does the flip come back, i.e. necessity):
  ACTIVATION splices ("act"): the other model's captured q/k/v/o projection OUTPUTS (pre-RoPE for q/k) or resid_pre,
    from its own run on the same biased prompt, at positions ans (final position) / cue (biased-prompt tokens
    inserted or replaced relative to the unbiased prompt, difflib on token ids) / all, over a layer scope. These say
    where the fix is CARRIED: a spliced activation already contains every upstream difference of the donor.
  WEIGHT splices ("w"): the other model's projection WEIGHTS applied to the recipient's own current input, at a
    layer scope (all positions, or only the final position for W_q_ans). These say which weight changes CREATE it.
Layer scopes (B0 = lowest reader layer, B1 = first layer above the circuit band): ALL, EARLY 0..B0-7, CARRIER = the six
layers below B0, CIRCUIT B0..B1-1, LATE B1..last.
  gemma2b: ALL 0-25, EARLY 0-6, CARRIER 7-12, CIRCUIT 13-18 (readers L13/L14/L18, writers L15/L16/L17), LATE 19-25.
  llama8b: ALL 0-31, EARLY 0-8, CARRIER 9-14, CIRCUIT 15-18 (readers L15/L18, writer L17), LATE 19-31.
Exactness checks, asserted on the first 3 items where M0 and M1 differ (max |letter-score difference| > 0.5):
  identity (a model's own q/k/v into itself; a no-op sanity check only);
  full o splice fwd == M1 and rev == M0 (tests the o hook);
  q/k/v hooks: M1's q,k,v at all positions and layers spliced into M0 must reproduce M1's o_proj INPUTS exactly;
  r hook: M1's resid_pre at L(B0-1) plus M1's o outputs at layers >= B0-1 spliced into M0 must reproduce M1 (gemma: L12);
  weight splice: all four projection weights of M1 applied everywhere in M0 must reproduce M1 (and rev M0).
Output: per item, M0/M1 letter probabilities on both prompts and, per arm and direction, (p_X, p_T, argmax, s_X - s_T)
with s = the logsumexp-over-variants letter score (unrounded log-odds).
"""
import argparse, difflib, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as Fnn
import transformers.tokenization_utils_base as _tub
if hasattr(_tub.PreTrainedTokenizerBase, "_patch_mistral_regex"):
    _tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, t, *a, **k: t)
sys.path.insert(0, "/data/code")
from utils import load_bct_source
import letters7; from letters7 import get_letter_token_ids   # v7: token sets of letters6, letter score = max over forms (letters7.agg)
DUMPS = "/data/dataset_dumps/test"; SCR = "/data/ct_root"; OUT = f"{SCR}/results/w7u"
DATASETS = ["mmlu", "hellaswag", "logiqa", "truthfulqa"]; TRAIN_CUE = "suggested_answer"
CUES = ["suggested_answer", "post_hoc", "distractor_fact", "wrong_few_shot", "distractor_argument", "spurious_few_shot_squares"]
# Per-model circuit band (B0, B1 as HEADS in ct_causal5.py) and resid_pre splice layers. R_LAYERS follows ct_route3's gemma
# list read as offsets from the band edges: [B0-5, B0-1, B1-3, B1-1, B1+1, B1+3] (two below the band, two inside, two above).
LAYOUT = {"gemma2b": dict(NL=26, B0=13, B1=19, R_LAYERS=[8, 12, 16, 18, 20, 22]),
          "llama8b": dict(NL=32, B0=15, B1=19, R_LAYERS=[10, 14, 16, 18, 20, 22])}
N_CARRIER = 6   # CARRIER = the six layers below the lowest reader layer


def layout(model):
    """(NL, SCOPES, R_LAYERS, CK_L): scopes tile 0..NL-1; CK_L = B0-1 is the layer of the r-hook exactness check (gemma: 12)."""
    lay = LAYOUT[model]; NL, B0, B1 = lay["NL"], lay["B0"], lay["B1"]; C0 = B0 - N_CARRIER
    scopes = {"ALL": list(range(NL)), "EARLY": list(range(0, C0)), "CARRIER": list(range(C0, B0)), "CIRCUIT": list(range(B0, B1)), "LATE": list(range(B1, NL))}
    assert 0 < C0 < B0 < B1 < NL and sorted(scopes["EARLY"] + scopes["CARRIER"] + scopes["CIRCUIT"] + scopes["LATE"]) == scopes["ALL"]
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
            prev_eq_start = j1; continue
        if tag == "insert" and prev_eq_start is not None:
            while j1 - 1 >= prev_eq_start and bt[j1 - 1] == bt[j2 - 1]:
                j1 -= 1; j2 -= 1
        if tag in ("insert", "replace"): pos.update(range(j1, j2))
        prev_eq_start = None
    return sorted(pos)


def build_arms(R_LAYERS):
    """(name, kind, components, position key, scope)."""
    A = [("full_o", "act", ("o",), "all", "ALL"),
         ("qkv_all", "act", ("q", "k", "v"), "all", "ALL"), ("qk_all", "act", ("q", "k"), "all", "ALL"),
         ("q_all", "act", ("q",), "all", "ALL"), ("k_all", "act", ("k",), "all", "ALL"), ("v_all", "act", ("v",), "all", "ALL"),
         ("q_ans", "act", ("q",), "ans", "ALL"), ("o_ans", "act", ("o",), "ans", "ALL"),
         ("kv_cue", "act", ("k", "v"), "cue", "ALL"), ("k_cue", "act", ("k",), "cue", "ALL"), ("v_cue", "act", ("v",), "cue", "ALL")]
    for sc in ("EARLY", "CARRIER", "CIRCUIT", "LATE"):
        A += [(f"qk_all@{sc}", "act", ("q", "k"), "all", sc), (f"q_ans@{sc}", "act", ("q",), "ans", sc), (f"kv_cue@{sc}", "act", ("k", "v"), "cue", sc)]
    for L in R_LAYERS:
        A += [(f"r_ans@L{L}", "act", ("r",), "ans", [L]), (f"r_all@L{L}", "act", ("r",), "all", [L])]
    A += [("W_all@ALL", "w", ("q", "k", "v", "o"), "all", "ALL"),
          ("W_q@ALL", "w", ("q",), "all", "ALL"), ("W_k@ALL", "w", ("k",), "all", "ALL"), ("W_v@ALL", "w", ("v",), "all", "ALL"), ("W_o@ALL", "w", ("o",), "all", "ALL"),
          ("W_qk@ALL", "w", ("q", "k"), "all", "ALL"), ("W_vo@ALL", "w", ("v", "o"), "all", "ALL"), ("W_q_ans@ALL", "w", ("q",), "ans", "ALL")]
    for sc in ("EARLY", "CARRIER", "CIRCUIT", "LATE"):
        A += [(f"W_all@{sc}", "w", ("q", "k", "v", "o"), "all", sc)]
    return A


class Splicer:
    """Hooks on q/k/v/o projections (output hooks; o also a pre-hook capturing its input z) and on resid_pre.
    Works on whole projection outputs (no per-head reshaping), so it is independent of head count / head_dim / kv grouping."""
    def __init__(self, model):
        self.mode = None; self.store = {}; self.spec = {}; self.hooks = []; self.capz = False; self.zstore = {}
        for L, layer in enumerate(model.model.layers):
            at = layer.self_attn
            for comp, mod in (("q", at.q_proj), ("k", at.k_proj), ("v", at.v_proj), ("o", at.o_proj)):
                assert mod.bias is None
                self.hooks.append(mod.register_forward_hook(self._out_hook(comp, L)))
            self.hooks.append(at.o_proj.register_forward_pre_hook(self._z_hook(L)))
            self.hooks.append(layer.register_forward_pre_hook(self._pre_hook(L), with_kwargs=True))

    def _z_hook(self, L):
        def h(mod, inp):
            if self.capz: self.zstore[L] = inp[0][0].detach().clone()
        return h

    def _out_hook(self, comp, L):
        def h(mod, inp, out):
            if self.mode == "capture":
                self.store[(comp, L)] = out[0].detach().clone()
            elif self.mode == "patch":
                if ("act", comp, L) in self.spec:
                    pos, src = self.spec[("act", comp, L)]
                    out = out.clone(); out[0, pos] = src[pos].to(out.dtype); return out
                if ("w", comp, L) in self.spec:
                    pos, W = self.spec[("w", comp, L)]   # W = the donor's projection as a callable (ct_route7u)
                    out = out.clone(); out[0, pos] = W(inp[0][0, pos]).to(out.dtype); return out
        return h

    def _pre_hook(self, L):
        def h(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            if self.mode == "capture":
                self.store[("r", L)] = hs[0].detach().clone()
            elif self.mode == "patch" and ("act", "r", L) in self.spec:
                pos, src = self.spec[("act", "r", L)]
                hs = hs.clone(); hs[0, pos] = src[pos].to(hs.dtype)
                if args: return (hs,) + tuple(args[1:]), kwargs
                kw = dict(kwargs); kw["hidden_states"] = hs; return args, kw
        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", default="1200:1300"); ap.add_argument("--train_rows", default="100:500")
    ap.add_argument("--max_per_cell", type=int, default=10, help="cap |S| per (cue, dataset) cell, first items in file order (0 = no cap)"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="gemma2b", choices=["gemma2b", "llama8b"])
    ap.add_argument("--agg", default="max", choices=["max", "sum"]); args = ap.parse_args(); letters7.MODE = args.agg; assert torch.cuda.is_available()
    NL, SCOPES, R_LAYERS, CK_L = layout(args.model)
    if args.smoke: args.rows, args.max_per_cell = "1200:1212", 1
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    sys.path.insert(0, "/data/code"); from pub_load2 import load_model   # shared loader v2 (= ct_route3's own loading for gemma2b q/k/v/o adapters)
    from pub_load3u import load_model as load_model_u   # UNMERGED (prereg audit NOTE "ROUTE UNMERGED")
    tok, m0 = load_model("none", model=args.model)      # M0: untrained, bf16, eager, eval
    _, m1 = load_model_u(args.adapter, model=args.model)  # M1: a second copy + adapter, UNMERGED (PEFT lora.Linear on the bf16 base weights; fp32 LoRA), eval
    import torch.nn as _nn, peft.tuners.lora as _pl
    assert isinstance(m0.model.layers[0].self_attn.o_proj, _nn.Linear) and "Peft" not in type(m0).__name__ and "Peft" not in type(m1).__name__
    for m in (m0, m1): assert m.config.num_hidden_layers == NL == len(m.model.layers) and not m.training
    lora = {n: mod for n, mod in m1.named_modules() if isinstance(mod, _pl.LoraLayer)}
    bad = [n for n in lora if not any(n.endswith(s) for s in ("q_proj", "k_proj", "v_proj", "o_proj"))]
    assert lora and not bad, f"the adapter must target q/k/v/o projections only ({len(lora)} LoRA modules; elsewhere: {bad[:5]})"
    assert all(mod.active_adapters == ["default"] and not mod.disable_adapters and not mod.merged for mod in lora.values()), "adapter not active / merged"
    p0 = dict(m0.named_parameters()); p1 = {n.replace(".base_layer.", "."): p for n, p in m1.named_parameters() if "lora_" not in n}
    assert set(p0) == set(p1) and all(torch.equal(p0[n], p1[n]) for n in p0), "M1's base weights must equal M0's exactly (unmerged adapter)"
    nB = sum(int(bool(mod.lora_B["default"].weight.abs().sum() > 0)) for mod in lora.values())
    print(f"[{args.tag}] {args.model}: UNMERGED {args.adapter}: {len(lora)} LoRA modules, all on q/k/v/o ({nB} with non-zero B); base weights identical to M0", flush=True)
    S0, S1 = Splicer(m0), Splicer(m1)
    WT = {}   # (model_id, comp, L) -> the projection as a callable (ct_route7u): M0 = F.linear with its weight (= its nn.Linear), M1 = its lora.Linear module (base + LoRA)
    import functools
    for mid, m in ((0, m0), (1, m1)):
        for L, layer in enumerate(m.model.layers):
            at = layer.self_attn
            for comp, mod in (("q", at.q_proj), ("k", at.k_proj), ("v", at.v_proj), ("o", at.o_proj)):
                WT[(mid, comp, L)] = functools.partial(Fnn.linear, weight=mod.weight) if mid == 0 else mod
    li = {L: torch.tensor(v) for L, v in get_letter_token_ids(tok).items() if v}
    letters = sorted(li)
    render = lambda msgs: tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").cuda()

    def lscores(row):
        return torch.stack([letters7.score(row.float(), li[L]) for L in letters])

    def run(model, spl, ids, mode=None, spec=None, capz=False):
        spl.mode, spl.spec, spl.capz = mode, (spec or {}), capz
        if mode == "capture": spl.store = {}
        if capz: spl.zstore = {}
        with torch.inference_mode():
            row = model(input_ids=ids, use_cache=False).logits[0, -1]
        spl.mode, spl.spec, spl.capz = None, {}, False
        return lscores(row)

    ta, tb_ = map(int, args.train_rows.split(":")); excl = set()
    for ds in DATASETS:
        pairs, _ = load_bct_source(f"{DUMPS}/{TRAIN_CUE}/{ds}_{TRAIN_CUE}.jsonl"); excl |= {p["original_question_hash"] for p in pairs[ta:tb_]}
    ARMS = build_arms(R_LAYERS); ea, eb = map(int, args.rows.split(":"))
    res = {"tag": args.tag, "adapter": args.adapter, "model": args.model, "rows": args.rows, "max_per_cell": args.max_per_cell, "arms": [a[0] for a in ARMS],
           "arm_kinds": {a[0]: a[1] for a in ARMS}, "scopes": SCOPES, "r_layers": R_LAYERS, "letters": letters, "counts": {}, "items": [], "checks": {}}
    n_checked = 0
    for cue in CUES:
        cnt = dict(cand=0, knew0=0, cave0=0, in_S=0)
        for ds in DATASETS:
            n_cell = 0
            pairs, _ = load_bct_source(f"{DUMPS}/{cue}/{ds}_{cue}.jsonl"); seg = pairs[ea:eb] if len(pairs) > ea + 50 else pairs[max(0, len(pairs) - (eb - ea)):]
            for pr in seg:
                X, T = pr["bias_target"], pr["correct_letter"]
                if X == T or X not in li or T not in li or pr["original_question_hash"] in excl or pr["biased_messages"] == pr["unbiased_messages"]: continue
                if args.max_per_cell and n_cell >= args.max_per_cell: break
                cnt["cand"] += 1; iu, ib = render(pr["unbiased_messages"]), render(pr["biased_messages"])
                xi, ti = letters.index(X), letters.index(T)
                s0u = run(m0, S0, iu)
                if int(s0u.argmax()) != ti: continue
                cnt["knew0"] += 1
                s0b = run(m0, S0, ib)
                if int(s0b.argmax()) != xi: continue
                cnt["cave0"] += 1
                s1u = run(m1, S1, iu)
                if int(s1u.argmax()) != ti: continue
                cnt["in_S"] += 1; n_cell += 1
                bt, ut = ib[0].tolist(), iu[0].tolist(); n = len(bt)
                cue_pos = cue_positions(ut, bt)
                assert cue_pos and cue_pos[-1] < n - 1, (cue, len(cue_pos))
                POS = {"ans": torch.tensor([n - 1], device="cuda"), "cue": torch.tensor(cue_pos, device="cuda"), "all": torch.arange(n, device="cuda")}
                sc0 = run(m0, S0, ib, "capture"); C0 = S0.store
                sc1 = run(m1, S1, ib, "capture", capz=True); C1 = S1.store; Z1 = S1.zstore
                p0u, p0b, p1u, p1b = (torch.softmax(v, 0) for v in (s0u, sc0, s1u, sc1))
                rec = dict(cue=cue, ds=ds, hash=pr["original_question_hash"], X=X, T=T, n_tok=n, n_cue=len(cue_pos),
                           p0u=[float(p0u[xi]), float(p0u[ti])], p0b=[float(p0b[xi]), float(p0b[ti])], p1u=[float(p1u[xi]), float(p1u[ti])],
                           p1b=[float(p1b[xi]), float(p1b[ti])], pred1b=letters[int(p1b.argmax())],
                           sd0b=float(sc0[xi] - sc0[ti]), sd1b=float(sc1[xi] - sc1[ti]), arms={})
                d_m = float((sc1 - sc0).abs().max())
                if n_checked < 3 and d_m > 0.5:
                    ALLP = POS["all"]
                    d_id = float((run(m0, S0, ib, "patch", {("act", c, L): (ALLP, C0[(c, L)]) for c in "qkv" for L in range(NL)}) - sc0).abs().max())
                    d_of = float((run(m0, S0, ib, "patch", {("act", "o", L): (ALLP, C1[("o", L)]) for L in range(NL)}) - sc1).abs().max())
                    d_or = float((run(m1, S1, ib, "patch", {("act", "o", L): (ALLP, C0[("o", L)]) for L in range(NL)}) - sc0).abs().max())
                    run(m0, S0, ib, "patch", {("act", c, L): (ALLP, C1[(c, L)]) for c in "qkv" for L in range(NL)}, capz=True)
                    d_z = max(float((S0.zstore[L] - Z1[L]).abs().max()) for L in range(NL))
                    spec_r = {("act", "r", CK_L): (ALLP, C1[("r", CK_L)])}; spec_r.update({("act", "o", L): (ALLP, C1[("o", L)]) for L in range(CK_L, NL)})
                    d_r = float((run(m0, S0, ib, "patch", spec_r) - sc1).abs().max())
                    d_wf = float((run(m0, S0, ib, "patch", {("w", c, L): (ALLP, WT[(1, c, L)]) for c in "qkvo" for L in range(NL)}) - sc1).abs().max())
                    d_wr = float((run(m1, S1, ib, "patch", {("w", c, L): (ALLP, WT[(0, c, L)]) for c in "qkvo" for L in range(NL)}) - sc0).abs().max())
                    ch = dict(m0_vs_m1=d_m, identity=d_id, full_o_fwd=d_of, full_o_rev=d_or, qkv_to_z=d_z, r_hook=d_r, weights_fwd=d_wf, weights_rev=d_wr)
                    res["checks"][f"item{n_checked}"] = ch; print("  check " + " ".join(f"{k} {v:.2e}" for k, v in ch.items()), flush=True)
                    assert d_id < 1e-3 and d_of < 5e-2 and d_or < 5e-2 and d_z < 1e-2 and d_r < 5e-2 and d_wf < 5e-2 and d_wr < 5e-2, ch
                    n_checked += 1
                for name, kind, comps, pkey, scope in ARMS:
                    layers = SCOPES[scope] if isinstance(scope, str) else scope; out = {}
                    for dirn, (mod, spl, src, oth) in (("fwd", (m0, S0, C1, 1)), ("rev", (m1, S1, C0, 0))):
                        if kind == "act": spec = {("act", c, L): (POS[pkey], src[(c, L)]) for c in comps for L in layers}
                        else: spec = {("w", c, L): (POS[pkey], WT[(oth, c, L)]) for c in comps for L in layers}
                        s = run(mod, spl, ib, "patch", spec); p = torch.softmax(s, 0)
                        out[dirn] = [round(float(p[xi]), 6), round(float(p[ti]), 6), letters[int(s.argmax())], round(float(s[xi] - s[ti]), 4)]
                    rec["arms"][name] = out
                del C0, C1, Z1; S0.store = {}; S1.store = {}; S1.zstore = {}; S0.zstore = {}
                res["items"].append(rec)
        res["counts"][cue] = cnt
        print(f"[{args.tag}] {cue:<26s} {cnt}  ({time.time() - t0:.0f}s)", flush=True)
    res["n_checked"] = n_checked   # the analysis refuses files with n_checked == 0 (no item had M0 and M1 differing by > 0.5)
    if n_checked == 0: print("!!! WARNING: no item qualified for the exactness checks", flush=True)
    for h in S0.hooks + S1.hooks: h.remove()
    json.dump(res, open(f"{OUT}/route_{args.tag}.json", "w"))
    print(f"saved -> {OUT}/route_{args.tag}.json  |S|={len(res['items'])}  checks={n_checked}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
