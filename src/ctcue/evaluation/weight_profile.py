#!/usr/bin/env python3
"""
V6 (27 Sep 2026) = ct_weights5.py with ONE change: head_dim falls back to hidden_size // num_attention_heads when
config.json has no head_dim (Llama-3.1-8B); gemma (explicit head_dim 256) is unchanged - regression vs weights_pub.json.
V5 (27 Sep 2026) = ct_weights4.py for the PUBLISHED-CODE adapters: rank and alpha read from each adapter_config (BCT r8
a16 q,k,v,o; ACT and AttCT r8 a16 q,v; RMCT r8 a32 q,k,v,o + MLP + lm_head). Attention modules without LoRA get an exact
zero update (as V4 did for layer bands); non-attention tensors (RMCT's MLP / lm_head LoRA, the embedding copy PEFT saves
with an lm_head target) are NOT part of the attention profile and are only counted (meta "skipped"). Per-head arithmetic
and the direct-materialisation check are V4's, unchanged.
V4 (26 Sep 2026) = ct_weights3.py generalised, per-head PROFILE only (no pairwise dW cosines, which are
uninformative across seeds): --model {gemma2b, llama8b}, --root (adapter root dir), --tags (comma list of
<cell>/step<S> under root, or "auto" = every */step* dir except step0), --out. Adapters restricted to a layer band are
allowed: modules without LoRA get an exact zero update. The per-head arithmetic is ct_weights3's, unchanged.
Verification: on the grid2 adapters it must reproduce ct_weights3's profile (weights_grid2.json) to float precision.
WEIGHT-SPACE PROFILE of the grid2 LoRA adapters (wave 3, 26 Sep 2026). CPU job, no model forward.
Adapters: {bct,act,rmct,attct}_s{0,1,2}/step{50,100,200,474} from ct_train2.py (LoRA r16, alpha 32, q/k/v/o of all
26 layers; lora_A initialised IDENTICALLY across methods at a given seed, so same-seed cross-method similarities
are inflated by the shared init: every similarity is therefore reported separately for SAME-seed and
DIFFERENT-seed pairs, and only different-seed pairs are used for the within- vs between-method test).
Per adapter and module (dW = (alpha/r) B A):
  * rel      = ||dW||_F / ||W0||_F
  * per head: q rows (head h), k/v rows (kv-group h//2), o columns (head h): ||dW_block|| / ||W0_block||
  * per head, attention composites with W = W0 + dW:  QK_h = Wq_h^T Wk_g (what the head attends to) and
    OV_h = Wo_h Wv_g (what it writes); relative change ||QK1 - QK0||_F / ||QK0||_F via 256x256 trace identities
    (||A^T B||_F^2 = tr((A A^T)(B B^T)), exact, never forming the 2304x2304 products). RoPE is ignored (weight-level).
Pairwise, for every pair of adapters at the same step and every module: cos(dW_i, dW_j) = tr((B_i^T B_j)(A_j A_i^T)) /
norms (exact, r x r), aggregated per module type and per layer band. CAVEAT (wave-3 audit): lora_A moves only ~10% from
its random seed-specific init, so raw dW cosines between DIFFERENT seeds are ~0 even within a method; they are reported
with the within-method different-seed value as the reliability ceiling, and method similarity is tested in FUNCTION space
(per-layer attention-contribution changes on real prompts, from ct_probe3 ao_b; see ana_probe3.py), not with raw dW cosines.
All arithmetic in float64 (fp32 Gram traces lose ~1e-3 relative precision at step-50 update sizes).
"""
import os, json, itertools, time, glob, argparse, numpy as np, torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
MODEL_IDS = {"gemma2b": "google/gemma-2-2b-it", "llama8b": "meta-llama/Llama-3.1-8B-Instruct"}
_ap = argparse.ArgumentParser(); _ap.add_argument("--model", default="gemma2b", choices=sorted(MODEL_IDS)); _ap.add_argument("--root", required=True)
_ap.add_argument("--tags", default="auto"); _ap.add_argument("--out", required=True); ARGS = _ap.parse_args()
assert not ARGS.out.endswith("weights_grid2.json"), "never overwrite the wave-3 output"
BASE = snapshot_download(MODEL_IDS[ARGS.model], local_files_only=True)
M = ARGS.root; OUT = ARGS.out
cfg = json.load(open(f"{BASE}/config.json")); NH, NKV, NL = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"]
HD = cfg["head_dim"] if cfg.get("head_dim") is not None else cfg["hidden_size"] // NH   # V6: Llama-3.1 config.json has no head_dim (4096 / 32 = 128)
G = NH // NKV; ATT = ["q_proj", "k_proj", "v_proj", "o_proj"]
torch.set_default_dtype(torch.float64)
TAGS = (sorted(os.path.relpath(d, M) for d in glob.glob(f"{M}/*/step*") if not d.endswith("/step0") and "smoke" not in d)
        if ARGS.tags == "auto" else ARGS.tags.split(","))
print(f"{len(TAGS)} adapters under {M}", flush=True)
idx = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
t0 = time.time()
W0 = {}
for L in range(NL):
    for mod in ATT:
        key = f"model.layers.{L}.self_attn.{mod}.weight"
        with safe_open(f"{BASE}/{idx[key]}", "pt") as f: W0[(L, mod)] = f.get_tensor(key).double()   # float64: Gram traces cancel at small updates
print(f"base weights loaded {time.time() - t0:.0f}s", flush=True)


SKIPPED = {}


def load_adapter(path):
    c = json.load(open(f"{path}/adapter_config.json")); r, alpha = int(c["r"]), float(c["lora_alpha"])
    assert not c.get("use_rslora", False) and not c.get("use_dora", False) and c.get("peft_type", "LORA") == "LORA", c
    assert not c.get("rank_pattern") and not c.get("alpha_pattern"), "per-module rank/alpha not handled"
    T = {}; skipped = []
    with safe_open(f"{path}/adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            if ".self_attn." in k and (".lora_A.weight" in k or ".lora_B.weight" in k): T[k] = f.get_tensor(k).double()
            else: skipped.append(k)
    AB = {}; n_present = 0
    for L in range(NL):
        for mod in ATT:
            p = f"base_model.model.model.layers.{L}.self_attn.{mod}"
            if f"{p}.lora_A.weight" in T:
                A, B = T[f"{p}.lora_A.weight"], T[f"{p}.lora_B.weight"]; assert A.shape[0] == r and B.shape[1] == r, (p, A.shape, B.shape)
                AB[(L, mod)] = (A, B * (alpha / r)); n_present += 1                                   # dW = B' A
            else:   # module without LoRA: exact zero update
                out_f, in_f = W0[(L, mod)].shape; AB[(L, mod)] = (torch.zeros(r, in_f), torch.zeros(out_f, r))
    assert len(T) == 2 * n_present and n_present > 0, (path, len(T), n_present)   # every attention LoRA tensor was used
    assert all("lora_" not in k or ".mlp." in k or "lm_head" in k for k in skipped), [k for k in skipped if "lora_" in k][:5]
    SKIPPED[path] = {"n_attn_modules": n_present, "r": r, "alpha": alpha, "n_skipped": len(skipped),
                     "skipped_lora_mlp": sum(".mlp." in k and "lora_" in k for k in skipped), "skipped_lora_lm_head": sum("lm_head" in k and "lora_" in k for k in skipped),
                     "skipped_other": sorted(k for k in skipped if "lora_" not in k)}
    return AB


def rows(mod, h):
    if mod == "q_proj": return slice(h * HD, (h + 1) * HD)
    return slice((h // G) * HD, (h // G + 1) * HD)          # k/v: kv-group of head h


# Every per-head block X (rows = the head's 256 dims, cols = 2304 residual dims) is updated by a rank-16 term
# dX = Lf @ Rf (Lf [256,16], Rf [16,2304]). All Frobenius quantities of the 2304x2304 composites follow exactly from
# 256x256 Gram matrices: ||A^T B||^2 = tr((A A^T)(B B^T)), <A1^T B1, A0^T B0> = tr((A1 A0^T)(B0 B1^T)).
GRAM = {}
for L in range(NL):
    for h in range(NH):
        q, kv, o = rows("q_proj", h), rows("k_proj", h), slice(h * HD, (h + 1) * HD)
        X = {"q": W0[(L, "q_proj")][q], "k": W0[(L, "k_proj")][kv], "v": W0[(L, "v_proj")][kv], "o": W0[(L, "o_proj")][:, o].T.contiguous()}
        GRAM[(L, h)] = {n: (x, x @ x.T) for n, x in X.items()}
print(f"base grams cached {time.time() - t0:.0f}s", flush=True)


def blocks(AB, L, h):
    """Lf, Rf of each head block update: q rows / k,v kv-group rows / o columns (transposed to [256, 2304])."""
    q, kv, o = rows("q_proj", h), rows("k_proj", h), slice(h * HD, (h + 1) * HD)
    Aq, Bq = AB[(L, "q_proj")]; Ak, Bk = AB[(L, "k_proj")]; Av, Bv = AB[(L, "v_proj")]; Ao, Bo = AB[(L, "o_proj")]
    return {"q": (Bq[q], Aq), "k": (Bk[kv], Ak), "v": (Bv[kv], Av), "o": (Ao[:, o].T, Bo.T)}   # dO_h^T = Ao_h^T Bo^T


def grams(x0, g0, Lf, Rf):
    """Return X1 X1^T and X1 X0^T for X1 = X0 + Lf Rf, exactly, without forming X1."""
    P = (x0 @ Rf.T) @ Lf.T                       # X0 dX^T  [256,256]
    D = Lf @ (Rf @ Rf.T) @ Lf.T                  # dX dX^T
    return g0 + P + P.T + D, g0 + P.T            # X1X1^T, X1X0^T (= G0 + dX X0^T)


def profile(AB):
    R = {"rel": {}, "head": {}}
    for L in range(NL):
        for mod in ATT:
            A, B = AB[(L, mod)]
            R["rel"][f"L{L}.{mod}"] = float(torch.trace((B.T @ B) @ (A @ A.T))) ** 0.5 / float(W0[(L, mod)].norm())
        for h in range(NH):
            gb, bl = GRAM[(L, h)], blocks(AB, L, h); G1, G10 = {}, {}
            for n in ("q", "k", "v", "o"):
                G1[n], G10[n] = grams(gb[n][0], gb[n][1], *bl[n])
            def comp(a, b):   # relative change of the composite a^T b: ||a1^T b1 - a0^T b0|| / ||a0^T b0||
                f0 = float(torch.trace(gb[a][1] @ gb[b][1])); f1 = float(torch.trace(G1[a] @ G1[b]))
                x = float(torch.trace(G10[a] @ G10[b].T))          # tr((A1 A0^T)(B0 B1^T)) with B0 B1^T = (B1 B0^T)^T
                return max(f1 + f0 - 2 * x, 0.0) ** 0.5 / f0 ** 0.5
            rel = {n: float(torch.trace((bl[n][0].T @ bl[n][0]) @ (bl[n][1] @ bl[n][1].T))) ** 0.5 / float(torch.trace(gb[n][1])) ** 0.5 for n in ("q", "k", "v", "o")}
            R["head"][f"L{L}.H{h}"] = {"qk": comp("q", "k"), "ov": comp("o", "v"), **rel}
    return R


def direct_check(AB, L, h):
    """Materialise one head's composites directly and compare with the Gram-based values."""
    q, kv, o = rows("q_proj", h), rows("k_proj", h), slice(h * HD, (h + 1) * HD)
    dW = {mod: AB[(L, mod)][1] @ AB[(L, mod)][0] for mod in ATT}
    Q0, K0, V0, O0 = W0[(L, "q_proj")][q], W0[(L, "k_proj")][kv], W0[(L, "v_proj")][kv], W0[(L, "o_proj")][:, o]
    Q1, K1, V1, O1 = Q0 + dW["q_proj"][q], K0 + dW["k_proj"][kv], V0 + dW["v_proj"][kv], O0 + dW["o_proj"][:, o]
    qk = float((Q1.T @ K1 - Q0.T @ K0).norm() / (Q0.T @ K0).norm()); ov = float((O1 @ V1 - O0 @ V0).norm() / (O0 @ V0).norm())
    return qk, ov


def cos_dW(ab1, ab2):   # cos(B1 A1, B2 A2) = tr((B1^T B2)(A2 A1^T)) / norms, all r x r
    (A1, B1), (A2, B2) = ab1, ab2
    ip = float(torch.trace((B1.T @ B2) @ (A2 @ A1.T)))
    n1 = float(torch.trace((B1.T @ B1) @ (A1 @ A1.T))) ** 0.5; n2 = float(torch.trace((B2.T @ B2) @ (A2 @ A2.T))) ** 0.5
    return ip / (n1 * n2) if n1 > 0 and n2 > 0 else float("nan")


res = {"meta": {"model": MODEL_IDS[ARGS.model], "root": M, "NL": NL, "NH": NH, "NKV": NKV, "HD": HD}, "profile": {}}
CHECK_HEADS = [(NL * 2 // 3, NH - 2), (NL // 2, 3)]
for i, tag in enumerate(TAGS):
    AD = load_adapter(f"{M}/{tag}"); key = tag.replace("/", "_"); res["profile"][key] = profile(AD)
    if i < 2:   # exactness of the Gram path against direct materialisation, on two heads
        for (L, h) in CHECK_HEADS:
            dq, dv = direct_check(AD, L, h); hq = res["profile"][key]["head"][f"L{L}.H{h}"]
            assert abs(dq - hq["qk"]) <= 1e-3 * max(dq, 1e-6) + 1e-7 and abs(dv - hq["ov"]) <= 1e-3 * max(dv, 1e-6) + 1e-7, (tag, L, h, dq, hq["qk"], dv, hq["ov"])
            print(f"  direct check {tag} L{L}.H{h}: qk {dq:.6f} vs {hq['qk']:.6f}; ov {dv:.6f} vs {hq['ov']:.6f}", flush=True)
    if (i + 1) % 8 == 0: print(f"  {i + 1}/{len(TAGS)} profiled {time.time() - t0:.0f}s", flush=True)
res["meta"]["adapters"] = SKIPPED
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(res, open(OUT, "w"))
print("saved ->", OUT, f"{time.time() - t0:.0f}s", flush=True)
