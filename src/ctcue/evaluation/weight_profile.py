"""Measure how much a LoRA adapter changes each attention head (CPU, no forward pass).

For every head: the relative change of its q, k, v and o weight blocks and of its OV and QK
products. Only attention projections count. LoRA weights on the MLP or lm_head (RMCT) are skipped.
With --tags auto it profiles every <root>/*/step* adapter except step0. The paper's OV depth
centroid is computed from these profiles. The script has no main function and runs at module level,
so importing it runs it.

Usage: python -m ctcue.evaluation.weight_profile --model gemma2b --root <adapter root> --tags auto
           --out <file>
Output: the JSON file given by --out
"""

import os, json, itertools, time, glob, argparse, numpy as np, torch
from safetensors import safe_open
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
MODEL_IDS = {"gemma2b": "google/gemma-2-2b-it", "llama8b": "meta-llama/Llama-3.1-8B-Instruct"}
_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="gemma2b", choices=sorted(MODEL_IDS))
_ap.add_argument("--root", required=True)
_ap.add_argument("--tags", default="auto")
_ap.add_argument("--out", required=True)
ARGS = _ap.parse_args()
assert not ARGS.out.endswith("weights_grid2.json"), "never overwrite the wave-3 output"
BASE = snapshot_download(MODEL_IDS[ARGS.model], local_files_only=True)
M = ARGS.root
OUT = ARGS.out
cfg = json.load(open(f"{BASE}/config.json"))
NH, NKV, NL = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"]
# V6: Llama-3.1 config.json has no head_dim (4096 / 32 = 128)
HD = cfg["head_dim"] if cfg.get("head_dim") is not None else cfg["hidden_size"] // NH
G = NH // NKV
ATT = ["q_proj", "k_proj", "v_proj", "o_proj"]
torch.set_default_dtype(torch.float64)
TAGS = (
    sorted(
        os.path.relpath(d, M)
        for d in glob.glob(f"{M}/*/step*")
        if not d.endswith("/step0") and "smoke" not in d
    )
    if ARGS.tags == "auto"
    else ARGS.tags.split(",")
)
print(f"{len(TAGS)} adapters under {M}", flush=True)
idx = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
t0 = time.time()
W0 = {}
for L in range(NL):
    for mod in ATT:
        key = f"model.layers.{L}.self_attn.{mod}.weight"
        # float64: Gram traces cancel at small updates
        with safe_open(f"{BASE}/{idx[key]}", "pt") as f:
            W0[(L, mod)] = f.get_tensor(key).double()
print(f"base weights loaded {time.time() - t0:.0f}s", flush=True)


SKIPPED = {}


def load_adapter(path):
    c = json.load(open(f"{path}/adapter_config.json"))
    r, alpha = int(c["r"]), float(c["lora_alpha"])
    assert (
        not c.get("use_rslora", False)
        and not c.get("use_dora", False)
        and c.get("peft_type", "LORA") == "LORA"
    ), c
    assert not c.get("rank_pattern") and not c.get("alpha_pattern"), "per-module rank/alpha not handled"
    T = {}
    skipped = []
    with safe_open(f"{path}/adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            if ".self_attn." in k and (".lora_A.weight" in k or ".lora_B.weight" in k):
                T[k] = f.get_tensor(k).double()
            else:
                skipped.append(k)
    AB = {}
    n_present = 0
    for L in range(NL):
        for mod in ATT:
            p = f"base_model.model.model.layers.{L}.self_attn.{mod}"
            if f"{p}.lora_A.weight" in T:
                A, B = T[f"{p}.lora_A.weight"], T[f"{p}.lora_B.weight"]
                assert A.shape[0] == r and B.shape[1] == r, (p, A.shape, B.shape)
                # dW = B' A
                AB[(L, mod)] = (A, B * (alpha / r))
                n_present += 1
            else:  # module without LoRA: exact zero update
                out_f, in_f = W0[(L, mod)].shape
                AB[(L, mod)] = (torch.zeros(r, in_f), torch.zeros(out_f, r))
    # every attention LoRA tensor was used
    assert len(T) == 2 * n_present and n_present > 0, (path, len(T), n_present)
    assert all("lora_" not in k or ".mlp." in k or "lm_head" in k for k in skipped), [
        k for k in skipped if "lora_" in k
    ][:5]
    SKIPPED[path] = {
        "n_attn_modules": n_present,
        "r": r,
        "alpha": alpha,
        "n_skipped": len(skipped),
        "skipped_lora_mlp": sum(".mlp." in k and "lora_" in k for k in skipped),
        "skipped_lora_lm_head": sum("lm_head" in k and "lora_" in k for k in skipped),
        "skipped_other": sorted(k for k in skipped if "lora_" not in k),
    }
    return AB


def rows(mod, h):
    if mod == "q_proj":
        return slice(h * HD, (h + 1) * HD)
    return slice((h // G) * HD, (h // G + 1) * HD)  # k/v: kv-group of head h


# Every per-head block X (rows = the head's 256 dims, cols = 2304 residual dims) is updated by a rank-16 term
# dX = Lf @ Rf (Lf [256,16], Rf [16,2304]). All Frobenius quantities of the 2304x2304 composites follow
# exactly from 256x256 Gram matrices: ||A^T B||^2 = tr((A A^T)(B B^T)), <A1^T B1, A0^T B0> = tr((A1 A0^T)(B0
# B1^T)).
GRAM = {}
for L in range(NL):
    for h in range(NH):
        q, kv, o = rows("q_proj", h), rows("k_proj", h), slice(h * HD, (h + 1) * HD)
        X = {
            "q": W0[(L, "q_proj")][q],
            "k": W0[(L, "k_proj")][kv],
            "v": W0[(L, "v_proj")][kv],
            "o": W0[(L, "o_proj")][:, o].T.contiguous(),
        }
        GRAM[(L, h)] = {n: (x, x @ x.T) for n, x in X.items()}
print(f"base grams cached {time.time() - t0:.0f}s", flush=True)


def blocks(AB, L, h):
    """Lf, Rf of each head block update: q rows / k,v kv-group rows / o columns (transposed to [256, 2304])."""
    q, kv, o = rows("q_proj", h), rows("k_proj", h), slice(h * HD, (h + 1) * HD)
    Aq, Bq = AB[(L, "q_proj")]
    Ak, Bk = AB[(L, "k_proj")]
    Av, Bv = AB[(L, "v_proj")]
    Ao, Bo = AB[(L, "o_proj")]
    # dO_h^T = Ao_h^T Bo^T
    return {"q": (Bq[q], Aq), "k": (Bk[kv], Ak), "v": (Bv[kv], Av), "o": (Ao[:, o].T, Bo.T)}


def grams(x0, g0, Lf, Rf):
    """Return X1 X1^T and X1 X0^T for X1 = X0 + Lf Rf, exactly, without forming X1."""
    P = (x0 @ Rf.T) @ Lf.T  # X0 dX^T  [256,256]
    D = Lf @ (Rf @ Rf.T) @ Lf.T  # dX dX^T
    return g0 + P + P.T + D, g0 + P.T  # X1X1^T, X1X0^T (= G0 + dX X0^T)


def profile(AB):
    R = {"rel": {}, "head": {}}
    for L in range(NL):
        for mod in ATT:
            A, B = AB[(L, mod)]
            R["rel"][f"L{L}.{mod}"] = float(torch.trace((B.T @ B) @ (A @ A.T))) ** 0.5 / float(
                W0[(L, mod)].norm()
            )
        for h in range(NH):
            gb, bl = GRAM[(L, h)], blocks(AB, L, h)
            G1, G10 = {}, {}
            for n in ("q", "k", "v", "o"):
                G1[n], G10[n] = grams(gb[n][0], gb[n][1], *bl[n])

            def comp(a, b):  # relative change of the composite a^T b: ||a1^T b1 - a0^T b0|| / ||a0^T b0||
                f0 = float(torch.trace(gb[a][1] @ gb[b][1]))
                f1 = float(torch.trace(G1[a] @ G1[b]))
                # tr((A1 A0^T)(B0 B1^T)) with B0 B1^T = (B1 B0^T)^T
                x = float(torch.trace(G10[a] @ G10[b].T))
                return max(f1 + f0 - 2 * x, 0.0) ** 0.5 / f0**0.5

            rel = {
                n: float(torch.trace((bl[n][0].T @ bl[n][0]) @ (bl[n][1] @ bl[n][1].T))) ** 0.5
                / float(torch.trace(gb[n][1])) ** 0.5
                for n in ("q", "k", "v", "o")
            }
            R["head"][f"L{L}.H{h}"] = {"qk": comp("q", "k"), "ov": comp("o", "v"), **rel}
    return R


def direct_check(AB, L, h):
    """Materialise one head's composites directly and compare with the Gram-based values."""
    q, kv, o = rows("q_proj", h), rows("k_proj", h), slice(h * HD, (h + 1) * HD)
    dW = {mod: AB[(L, mod)][1] @ AB[(L, mod)][0] for mod in ATT}
    Q0, K0, V0, O0 = (
        W0[(L, "q_proj")][q],
        W0[(L, "k_proj")][kv],
        W0[(L, "v_proj")][kv],
        W0[(L, "o_proj")][:, o],
    )
    Q1, K1, V1, O1 = (
        Q0 + dW["q_proj"][q],
        K0 + dW["k_proj"][kv],
        V0 + dW["v_proj"][kv],
        O0 + dW["o_proj"][:, o],
    )
    qk = float((Q1.T @ K1 - Q0.T @ K0).norm() / (Q0.T @ K0).norm())
    ov = float((O1 @ V1 - O0 @ V0).norm() / (O0 @ V0).norm())
    return qk, ov


def cos_dW(ab1, ab2):  # cos(B1 A1, B2 A2) = tr((B1^T B2)(A2 A1^T)) / norms, all r x r
    (A1, B1), (A2, B2) = ab1, ab2
    ip = float(torch.trace((B1.T @ B2) @ (A2 @ A1.T)))
    n1 = float(torch.trace((B1.T @ B1) @ (A1 @ A1.T))) ** 0.5
    n2 = float(torch.trace((B2.T @ B2) @ (A2 @ A2.T))) ** 0.5
    return ip / (n1 * n2) if n1 > 0 and n2 > 0 else float("nan")


res = {
    "meta": {"model": MODEL_IDS[ARGS.model], "root": M, "NL": NL, "NH": NH, "NKV": NKV, "HD": HD},
    "profile": {},
}
CHECK_HEADS = [(NL * 2 // 3, NH - 2), (NL // 2, 3)]
for i, tag in enumerate(TAGS):
    AD = load_adapter(f"{M}/{tag}")
    key = tag.replace("/", "_")
    res["profile"][key] = profile(AD)
    if i < 2:  # exactness of the Gram path against direct materialisation, on two heads
        for L, h in CHECK_HEADS:
            dq, dv = direct_check(AD, L, h)
            hq = res["profile"][key]["head"][f"L{L}.H{h}"]
            assert (
                abs(dq - hq["qk"]) <= 1e-3 * max(dq, 1e-6) + 1e-7
                and abs(dv - hq["ov"]) <= 1e-3 * max(dv, 1e-6) + 1e-7
            ), (tag, L, h, dq, hq["qk"], dv, hq["ov"])
            print(
                f"  direct check {tag} L{L}.H{h}: qk {dq:.6f} vs {hq['qk']:.6f}; ov {dv:.6f} vs {hq['ov']:.6f}",
                flush=True,
            )
    if (i + 1) % 8 == 0:
        print(f"  {i + 1}/{len(TAGS)} profiled {time.time() - t0:.0f}s", flush=True)
res["meta"]["adapters"] = SKIPPED
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"))
print("saved ->", OUT, f"{time.time() - t0:.0f}s", flush=True)
