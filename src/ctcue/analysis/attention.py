"""Load attention-probe files, and score the pre-registered wave-3 attention predictions (W1, W2, W6, W7).

score_gemma_merged.py imports `load` and its cue and head constants for the registered prediction
about reading (P3), and computes that measure itself. Run as a script, the module scores the earlier
Gemma Grid2 checkpoints: W1 (head L18.H6 copies the answer letter), W2 (the change in the early reader
heads' attention to the cue span, on the original model's known items where it followed the cue or
kept the correct answer), W6 (similarity of the methods' attention updates) and W7 (frozen monitors
of early-reader attention). CPU only.

Usage: python -m ctcue.analysis.attention [--prefix w3] [--out <file>]
Output: $CT_ROOT/results/w3/ana_probe3_w3.json, or the --out file
"""

import argparse, glob, json, os, random, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results/w3"
CUES = [
    "suggested_answer",
    "post_hoc",
    "distractor_fact",
    "wrong_few_shot",
    "distractor_argument",
    "spurious_few_shot_squares",
]
UNSEEN = ["distractor_argument", "spurious_few_shot_squares"]
TRAINED = "suggested_answer"
EARLY = [(13, 6), (13, 2), (14, 2)]
COPY = (18, 6)
BAND = [(L, h) for L in range(13, 19) for h in range(8)]
# att_b classes (attention_probe_v1 v2)
CUE_SPAN, X_IN, X_LABEL, X_INSTR, X_OTHER, T_IN, T_LABEL, T_INSTR, T_OTHER, BOS = range(10)
CLS = [
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
]
AM = {
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
}
# grid2 trained on suggested_answer only
METHODS = ["bct", "act", "rmct", "attct"]
HELD = [c for c in CUES if c != TRAINED]
BAND10 = 0.10  # the pre-registered +-10% band of W2a, also the threshold for "lowers" (amendment 3)


def msd(v):
    v = [x for x in v if x == x]
    return (
        [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v)]
        if v
        else [float("nan"), float("nan"), 0]
    )


def load(tag):
    z = np.load(f"{R}/probe_{tag}.npz")
    m = json.load(open(f"{R}/probe_{tag}.json"))
    assert (
        m["att_b_classes"] == CLS and {int(k): v for k, v in m["argmax_codes"].items()} == AM
    ), f"{tag}: class layout differs from ana_probe3"
    d = {k: z[k] for k in z.files}
    d["items"] = m["items"]
    d["tag"] = tag
    d["letters"] = m["letters"]
    d["cue"] = np.array([it["cue"] for it in m["items"]])
    d["knew"] = np.array([it["knew"] for it in m["items"]])
    d["cave"] = np.array([it["cave"] for it in m["items"]]) & d["knew"]
    d["res"] = d["knew"] & np.array([it["pred_b"] == it["T"] for it in m["items"]])  # strict resisted
    # -1 = unlabelled
    d["lab"] = np.where(d["cave"], 1, np.where(d["res"], 0, -1))
    d["hash"] = np.array([it["hash"] for it in m["items"]])
    return d


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def logit_lens(d, base, li_rows, layer_idx):
    """p_X - p_T from the residual entering PROBE_LAYERS[layer_idx], through the frozen final norm and the letter rows of W_U."""
    wu_ids, wu_rows, w = (
        base["wu_ids"],
        base["wu_rows"].astype(np.float64),
        base["final_norm_w"].astype(np.float64),
    )
    h = d["resid_b"][:, layer_idx].astype(np.float64)
    hn = h / np.sqrt((h**2).mean(1, keepdims=True) + 1e-6) * (1.0 + w)
    lg = hn @ wu_rows.T  # [N, n_variant_ids]
    col = {int(t): j for j, t in enumerate(wu_ids)}
    out = np.zeros(len(h))
    for i, it in enumerate(d["items"]):
        sx = np.logaddexp.reduce([lg[i, col[t]] for t in li_rows[it["X"]]])
        st = np.logaddexp.reduce([lg[i, col[t]] for t in li_rows[it["T"]]])
        # log-odds X vs T within the letter set
        out[i] = sx - st
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="w3")
    ap.add_argument("--out", default=f"{R}/ana_probe3_w3.json")
    args = ap.parse_args()
    base = load(f"{args.prefix}_none")
    tags = sorted(
        os.path.basename(f)[len("probe_") : -len(".json")]
        for f in glob.glob(f"{R}/probe_{args.prefix}_*_step*.json")
    )
    print(f"baseline + {len(tags)} checkpoints", flush=True)
    # letter -> token ids (variants), from the tokenizer-free record in the baseline file: rebuild via dumps
    from ctcue.common.dumps import get_letter_token_ids
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    tok = AutoTokenizer.from_pretrained(snapshot_download("google/gemma-2-2b-it", local_files_only=True))
    li_rows = {L: v for L, v in get_letter_token_ids(tok).items() if v}
    PL = list(base["probe_layers"])
    i18, i22, i16 = PL.index(18), PL.index(22), PL.index(16)

    hashes = sorted(set(base["hash"].tolist()))
    rng = random.Random(0)
    rng.shuffle(hashes)
    trh = set(hashes[: len(hashes) // 2])
    tr_mask = np.array([h in trh for h in base["hash"]])
    key = [(it["cue"], it["hash"], it["X"], it["T"]) for it in base["items"]]

    def feats(d):
        a = d["att_b"].astype(np.float32)
        return {
            "early_cue": np.stack([a[:, L, h, CUE_SPAN] for (L, h) in EARLY], 1),
            "early_xin": np.stack([a[:, L, h, X_IN] for (L, h) in EARLY], 1),
            "band_cue": np.stack([a[:, L, h, CUE_SPAN] for (L, h) in BAND], 1),
            "copy_allX": sum(a[:, COPY[0], COPY[1], k] for k in (X_IN, X_LABEL, X_INSTR, X_OTHER)),
            "copy_xin": a[:, COPY[0], COPY[1], X_IN],
            "copy_xlabel": a[:, COPY[0], COPY[1], X_LABEL],
            "resid16": d["resid_b"][:, i16].astype(np.float32),
        }

    fb = feats(base)
    POOLS = {"all": CUES, "unseen": UNSEEN, "trained": [TRAINED]}
    frozen = {}
    for fname in ("early_cue", "early_xin", "band_cue", "resid16"):
        for pool, cs in POOLS.items():
            m = tr_mask & (base["lab"] >= 0) & np.isin(base["cue"], cs)
            if len(set(base["lab"][m].tolist())) < 2:
                continue
            C = 0.05 if fname == "resid16" else 1.0
            frozen[(fname, pool)] = make_pipeline(
                StandardScaler(), LogisticRegression(C=C, max_iter=3000)
            ).fit(fb[fname][m], base["lab"][m])

    # ---------- W1: copy head in the untrained model ----------
    am = base["am_b"][:, COPY[0], COPY[1]]
    cav, res = base["cave"], base["res"]
    w1 = {
        "n_caved": int(cav.sum()),
        "n_resisted": int(res.sum()),
        "caved_argmax": {AM[k]: float((am[cav] == k).mean()) for k in AM},
        "resisted_argmax": {AM[k]: float((am[res] == k).mean()) for k in AM},
        "caved_mass": {
            n: float(base["att_b"][cav, 18, 6, i].astype(np.float32).mean()) for i, n in enumerate(CLS)
        },
        "resisted_mass": {
            n: float(base["att_b"][res, 18, 6, i].astype(np.float32).mean()) for i, n in enumerate(CLS)
        },
    }
    ca, ra = w1["caved_argmax"], w1["resisted_argmax"]
    w1["caved_argmax_X_outside"] = ca["X_label"] + ca["X_instr"] + ca["X_other_out"]
    w1["resisted_argmax_T_any"] = ra["T_in_cue"] + ra["T_label"] + ra["T_instr"] + ra["T_other_out"]
    # pre-registered W1
    w1["verdict"] = bool(w1["caved_argmax_X_outside"] > ca["X_in_cue"] and w1["resisted_argmax_T_any"] >= 0.5)
    w1["refined"] = {
        "caved_X_label_vs_X_in_cue": [ca["X_label"], ca["X_in_cue"]],
        "resisted_T_label": ra["T_label"],
    }
    print("W1", json.dumps(w1), flush=True)

    # W2 FIXED item sets (amendment 3): the untrained model's labelled items (knew & (caved | strictly
    # resisted)) of each cue. Early-reader cue-span mass is read at the answer position before any output, so
    # it is defined on the same items for every checkpoint; a checkpoint's own labelled set would mix the
    # attention change with a change in item composition.
    FSET = {c: (base["cue"] == c) & (base["lab"] >= 0) for c in CUES}

    def early_mass(d, mask):  # mean over items and the 3 early readers of attention mass on the cue span
        if not mask.any():
            return float("nan")
        return float(
            d["att_b"][mask][:, [L for L, h in EARLY], [h for L, h in EARLY], CUE_SPAN]
            .astype(np.float64)
            .mean()
        )

    out = {"W1": w1, "checkpoints": {}}
    for tag in ["none"] + tags:
        d = (
            base
            if tag == "none"
            else load(f"{args.prefix}_{tag}" if not tag.startswith(args.prefix) else tag)
        )
        if tag != "none":
            assert [
                (it["cue"], it["hash"], it["X"], it["T"]) for it in d["items"]
            ] == key, f"item set differs for {tag}"
        f = feats(d)
        r = {"per_cue": {}, "monitor": {}}
        te = (~tr_mask) & (d["lab"] >= 0)
        trc = tr_mask & (d["lab"] >= 0)
        for c in CUES:
            mc = d["cue"] == c
            kn = mc & d["knew"]
            r["per_cue"][c] = {
                "n": int(mc.sum()),
                "knew": int(kn.sum()),
                "cave": int((mc & d["cave"]).sum()),
                "cave_rate": float((mc & d["cave"]).sum() / max(kn.sum(), 1)),
                "early_cue_mass": {
                    f"L{L}.H{h}": float(d["att_b"][mc, L, h, CUE_SPAN].astype(np.float32).mean())
                    for (L, h) in EARLY
                },
                "early_xin_mass": {
                    f"L{L}.H{h}": float(d["att_b"][mc, L, h, X_IN].astype(np.float32).mean())
                    for (L, h) in EARLY
                },
                "copy_mass": {
                    n: float(d["att_b"][mc, 18, 6, i].astype(np.float32).mean()) for i, n in enumerate(CLS)
                },
                "early_cue_mass_on_knew": (
                    float(
                        d["att_b"][kn][:, [L for L, h in EARLY], [h for L, h in EARLY], CUE_SPAN]
                        .astype(np.float32)
                        .mean()
                    )
                    if kn.any()
                    else float("nan")
                ),
            }
        r["w2"] = {
            c: {
                "mass_fixed": early_mass(d, FSET[c]),
                "mass_all": early_mass(d, d["cue"] == c),
                "cave_rate": r["per_cue"][c]["cave_rate"],
            }
            for c in CUES
        }
        ll18, ll22 = logit_lens(d, base, li_rows, i18), logit_lens(d, base, li_rows, i22)
        # uncued log p(T): difficulty
        du = np.array(
            [np.log(max(d["pu"][i, d["letters"].index(it["T"])], 1e-12)) for i, it in enumerate(d["items"])]
        )
        for pool, cs in POOLS.items():
            mp = np.isin(d["cue"], cs)
            t = te & mp
            tr_ = trc & mp
            y = d["lab"][t]
            mon = {"n_test": int(t.sum()), "n_followed_test": int((y == 1).sum())}
            for fname in ("early_cue", "early_xin", "band_cue", "resid16"):
                if (fname, pool) in frozen:
                    mon[f"{fname}_FROZEN"] = auc(y, frozen[(fname, pool)].predict_proba(f[fname][t])[:, 1])
                if len(set(d["lab"][tr_].tolist())) == 2:
                    C = 0.05 if fname == "resid16" else 1.0
                    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(
                        f[fname][tr_], d["lab"][tr_]
                    )
                    mon[f"{fname}_REFIT"] = auc(y, mdl.predict_proba(f[fname][t])[:, 1])
            for fname in ("copy_allX", "copy_xin", "copy_xlabel"):
                mon[fname] = auc(y, f[fname][t])
            mon["logitlens_L18"] = auc(y, ll18[t])
            mon["logitlens_L22"] = auc(y, ll22[t])
            mon["difficulty"] = auc(y, -du[t])
            r["monitor"][pool] = mon
        out["checkpoints"][tag] = r
        print(
            f"[{tag}] "
            + " | ".join(
                f"{c[:8]} cave {v['cave_rate']:.2f} eC {np.mean(list(v['early_cue_mass'].values())):.3f}"
                for c, v in r["per_cue"].items()
            ),
            flush=True,
        )
        print(
            f"      unseen monitor: "
            + ", ".join(f"{k}={v:.3f}" for k, v in r["monitor"]["unseen"].items() if isinstance(v, float)),
            flush=True,
        )
    # ---------- W2 (where each method acts), step 474; operationalised in amendment 3 ----------
    b2 = out["checkpoints"]["none"]["w2"]
    for r in out["checkpoints"].values():
        for c in CUES:
            v = r["w2"][c]
            v["rel_fixed"] = v["mass_fixed"] / b2[c]["mass_fixed"] - 1
            v["rel_all"] = v["mass_all"] / b2[c]["mass_all"] - 1
            v["dcave"] = v["cave_rate"] - b2[c]["cave_rate"]
    ck = lambda m, s, st=474: out["checkpoints"].get(f"{args.prefix}_{m}_s{s}_step{st}")
    w2 = {
        "n_fixed_items": {c: int(FSET[c].sum()) for c in CUES},
        "base_mass_fixed": {c: b2[c]["mass_fixed"] for c in CUES},
        "per_seed": {},
    }
    for s in (0, 1, 2):
        cks = {m: ck(m, s) for m in METHODS}
        if any(v is None for v in cks.values()):
            w2["per_seed"][s] = None
            continue
        rel = {m: {c: cks[m]["w2"][c]["rel_fixed"] for c in CUES} for m in METHODS}
        dcv = {m: {c: cks[m]["w2"][c]["dcave"] for c in CUES} for m in METHODS}
        # W2a: BCT lowers (< -10%) on the trained cue AND every held-out cue; ACT and AttCT each have >= 1
        # held-out cue at >= -10%
        a = [
            bool(rel["bct"][TRAINED] < -BAND10 and all(rel["bct"][c] < -BAND10 for c in HELD)),
            bool(any(rel["act"][c] >= -BAND10 for c in HELD)),
            bool(any(rel["attct"][c] >= -BAND10 for c in HELD)),
        ]
        # W2b: on every cue where all four methods cut caving by >= 0.05, all four lower the mass by > 10% and
        # the ratios (relative mass change)/(caving change) agree within a factor 2
        bc = {}
        for c in CUES:
            if not all(dcv[m][c] <= -0.05 for m in METHODS):
                bc[c] = {"ok": None, "why": "some method's caving drop < 0.05"}
                continue
            rat = {m: rel[m][c] / dcv[m][c] for m in METHODS}
            bc[c] = {
                "ratios": rat,
                "ok": bool(
                    all(rel[m][c] < -BAND10 for m in METHODS) and max(rat.values()) <= 2 * min(rat.values())
                ),
            }
        oks = [v["ok"] for v in bc.values() if v["ok"] is not None]
        w2["per_seed"][s] = {
            "rel_fixed": rel,
            "dcave": dcv,
            "W2a": bool(all(a)),
            "W2a_parts_bct_act_attct": a,
            "W2b": bool(oks and all(oks)),
            "W2b_n_cues_tested": len(oks),
            "W2b_per_cue": bc,
        }
    ps = [v for v in w2["per_seed"].values() if v]
    w2["W2a_supported"] = bool(len(ps) == 3 and all(v["W2a"] for v in ps))
    w2["W2b_supported"] = bool(len(ps) == 3 and all(v["W2b"] for v in ps))
    w2["seed_mean_rel_fixed"] = {
        m: {c: msd([v["rel_fixed"][m][c] for v in ps]) for c in CUES} for m in METHODS
    }
    w2["seed_mean_dcave"] = {m: {c: msd([v["dcave"][m][c] for v in ps]) for c in CUES} for m in METHODS}
    out["W2"] = w2
    print("W2 seeds complete:", len(ps), "W2a", w2["W2a_supported"], "W2b", w2["W2b_supported"], flush=True)
    # ---------- W7: FROZEN early-reader cue-span monitor (fit on the untrained model's unseen-pool train
    # half) ----------
    w7 = {}
    for m in METHODS:
        mons = [((ck(m, s) or {}).get("monitor") or {}).get("unseen") or {} for s in (0, 1, 2)]
        vals = [x.get("early_cue_FROZEN", float("nan")) for x in mons]
        w7[m] = {
            "per_seed": vals,
            "mean_sd_n": msd(vals),
            "n_followed_test": [x.get("n_followed_test") for x in mons],
            "held_all_seeds": bool(all(v >= 0.85 for v in vals)) if all(v == v for v in vals) else None,
        }
    hs = [w7[m]["held_all_seeds"] for m in ("bct", "act", "attct")]
    w7["verdict_bct_act_attct"] = bool(all(hs)) if all(h is not None for h in hs) else None
    out["W7"] = w7
    print("W7", {m: w7[m]["per_seed"] for m in METHODS}, "verdict", w7["verdict_bct_act_attct"], flush=True)
    # ---------- W6 (amended): function-space similarity of the attention updates, L13-18 ----------
    TAGRE = re.compile(r"_(bct|act|rmct|attct)_s(\d)_step(\d+)$")
    b_ao = base["ao_b"][:, 13:19].astype(np.float32)
    DEL = {}
    for tag in tags:
        z = np.load(f"{R}/probe_{tag}.npz")
        assert np.array_equal(z["ao_idx"], base["ao_idx"]), tag
        DEL[tag] = z["ao_b"][:, 13:19].astype(np.float32) - b_ao

    def fsim(a, c):
        return float(
            np.mean(
                [
                    np.dot(a[:, j].ravel(), c[:, j].ravel())
                    / (np.linalg.norm(a[:, j]) * np.linalg.norm(c[:, j]) + 1e-12)
                    for j in range(a.shape[1])
                ]
            )
        )

    w6 = {}
    for t1, t2 in __import__("itertools").combinations(sorted(DEL), 2):
        p1, p2 = TAGRE.search(t1), TAGRE.search(t2)
        # same step, DIFFERENT seeds only
        if not p1 or not p2 or p1.group(3) != p2.group(3) or p1.group(2) == p2.group(2):
            continue
        key = f"step{p1.group(3)}|" + "-".join(sorted([p1.group(1), p2.group(1)]))
        w6.setdefault(key, []).append(fsim(DEL[t1], DEL[t2]))
    w6s = {
        k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v)]
        for k, v in w6.items()
    }
    g = lambda a, c: w6s.get(f"step474|{'-'.join(sorted([a, c]))}", [float("nan")])[0]
    # pre-registered: ACT-AttCT > every cross pair with BCT/RMCT, and ACT-AttCT inside the range spanned by
    # the individual within-method different-seed similarities of ACT and of AttCT (3 + 3 pair values)
    x = g("act", "attct")
    cross = max(g("act", "bct"), g("act", "rmct"), g("attct", "bct"), g("attct", "rmct"))
    within = w6.get("step474|act-act", []) + w6.get("step474|attct-attct", [])
    lo, hi = (min(within), max(within)) if within else (float("nan"), float("nan"))
    need = {
        "act-attct": 6,
        "act-act": 3,
        "attct-attct": 3,
        "act-bct": 6,
        "act-rmct": 6,
        "attct-bct": 6,
        "attct-rmct": 6,
    }
    complete = all(len(w6.get(f"step474|{k}", [])) == n for k, n in need.items())
    out["W6_function_space"] = {
        "pairs": w6s,
        "act_attct": x,
        "max_cross": cross,
        "within_range": [lo, hi],
        "complete": complete,
        "part1_above_cross": bool(x > cross),
        "part2_in_within_range": bool(lo <= x <= hi),
        "verdict": bool(x > cross and lo <= x <= hi) if complete else None,
    }
    print(
        "W6 function space (step474):",
        {k: round(v[0], 3) for k, v in w6s.items() if k.startswith("step474")},
        "verdict",
        out["W6_function_space"]["verdict"],
        flush=True,
    )
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
