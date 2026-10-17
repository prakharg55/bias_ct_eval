"""Additional tables: method contrasts, bias rates, relapse and layer-swap intervals, and verdict margins.

Usage: python -m ctcue.paper.tables_extra --unmerged --B 10000 --seed 0 --tables <dir>/paper_tables.json
           --out_dir <dir>
Output: <dir>/paper_tables_extra.json and paper_tables_extra.md
"""

import sys

sys.dont_write_bytecode = True
import argparse, json, os, time
import numpy as np

from ctcue.paper.tables import strat_weights, t1_stats, ci95, CUES, TRAINED, HELD, SEEDS, PREREG, NAME
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
J = lambda f: json.load(open(f))
MODELS = {
    "gemma": {"prefix": "gemma2b", "arms": PREREG + ["rmcta"]},
    "llama": {"prefix": "llama8b", "arms": PREREG},
}
f3 = lambda x: "n/a" if x is None or x != x else f"{x:.3f}"
br = lambda c: "n/a" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"


def tab(head, rows):
    return (
        ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
        + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
        + [""]
    )


T975_DF2 = 4.302652729911275  # t quantile .975, 2 df
UNM = {"on": False}  # --unmerged


def etag(pre, m, s):
    if not UNM["on"]:
        return f"{pre}_final_none" if m is None else f"{pre}_final_{m}_s{s}"
    from ctcue.common import checkpoints

    return checkpoints.eval_tag(pre, m, s)


def relf(pre, t_or_ms):
    """relapse file: merged results/relapse7/relapse_<pre>_<tag>.json; --unmerged: relapse7u per checkpoints (untrained: relapse7, no adapter)."""
    if not UNM["on"] or t_or_ms == "none":
        return (
            f"{R}/relapse7/relapse_{pre}_{t_or_ms if isinstance(t_or_ms, str) else '%s_s%d' % t_or_ms}.json"
        )
    from ctcue.common import checkpoints

    m, s = t_or_ms
    return f"{R}/relapse7u/relapse_{checkpoints.relapse_tag('gemma2b' if pre == 'gemma' else 'llama8b', m, s)}.json"


def load_items(root, key):
    pre, arms = MODELS[key]["prefix"], MODELS[key]["arms"]
    cols = [(None, None)] + [(m, s) for m in arms for s in SEEDS]
    its = {c: J(f"{R}/{root}/evalitems_{etag(pre, *c)}.json")["items"] for c in cols}
    base = its[(None, None)]
    ident = [(i["cue"], i["ds"], i["hash"], i["X"], i["T"]) for i in base]
    for c, v in its.items():
        assert [(i["cue"], i["ds"], i["hash"], i["X"], i["T"]) for i in v] == ident, (key, c)
    N, M = len(base), len(cols)
    K = np.zeros((N, M))
    F = np.zeros((N, M))
    XB = np.zeros((N, M))
    XU = np.zeros((N, M))
    for j, c in enumerate(cols):
        v = its[c]
        K[:, j] = [i["gen_u"] == i["T"] for i in v]
        F[:, j] = K[:, j] * np.array([i["gen_b"] == i["X"] for i in v])
        XB[:, j] = [i["gen_b"] == i["X"] for i in v]
        XU[:, j] = [i["gen_u"] == i["X"] for i in v]
    cue_idx = {c: np.array([n for n, i in enumerate(base) if i["cue"] == c]) for c in CUES}
    qk = sorted({(i["ds"], i["hash"]) for i in base})
    qid = {k: n for n, k in enumerate(qk)}
    qstrata = [np.array([n for n, k in enumerate(qk) if k[0] == ds]) for ds in sorted({k[0] for k in qk})]
    item_q = np.array([qid[(i["ds"], i["hash"])] for i in base])
    return {
        "cols": cols,
        "arms": arms,
        "K": K,
        "F": F,
        "XB": XB,
        "XU": XU,
        "cue_idx": cue_idx,
        "qstrata": qstrata,
        "nq": len(qk),
        "item_q": item_q,
        "N": N,
    }


def brr_stats(W, XB, XU, cue_idx):
    out = {}
    for c in CUES:
        ix = cue_idx[c]
        w = W[:, ix]
        out[f"cue:{c}"] = (w @ (XB[ix] - XU[ix])) / w.sum(1, keepdims=True)
    out["trained"] = out[f"cue:{TRAINED}"]
    out["held"] = np.mean(np.stack([out[f"cue:{c}"] for c in HELD]), axis=0)
    return out


def x1_x2(root, key, B, seed):
    D = load_items(root, key)
    cols = D["cols"]
    arms = D["arms"]
    N = D["N"]
    W = strat_weights(
        np.random.default_rng([seed, 11, 0 if key == "gemma" else 1]), D["qstrata"], D["nq"], B
    )[:, D["item_q"]]
    one = np.ones((1, N))
    P, BT = t1_stats(one, D["K"], D["F"], D["cue_idx"]), t1_stats(W, D["K"], D["F"], D["cue_idx"])
    PB, BB = brr_stats(one, D["XB"], D["XU"], D["cue_idx"]), brr_stats(W, D["XB"], D["XU"], D["cue_idx"])
    js = {m: [1 + 3 * a + i for i in range(3)] for a, m in enumerate(arms)}
    X1 = {}
    for st in ("trained", "held", "acc"):
        for a in range(len(arms)):
            for b in range(a + 1, len(arms)):
                ma, mb = arms[a], arms[b]
                va, vb = P[st][0, js[ma]], P[st][0, js[mb]]
                d = float(va.mean() - vb.mean())
                bd = BT[st][:, js[ma]].mean(1) - BT[st][:, js[mb]].mean(1)
                se_i = float(np.std(bd, ddof=1))
                se_s = float(np.sqrt(va.var(ddof=1) / 3 + vb.var(ddof=1) / 3))
                se_t = float(np.hypot(se_i, se_s))
                X1[f"{st}|{ma}-{mb}"] = {
                    "diff": d,
                    "se_item": se_i,
                    "ci95_item": ci95(bd),
                    "se_seed": se_s,
                    "se_total": se_t,
                    "ci95_total": [d - 1.96 * se_t, d + 1.96 * se_t],
                    "excludes_0": bool(abs(d) > 1.96 * se_t),
                }
    X2 = {"untrained": {st: float(PB[st][0, 0]) for st in ("trained", "held")}, "arms": {}}
    for m in arms:
        o = {}
        for st in ("trained", "held"):
            v = PB[st][0, js[m]]
            bm = BB[st][:, js[m]].mean(1)
            o[st] = {
                "seeds": v.tolist(),
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)),
                "diff": float(v.mean() - PB[st][0, 0]),
                "diff_ci95_qcluster": ci95(bm - BB[st][:, 0]),
            }
            fk = P[st][0, js[m]]
            o[st]["flip_given_knew_mean"] = float(fk.mean())
            o[st]["flip_given_knew_diff"] = float(fk.mean() - P[st][0, 0])
            o[st]["flip_given_knew_diff_ci95_qcluster"] = ci95(BT[st][:, js[m]].mean(1) - BT[st][:, 0])
        X2["arms"][m] = o
    X2["untrained_flip_given_knew"] = {st: float(P[st][0, 0]) for st in ("trained", "held", "acc")}
    T1boot = {"P": {st: P[st][0].tolist() for st in ("trained", "held", "acc")}, "boot": BT, "js": js}
    return X1, X2, T1boot


# -------------------------------------------------------------------------------------------------------------------
# relapse
def rel_rates(blk, rng=None, R_=None):
    """blk = {"flip_rates": {cue: {knew, flips}}}; returns arrays [R_] (or scalars) of trained rate and held-out mean."""

    def rate(c):
        k, f = blk["flip_rates"][c]["knew"], blk["flip_rates"][c]["flips"]
        if rng is None:
            return f / max(k, 1)
        return rng.binomial(k, f / k, size=R_) / k if k > 0 else np.zeros(R_)

    return rate(CUES[0]), np.mean([rate(c) for c in CUES[1:]], axis=0)


def x3(B, seed):
    out = {}
    for key, pre, arms in (("gemma", "gemma", PREREG + ["rmcta"]), ("llama", "llama8b", PREREG)):
        rng = np.random.default_rng([seed, 13, 0 if key == "gemma" else 1])
        rd = lambda t: J(relf(pre, t))["checkpoints"]
        u = rd("none")
        gen = lambda ck, st: ck[st]["gen"]
        U0, U125 = rel_rates(gen(u, "0")), rel_rates(gen(u, "125"))
        bU0, bU125 = rel_rates(gen(u, "0"), rng, B), rel_rates(gen(u, "125"), rng, B)
        res = {
            "untrained": {
                "step0": {"trained": U0[0], "held": float(U0[1])},
                "step125": {"trained": U125[0], "held": float(U125[1])},
            },
            "arms": {},
        }
        for m in arms:
            per, bper = [], []
            for s in SEEDS:
                f = relf(pre, (m, s))
                if not os.path.exists(f):
                    continue
                ck = rd((m, s))
                A0, A125 = rel_rates(gen(ck, "0")), rel_rates(gen(ck, "125"))
                bA0, bA125 = rel_rates(gen(ck, "0"), rng, B), rel_rates(gen(ck, "125"), rng, B)
                rec, brec = {}, {}
                for i, k in enumerate(("trained", "held")):
                    gap = U0[i] - A0[i]
                    rec[f"frac_{k}"] = (A125[i] - A0[i]) / gap
                    rec[f"adj_{k}"] = (A125[i] - A0[i] - (U125[i] - U0[i])) / gap
                    rec[f"gap_{k}"] = gap
                    rec[f"abs_change_{k}"] = A125[i] - A0[i]
                    rec[f"abs_change_beyond_drift_{k}"] = A125[i] - A0[i] - (U125[i] - U0[i])
                    bg = bU0[i] - bA0[i]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        brec[f"frac_{k}"] = (bA125[i] - bA0[i]) / bg
                        brec[f"adj_{k}"] = (bA125[i] - bA0[i] - (bU125[i] - bU0[i])) / bg
                    brec[f"gap_{k}"] = bg
                    brec[f"abs_change_beyond_drift_{k}"] = bA125[i] - bA0[i] - (bU125[i] - bU0[i])
                per.append(rec)
                bper.append(brec)
            if not per:
                continue
            agg = {"n_seeds": len(per)}
            for q in (
                "frac_trained",
                "adj_trained",
                "frac_held",
                "adj_held",
                "abs_change_beyond_drift_trained",
                "abs_change_beyond_drift_held",
            ):
                v = np.array([p[q] for p in per])
                bm = np.mean([b[q] for b in bper], axis=0)
                mean, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                agg[q] = {
                    "seeds": v.tolist(),
                    "mean": mean,
                    "sd": sd,
                    "ci95_item_binomial": ci95(bm),
                    "ci95_seed_t": (
                        [mean - T975_DF2 * sd / np.sqrt(len(v)), mean + T975_DF2 * sd / np.sqrt(len(v))]
                        if len(v) == 3
                        else None
                    ),
                }
            for k in ("trained", "held"):
                g = np.array([b[f"gap_{k}"] for b in bper])
                agg[f"share_replicates_gap_lt_.05_{k}"] = float((g < 0.05).any(0).mean())
                agg[f"gap_{k}_mean"] = float(np.mean([p[f"gap_{k}"] for p in per]))
            res["arms"][m] = agg
        out[key] = res
    return out


# ---------------------------------------------------------------------------------------------------------------------
# route
def x4(B, seed):
    if UNM["on"]:
        from ctcue.common import checkpoints

        S = J(f"{R}/w7u/score_route_u.json")["llama"]
        rfile = lambda m: f"{R}/w7u/route_{checkpoints.inst_tag('llama8b', m, 42)}.json"
    else:
        S = J(f"{R}/w7/score_route_llama7.json")
        rfile = lambda m: f"{R}/w7/route_llama8b_{m}_s42.json"
    out = {}
    for m in ("bct", "act", "attct"):
        d = J(rfile(m))
        its = d["items"]
        n = len(its)
        X = np.array([i["X"] for i in its])
        M1 = np.array([i["pred1b"] for i in its]) == X
        scopes = ("EARLY", "CARRIER", "CIRCUIT", "LATE")
        cf = {
            sc: np.array([i["arms"][f"W_all@{sc}"]["fwd"][2] for i in its]) == X for sc in scopes + ("ALL",)
        }
        fixed = 1 - M1.mean()
        med = {sc: float((1 - cf[sc].mean()) / fixed) for sc in cf}
        for sc in med:
            assert abs(med[sc] - S["arms"][m]["fracs"][sc]) < 1e-12, (
                m,
                sc,
                med[sc],
                S["arms"][m]["fracs"][sc],
            )
        qk = sorted({(i["ds"], i["hash"]) for i in its})
        qid = {k: q for q, k in enumerate(qk)}
        iq = np.array([qid[(i["ds"], i["hash"])] for i in its])
        qstrata = [np.array([q for q, k in enumerate(qk) if k[0] == ds]) for ds in sorted({k[0] for k in qk})]
        W = strat_weights(
            np.random.default_rng([seed, 14, ("bct", "act", "attct").index(m)]), qstrata, len(qk), B
        )[:, iq]
        bfix = 1 - (W @ M1) / W.sum(1)
        bmed = {sc: (1 - (W @ cf[sc]) / W.sum(1)) / bfix for sc in scopes}
        largest = np.argmax(np.stack([bmed[sc] for sc in scopes]), axis=0)
        cond = {sc: float(((~cf[sc]) & (~M1)).sum() / (~M1).sum()) for sc in scopes}
        out[m] = {
            "n": n,
            "n_fixed_by_M1": int((~M1).sum()),
            "med_fwd": med,
            "ci95_qcluster": {sc: ci95(bmed[sc]) for sc in scopes},
            "p_largest": {sc: float((largest == k).mean()) for k, sc in enumerate(scopes)},
            "conditional_share": cond,
            "late_ge_.15_share": float((bmed["LATE"] >= 0.15).mean()),
            "scope_layers": {"EARLY": "0-8", "CARRIER": "9-14", "CIRCUIT": "15-18", "LATE": "19-31"},
        }
    return out


# ------------------------------------------------------------------------------------------------------------------
# verdicts
def x5(T1b, X3, X4, tables):
    rows = []

    def add(v, quantity, threshold, value, margin, interval, prob, note=""):
        unres = (
            interval is not None
            and isinstance(threshold, (int, float))
            and interval[0] <= threshold <= interval[1]
        )
        rows.append(
            {
                "verdict": v,
                "quantity": quantity,
                "threshold": threshold,
                "value": value,
                "margin": margin,
                "interval95": interval,
                "p_criterion": prob,
                "resolved": (not unres) if interval is not None else None,
                "note": note,
            }
        )

    for key in ("gemma", "llama"):
        tb = T1b[key]
        P, BT, js = tb["P"], tb["boot"], tb["js"]
        U = P["trained"][0]
        if key == "gemma":  # P1: arm trained <= .5 x untrained; P2: BCT lowest held-out
            for m in PREREG:
                v = float(np.mean([P["trained"][j] for j in js[m]]))
                b = BT["trained"][:, js[m]].mean(1) - 0.5 * BT["trained"][:, 0]
                add(
                    f"P1 ({NAME[m]})",
                    "trained flip|knew - .5 x untrained (<= 0 holds)",
                    0.0,
                    v - 0.5 * U,
                    v - 0.5 * U,
                    ci95(b),
                    float((b <= 0).mean()),
                )
            hb = {m: BT["held"][:, js[m]].mean(1) for m in PREREG}
            others = np.min(np.stack([hb[m] for m in PREREG if m != "bct"]), axis=0)
            hv = {m: float(np.mean([P["held"][j] for j in js[m]])) for m in PREREG}
            d = hv["bct"] - min(hv[m] for m in PREREG if m != "bct")
            add(
                "P2",
                "BCT held-out - lowest other arm (< 0 holds)",
                0.0,
                d,
                d,
                ci95(hb["bct"] - others),
                float((hb["bct"] - others < 0).mean()),
            )
        else:  # L1: every arm lowers trained and held-out vs untrained
            for m in PREREG:
                for st in ("trained", "held"):
                    v = float(np.mean([P[st][j] for j in js[m]])) - P[st][0]
                    b = BT[st][:, js[m]].mean(1) - BT[st][:, 0]
                    add(
                        f"L1 ({NAME[m]}, {st})",
                        f"{st} arm - untrained (< 0 holds)",
                        0.0,
                        v,
                        v,
                        ci95(b),
                        float((b < 0).mean()),
                    )
    for key, name, arms, thr in (("gemma", "R1", PREREG, 0.25), ("llama", "LR1", ["act", "attct"], 0.25)):
        for m in arms:
            a = X3[key]["arms"][m]["frac_trained"]
            add(
                f"{name} ({NAME[m]})",
                "trained-cue relapse fraction, generated (>= .25 holds)",
                thr,
                a["mean"],
                a["mean"] - thr,
                a["ci95_item_binomial"],
                None,
                f"seed t-interval {br(a['ci95_seed_t'])}",
            )
    lb = X3["llama"]["arms"]
    for m in ("act", "attct"):
        d = lb["bct"]["frac_trained"]["mean"] - lb[m]["frac_trained"]["mean"]
        sd = np.sqrt(
            np.var(lb["bct"]["frac_trained"]["seeds"], ddof=1) / 3
            + np.var(lb[m]["frac_trained"]["seeds"], ddof=1) / 3
        )
        add(
            f"LR2 (BCT vs {NAME[m]})",
            "BCT fraction - arm fraction (< 0 holds)",
            0.0,
            d,
            d,
            [d - T975_DF2 * sd, d + T975_DF2 * sd],
            None,
            "seed-level interval (t, df 2, Welch-type SE)",
        )
    for m, r in X4.items():
        b = r["ci95_qcluster"]["LATE"]
        add(
            f"D1 ({NAME[m]}, LATE < .15)",
            "LATE med_fwd (< .15 holds)",
            0.15,
            r["med_fwd"]["LATE"],
            r["med_fwd"]["LATE"] - 0.15,
            b,
            1 - r["late_ge_.15_share"],
        )
    # S1 / S2 (prereg NOTE 19:11), seed-level Welch-type t intervals (df 2)
    if UNM["on"] and os.path.exists(f"{R}/w7u/ana_selftrain_u.json"):
        st = J(f"{R}/w7u/ana_selftrain_u.json")
        for mk in ("gemma2b", "llama8b"):
            o = st[mk]
            u = o["untrained"]["trained"]
            a, b = np.array(o["test"]["selftrain"]["trained"]), np.array(o["test"]["bct"]["trained"])
            d = (u - a.mean()) - 0.5 * (u - b.mean())
            se = np.sqrt(a.var(ddof=1) / 3 + 0.25 * b.var(ddof=1) / 3)
            add(
                f"S1 ({mk})",
                "control's cut - .5 x BCT's cut in trained-cue flip|knew (< 0 holds)",
                0.0,
                d,
                d,
                [d - T975_DF2 * se, d + T975_DF2 * se],
                None,
                "seed-level interval (t, df 2)",
            )
            a, b = np.array(o["gen_paired"]["selftrain"]["helpful_drop"]), np.array(
                o["gen_paired"]["bct"]["helpful_drop"]
            )
            d = a.mean() - 0.5 * b.mean()
            se = np.sqrt(a.var(ddof=1) / 3 + 0.25 * b.var(ddof=1) / 3)
            add(
                f"S2 ({mk})",
                "control's helpful drop - .5 x BCT's (< 0 holds)",
                0.0,
                d,
                d,
                [d - T975_DF2 * se, d + T975_DF2 * se],
                None,
                "seed-level interval (t, df 2)",
            )
    if tables is not None:  # L3 (front - post / front - end >= .30), from paper_tables.json T3 llama
        try:
            t3 = tables["llama"]["gen"]["arms"]
            for m in PREREG:
                for k in ("front_minus_post", "front_minus_end"):
                    q = t3[m][f"relred_{k}"]
                    add(
                        f"L3 ({NAME[m]}, {k})",
                        f"relative reduction {k.replace('_', ' ')} (>= .30 holds)",
                        0.30,
                        q["mean"],
                        q["mean"] - 0.30,
                        q.get("ci95"),
                        None,
                    )
        except Exception as e:
            rows.append({"verdict": "L3", "note": f"not read from tables: {e!r}"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="pub7")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--unmerged", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.unmerged:
        UNM["on"] = True
        a.root = "pub7uc" if a.root == "pub7" else a.root
        assert a.root == "pub7uc", a.root
    a.tables = a.tables or f"{R}/{'paper_u' if a.unmerged else 'paper'}/paper_tables.json"
    a.out_dir = a.out_dir or f"{R}/{'paper_u' if a.unmerged else 'paper8'}"
    res = {
        "meta": {
            "root": a.root,
            "B": a.B,
            "seed": a.seed,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unmerged": a.unmerged,
        }
    }
    T1b = {}
    for key in ("gemma", "llama"):
        X1, X2, T1b[key] = x1_x2(a.root, key, a.B, a.seed)
        res[f"X1_{key}"], res[f"X2_{key}"] = X1, X2
    res["X3"] = x3(a.B, a.seed)
    res["X4"] = x4(a.B, a.seed)
    tables = J(a.tables) if os.path.exists(a.tables) else None
    res["X5"] = x5(T1b, res["X3"], res["X4"], tables)
    L = [
        f"# Extra uncertainty / robustness tables (paper_tables_extra.py; eval root {a.root}; B = {a.B}, seed {a.seed}; {res['meta']['created']})",
        "",
    ]
    for key in ("gemma", "llama"):
        L += [
            f"## X1 ({key}). Method contrasts (difference of seed means; generated answers)",
            "",
            "SE_item = SD of the paired question-cluster bootstrap of the difference; SE_seed = sqrt(SD_a^2/3 + SD_b^2/3); 95% = diff +- 1.96 sqrt(SE_item^2 + SE_seed^2).",
            "",
        ]
        L += tab(
            [
                "metric",
                "contrast",
                "diff",
                "item-only 95% CI",
                "SE item",
                "SE seed",
                "95% (item + seed)",
                "excludes 0",
            ],
            [
                [
                    k.split("|")[0],
                    " - ".join(NAME[x] for x in k.split("|")[1].split("-")),
                    f"{v['diff']:+.3f}",
                    br(v["ci95_item"]),
                    f3(v["se_item"]),
                    f3(v["se_seed"]),
                    br(v["ci95_total"]),
                    "yes" if v["excludes_0"] else "no",
                ]
                for k, v in res[f"X1_{key}"].items()
            ],
        )
        X2 = res[f"X2_{key}"]
        L += [
            f"## X2 ({key}). Biased reasoning rate (Chua et al.) next to flip|knew",
            "",
            f"BRR = P(X | biased) - P(X | unbiased) over all test items (unconditional). Untrained BRR trained {f3(X2['untrained']['trained'])}, held-out {f3(X2['untrained']['held'])}; "
            f"flip|knew trained {f3(X2['untrained_flip_given_knew']['trained'])}, held-out {f3(X2['untrained_flip_given_knew']['held'])}.",
            "",
        ]
        L += tab(
            [
                "arm",
                "BRR trained (Δ, 95% CI)",
                "flip|knew trained Δ (95% CI)",
                "BRR held-out (Δ, 95% CI)",
                "flip|knew held-out Δ (95% CI)",
            ],
            [
                [
                    NAME[m],
                    f"{f3(o['trained']['mean'])} ({o['trained']['diff']:+.3f}, {br(o['trained']['diff_ci95_qcluster'])})",
                    f"{o['trained']['flip_given_knew_diff']:+.3f} {br(o['trained']['flip_given_knew_diff_ci95_qcluster'])}",
                    f"{f3(o['held']['mean'])} ({o['held']['diff']:+.3f}, {br(o['held']['diff_ci95_qcluster'])})",
                    f"{o['held']['flip_given_knew_diff']:+.3f} {br(o['held']['flip_given_knew_diff_ci95_qcluster'])}",
                ]
                for m, o in X2["arms"].items()
            ],
        )
    for key in ("gemma", "llama"):
        r = res["X3"][key]
        U = r["untrained"]
        L += [
            f"## X3 ({key}). Relapse with intervals (generated answers, steps 0 / 125)",
            "",
            f"Untrained control: trained {f3(U['step0']['trained'])} -> {f3(U['step125']['trained'])}, held-out {f3(U['step0']['held'])} -> {f3(U['step125']['held'])}. "
            "Item interval = parametric binomial bootstrap of the stored counts (conservative); seed interval = t (df 2). 'gap<.05' = share of replicates in which some seed's gap falls below .05.",
            "",
        ]
        L += tab(
            [
                "arm",
                "n",
                "fraction trained (item CI; seed CI)",
                "beyond drift trained (item CI; seed CI)",
                "abs. change beyond drift trained",
                "fraction held-out (item CI)",
                "beyond drift held-out (item CI)",
                "gap trained / held",
                "gap<.05 held",
            ],
            [
                [
                    NAME[m],
                    o["n_seeds"],
                    f"{f3(o['frac_trained']['mean'])} {br(o['frac_trained']['ci95_item_binomial'])}; {br(o['frac_trained']['ci95_seed_t'])}",
                    f"{o['adj_trained']['mean']:+.3f} {br(o['adj_trained']['ci95_item_binomial'])}; {br(o['adj_trained']['ci95_seed_t'])}",
                    f"{o['abs_change_beyond_drift_trained']['mean']:+.3f} {br(o['abs_change_beyond_drift_trained']['ci95_item_binomial'])}",
                    f"{f3(o['frac_held']['mean'])} {br(o['frac_held']['ci95_item_binomial'])}",
                    f"{o['adj_held']['mean']:+.3f} {br(o['adj_held']['ci95_item_binomial'])}",
                    f"{f3(o['gap_trained_mean'])} / {f3(o['gap_held_mean'])}",
                    f"{o['share_replicates_gap_lt_.05_held']:.2f}",
                ]
                for m, o in r["arms"].items()
            ],
        )
    L += ["## X4 (llama). Route with question-cluster bootstrap CIs (seed 42; med_fwd as T5)", ""]
    L += tab(
        [
            "arm",
            "|S| / fixed by M1",
            "EARLY",
            "CARRIER",
            "CIRCUIT",
            "LATE",
            "P(largest) E/Ca/Ci/L",
            "conditional share E/Ca/Ci/L",
            "P(LATE >= .15)",
        ],
        [
            [NAME[m], f"{r['n']} / {r['n_fixed_by_M1']}"]
            + [
                f"{r['med_fwd'][sc]:.3f} {br(r['ci95_qcluster'][sc])}"
                for sc in ("EARLY", "CARRIER", "CIRCUIT", "LATE")
            ]
            + [
                "/".join(f"{r['p_largest'][sc]:.2f}" for sc in ("EARLY", "CARRIER", "CIRCUIT", "LATE")),
                "/".join(
                    f"{r['conditional_share'][sc]:.2f}" for sc in ("EARLY", "CARRIER", "CIRCUIT", "LATE")
                ),
                f"{r['late_ge_.15_share']:.2f}",
            ]
            for m, r in res["X4"].items()
        ],
    )
    L += [
        "## X5. Pre-registered verdicts: deciding quantity, margin and uncertainty",
        "",
        "'unresolved' = the 95% interval contains the threshold (the verdict follows the point estimate, as pre-registered).",
        "",
    ]
    L += tab(
        [
            "verdict",
            "quantity",
            "threshold",
            "value",
            "margin",
            "95% interval",
            "P(criterion)",
            "status",
            "note",
        ],
        [
            [
                r["verdict"],
                r.get("quantity", ""),
                r.get("threshold", ""),
                f3(r.get("value")),
                f"{r['margin']:+.3f}" if isinstance(r.get("margin"), float) else "",
                br(r.get("interval95")),
                "" if r.get("p_criterion") is None else f"{r['p_criterion']:.2f}",
                "" if r.get("resolved") is None else ("resolved" if r["resolved"] else "UNRESOLVED"),
                r.get("note", ""),
            ]
            for r in res["X5"]
        ],
    )

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if not isinstance(v, np.ndarray) or v.ndim <= 1}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(
        clean({k: v for k, v in res.items()}), open(f"{a.out_dir}/paper_tables_extra.json", "w"), indent=1
    )
    open(f"{a.out_dir}/paper_tables_extra.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved -> {a.out_dir}/paper_tables_extra.{{json,md}} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
