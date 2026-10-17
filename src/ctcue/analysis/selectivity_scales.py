"""Test the two main cue-battery findings on other scales and against cue strength (exploratory).

  1. The ratio of the helpful to the harmful drop as absolute, proportional and log-odds drops,
     pooled, before or after the options and per position, with question-bootstrap intervals
     (2,000 resamples). A same-question version compares a cue that names the correct answer with
     one that names a wrong answer, on items that the original and the trained model both answer
     wrongly without a cue, with a cued wrong letter that differs from both of their answers.
  2. Position against cue strength: over the 24 combinations of wording and position, a regression
     of the share removed on the original model's following and an after-the-options indicator,
     and a comparison at matched strength. Also the share removed at each position when any answer
     other than the correct one counts as following.

Usage: python -m ctcue.analysis.selectivity_scales
Output: $CT_ROOT/results/w7u/ana_scale_u.json
"""

import json, os
import numpy as np

from ctcue.common import checkpoints
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
SEEDS = (42, 43, 44)
B = 2000
GROUPS = {
    "all": ("front", "pre", "post", "end"),
    "before": ("front", "pre"),
    "after": ("post", "end"),
    "front": ("front",),
    "pre": ("pre",),
    "post": ("post",),
    "end": ("end",),
}
ARMS = {
    "gemma2b": (
        "bct",
        "act",
        "attct",
        "rmct",
        "rmcta",
        "fix_bayes_r0.5",
        "fix_bayes_r0.8",
        "fix_goldgated",
        "fix_posdiv",
        "bctro",
        "selftrain",
    ),
    "llama8b": ("bct", "act", "attct", "rmct", "fix_bayes6_r0.8", "fix_goldgated", "fix_posdiv", "selftrain"),
}


def load(path):
    d = json.load(open(path))
    return d, {(it["ds"], it["hash"]): it for it in d["items"]}


def per_item(d, key, items, pos):
    """[N, n_pos] fraction of wordings giving T under the right hint (h) and X under the wrong hint (c), per position."""
    W = [w[0] for w in d["wordings"]]
    its = [key[k] for k in items]
    h = np.array(
        [[np.mean([it["cond"][f"{w}|{p}|right"][2] == it["T"] for w in W]) for p in pos] for it in its]
    )
    c = np.array(
        [[np.mean([it["cond"][f"{w}|{p}|wrong"][2] == it["X"] for w in W]) for p in pos] for it in its]
    )
    return h, c, len(W)


def wrong_hint_T_other(d, key, items, pos):
    """[N, n_pos] fraction of wordings answering T, and answering neither T nor X, under the WRONG hint."""
    W = [w[0] for w in d["wordings"]]
    its = [key[k] for k in items]
    t = np.array(
        [[np.mean([it["cond"][f"{w}|{p}|wrong"][2] == it["T"] for w in W]) for p in pos] for it in its]
    )
    o = np.array(
        [
            [np.mean([it["cond"][f"{w}|{p}|wrong"][2] not in (it["T"], it["X"]) for w in W]) for p in pos]
            for it in its
        ]
    )
    return t, o


def logit(p, n):
    e = 1 / (2 * n)
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def drops(base, ck, n):
    return {"absolute": base - ck, "proportional": 1 - ck / base, "log-odds": logit(base, n) - logit(ck, n)}


def main():
    out = {}
    for model, arms in ARMS.items():
        bd, bkey = load(f"{R}/w7/gen_{model}_none.json")
        items = sorted(bkey)
        POS = bd["positions"]
        ds = np.array([k[0] for k in items])
        rng = np.random.default_rng(0)
        idx = [np.where(ds == g)[0] for g in sorted(set(ds))]
        boot = np.zeros((B, len(items)))
        for b in range(B):
            for ix in idx:
                np.add.at(boot[b], rng.choice(ix, len(ix)), 1)
        hb_all, cb_all, nw = per_item(bd, bkey, items, POS)
        right0 = np.array([bkey[k]["pred0"] == bkey[k]["T"] for k in items])
        tw0, ow0 = wrong_hint_T_other(bd, bkey, items, POS)
        x0 = np.array([bkey[k]["pred0"] == bkey[k]["X"] for k in items])
        cells0 = {
            (w[0], p): np.array([bkey[k]["cond"][f"{w[0]}|{p}|wrong"][2] == bkey[k]["X"] for k in items])
            for w in bd["wordings"]
            for p in POS
        }
        mo = {}
        for arm in arms:
            files = [f"{R}/w7u/gen_{checkpoints.inst_tag(model, arm, s)}.json" for s in SEEDS]
            if not all(os.path.exists(f) for f in files):
                print(f"{model} {arm}: missing gen files, skipped", flush=True)
                continue
            S = []
            for f in files:
                d, key = load(f)
                assert sorted(key) == items, f
                assert all(key[k]["T"] == bkey[k]["T"] and key[k]["X"] == bkey[k]["X"] for k in items), f
                hc, cc, _ = per_item(d, key, items, POS)
                right1 = np.array([key[k]["pred0"] == key[k]["T"] for k in items])
                cells1 = {
                    (w[0], p): np.array(
                        [key[k]["cond"][f"{w[0]}|{p}|wrong"][2] == key[k]["X"] for k in items]
                    )
                    for w in d["wordings"]
                    for p in POS
                }
                tw1, ow1 = wrong_hint_T_other(d, key, items, POS)
                x1 = np.array([key[k]["pred0"] == key[k]["X"] for k in items])
                S.append(
                    dict(
                        I=(~right0) & (~right1),
                        J=right0 & right1,
                        K=(~right0) & (~right1) & (~x0) & (~x1),
                        hc=hc,
                        cc=cc,
                        right1=right1,
                        cells1=cells1,
                        tw=tw1,
                        ow=ow1,
                    )
                )
            res = {}
            for g, gp in GROUPS.items():
                gi = [POS.index(p) for p in gp]
                n_cnt = nw * len(gp)

                def scale_ratios(wt):  # wt [N] item weights (1 = observed sample) or a bootstrap row
                    hd = {k: [] for k in ("absolute", "proportional", "log-odds")}
                    cd = {k: [] for k in hd}
                    for s in S:
                        wi, wj = wt * s["I"], wt * s["J"]
                        hb = (wi @ hb_all[:, gi].mean(1)) / wi.sum()
                        hc = (wi @ s["hc"][:, gi].mean(1)) / wi.sum()
                        cb = (wj @ cb_all[:, gi].mean(1)) / wj.sum()
                        cc = (wj @ s["cc"][:, gi].mean(1)) / wj.sum()
                        for k, v in drops(hb, hc, n_cnt * wi.sum()).items():
                            hd[k].append(v)
                        for k, v in drops(cb, cc, n_cnt * wj.sum()).items():
                            cd[k].append(v)
                    return {k: (float(np.mean(hd[k])), float(np.mean(cd[k]))) for k in hd}

                obs = scale_ratios(np.ones(len(items)))
                bt = [scale_ratios(boot[b]) for b in range(B)]
                res[g] = {}
                for k, (h, c) in obs.items():
                    rb = np.array([x[k][0] / x[k][1] if x[k][1] > 0.02 else np.nan for x in bt])
                    res[g][k] = {
                        "helpful_drop": h,
                        "harmful_drop": c,
                        "ratio": h / c if c > 0.02 else None,
                        "ratio_ci95": (
                            [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))]
                            if c > 0.02
                            else None
                        ),
                        "frac_boot_harmful_gt_0.02": float(np.mean(~np.isnan(rb))),
                    }
            # SAME-ITEM content test: items both models answer wrong uncued, X different from both uncued
            # answers; uptake of a hint naming T (correct) vs uptake of a hint naming X (wrong), base and
            # checkpoint; seed means; bootstrap over questions
            res["same_item"] = {}
            for g, gp in GROUPS.items():
                gi = [POS.index(p) for p in gp]
                n_cnt = nw * len(gp)

                def same(wt):
                    v = {k: [] for k in ("Tb", "Tc", "Xb", "Xc", "dT", "dX", "gap_b", "gap_c")}
                    for s_ in S:
                        wk = wt * s_["K"]
                        n = n_cnt * wk.sum()
                        Tb = (wk @ hb_all[:, gi].mean(1)) / wk.sum()
                        Tc = (wk @ s_["hc"][:, gi].mean(1)) / wk.sum()
                        Xb = (wk @ cb_all[:, gi].mean(1)) / wk.sum()
                        Xc = (wk @ s_["cc"][:, gi].mean(1)) / wk.sum()
                        for k_, x_ in (
                            ("Tb", Tb),
                            ("Tc", Tc),
                            ("Xb", Xb),
                            ("Xc", Xc),
                            ("dT", Tb - Tc),
                            ("dX", Xb - Xc),
                            ("gap_b", logit(Tb, n) - logit(Xb, n)),
                            ("gap_c", logit(Tc, n) - logit(Xc, n)),
                        ):
                            v[k_].append(x_)
                    return {k_: float(np.mean(x_)) for k_, x_ in v.items()}

                ob = same(np.ones(len(items)))
                bt = [same(boot[b]) for b in range(B)]
                ob["dT_minus_dX_ci95"] = [
                    float(np.percentile([x["dT"] - x["dX"] for x in bt], q)) for q in (2.5, 97.5)
                ]
                ob["gap_change_ci95"] = [
                    float(np.percentile([x["gap_c"] - x["gap_b"] for x in bt], q)) for q in (2.5, 97.5)
                ]
                ob["n_items"] = [int(s_["K"].sum()) for s_ in S]
                res["same_item"][g] = ob
            # position measured on a P(T) basis (a switch to a third letter does not count as resisting): own
            # known items, wrong hint
            res["pt_basis"] = {}
            for p in POS:
                i_ = POS.index(p)
                nt0 = 1 - tw0[right0, i_].mean()
                oth0 = ow0[right0, i_].mean()
                nt1 = float(np.mean([1 - s_["tw"][s_["right1"], i_].mean() for s_ in S]))
                oth1 = float(np.mean([s_["ow"][s_["right1"], i_].mean() for s_ in S]))
                res["pt_basis"][p] = {
                    "relred_not_T": float(1 - nt1 / nt0),
                    "third_letter_untrained": float(oth0),
                    "third_letter_arm": oth1,
                }
            # position versus strength (cells on each model's own known items)
            strength = np.array([cells0[c][right0].mean() for c in sorted(cells0)])
            cav = np.mean([[s["cells1"][c][s["right1"]].mean() for c in sorted(cells0)] for s in S], 0)
            rr = 1 - cav / strength
            after = np.array([c[1] in ("post", "end") for c in sorted(cells0)], float)
            X = np.column_stack([np.ones(len(rr)), strength, after])
            coef, *_ = np.linalg.lstsq(X, rr, rcond=None)
            X1 = np.column_stack([np.ones(len(rr)), after])
            coef1, *_ = np.linalg.lstsq(X1, rr, rcond=None)
            lo, hi = max(strength[after == 0].min(), strength[after == 1].min()), min(
                strength[after == 0].max(), strength[after == 1].max()
            )
            ov = (strength >= lo) & (strength <= hi)
            matched = {
                "range": [float(lo), float(hi)],
                "n_before": int((ov & (after == 0)).sum()),
                "n_after": int((ov & (after == 1)).sum()),
                "relred_before": float(rr[ov & (after == 0)].mean()) if (ov & (after == 0)).any() else None,
                "relred_after": float(rr[ov & (after == 1)].mean()) if (ov & (after == 1)).any() else None,
            }
            e_ = 1 / (2 * 6 * max(int(right0.sum()), 1))
            lgt = lambda q: np.log(np.clip(q, e_, 1 - e_) / (1 - np.clip(q, e_, 1 - e_)))
            shift = lgt(strength) - lgt(cav)
            posd = np.array([[c[1] == p for p in POS] for c in sorted(cells0)], float)
            cs_, *_ = np.linalg.lstsq(np.column_stack([strength, posd]), shift, rcond=None)
            cr_, *_ = np.linalg.lstsq(np.column_stack([strength, posd]), rr, rcond=None)
            res["stronger_hints"] = {
                "logodds_shift_slope_pos_fe": float(cs_[0]),
                "relred_slope_pos_fe": float(cr_[0]),
            }
            res["position_vs_strength"] = {
                "matched_strength": matched,
                "after_coef_controlling_strength": float(coef[2]),
                "strength_coef": float(coef[1]),
                "after_coef_alone": float(coef1[1]),
                "strength_range_before": [
                    float(strength[after == 0].min()),
                    float(strength[after == 0].max()),
                ],
                "strength_range_after": [
                    float(strength[after == 1].min()),
                    float(strength[after == 1].max()),
                ],
            }
            mo[arm] = res
            a_, p_, l_ = (res["all"][k]["ratio"] for k in ("absolute", "proportional", "log-odds"))
            f_ = lambda x: "  --" if x is None else f"{x:.2f}"
            print(
                f"{model} {arm:15s} ratio abs {f_(a_)} prop {f_(p_)} logodds {f_(l_)} | before: prop {f_(res['before']['proportional']['ratio'])} | "
                f"position effect at equal strength {coef[2]:+.2f} (alone {coef1[1]:+.2f}, strength coef {coef[1]:+.2f}) | matched [{lo:.2f},{hi:.2f}] "
                f"before {matched['relred_before'] if matched['relred_before'] is None else round(matched['relred_before'], 2)} (n {matched['n_before']}) after {matched['relred_after'] if matched['relred_after'] is None else round(matched['relred_after'], 2)} (n {matched['n_after']})",
                flush=True,
            )
        out[model] = mo
    json.dump(out, open(f"{R}/w7u/ana_scale_u.json", "w"), indent=1)
    print(f"-> {R}/w7u/ana_scale_u.json")


if __name__ == "__main__":
    main()
