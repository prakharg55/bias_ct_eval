#!/usr/bin/env python3
"""
ana_mech_u (9 Oct 2026): EXPLORATORY robustness checks of the mechanism, relapse and data statements, from existing unmerged outputs.
CPU, seconds. Output results/w7u/ana_mech_u.json.
 1. READING on fixed vs still-caved items (gemma; ct_probe7u files; untrained = w3/probe_pub_none): early readers' (L13.H6, L13.H2,
    L14.H2) cue-span attention mass, on the untrained model's known-and-caved items split by whether the checkpoint still caves (pred_b
    == X) or not, against the untrained model's own caved-vs-resisted gap. Trained cue and all cues.
 2. PATCH RECOVERY by head subset (circuit, readers, writers, early readers, best random set), all cues and the trained cue only, pooled
    over the causal3 files of each arm (gemma 3 seeds; Llama seed 42).
 3. ROUTE readouts per scope: the pre-registered binary med_fwd = (1 - cave_fwd) / (1 - cave_M1) on S; an own-fix version (the share of
    items M1 fixes that the splice also fixes); a probability version (pX0 - pX_fwd) / (pX0 - pX1); and the share of items of S that are
    trained-cue items.
 4. RELAPSE on the log-odds scale: change in logit(trained-cue flip|knew) from step 0 to 125 (generated answers), arm vs untrained.
 5. TRAINING DATA: letters named by the cue and correct letters in the shared training set; BCT targets on correct-cue rows.
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, "/data/code")
import utags
C = "/data/ct_root"; R = f"{C}/results"; SEEDS = (42, 43, 44); ARMS = ("bct", "act", "attct", "rmct")
EARLY = [(13, 6), (13, 2), (14, 2)]; SCOPES = ("EARLY", "CARRIER", "CIRCUIT", "LATE"); J = lambda p: json.load(open(p))


def reading():
    bj = J(f"{R}/w3/probe_pub_none.json"); bz = np.load(f"{R}/w3/probe_pub_none.npz"); bi = bj["items"]
    mass = lambda z: np.mean([z["att_b"][:, L, h, 0].astype(np.float32) for L, h in EARLY], 0)
    m0 = mass(bz); knew0 = np.array([it["knew"] for it in bi]); cave0 = knew0 & np.array([it["cave"] for it in bi])
    res0 = knew0 & np.array([it["pred_b"] == it["T"] for it in bi]); cue = np.array([it["cue"] for it in bi]); out = {}
    for arm in ARMS:
        per = []
        for s in SEEDS:
            t = utags.inst_tag("gemma2b", arm, s); aj = J(f"{R}/w7u/probe_{t}.json"); az = np.load(f"{R}/w7u/probe_{t}.npz")
            assert all((a["ds"], a["hash"], a["cue"]) == (b["ds"], b["hash"], b["cue"]) for a, b in zip(aj["items"], bi)), t
            m1 = mass(az); still = np.array([it["pred_b"] == it["X"] for it in aj["items"]])
            row = {}
            for scope, sel in (("trained", cue == "suggested_answer"), ("all", np.ones(len(bi), bool))):
                F, S_ = cave0 & ~still & sel, cave0 & still & sel; gap = m0[cave0 & sel].mean() - m0[res0 & sel].mean()
                row[scope] = {"n_fixed": int(F.sum()), "n_still": int(S_.sum()), "base_caved": float(m0[cave0 & sel].mean()), "base_resisted": float(m0[res0 & sel].mean()),
                              "rel_change_fixed": float(m1[F].mean() / m0[F].mean() - 1) if F.any() else None,
                              "rel_change_still": float(m1[S_].mean() / m0[S_].mean() - 1) if S_.any() else None,
                              "fixed_drop_over_gap": float((m0[F].mean() - m1[F].mean()) / gap) if F.any() and gap > 0 else None}
            per.append(row)
        out[arm] = {sc: {k: (float(np.mean([p[sc][k] for p in per if p[sc][k] is not None])) if any(p[sc][k] is not None for p in per) else None) for k in per[0][sc]} for sc in ("trained", "all")}
        out[arm]["per_seed"] = per
        print("reading", arm, {sc: {k: round(v, 3) for k, v in out[arm][sc].items() if v is not None} for sc in ("trained", "all")}, flush=True)
    return out


def patch():
    out = {}
    for model, base, arm_files in (("gemma2b", f"{R}/w3/causal3_pub_none.json", lambda a: [f"{R}/w7u/causal3_{utags.inst_tag('gemma2b', a, s)}.json" for s in SEEDS]),
                                   ("llama8b", f"{R}/w7/causal3_llama8b_none.json", lambda a: [f"{R}/w7u/causal3_{utags.inst_tag('llama8b', a, 42)}.json"])):
        def pool(files):
            P = [J(f)["patch"] for f in files]; r = {}
            for sub in [x for x in ("circuit", "readers", "writers", "early_readers") if all(x in p for p in P)]:
                n = sum(p[sub]["n_flips"] for p in P); k = sum(p[sub]["recovered"] for p in P)
                nt = sum(p[sub]["per_cue"].get("suggested_answer", {}).get("n", 0) for p in P); kt = sum(p[sub]["per_cue"].get("suggested_answer", {}).get("recovered", 0) for p in P)
                r[sub] = {"recovery": k / n, "n": n, "trained_cue_recovery": (kt / nt) if nt else None, "trained_cue_n": nt}
            r["best_random"] = max(max(p[f"random{i}"]["recovery"] for i in range(3) if f"random{i}" in p) for p in P)
            return r
        out[model] = {"untrained": pool([base])} | {a: pool(arm_files(a)) for a in ARMS if all(os.path.exists(f) for f in arm_files(a))}
        for k, v in out[model].items():
            print("patch", model, k, {s: (round(v[s]["recovery"], 3), v[s]["trained_cue_recovery"] and round(v[s]["trained_cue_recovery"], 3), v[s]["trained_cue_n"]) for s in ("circuit", "readers", "writers", "early_readers") if s in v}, "rand", round(v["best_random"], 3), flush=True)
    return out


def route():
    out = {}
    for model, arms, seeds in (("llama8b", ("bct", "act", "attct"), (42,)), ("gemma2b", ("bct", "act", "attct"), SEEDS)):
        out[model] = {}
        for a in arms:
            per = []
            for s in seeds:
                its = J(f"{R}/w7u/route_{utags.inst_tag(model, a, s)}.json")["items"]
                c1 = np.mean([it["pred1b"] == it["X"] for it in its]); fixedset = [it for it in its if it["pred1b"] != it["X"]]
                pX0 = np.mean([it["p0b"][0] for it in its]); pX1 = np.mean([it["p1b"][0] for it in its]); row = {"n": len(its), "n_fixed": len(fixedset),
                       "trained_cue_share": float(np.mean([it["cue"] == "suggested_answer" for it in its]))}
                for sc in SCOPES:
                    arm = f"W_all@{sc}"; cf = np.mean([it["arms"][arm]["fwd"][2] == it["X"] for it in its])
                    row[sc] = {"binary": float((1 - cf) / (1 - c1)), "own_fix": float(np.mean([it["arms"][arm]["fwd"][2] != it["X"] for it in fixedset])),
                               "probability": float((pX0 - np.mean([it["arms"][arm]["fwd"][0] for it in its])) / (pX0 - pX1))}
                per.append(row)
            out[model][a] = {sc: {k: float(np.mean([p[sc][k] for p in per])) for k in ("binary", "own_fix", "probability")} for sc in SCOPES}
            out[model][a]["n_fixed"] = [p["n_fixed"] for p in per]; out[model][a]["trained_cue_share"] = float(np.mean([p["trained_cue_share"] for p in per]))
            print("route", model, a, {sc: {k: round(v, 2) for k, v in out[model][a][sc].items()} for sc in SCOPES}, "fixed", out[model][a]["n_fixed"], "trained share", round(out[model][a]["trained_cue_share"], 2), flush=True)
    return out


def relapse():
    T = J(f"{R}/paper_u/paper_tables.json"); out = {}
    lg = lambda p: float(np.log(np.clip(p, 0.005, 0.995) / (1 - np.clip(p, 0.005, 0.995))))
    for m in ("gemma", "llama"):
        rl = T[m]["relapse_u"]; u = rl["untrained_gen"]; du = lg(u["125"]["trained"]) - lg(u["0"]["trained"]); out[m] = {"untrained_dlogit": du, "arms": {}}
        for a, o in rl["arms"].items():
            d = [lg(p["gen"]["125"]["trained"]) - lg(p["gen"]["0"]["trained"]) for p in o["per_seed"]]
            out[m]["arms"][a] = {"dlogit_mean": float(np.mean(d)), "dlogit_sd": float(np.std(d, ddof=1)) if len(d) > 1 else None, "seeds": d, "ratio_to_untrained": float(np.mean(d) / du)}
        print("relapse", m, "untrained", round(du, 2), {a: (round(v["dlogit_mean"], 2), round(v["ratio_to_untrained"], 2)) for a, v in out[m]["arms"].items()}, flush=True)
    return out


def data():
    rows = [json.loads(l) for l in open(f"{C}/shared/shared_train.jsonl")]
    X = [r["X"] for r in rows]; T_ = [r["T"] for r in rows]; out = {"n": len(rows), "cue_names_correct": int(sum(x == t for x, t in zip(X, T_))),
           "cue_letter_counts": {L: X.count(L) for L in sorted(set(X))}, "correct_letter_counts": {L: T_.count(L) for L in sorted(set(T_))}}
    for model in ("gemma2b", "llama8b"):
        bct = [json.loads(l) for l in open(f"{C}/shared/bct_{model}/bct_non_cot.jsonl")]
        by_user = {}
        for b in bct:
            u = next((m["content"] for m in b["messages"] if m["role"] == "user"), None); a = next((m["content"] for m in b["messages"] if m["role"] == "assistant"), "")
            if u is not None: by_user[u] = a
        hit = contra = 0
        for r in rows:
            if r["X"] != r["T"]: continue
            u = f"{r['cue']}\n{r['clean']}"
            if u not in by_user: continue
            m_ = re.search(r"\b([A-P])\b", by_user[u]); hit += 1; contra += bool(m_ and m_.group(1) != r["T"])
        out[f"bct_{model}_correct_cue_rows_matched"] = hit; out[f"bct_{model}_target_contradicts_correct_cue"] = contra
    print("data", out, flush=True)
    return out


def main():
    res = {"reading_gemma": reading(), "patch": patch(), "route": route(), "relapse_logodds": relapse(), "training_data": data()}
    json.dump(res, open(f"{R}/w7u/ana_mech_u.json", "w"), indent=1); print(f"-> {R}/w7u/ana_mech_u.json")


if __name__ == "__main__":
    main()
