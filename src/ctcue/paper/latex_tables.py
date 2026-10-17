#!/usr/bin/env python3
"""
paper_tables_tex (2 Oct 2026): the ARR draft's LaTeX tables, generated from paper_tables.json / paper_tables_extra.json / appendix_u.json so
that no table cell is typed by hand. Writes <out>/tables.tex defining one macro per table (main text: \\tabMain, \\tabContentPosition; appendix, 9 Oct lean set: \\tabVerdicts,
\\tabSelection, \\tabMerge, \\tabPerCue, \\tabLetterPrior, \\tabScale, \\tabContentByPos, \\tabSameItem, \\tabPosCheck (with wording families),
\\tabLRSweep, \\tabPatch, \\tabReading, \\tabRoute (both models, three readouts), \\tabRelapse (with log-odds), \\tabVariants, \\tabAuthorsEval, \\tabRmctPaper)
whose bodies are tabular environments; the manuscript places them in table / table* floats with its own captions.
Usage: paper_tables_tex.py --tables_dir <dir> [--extra_dir <dir>] --out <dir>
"""
import argparse, json, os
NAME = {"untrained": "original", "bct": "BCT", "act": "ACT", "attct": "AttCT", "rmct": "RMCT", "rmcta": "RMCT+anchor$^\\dagger$", "bayes_r0.5": "Bayes-BCT ($r{=}.5$)",
        "bayes_r0.8": "Bayes-BCT ($r{=}.8$)", "goldgated": "gold-gated BCT", "posdiv": "position-diverse BCT", "bctro": "response-only BCT", "selftrain": "self-training control"}
MAIN_EXCLUDE = ("rmcta",)   # (10 Oct) the exploratory RMCT+anchor arm appears in the appendix tables only
MODELS = (("gemma", "Gemma-2-2B-it"), ("llama", "Llama-3.1-8B-Instruct")); POS = ("front", "pre", "post", "end"); SCOPES = ("EARLY", "CARRIER", "CIRCUIT", "LATE")
T = {}


def g(o, *keys):
    for k in keys:
        if isinstance(o, dict) and k in o: o = o[k]
        elif isinstance(o, list) and isinstance(k, int) and -len(o) <= k < len(o): o = o[k]
        else: return None
    return o


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and x != x): return "--"
    if isinstance(x, float) and round(x, d) == 0: x = 0.0
    if isinstance(x, bool): return "yes" if x else "no"
    if isinstance(x, int): return str(x)
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _nz(x, d): return 0.0 if round(x, d) == 0 else x   # no negative zero
def ci(c, d=3): return "--" if not c or c[0] is None else f"[{_nz(c[0], d):.{d}f}, {_nz(c[1], d):.{d}f}]"
def pm(m, s, d=3): return f"{f(m, d)} $\\pm$ {f(s, d)}"
def esc(s): return s.replace("_", "\\_").replace("%", "\\%")


def tab(cols, rows, align=None, groups=None):
    """groups: optional [(first_col, last_col, label), ...] (1-based) printed as a spanning header row above cols."""
    align = align or ("l" + "r" * (len(cols) - 1))
    L = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    if groups:
        cells, c = [], 1
        for a_, b_, lab in groups:
            cells += [""] * (a_ - c) + [f"\\multicolumn{{{b_ - a_ + 1}}}{{c}}{{{lab}}}"]; c = b_ + 1
        cells += [""] * (len(cols) + 1 - c)
        L += [" & ".join(cells) + " \\\\", "".join(f"\\cmidrule(lr){{{a_}-{b_}}}" for a_, b_, _ in groups)]
    L += [" & ".join(cols) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MIDRULE": L.append("\\midrule")
        else: L.append(" & ".join(str(x) for x in r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; return "\n".join(L)


def define(name, body): T[name] = body


import re as _re
MODEL_SHORT = {"gemma2b": "Gemma", "llama8b": "Llama", "gemma": "Gemma", "llama": "Llama"}


def pretty(t):
    """checkpoint tag -> readable label, e.g. gemma2b_fix_bayes_r0.5_s44 -> Gemma, Bayes-BCT (r=.5), seed 44."""
    m = _re.match(r"(gemma2b|llama8b)_(.*)$", t)
    if not m: return esc(t)
    model, rest = m.groups(); seed = None
    ms = _re.match(r"(.*)_s(\d+)$", rest)
    if ms: rest, seed = ms.groups()
    rest = _re.sub(r"^(final7u|final)_", "", rest); rest = _re.sub(r"^fix_", "", rest); rest = rest.replace("bayes6_", "bayes_"); rest = _re.sub(r"7u$", "", rest)
    arm = "original" if rest == "none" else NAME.get(rest, esc(rest))
    return f"{MODEL_SHORT[model]}, {arm}" + (f", seed {seed}" if seed else "")


def vq(q):
    """verdict-table quantity text -> LaTeX."""
    for a, b in (("flip|knew", "flip$\\mid$knew"), (" - ", " $-$ "), (".5 x", "$0.5\\times$"), ("<=", "$\\le$"), (">=", "$\\ge$"), ("(< ", "($<$ "),
                 ("med_fwd", "mediated fraction"), ("untrained", "original"), ("arm ", "method "), ("front_minus_post", "front $-$ post"), ("front_minus_end", "front $-$ end"), ("gemma2b", "Gemma"), ("llama8b", "Llama")):
        q = q.replace(a, b)
    return q.replace("_", "\\_")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tables_dir", required=True); ap.add_argument("--extra_dir", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--extra_u", default="/data/ct_root/results/w7u/ana_extra_u.json"); a = ap.parse_args()
    P = json.load(open(f"{a.tables_dir}/paper_tables.json")); xd = a.extra_dir or a.tables_dir
    X = json.load(open(f"{xd}/paper_tables_extra.json")) if os.path.exists(f"{xd}/paper_tables_extra.json") else {}
    A = json.load(open(f"{xd}/appendix_u.json")) if os.path.exists(f"{xd}/appendix_u.json") else {}
    # ---- Table: main behavioural (both models) ----
    rows = []
    for key, mname in MODELS:
        t1 = P[key]["T1"]; u = t1["untrained"]
        rows.append([f"\\multicolumn{{6}}{{l}}{{\\textit{{{mname}}}}}"])
        rows.append(["original"] + [x for st in ("trained", "held") for x in (f"{f(u[st]['value'])} {ci(u[st].get('ci95_qcluster') or u[st]['ci95'], 2)}", "--")] + [f"{f(u['acc']['value'])} {ci(u['acc'].get('ci95_qcluster') or u['acc']['ci95'], 2)}"])
        for arm, o in t1["arms"].items():
            if arm in MAIN_EXCLUDE: continue
            rows.append([NAME[arm]] + [x for st in ("trained", "held") for x in (f"{pm(o[st]['mean'], o[st]['sd'])}", f"{f(o[st]['diff'], 2, sign=True)} {ci(o[st].get('diff_ci95_qcluster') or o[st]['diff_ci95'], 2)}")]
                        + [f"{pm(o['acc']['mean'], o['acc']['sd'])} ({f(o['acc']['diff'], 2, sign=True)})"])
        if key == "gemma": rows.append("MIDRULE")
    define("tabMain", tab(["method", "trained-cue flip$\\mid$knew", "$\\Delta$ [95\\% CI]", "held-out (5 cues)", "$\\Delta$ [95\\% CI]", "uncued accuracy ($\\Delta$)"], rows, "lrrrrr"))
    # ---- Table: content-blindness + position binding ----
    rows = []
    for key, mname in MODELS:
        gen = P[key]["gen"]; rows.append([f"\\multicolumn{{8}}{{l}}{{\\textit{{{mname}}} (original model's harmful-cue following, " + ", ".join(f"{p} {f(gen['untrained']['cave'][p]['value'])}" for p in POS) + ")}"])
        for arm in [x for x in P[key]["T1"]["arms"] if x not in MAIN_EXCLUDE]:
            o = gen["arms"][arm]; r = o["ratio_of_seed_means"]
            rows.append([NAME[arm], pm(o["helpful_drop"]["mean"], o["helpful_drop"]["sd"]), pm(o["harmful_drop"]["mean"], o["harmful_drop"]["sd"]), f"{f(r['value'], 2)} {ci(r['ci95'], 2)}"]
                        + [f(o["relred"][p]["mean"]) for p in POS])
        if key == "gemma": rows.append("MIDRULE")
    define("tabContentPosition", tab(["method", "helpful drop", "harmful drop", "ratio [95\\% CI]", "rel. red. front", "pre", "post", "end"], rows, "lrrrrrrr"))
    # ---- Appendix, mechanism (9 Oct: one table per instrument; head subsets, fixed/still-caved reading and route readouts from ana_mech_u.py) ----
    R7 = os.path.dirname(a.extra_u); J7 = lambda n: json.load(open(os.path.join(R7, n))) if os.path.exists(os.path.join(R7, n)) else None
    me, SRJ = J7("ana_mech_u.json"), J7("score_route_u.json"); assert me and SRJ, R7
    # patch recovery: circuit with its Wilson interval (paper_tables causal block), head subsets, trained-cue-only recovery and best random set
    rows = []
    for key, mk, mname in (("gemma", "gemma2b", "Gemma-2-2B-it"), ("llama", "llama8b", "Llama-3.1-8B-Instruct")):
        ca = P[key]["causal"]; pp = me["patch"][mk]; rows.append([f"\\multicolumn{{6}}{{l}}{{\\textit{{{mname}}}}}"])
        for arm in ["untrained"] + [x for x in P[key]["T1"]["arms"] if x in pp]:
            c = ca["untrained"] if arm == "untrained" else ca["arms"][arm]["pooled"]; o = pp[arm]
            br = ca["untrained"]["best_random"] if arm == "untrained" else ca["arms"][arm]["best_random_max"]
            assert abs(c["circuit"] - o["circuit"]["recovery"]) < 1e-9 and c["n_flips"] == o["circuit"]["n"] and abs(br - o["best_random"]) < 1e-9, (key, arm)
            rows.append([NAME[arm], f"{f(c['circuit'])} {ci(c['wilson95'])} ({c['recovered']}/{c['n_flips']})"] + [f(o[s_]["recovery"], 2) if s_ in o else "--" for s_ in ("writers", "readers")]
                        + [f"{f(o['circuit']['trained_cue_recovery'], 2)} ({o['circuit']['trained_cue_n']})", f(br, 2)])
        if key == "gemma": rows.append("MIDRULE")
    define("tabPatch", tab(["method", "circuit [Wilson 95\\%] (recovered / flips)", "writers", "readers", "circuit, trained-cue flips ($n$)", "best random set"], rows, "lrrrrr"))
    # reading (Gemma early readers): average relative change, then fixed vs still-caved trained-cue items; weight-edit depth centroids
    mech = P["gemma"]["mechanism"]; rows = []
    for arm, o in mech["arms"].items():
        t = me["reading_gemma"].get(arm, {}).get("trained")
        rows.append([NAME[arm], f"{f(o['early_reader_rel_trained']['mean'], sign=True)} $\\pm$ {f(o['early_reader_rel_trained']['sd'])}", " ".join(f"{v:+.2f}" for v in o["early_reader_rel_held_seed_mean"].values())]
                    + ([f"{f(-t['rel_change_fixed'] * 100, 0)}\\%", f"{f(-t['rel_change_still'] * 100, 0)}\\%", f(t["fixed_drop_over_gap"], 2), f"{t['n_fixed']:.0f} / {t['n_still']:.0f}"] if t else ["--"] * 4)
                    + [f"{f(o['ov_centroid']['mean'], 1)} $\\pm$ {f(o['ov_centroid']['sd'], 1)}"])
    define("tabReading", tab(["method", "trained cue", "held-out cues", "fixed", "still followed", "fixed / gap", "$n$ fixed / still", "OV depth centroid"], rows, "lrrrrrrr",
                             [(2, 3, "rel.\\ change, labeled items"), (4, 7, "drop on followed trained-cue items")]))
    # route: both models, three readouts per scope (binary pre-registered / own fixes / probability), items fixed, D1 (Llama) or D1-style (Gemma, per seed)
    import numpy as np
    rows = []
    for key, mk, mname in (("llama", "llama8b", "Llama-3.1-8B-Instruct"), ("gemma", "gemma2b", "Gemma-2-2B-it")):
        rows.append([f"\\multicolumn{{7}}{{l}}{{\\textit{{{mname}}}}}"])
        for arm, o in me["route"][mk].items():
            if key == "llama":
                r = P["llama"]["route"]["arms"][arm]; nfix = str(r["n_fixed"]); d1 = ("holds" if r["D1"]["holds"] else "fails") + f" ({r['D1']['largest_scope']})"
                assert all(abs(r["fracs"][s_] - o[s_]["binary"]) < 1e-6 for s_ in SCOPES), arm
            else:
                per = [x for x in SRJ["gemma"]["arms"][arm] if x and x.get("fracs")]; nfix = f"{min(x['n_fixed'] for x in per)}--{max(x['n_fixed'] for x in per)}"
                d1 = f"{sum(bool(x['D1_style']['holds']) for x in per)}/{len(per)} seeds"
                assert all(abs(float(np.mean([x["fracs"][s_] for x in per])) - o[s_]["binary"]) < 1e-6 for s_ in SCOPES), arm
            rows.append([NAME[arm], nfix] + [" / ".join(f(o[s_][k], 2) for k in ("binary", "own_fix", "probability")) for s_ in SCOPES] + [d1])
        if key == "llama": rows.append("MIDRULE")
    define("tabRoute", tab(["method", "fixed by trained model", "EARLY", "CARRIER", "CIRCUIT", "LATE", "D1"], rows, "lrrrrrl", [(3, 6, "mediated fraction: binary / own fixes / probability")]))
    # ---- Table: relapse ----
    rows = []
    for key, mname in MODELS:
        r = P[key]; ru = r.get("relapse_u"); x3 = g(X, "X3", key, "arms") or {}
        if ru:
            U = ru["untrained_gen"]; rl = me["relapse_logodds"][key]
            rows.append([f"\\multicolumn{{8}}{{l}}{{\\textit{{{mname}}}; original model's drift {f(U['0']['trained'])} $\\to$ {f(U['125']['trained'])}, log-odds {f(rl['untrained_dlogit'], 2, sign=True)} (held-out {f(U['0']['held'])} $\\to$ {f(U['125']['held'])})}}"])
            for arm, o in ru["arms"].items():
                ag = o["agg"]; ps = o["per_seed"]; n = ag["n_seeds"]; assert n == 3, (key, arm)
                s0 = sum(p["gen"]["0"]["trained"] for p in ps) / max(n, 1); s1 = sum(p["gen"]["125"]["trained"] for p in ps) / max(n, 1); lo = rl["arms"].get(arm)
                rows.append([NAME[arm], f"{f(s0)} $\\to$ {f(s1)}", f"{f(g(ag, 'frac_trained', 0), sign=True)} $\\pm$ {f(g(ag, 'frac_trained', 1))}", ci(g(x3, arm, "frac_trained", "ci95_seed_t")),
                             f"{f(g(ag, 'adj_trained', 0), sign=True)} $\\pm$ {f(g(ag, 'adj_trained', 1))}", f"{f(lo['dlogit_mean'], 2, sign=True)} ({f(lo['ratio_to_untrained'], 1)}$\\times$)" if lo else "--",
                             f"{f(g(ag, 'frac_held', 0), sign=True)}" if g(ag, "frac_held") else "--",
                             f"{f(g(ag, 'adj_held', 0), sign=True)} $\\pm$ {f(g(ag, 'adj_held', 1))}" if g(ag, "adj_held") else "--"])
        else:
            blk = r.get("relapse7") if key == "gemma" else r.get("relapse")
            if not blk: continue
            U = blk["untrained_gen"]; rows.append([f"\\multicolumn{{8}}{{l}}{{\\textit{{{mname}}}; original model's drift {f(U['0']['trained'])} $\\to$ {f(U['125']['trained'])} (held-out {f(U['0']['held'])} $\\to$ {f(U['125']['held'])})}}"])
            for arm, o in blk["arms"].items():
                ag = o["agg"]; ps = [p for p in o["per_seed"] if p]; n = ag["n_seeds"]
                if key == "gemma": s0, s1 = (sum(g(p, "gen_rates", st, "trained") or 0 for p in ps) / max(n, 1) for st in ("0", "125")) if all(g(p, "gen_rates") for p in ps) else (None, None)
                else: s0, s1 = (sum(p["gen"][st]["trained"] for p in ps) / max(n, 1) for st in ("0", "125"))
                rows.append([NAME[arm], f"{f(s0)} $\\to$ {f(s1)}", f"{f(g(ag, 'gen_frac_trained_125', 0), sign=True)} $\\pm$ {f(g(ag, 'gen_frac_trained_125', 1))}", ci(g(x3, arm, "frac_trained", "ci95_seed_t")),
                             f"{f(g(ag, 'gen_frac_trained_125_ctrl_adj', 0), sign=True)} $\\pm$ {f(g(ag, 'gen_frac_trained_125_ctrl_adj', 1))}", "--", f"{f(g(ag, 'gen_frac_held_125', 0), sign=True)}" if g(ag, "gen_frac_held_125") else "--", "--"])
        if key == "gemma": rows.append("MIDRULE")
    define("tabRelapse", tab(["method", "flip$\\mid$knew, step 0 $\\to$ 125", "relapse fraction", "seed 95\\% CI", "beyond drift", "log-odds rise", "fraction", "beyond drift"], rows, "lrrrrrrr",
                             [(2, 6, "trained cue"), (7, 8, "held-out cues")]))
    # ---- Table: training-data variants, response-only BCT, self-training ----
    rows = []; st = P.get("selftrain")
    for key, mname in MODELS:
        fx = P[key]["fix6"]; gen = P[key]["gen"]; rows.append([f"\\multicolumn{{9}}{{l}}{{\\textit{{{mname}}}}}"])
        mean = lambda v: sum(v) / len(v)
        for arm, o in fx["arms"].items():
            gg = gen["arms"].get(arm, {}); hd, hm = mean(o["helpful_drop"]), mean(o["harmful_drop"])
            cai = f(o["CAI"], 2) if (o["CAI"] != float("inf") and hm >= 0.05) else ("undef." if hm <= 0 else f"{f(o['CAI'], 2)}$^{{*}}$")
            rows.append([NAME[arm], f(mean(o["test_trained"])), f(mean(o["test_held"])), f(mean(o["test_acc"])), f(hd), f(hm), cai, f(o["relred_mean"]["front"]), f(o["relred_mean"]["end"])])
        if st:
            mk = "gemma2b" if key == "gemma" else "llama8b"; so = st[mk]; t = so["test"]["selftrain"]; gp = so["gen_paired"]["selftrain"]
            sh, sm = mean(gp["helpful_drop"]), mean(gp["harmful_drop"]); scai = (f(sh / sm, 2) + ("" if sm >= 0.05 else "$^{*}$")) if sm > 0 else "undef."
            rows.append([NAME["selftrain"], f(mean(t["trained"])), f(mean(t["held_fs_corrected"])), f(mean(t["acc"])), f(sh), f(sm), scai, "--", "--"])
        if key == "gemma": rows.append("MIDRULE")
    define("tabVariants", tab(["method", "trained-cue", "held-out", "accuracy", "helpful drop", "harmful drop", "CAI", "rel. red. front", "end"], rows, "lrrrrrrrr"))
    # ---- Appendix: verdicts with margins ----
    rows = []
    for row in X.get("X5", []):
        if "quantity" not in row: continue
        val = f(row["value"], sign=True) if round(row["value"], 3) != 0 else f"{row['value']:+.4f}"   # (9 Oct) a near-zero deciding value keeps its sign
        rows.append([vq(row["verdict"]), vq(row["quantity"]), f(row["threshold"], 2), val, ci(row["interval95"]), ("resolved" if row["resolved"] else "unresolved") if row["resolved"] is not None else "--"])
    define("tabVerdicts", tab(["verdict", "deciding quantity", "threshold", "value", "95\\% interval", "status"], rows, "llrrrl"))
    # (9 Oct) the readout-gate table is no longer printed: the corrections appendix states the counts, failures and minima as macros (paper_numbers.py)
    # ---- Appendix: authors' evaluator ----
    rows = []; pr = P["attct_eval"]["paper_reported_llama"]
    AE = [("gemma_base", "original"), ("gemma_ours_bct_s42", "BCT"), ("gemma_ours7u_act_s42", "ACT"), ("gemma_ours7u_attct_s42", "AttCT"), ("gemma_bctro_s42", "response-only BCT"),
          ("gemma_sweep_bct_lr1e-5", "BCT sweep, LR 1e-5"), ("gemma_sweep_bct_lr3.16e-5", "BCT sweep, LR 3.16e-5"), ("gemma_sweep_bct_lr3.16e-4", "BCT sweep, LR 3.16e-4"), "MIDRULE",
          ("llama_base", "original"), ("llama_ours_bct_s42", "BCT"), ("llama_ours_act_s42", "ACT"), ("llama_ours_attct_s42", "AttCT"),
          ("llama_repro_bct", "BCT, authors' recipe$^\\S$"), ("llama_repro_act", "ACT, authors' recipe$^\\S$"), ("llama_repro_attct", "AttCT, authors' recipe$^\\S$")]
    assert {x[0] for x in AE if x != "MIDRULE"} | {"gemma_ours_act_s42", "gemma_ours_attct_s42"} == set(P["attct_eval"]["rows"]), sorted(set(P["attct_eval"]["rows"]) ^ {x[0] for x in AE if x != "MIDRULE"})
    for x in AE:
        if x == "MIDRULE": rows.append("MIDRULE"); continue
        t, label = x; r = P["attct_eval"]["rows"][t]
        pk = {"llama_base": "base", "llama_repro_bct": "bct", "llama_repro_act": "act", "llama_repro_attct": "attct"}.get(t); pp = pr.get(pk, {})
        rows.append([("Gemma-2-2B-it " if t.startswith("gemma") else "Llama-3.1-8B-Instruct ") + label, f"{f(r['brr'])} $\\pm$ {f(r['brr_se_unpaired'])}" + (f" ({f(pp['brr'])})" if "brr" in pp else ""), f"{f(r['held_out_brr'])} $\\pm$ {f(r['held_out_brr_se_unpaired'])}" + (f" ({f(pp['held_out_brr'])})" if "held_out_brr" in pp else ""),
                     f"{f(r['clean_acc'])} {ci(r['clean_acc_wilson95'])}" + (f" ({f(pp['clean_acc'])})" if "clean_acc" in pp else ""), (f(r["anthropic_rate"]) if r["anthropic_rate"] is not None else "--") + (f" ({f(pp['anthropic'])})" if "anthropic" in pp else "")])
    define("tabAuthorsEval", tab(["checkpoint", "MMLU BRR $\\pm$ SE", "held-out BRR $\\pm$ SE", "MMLU clean acc.", "Anthropic rate"], rows, "lrrrr"))
    # ---- Appendix: RMCT paper regime ----
    rows = []
    for lr, o in P["rmct_paper"]["pooled_final_step"].items():
        pp = P["rmct_paper"]["paper_reported"].get(lr, {})
        rows.append([lr, o["n_evals"], f"{f(o['train_bias_pro_bsr'])} {ci(o['wilson95'])}", f"{f(o['held_out_pro_bsr'])} {ci(o['held_out_wilson95'])}", "/".join(f"{x:.3f}" for x in pp.get("da", [])) or "--", "/".join(f"{x:.3f}" for x in pp.get("held", [])) or "--"])
    define("tabRmctPaper", tab(["LR", "evals", "training-bias switch rate", "held-out switch rate", "authors' logs (distractor argument)", "authors' logs (held-out)"], rows, "lrrrrr"))
    # ---- Appendix (unmerged only): merge fidelity, selection, few-shot ----
    if A.get("A1"):   # merge fidelity, summarised per model x arm over its three seeds
        grp = {}
        for t, r in A["A1"].items():
            m_ = _re.match(r"(gemma2b|llama8b)_(.*)_s(\d+)$", t); assert m_, t
            arm = _re.sub(r"^(final7u|final)_", "", m_.group(2)); arm = _re.sub(r"^fix_", "", arm).replace("bayes6_", "bayes_"); arm = _re.sub(r"7u$", "", arm)
            assert arm in NAME, t; grp.setdefault((m_.group(1), arm), []).append(r)
        ORD = ("bct", "act", "attct", "rmct", "rmcta", "bctro", "bayes_r0.5", "bayes_r0.8", "goldgated", "posdiv"); rows = []
        for model in ("gemma2b", "llama8b"):
            rows.append([f"\\multicolumn{{6}}{{l}}{{\\textit{{{dict(gemma2b='Gemma-2-2B-it', llama8b='Llama-3.1-8B-Instruct')[model]}}}}}"])
            for arm in [x for x in ORD if (model, x) in grp]:
                rs = grp[(model, arm)]; ag = [r["agree_generated"] for r in rs]; d = lambda k: sum(r["merged"][k] - r["unmerged"][k] for r in rs) / len(rs)
                rows.append([NAME[arm], f"{min(ag):.3f}--{max(ag):.3f}"] + [f(d(k), sign=True) for k in ("trained", "held", "acc")] + [f"{sum(r['merged_results_stand'] for r in rs)}/{len(rs)}"])
            if model == "gemma2b": rows.append("MIDRULE")
        assert sum(len(v) for v in grp.values()) == len(A["A1"])
        define("tabMerge", tab(["method", "agreement (range)", "trained cue", "held-out", "accuracy", "rule met"], rows, "lrrrrr", [(3, 5, "merged $-$ unmerged, seed mean")]))
    if A.get("A2"):
        rows = []
        for model in ("gemma2b", "llama8b"):
            for arm, r in (A["A2"].get(model) or {}).items():
                if arm == "untrained": continue
                grid = (" ".join(f"{lr}: {v['F1']:.3f}{'$^d$' if v.get('degenerate') else ''}" for lr, v in (r.get("rows") or {}).items()) if r.get("rows") else
                        " ".join(f"{lr}: {p['F1']:.3f}{'$^d$' if p['modal_frac'] >= .95 else ''}" for lr, p in sorted((r.get("points") or {}).items(), key=lambda kv: float(kv[0]))))
                rows.append([MODEL_SHORT[model], NAME.get(arm, arm), r.get("stored_selection") or "--", r.get("selected") or "--", "yes" if r.get("changed") else "no", esc(grid)])
        define("tabSelection", tab(["model", "method", "stored", "re-selected", "changed", "validation F1 by LR (unmerged, generated)"], rows, "lllllp{7cm}"))
    os.makedirs(a.out, exist_ok=True)
    # ---- Appendix: per-cue test rates (T1 per-cue block; generated answers, few-shot cues at the test slot) ----
    CUES = [("suggested_answer", "trained cue"), ("post_hoc", "post hoc"), ("distractor_fact", "distractor fact"), ("wrong_few_shot", "wrong few-shot"),
            ("distractor_argument", "distractor arg."), ("spurious_few_shot_squares", "squares")]
    rows = []
    for key, mname in MODELS:
        t1 = P[key]["T1"]; u = t1["untrained"]; rows.append([f"\\multicolumn{{7}}{{l}}{{\\textit{{{mname}}}}}"])
        rows.append(["original"] + [f(u[f"cue:{c}"]["value"]) for c, _ in CUES])
        for arm, o in t1["arms"].items():
            cell = lambda c: (lambda q: f"{f(q['mean'])} ({f(q['diff'], 2, sign=True)}{'' if (q['diff_ci95_qcluster'][0] > 0 or q['diff_ci95_qcluster'][1] < 0) else '$^{\\circ}$'})")(o[f"cue:{c}"])
            rows.append([NAME[arm]] + [cell(c) for c, _ in CUES])
    define("tabPerCue", tab(["method"] + [n for _, n in CUES], rows, "lrrrrrr"))
    EU = json.load(open(a.extra_u)) if os.path.exists(a.extra_u) else None; assert EU, a.extra_u
    # ---- Appendix: letter-prior control per arm, with modal-letter shares ----
    rows = []
    for key, mname, blocks in (("gemma", "Gemma-2-2B-it", ("letterprior",)), ("llama", "Llama-3.1-8B-Instruct", ("letterprior",))):
        rows.append([f"\\multicolumn{{6}}{{l}}{{\\textit{{{mname}}}}}"]); seen = set()
        for b in blocks:
            for arm, o in P[key][b]["arms"].items():
                if arm in seen: continue
                seen.add(arm); bt, bh = o["beyond_prior_drop_trained"], o["beyond_prior_drop_held"]
                rows.append([NAME[arm], f"{f(o['arm_trained'][0])} / {f(o['control_trained'][0])}", pm(bt["mean"], bt["sd"]), f"{f(o['arm_held'][0])} / {f(o['control_held'][0])}", pm(bh["mean"], bh["sd"]),
                             f"{f(sum(o['modal_share']) / len(o['modal_share']), 2)} ({'/'.join(o['modal_letter'])})"])
    define("tabLetterPrior", tab(["method", "trained: method / control", "beyond prior", "held-out: method / control", "beyond prior", "modal-letter share"], rows, "lrrrrr"))
    # ---- Appendix, selectivity and position (ana_scale_u.py; wording families from ana_extra_u.py): the four methods and RMCT+anchor ----
    sc = J7("ana_scale_u.json"); assert sc, R7
    SHOW = ("bct", "act", "attct", "rmct", "rmcta"); SC3 = ("absolute", "proportional", "log-odds"); rows, rowsB, rowsP, rowsS = [], [], [], []
    for key, mname in (("gemma2b", "Gemma-2-2B-it"), ("llama8b", "Llama-3.1-8B-Instruct")):
        for R_, n_ in ((rows, 4), (rowsB, 6), (rowsP, 8), (rowsS, 6)): R_.append([f"\\multicolumn{{{n_}}}{{l}}{{\\textit{{{mname}}}}}"])
        for arm in [x for x in SHOW if x in sc[key]]:
            o = sc[key][arm]; ms = o["position_vs_strength"]["matched_strength"]; fr = EU[key]["arms"][arm]["family_relred"]
            cell = lambda v: "--" if v["ratio"] is None else f"{f(v['ratio'], 2)} " + (ci(v['ratio_ci95'], 2) if v.get("frac_boot_harmful_gt_0.02", 1.0) >= 0.975 else "--")
            tri = lambda p: " / ".join("--" if o[p][k]["ratio"] is None else f(o[p][k]["ratio"], 2) for k in SC3)
            rows.append([NAME[arm]] + [cell(o["all"][k]) for k in SC3])
            rowsB.append([NAME[arm]] + [tri(p) for p in ("before",) + POS])
            rowsP.append([NAME[arm], f"{f(ms['relred_before'], 2)} / {f(ms['relred_after'], 2)}", f(o["position_vs_strength"]["after_coef_controlling_strength"], 2, sign=True), f(o["pt_basis"]["end"]["relred_not_T"], 2)]
                         + [f"{f(fr[f'train|{p}'], 2)} / {f(fr[f'novel|{p}'], 2)}" for p in POS])
        for arm in [x for x in ("bct", "act", "attct", "rmct", "fix_goldgated") if x in sc[key]]:
            v = sc[key][arm]["same_item"]["before"]
            rowsS.append([NAME[arm.replace("fix_", "")], f"{f(v['Tb'], 2)} $\\to$ {f(v['Tc'], 2)}", f"{f(v['Xb'], 2)} $\\to$ {f(v['Xc'], 2)}",
                          f"{f(v['dT'] - v['dX'], 3, sign=True)} {ci(v['dT_minus_dX_ci95'], 3)}", f"{f(v['gap_b'], 2, sign=True)} $\\to$ {f(v['gap_c'], 2, sign=True)}", ci(v["gap_change_ci95"], 2)])
    define("tabScale", tab(["method", "absolute [95\\% CI]", "proportional [95\\% CI]", "log-odds [95\\% CI]"], rows, "lrrr"))
    define("tabContentByPos", tab(["method", "before options", "front", "pre", "post", "end"], rowsB, "lrrrrr", [(2, 6, "helpful / harmful drop: absolute / proportional / log-odds")]))
    define("tabPosCheck", tab(["method", "matched: before / after", "after coef.", "end, non-correct", "front", "pre", "post", "end"], rowsP, "lrrrrrrr",
                              [(2, 4, "strength and letter checks"), (5, 8, "training-family / novel wordings")]))
    define("tabSameItem", tab(["method", "correct cue", "wrong cue", "drop difference [95\\%]", "log-odds gap", "change [95\\%]"], rowsS, "lrrrrr"))
    ly = J7("ana_layout_u.json")   # (9 Oct, post hoc, exploratory) the test battery's trained cue split by the dumps' own cue layout
    if ly:
        rows = []
        for key, mname in (("gemma2b", "Gemma-2-2B-it"), ("llama8b", "Llama-3.1-8B-Instruct")):
            u = ly["models"][key]["untrained"]
            rows.append([f"\\multicolumn{{5}}{{l}}{{\\textit{{{mname}}}}}"]); rows.append(["original flip$\\mid$knew ($n$)"] + [f"{f(u[l_]['rate'])} ({u[l_]['n_knew']})" for l_ in POS[:3]] + ["--"])
            for arm, o in ly["models"][key]["arms"].items():
                rows.append([NAME[arm]] + [f(o["relred"][l_], 2) for l_ in POS[:3]] + [f"{f(o['front_minus_post'], 2, sign=True)} {ci(o['front_minus_post_ci95'], 3)}"])
        define("tabLayout", tab(["method", "front (trained)", "pre", "post", "front $-$ post [95\\%]"], rows, "lrrrr", [(2, 4, "relative reduction of trained-cue flip$\\mid$knew")]))
    LRF = os.path.join(os.path.dirname(a.extra_u), "ana_lrsweep_u.json")
    if os.path.exists(LRF):
        lr = json.load(open(LRF)); rows = []
        row_ = lambda lab, v: [lab, f(v["scales"]["absolute"]["harmful_drop"]), f(v["scales"]["absolute"]["helpful_drop"])] + [f(v["scales"][k]["ratio"], 2) for k in ("absolute", "proportional", "log-odds")] + [f(v["relred"][p_], 2) for p_ in POS]
        for model, mname in (("gemma2b", "Gemma"), ("llama8b", "Llama")):
            rows.append([f"\\multicolumn{{10}}{{l}}{{\\textit{{{mname}}}}}"])
            for arm, row in lr[model].items():
                for k, v in row.items():
                    if v: rows.append(row_(f"{NAME[arm]}, " + ("selected LR" if k == "selected" else f"LR {k}"), v))
        rows.append([f"\\multicolumn{{10}}{{l}}{{\\textit{{Llama, authors' recipe}}}}"])
        for arm, v in lr.get("llama8b_repro", {}).items():
            if v: rows.append(row_(NAME[arm], v))
        define("tabLRSweep", tab(["checkpoint (seed 42)", "harmful drop", "helpful drop", "ratio abs.", "prop.", "log-odds", "rel. red. front", "pre", "post", "end"], rows, "lrrrrrrrrr"))
    with open(f"{a.out}/tables.tex", "w") as fh:
        fh.write(f"% generated by paper_tables_tex.py from {a.tables_dir}; do not edit\n")
        for k, v in T.items(): fh.write(f"\\newcommand{{\\{k}}}{{%\n{v}}}\n\n")
    print(f"{len(T)} tables -> {a.out}/tables.tex: {', '.join(T)}")


if __name__ == "__main__":
    main()
