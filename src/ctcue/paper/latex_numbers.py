#!/usr/bin/env python3
"""
paper_numbers (2 Oct 2026): every number the ARR draft cites, as LaTeX macros generated from the analysis outputs (no number is typed into
the manuscript). Reads <tables_dir>/paper_tables.json (paper_tables.py), <tables_dir>/paper_tables_extra.json (paper_tables_extra.py) and,
if present, <tables_dir>/appendix_u.json (appendix_u.py); writes <out>/numbers.tex (+ numbers_report.txt listing every macro).
Macro names: \\v<Model><Arm><Quantity>[<Field>] in CamelCase letters only (LaTeX forbids digits), e.g. \\vGemBctTrainedMean, \\vLlaRmctHeldDiffCI.
A quantity missing from the JSON becomes "\\missingnum" (rendered as a visible ??) so that a stale draft can never show a wrong number silently.
Usage: paper_numbers.py --tables_dir <dir with paper_tables.json> [--extra_dir <dir with paper_tables_extra.json / appendix_u.json>] --out <dir>
"""
import argparse, json, os, re
MODEL = {"gemma": "Gem", "llama": "Lla"}; ARM = {"bct": "Bct", "act": "Act", "attct": "Attct", "rmct": "Rmct", "rmcta": "Rmcta", "untrained": "Base",
                                                "bayes_r0.5": "BayesHalf", "bayes_r0.8": "BayesEight", "goldgated": "Gold", "posdiv": "Posdiv", "bctro": "Bctro", "selftrain": "Selftrain"}
CUE = {"suggested_answer": "Sugg", "post_hoc": "Posthoc", "distractor_fact": "Dfact", "wrong_few_shot": "Wfs", "distractor_argument": "Darg", "spurious_few_shot_squares": "Squares"}
POS = {"front": "Front", "pre": "Pre", "post": "Post", "end": "End"}; SCOPE = {"EARLY": "Early", "CARRIER": "Carrier", "CIRCUIT": "Circuit", "LATE": "Late"}
M = {}   # name -> text


def g(o, *keys):
    for k in keys:
        if isinstance(o, dict) and k in o: o = o[k]
        elif isinstance(o, list) and isinstance(k, int) and -len(o) <= k < len(o): o = o[k]
        else: return None
    return o


def neg(s):   # typeset a leading minus as a math minus that cannot break across a line
    return f"\\mbox{{$-${s[1:]}}}" if s.startswith("-") else s


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and x != x): return None
    if isinstance(x, bool): return "TRUE" if x else "FALSE"
    if isinstance(x, int): return str(x)
    if round(x, d) == 0: return f"{0:.{d}f}"            # no negative or signed zero
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return neg(s)


def ci(c, d=3):
    if not c or c[0] is None: return None
    z = lambda v: 0.0 if round(v, d) == 0 else v
    return "\\mbox{[" + ", ".join((f"$-${-z(v):.{d}f}" if z(v) < 0 else f"{z(v):.{d}f}") for v in c[:2]) + "]}"


DIGITS = {str(i): w for i, w in enumerate(("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"))}


def put(name, text):
    name = "".join(DIGITS.get(c, c) for c in name); name = re.sub(r"[^A-Za-z]", "", name)   # LaTeX macro names: letters only
    assert name not in M, f"duplicate macro {name}"
    if isinstance(text, str) and "_" in text: text = text.replace("_", "\\_")   # tag lists (gate failures) are printed in text mode
    M[name] = text if text is not None else "\\missingnum"


def same(name, text):
    """A macro already defined elsewhere from another source: assert it agrees (or define it if absent)."""
    n = re.sub(r"[^A-Za-z]", "", "".join(DIGITS.get(c, c) for c in name))
    if n in M: assert M[n] == text, f"{n}: {M[n]} != {text}"
    else: put(name, text)


def pct(x):   # a rate as a percentage with no decimals ("41\%")
    return None if x is None else f"{100 * x:.0f}\\%"


R_W7U = "/data/ct_root/results/w7u"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tables_dir", required=True); ap.add_argument("--extra_dir", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args(); T = json.load(open(f"{a.tables_dir}/paper_tables.json")); xd = a.extra_dir or a.tables_dir
    X = json.load(open(f"{xd}/paper_tables_extra.json")) if os.path.exists(f"{xd}/paper_tables_extra.json") else {}
    A = json.load(open(f"{xd}/appendix_u.json")) if os.path.exists(f"{xd}/appendix_u.json") else {}
    basis = "unmerged" if g(T, "meta", "unmerged") else "merged"
    put("vBasis", basis); put("vBasisNote", "UNMERGED adapters (primary results)" if basis == "unmerged" else "MERGED adapters (SUPERSEDED pipeline; placeholder numbers)")
    put("vIsMerged", "0" if basis == "unmerged" else "1")
    put("vBootB", str(g(T, "meta", "B"))); put("vGenerated", str(g(T, "meta", "created")))
    for key, mk in MODEL.items():
        r = T[key]; t1 = r["T1"]
        put(f"v{mk}NItems", str(sum(t1["n_items"].values()))); put(f"v{mk}NQuestions", str(t1["n_questions"])); put(f"v{mk}NTrainedItems", str(t1["n_items"]["suggested_answer"]))
        for st, sn in (("trained", "Trained"), ("held", "Held"), ("acc", "Acc")):
            u = t1["untrained"][st]; put(f"v{mk}Base{sn}", f(u["value"])); put(f"v{mk}Base{sn}CI", ci(u.get("ci95_qcluster") or u.get("ci95"))); put(f"v{mk}Base{sn}Pct", pct(u["value"]))
            for arm, o in t1["arms"].items():
                an = ARM[arm]; q = o[st]
                put(f"v{mk}{an}{sn}Mean", f(q["mean"])); put(f"v{mk}{an}{sn}SD", f(q["sd"])); put(f"v{mk}{an}{sn}CI", ci(q.get("ci95_qcluster") or q.get("ci95")))
                put(f"v{mk}{an}{sn}Diff", f(q["diff"], sign=True)); put(f"v{mk}{an}{sn}DiffCI", ci(q.get("diff_ci95_qcluster") or q.get("diff_ci95")))
                put(f"v{mk}{an}{sn}Pct", pct(q["mean"]))
                if st == "trained": put(f"v{mk}{an}TrainedRelCut", pct(1 - q["mean"] / t1["untrained"]["trained"]["value"]))
                if st == "held": put(f"v{mk}{an}HeldRelCut", pct(1 - q["mean"] / t1["untrained"]["held"]["value"]))
        for c, cn in CUE.items():
            put(f"v{mk}Base{cn}", f(g(t1, "untrained", f"cue:{c}", "value"))); put(f"v{mk}BaseKnew{cn}", str(g(t1, "untrained", "knew", c)))
            for arm, o in t1["arms"].items(): put(f"v{mk}{ARM[arm]}{cn}", f(g(o, f"cue:{c}", "mean")))
        gen = r["gen"]; put(f"v{mk}GenNItems", str(gen["n_items"]))
        for p, pn in POS.items():
            put(f"v{mk}BaseCave{pn}", f(g(gen, "untrained", "cave", p, "value"))); put(f"v{mk}BaseHelp{pn}", f(g(gen, "untrained", "helpful_uptake", p, "value")))
        for arm, o in gen["arms"].items():
            an = ARM[arm]
            for k, kn in (("helpful_drop", "HelpDrop"), ("harmful_drop", "HarmDrop")):
                put(f"v{mk}{an}{kn}", f(o[k]["mean"])); put(f"v{mk}{an}{kn}SD", f(o[k]["sd"])); put(f"v{mk}{an}{kn}CI", ci(o[k]["ci95"]))
            put(f"v{mk}{an}HelpMinusHarm", f(o["helpful_minus_harmful"]["mean"], sign=True)); put(f"v{mk}{an}HelpMinusHarmCI", ci(o["helpful_minus_harmful"]["ci95"]))
            put(f"v{mk}{an}Ratio", f(o["ratio_of_seed_means"]["value"], 2)); put(f"v{mk}{an}RatioCI", ci(o["ratio_of_seed_means"]["ci95"], 2))
            for p, pn in POS.items():
                put(f"v{mk}{an}Relred{pn}", f(o["relred"][p]["mean"])); put(f"v{mk}{an}Relred{pn}CI", ci(o["relred"][p]["ci95"])); put(f"v{mk}{an}Help{pn}", f(o["helpful_uptake"][p]["mean"]))
                put(f"v{mk}{an}Relred{pn}Pct", pct(o["relred"][p]["mean"]))
            for p in ("post", "end"):
                q = o[f"relred_front_minus_{p}"]; put(f"v{mk}{an}FrontMinus{POS[p]}", f(q["mean"], sign=True)); put(f"v{mk}{an}FrontMinus{POS[p]}CI", ci(q["ci95"]))
            for k in ("helpful", "harmful"):
                q = o.get(f"{k}_ratio_to_bct")
                if q: put(f"v{mk}{an}{k.capitalize()}RatioToBct", f(q["value"], 2)); put(f"v{mk}{an}{k.capitalize()}RatioToBctCI", ci(q["ci95"], 2))
        ca = r["causal"]; cu = ca["untrained"]
        put(f"v{mk}BasePatch", f(cu["circuit"])); put(f"v{mk}BasePatchCI", ci(cu["wilson95"])); put(f"v{mk}BasePatchN", f"{cu['recovered']}/{cu['n_flips']}"); put(f"v{mk}BasePatchRandom", f(cu["best_random"]))
        put(f"v{mk}PatchSeeds", ", ".join(map(str, ca["seeds"])))
        for arm, o in ca["arms"].items():
            an = ARM[arm]; p = o["pooled"]
            put(f"v{mk}{an}Patch", f(p["circuit"])); put(f"v{mk}{an}PatchCI", ci(p["wilson95"])); put(f"v{mk}{an}PatchN", f"{p['recovered']}/{p['n_flips']}"); put(f"v{mk}{an}PatchRandom", f(o["best_random_max"]))
        lp = r["letterprior"]; put(f"v{mk}LpBaseTrained", f(lp["untrained"]["trained"])); put(f"v{mk}LpBaseHeld", f(lp["untrained"]["held"]))
        for arm, o in lp["arms"].items():
            an = ARM[arm]
            for k, kn in (("trained", "Trained"), ("held", "Held")):
                put(f"v{mk}{an}Bpd{kn}", f(o[f"beyond_prior_drop_{k}"]["mean"])); put(f"v{mk}{an}Bpd{kn}SD", f(o[f"beyond_prior_drop_{k}"]["sd"]))
                put(f"v{mk}{an}LpArm{kn}", f(o[f"arm_{k}"][0])); put(f"v{mk}{an}LpControl{kn}", f(o[f"control_{k}"][0])); put(f"v{mk}{an}PriorExplained{kn}", f(o[f"prior_explained_drop_{k}"]))
            put(f"v{mk}{an}ModalShare", f(sum(o["modal_share"]) / len(o["modal_share"]), 2)); put(f"v{mk}{an}ModalLetters", "/".join(o["modal_letter"]))
            put(f"v{mk}{an}LpFlagged", ", ".join(map(str, o.get("flagged_seeds", []))) or "none")
        if key == "gemma":
            for arm, o in r["letterprior_fix"]["arms"].items():
                an = ARM[arm]; put(f"vGem{an}FixBpdTrained", f(o["beyond_prior_drop_trained"]["mean"])); put(f"vGem{an}FixBpdHeld", f(o["beyond_prior_drop_held"]["mean"]))
            mech = r["mechanism"]; put("vGemEarlyReaders", ", ".join(mech["early_reader_heads"]))
            for arm, o in mech["arms"].items():
                an = ARM[arm]; put(f"vGem{an}EarlyRel", f(o["early_reader_rel_trained"]["mean"], sign=True)); put(f"vGem{an}EarlyRelSD", f(o["early_reader_rel_trained"]["sd"]))
                put(f"vGem{an}OV", f(o["ov_centroid"]["mean"], 1)); put(f"vGem{an}OVSD", f(o["ov_centroid"]["sd"], 1)); put(f"vGem{an}QK", f(o["qk_centroid"]["mean"], 1)); put(f"vGem{an}QKSD", f(o["qk_centroid"]["sd"], 1))
            fx = r["fix"]
            for arm, o in fx["arms"].items():
                an = ARM[arm]; mean = lambda v: sum(v) / len(v)
                put(f"vGem{an}FixTestTrained", f(mean(o["test_trained"]))); put(f"vGem{an}FixTestHeld", f(mean(o["test_held"]))); put(f"vGem{an}FixTestAcc", f(mean(o["test_acc"])))
            for k in ("F1_bayes_r0.8", "F2_goldgated", "F3_posdiv"): put(f"vGemFixVerdict{k.split('_')[0]}", f(fx[k]["verdict"]))
            put("vGemF3PosdivEnd", f(fx["F3_posdiv"]["relred_end_posdiv"])); put("vGemF3BctEnd", f(fx["F3_posdiv"]["relred_end_bct"])); put("vGemF3Ratio", f(fx["F3_posdiv"]["ratio"], 2))
        fx6 = r["fix6"]
        for arm, o in fx6["arms"].items():
            an = ARM[arm]; mean = lambda v: sum(v) / len(v)
            put(f"v{mk}{an}FxTrained", f(mean(o["test_trained"]))); put(f"v{mk}{an}FxHeld", f(mean(o["test_held"]))); put(f"v{mk}{an}FxAcc", f(mean(o["test_acc"])))
            put(f"v{mk}{an}FxHelp", f(mean(o["helpful_drop"]))); put(f"v{mk}{an}FxHarm", f(mean(o["harmful_drop"]))); put(f"v{mk}{an}CAI", f(o["CAI"], 2) if o["CAI"] != float("inf") else "undefined")
            put(f"v{mk}{an}ContentAware", "yes" if o["content_aware"] else "no")
            for p, pn in POS.items(): put(f"v{mk}{an}FxRelred{pn}", f(o["relred_mean"][p])); put(f"v{mk}{an}FxHelp{pn}", f(mean(o["helpful"][p])))
        for k, v in fx6["criteria"].items():
            if k.startswith(("LF", "B")): put(f"v{mk}Crit{k.split('_')[0]}", f(v["prediction_holds"]))   # per-model criteria (the pre-registered ones are in the verdict table)
        # relapse: merged pipeline (relapse7 for gemma, relapse for llama) or the unmerged block
        ru = r.get("relapse_u")
        if ru:
            U = ru["untrained_gen"]; put(f"v{mk}RelBaseStepZero", f(U["0"]["trained"])); put(f"v{mk}RelBaseStepEnd", f(U["125"]["trained"])); put(f"v{mk}RelBaseHeldZero", f(U["0"]["held"])); put(f"v{mk}RelBaseHeldEnd", f(U["125"]["held"]))
            for arm, o in ru["arms"].items():
                an = ARM[arm]; ag = o["agg"]
                for q, qn in (("frac_trained", "RelFrac"), ("adj_trained", "RelAdj"), ("frac_held", "RelFracHeld"), ("adj_held", "RelAdjHeld")):
                    put(f"v{mk}{an}{qn}", f(g(ag, q, 0), sign=True)); put(f"v{mk}{an}{qn}SD", f(g(ag, q, 1)))
                put(f"v{mk}{an}RelN", str(ag["n_seeds"])); put(f"v{mk}{an}RelGap", f(sum(p["gap_trained"] for p in o["per_seed"]) / max(len(o["per_seed"]), 1)))
                put(f"v{mk}{an}RelStepZero", f(sum(p["gen"]["0"]["trained"] for p in o["per_seed"]) / max(len(o["per_seed"]), 1))); put(f"v{mk}{an}RelStepEnd", f(sum(p["gen"]["125"]["trained"] for p in o["per_seed"]) / max(len(o["per_seed"]), 1)))
            for k in ("R1", "LR1", "LR2"):
                if k in ru: put(f"v{mk}RelVerdict{k}", f(ru[k]))
        elif key == "gemma" and r.get("relapse7"):
            r7 = r["relapse7"]; U = r7["untrained_gen"]; put("vGemRelBaseStepZero", f(U["0"]["trained"])); put("vGemRelBaseStepEnd", f(U["125"]["trained"])); put("vGemRelBaseHeldZero", f(U["0"]["held"])); put("vGemRelBaseHeldEnd", f(U["125"]["held"]))
            for arm, o in r7["arms"].items():
                an = ARM[arm]; ag = o["agg"]
                put(f"vGem{an}RelFrac", f(g(ag, "gen_frac_trained_125", 0), sign=True)); put(f"vGem{an}RelFracSD", f(g(ag, "gen_frac_trained_125", 1)))
                put(f"vGem{an}RelAdj", f(g(ag, "gen_frac_trained_125_ctrl_adj", 0), sign=True)); put(f"vGem{an}RelAdjSD", f(g(ag, "gen_frac_trained_125_ctrl_adj", 1)))
                put(f"vGem{an}RelFracHeld", f(g(ag, "gen_frac_held_125", 0), sign=True)); put(f"vGem{an}RelAdjHeld", f(g(ag, "gen_frac_held_125_ctrl_adj", 0), sign=True))
                put(f"vGem{an}RelN", str(ag["n_seeds"])); put(f"vGem{an}RelGap", f(sum(p["gen_gap_trained"] for p in o["per_seed"]) / max(len(o["per_seed"]), 1)))
                gr = [g(p, "gen_rates") for p in o["per_seed"]]
                if all(gr): put(f"vGem{an}RelStepZero", f(sum(x["0"]["trained"] for x in gr) / len(gr))); put(f"vGem{an}RelStepEnd", f(sum(x["125"]["trained"] for x in gr) / len(gr)))
                else: put(f"vGem{an}RelStepZero", None); put(f"vGem{an}RelStepEnd", None)
            put("vGemRelVerdictR1", f(r7["R1_gen"]))
        elif key == "llama" and r.get("relapse"):
            rl = r["relapse"]; U = rl["untrained_gen"]; put("vLlaRelBaseStepZero", f(U["0"]["trained"])); put("vLlaRelBaseStepEnd", f(U["125"]["trained"])); put("vLlaRelBaseHeldZero", f(U["0"]["held"])); put("vLlaRelBaseHeldEnd", f(U["125"]["held"]))
            for arm, o in rl["arms"].items():
                an = ARM[arm]; ag = o["agg"]
                put(f"vLla{an}RelFrac", f(g(ag, "gen_frac_trained_125", 0), sign=True)); put(f"vLla{an}RelFracSD", f(g(ag, "gen_frac_trained_125", 1)))
                put(f"vLla{an}RelAdj", f(g(ag, "gen_frac_trained_125_ctrl_adj", 0), sign=True)); put(f"vLla{an}RelAdjSD", f(g(ag, "gen_frac_trained_125_ctrl_adj", 1)))
                put(f"vLla{an}RelFracHeld", f(g(ag, "gen_frac_held_125", 0), sign=True)); put(f"vLla{an}RelAdjHeld", f(g(ag, "gen_frac_held_125_ctrl_adj", 0), sign=True))
                put(f"vLla{an}RelN", str(ag["n_seeds"])); put(f"vLla{an}RelGap", f(sum(p["gen_gap_trained"] for p in o["per_seed"] if p) / max(len([p for p in o["per_seed"] if p]), 1)))
                ps = [p for p in o["per_seed"] if p]; put(f"vLla{an}RelStepZero", f(sum(p["gen"]["0"]["trained"] for p in ps) / max(len(ps), 1))); put(f"vLla{an}RelStepEnd", f(sum(p["gen"]["125"]["trained"] for p in ps) / max(len(ps), 1)))
            put("vLlaRelVerdictLR1", f(rl["LR1"])); put("vLlaRelVerdictLR2", f(rl["LR2"]))
        if key == "llama" and r.get("route"):
            rt = r["route"]
            for arm, o in rt["arms"].items():
                if not o: continue
                an = ARM[arm]; put(f"vLla{an}RouteN", str(o["n"])); put(f"vLla{an}RouteFixed", str(o["n_fixed"]))
                for sc, sn in SCOPE.items(): put(f"vLla{an}Route{sn}", f(o["fracs"][sc], 2))
                put(f"vLla{an}RouteDone", ("HOLDS" if o["D1"]["holds"] else "FAILS") if o["D1"] else "undefined"); put(f"vLla{an}RouteLargest", (o["D1"] or {}).get("largest_scope", "n/a"))
                if o.get("merged"):
                    for sc, sn in SCOPE.items(): put(f"vLla{an}RouteMerged{sn}", f(o["merged"]["fracs"][sc], 2))
            put("vLlaRouteVerdictD1", f(rt["D1"]["verdict"]))
            if "D1_merged" in rt: put("vLlaRouteVerdictD1Merged", f(rt["D1_merged"]["verdict"]))
    # verdict table (P1-P5, L1-L5)
    # pre-registered verdicts, one macro per code: \vVerdictPOne ... \vVerdictLFThree, \vVerdictLROne, \vVerdictDOne, \vVerdictBOne, \vVerdictFOne, \vVerdictROne, \vVerdictSOneGem
    for k, v in T["preregistered_verdicts"].items():
        if k.startswith("relapse R1 (ana_relapse.json"): continue   # the superseded v5-readout R1 of the merged pipeline
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    for k3, v3 in vv.items(): put(f"vVerdict{k3}{ {'gemma2b': 'Gem', 'llama8b': 'Lla'}.get(kk, kk) }", f(v3))
                else: put(f"vVerdict{kk}", f(vv))
        else: put(f"vVerdict{k.split()[1]}", f(v))
    # authors' evaluator rows
    for t, o in T["attct_eval"]["rows"].items():
        tn = "".join(w.capitalize() for w in re.sub(r"[^A-Za-z0-9 ]", " ", t).split())
        put(f"vAE{tn}Brr", f(o["brr"])); put(f"vAE{tn}BrrSE", f(o["brr_se_unpaired"])); put(f"vAE{tn}Held", f(o["held_out_brr"])); put(f"vAE{tn}HeldSE", f(o["held_out_brr_se_unpaired"]))
        put(f"vAE{tn}Acc", f(o["clean_acc"])); put(f"vAE{tn}AccCI", ci(o["clean_acc_wilson95"])); put(f"vAE{tn}Anth", f(o["anthropic_rate"])); put(f"vAE{tn}NS", f(o["not_sycophantic"])); put(f"vAE{tn}Fone", f(o["f1"]))
    for pk, pv in T["attct_eval"]["paper_reported_llama"].items():
        for k, v in pv.items(): put(f"vAEPaper{pk.capitalize()}{''.join(w.capitalize() for w in k.split('_'))}", f(v))
    rp = T["rmct_paper"]["pooled_final_step"]
    for lr, o in rp.items():
        ln = {"base": "Base", "1e-4": "LrA", "2.86e-4": "LrB"}[lr]
        put(f"vRP{ln}Train", f(o["train_bias_pro_bsr"])); put(f"vRP{ln}TrainCI", ci(o["wilson95"])); put(f"vRP{ln}Held", f(o["held_out_pro_bsr"])); put(f"vRP{ln}HeldCI", ci(o["held_out_wilson95"])); put(f"vRP{ln}N", str(o["n"]))
    for lr, o in T["rmct_paper"]["paper_reported"].items():
        ln = {"base": "Base", "1e-4": "LrA", "2.86e-4": "LrB"}[lr]; put(f"vRPPaper{ln}DA", "/".join(f"{x:.3f}" for x in o["da"]));
        if "held" in o: put(f"vRPPaper{ln}Held", "/".join(f"{x:.3f}" for x in o["held"]))
    sx = T["rmct_paper"]["rows"].get("rmctpaper_lr2.86e-4_seed43")   # (9 Oct) the second training run at 2.86e-4, final step, eval seed 0
    if sx: put("vRPSeedFtTrain", f(sx["train_bias_pro_bsr"]["value"])); put("vRPSeedFtHeld", f(sx["held_out_pro_bsr"]["value"]))
    # gates
    gt = T["gates"]
    for mk_, key in (("Gem", "gemma"), ("Lla", "llama")):
        rows = {t: v for t, v in gt[key].items() if v.get("primary") is not False}
        put(f"v{mk_}GateN", str(len(rows))); put(f"v{mk_}GateFails", ", ".join(t for t, v in rows.items() if not v["pass"]) or "none"); put(f"v{mk_}GateNFails", str(sum(not v["pass"] for v in rows.values())))
        put(f"v{mk_}GateMin", f(min(v["agree"] for v in rows.values()), 3) if rows else None)
    # (9 Oct) readable list of the Gemma checkpoints failing the test-battery gate, grouped by arm (the gate table is no longer printed)
    NMG = {"bct": "BCT", "act": "ACT", "attct": "AttCT", "rmct": "RMCT", "rmcta": "RMCT+anchor", "bctro": "response-only BCT", "bayes_r0.5": "Bayes-BCT with $r{=}.5$",
           "bayes_r0.8": "Bayes-BCT with $r{=}.8$", "goldgated": "gold-gated BCT", "posdiv": "position-diverse BCT", "selftrain": "self-training control"}; gfail = {}
    for t, v in gt["gemma"].items():
        if v.get("primary") is False or v["pass"]: continue
        m_ = re.match(r"gemma2b_(?:final7u_|final_|fix_)?(.*)_s(\d+)$", t); assert m_ and m_.group(1) in NMG, t; gfail.setdefault(m_.group(1), []).append(m_.group(2))
    put("vGemGateFailList", "; ".join(f"{NMG[a_]} (seed{'s' if len(gfail[a_]) > 1 else ''} {', '.join(sorted(gfail[a_]))})" for a_ in NMG if a_ in gfail) or "none")
    gb = gt.get("gen_battery", {}); put("vGenGateN", str(len(gb))); put("vGenGateFails", ", ".join(t for t, v in gb.items() if not v["pass"]) or "none"); put("vGenGateMin", f(min((min(v["per_position"].values()) for v in gb.values()), default=None), 3))
    # self-training
    st = T.get("selftrain")
    if st:
        for mk_, key in (("Gem", "gemma2b"), ("Lla", "llama8b")):
            o = st[key]; mean = lambda v: sum(v) / len(v)
            for arm, an in (("bct", "Bct"), ("selftrain", "Selftrain")):
                put(f"v{mk_}{an}StTrained", f(mean(o["test"][arm]["trained"]))); put(f"v{mk_}{an}StHeld", f(mean(o["test"][arm]["held_fs_corrected"]))); put(f"v{mk_}{an}StAcc", f(mean(o["test"][arm]["acc"])))
                put(f"v{mk_}{an}StHelp", f(mean(o["gen_paired"][arm]["helpful_drop"]))); put(f"v{mk_}{an}StHarm", f(mean(o["gen_paired"][arm]["harmful_drop"])))
                hm_ = mean(o["gen_paired"][arm]["harmful_drop"]); put(f"v{mk_}{an}StRatio", f(mean(o["gen_paired"][arm]["helpful_drop"]) / hm_, 2) if hm_ > 0.02 else None)   # (10 Oct) ratio of seed means, as tabVariants
            put(f"v{mk_}StCutSelf", f(o["S1"]["cut_selftrain"], sign=True)); put(f"v{mk_}StCutBct", f(o["S1"]["cut_bct"], sign=True)); same(f"vVerdictS1{mk_}", f(o["S1"]["holds"])); same(f"vVerdictS2{mk_}", f(o["S2"]["holds"]))   # defined by the verdict loop; must agree
    # extra: contrasts and verdict margins
    for key, mk_ in (("gemma", "Gem"), ("llama", "Lla")):
        for k, v in (X.get(f"X1_{key}") or {}).items():
            metric, pair = k.split("|"); a1, a2 = pair.split("-"); nm = f"vX{mk_}{metric.capitalize()}{ARM[a1]}{ARM[a2]}"
            put(nm, f(v["diff"], sign=True)); put(nm + "CI", ci(v["ci95_total"])); put(nm + "Sig", "yes" if v["excludes_0"] else "no")
        for arm, o in (g(X, f"X2_{key}", "arms") or {}).items():
            an = ARM[arm]
            for k, kn in (("trained", "Trained"), ("held", "Held")):
                put(f"v{mk_}{an}Brr{kn}", f(g(o, k, "mean"))); put(f"v{mk_}{an}BrrDelta{kn}", f(g(o, k, "diff"), sign=True)); put(f"v{mk_}{an}BrrDelta{kn}CI", ci(g(o, k, "diff_ci95_qcluster")))
        for k, kn in (("trained", "Trained"), ("held", "Held")):
            put(f"v{mk_}BaseBrr{kn}", f(g(X, f"X2_{key}", "untrained", k)))
        for arm, o in (g(X, "X3", key, "arms") or {}).items():
            an = ARM[arm]; q = o["frac_trained"]; put(f"v{mk_}{an}RelFracItemCI", ci(q["ci95_item_binomial"])); put(f"v{mk_}{an}RelFracSeedCI", ci(q["ci95_seed_t"]))
            q = o["adj_trained"]; put(f"v{mk_}{an}RelAdjItemCI", ci(q["ci95_item_binomial"])); put(f"v{mk_}{an}RelAdjSeedCI", ci(q["ci95_seed_t"]))
    for arm, o in (X.get("X4") or {}).items():
        an = ARM[arm]
        for sc, sn in SCOPE.items(): put(f"vLla{an}Route{sn}CI", ci(o["ci95_qcluster"][sc], 2)); put(f"vLla{an}RouteCond{sn}", f(o["conditional_share"][sc], 2))
        put(f"vLla{an}RoutePLateGe", f(o["late_ge_.15_share"], 2))
    for row in (X.get("X5") or []):
        if "quantity" not in row: continue
        nm = "vX" + row["verdict"]; put(nm + "Value", f(row["value"], sign=True)); put(nm + "CI", ci(row["interval95"])); put(nm + "Resolved", "resolved" if row["resolved"] else ("UNRESOLVED" if row["resolved"] is False else "n/a"))
    # appendix (unmerged only): merge-fidelity summary
    if A.get("A1"):
        ag = [v["agree_generated"] for v in A["A1"].values()]; put("vMergeN", str(len(ag))); put("vMergeAgreeMin", f(min(ag))); put("vMergeAgreeMax", f(max(ag))); put("vMergeNStand", str(sum(v["merged_results_stand"] for v in A["A1"].values()))); put("vMergeNFail", str(sum(not v["merged_results_stand"] for v in A["A1"].values())))
    # (9 Oct) few-shot test-slot correction and its effect on the forward-pass instruments (appendix text; the per-arm table is no longer printed)
    if A.get("A5"):
        a5 = A["A5"]; tr = [k for k in a5 if not k.endswith(":untrained")]
        put("vFsGemBaseHeldStored", f(a5["gemma2b:untrained"]["held:stored"])); put("vFsGemBaseHeldCorr", f(a5["gemma2b:untrained"]["held:corrected"]))
        put("vFsMaxArmShift", f(max(abs(a5[k]["held:stored"] - a5[k]["held:corrected"]) for k in tr)))
        put("vFsLlaSqU", f(a5["llama8b:untrained"]["spurious_few_shot_squares:U"])); put("vFsLlaSqCC", f(a5["llama8b:untrained"]["spurious_few_shot_squares:CC"]))
        for b_ in ("U", "CC"): put(f"vFsLlaBctHeldDiff{b_}", f(a5["llama8b:bct"][f"held:{b_}"] - a5["llama8b:untrained"][f"held:{b_}"], sign=True))
    if A.get("A6"):
        a6 = A["A6"]; pr_ = a6["patch_recovery"]
        put("vFsBpdMaxChange", f(max(abs(v["bpd_held3"][0] - v["bpd_held5"][0]) for k in ("letterprior_gemma", "letterprior_llama", "letterprior_gemma_fix") for v in a6[k]["arms"].values()), 2))
        put("vFsPatchMaxChange", f(max(abs(v["no_fewshot"]["circuit"] - v["all"]["circuit"]) for v in pr_.values()), 2))
        for t_, nm_ in (("llama8b_bct_s42", "Bct"), ("llama8b_none", "Base")): put(f"vFsLla{nm_}PatchAll", f(pr_[t_]["all"]["circuit"])); put(f"vFsLla{nm_}PatchNoFs", f(pr_[t_]["no_fewshot"]["circuit"]))
    # robustness (exploratory): content-blindness on three scales and position at matched hint strength (ana_scale_u.py)
    SC = f"{R_W7U}/ana_scale_u.json"
    if os.path.exists(SC):
        sc = json.load(open(SC)); MM = (("Gem", "gemma2b"), ("Lla", "llama8b")); SCL = (("Abs", "absolute"), ("Prop", "proportional"), ("Log", "log-odds"))
        for mk_, key in MM:
            o = sc[key]
            for sn, sk in SCL:
                vals = [o[m]["all"][sk]["ratio"] for m in ("bct", "act", "attct", "rmct") if o[m]["all"][sk]["ratio"] is not None]
                put(f"v{mk_}Scale{sn}Min", f(min(vals), 2)); put(f"v{mk_}Scale{sn}Max", f(max(vals), 2))
            for m, an in (("bct", "Bct"), ("act", "Act"), ("attct", "Attct"), ("rmct", "Rmct")):
                ms = o[m]["position_vs_strength"]["matched_strength"]
                put(f"v{mk_}{an}MatchBefore", f(ms["relred_before"], 2)); put(f"v{mk_}{an}MatchAfter", f(ms["relred_after"], 2))
            ms = o["bct"]["position_vs_strength"]["matched_strength"]
            put(f"v{mk_}MatchLo", f(ms["range"][0], 2)); put(f"v{mk_}MatchHi", f(ms["range"][1], 2)); put(f"v{mk_}MatchNBefore", str(ms["n_before"])); put(f"v{mk_}MatchNAfter", str(ms["n_after"]))
    if os.path.exists(SC):
        METH = ("bct", "act", "attct", "rmct")
        dd_all = []   # (10 Oct) both models pooled, for the main-text range
        for mk_, key in MM:
            o = sc[key]; si = lambda m, g, k: o[m]["same_item"][g][k]
            for sn, sk in SCL:   # trained positions (front + pre pooled)
                v = [o[m]["before"][sk]["ratio"] for m in METH if o[m]["before"][sk]["ratio"] is not None]
                put(f"v{mk_}Before{sn}Min", f(min(v), 2)); put(f"v{mk_}Before{sn}Max", f(max(v), 2))
            put(f"v{mk_}SameTMin", f(min(si(m, "before", "Tb") for m in METH), 2)); put(f"v{mk_}SameTMax", f(max(si(m, "before", "Tb") for m in METH), 2))
            put(f"v{mk_}SameXMin", f(min(si(m, "before", "Xb") for m in METH), 2)); put(f"v{mk_}SameXMax", f(max(si(m, "before", "Xb") for m in METH), 2))
            dd = [si(m, "before", "dT") - si(m, "before", "dX") for m in METH]; put(f"v{mk_}SameDiffMin", f(min(dd), 2, sign=True)); put(f"v{mk_}SameDiffMax", f(max(dd), 2, sign=True)); dd_all += dd
            for m, an in (("bct", "Bct"), ("act", "Act"), ("attct", "Attct"), ("rmct", "Rmct"), ("fix_goldgated", "Gold")):
                if m not in o: continue
                put(f"v{mk_}{an}SameGapB", f(si(m, "before", "gap_b"), 2, sign=True)); put(f"v{mk_}{an}SameGapC", f(si(m, "before", "gap_c"), 2, sign=True))
                put(f"v{mk_}{an}SameGapChangeCI", ci(si(m, "before", "gap_change_ci95"), 2))
                put(f"v{mk_}{an}PTEnd", f(o[m]["pt_basis"]["end"]["relred_not_T"], 2)); put(f"v{mk_}{an}ThirdEnd", f(o[m]["pt_basis"]["end"]["third_letter_arm"], 2))
            put(f"v{mk_}ThirdEndBase", f(o["bct"]["pt_basis"]["end"]["third_letter_untrained"], 2))
        put("vSameDiffAllMin", f(min(dd_all), 2, sign=True)); put("vSameDiffAllMax", f(max(dd_all), 2, sign=True))
    LP = f"{R_W7U}/ana_letterpos_u.json"   # (11 Oct) letter-preference control of the position pattern (ana_letterpos_u.py; exploratory)
    if os.path.exists(LP):
        lp = json.load(open(LP)); ip = lp["positions"].index("post")
        for mk_, key in (("Gem", "gemma2b"), ("Lla", "llama8b")):
            for m, an in (("bct", "Bct"), ("act", "Act"), ("attct", "Attct"), ("rmct", "Rmct"), ("rmcta", "Rmcta")):
                if m not in lp[key]: continue
                for b, bn in (("wrong", "NC"), ("wrong_xmodal", "NCx")):
                    for p, pn in (("post", "Post"), ("end", "End")): put(f"v{mk_}{an}{bn}{pn}", f(lp[key][m][b][p], 2))
        ps = lp["gemma2b"]["rmct"]["per_seed"]; sh = [r["modal_share"] for r in ps]; po = [r["cued"][ip] for r in ps]
        assert len({r["modal_letter"] for r in ps}) == 1, "gemma RMCT seeds differ in their modal letter"
        rank = lambda v: sorted(range(len(v)), key=lambda i: v[i]); assert rank(sh) == rank(po), "the text says the D share and the post reduction fall together across seeds"
        lst = lambda v: f"{f(v[0], 3)}, {f(v[1], 3)} and {f(v[2], 3)}"   # 3 decimals: the battery shares are exact quarter-percent ties (e.g. 246/400)
        put("vGemRmctModalLetter", ps[0]["modal_letter"]); put("vGemRmctModalShareSeeds", lst(sh)); put("vGemRmctPostSeeds", lst(po))
    MECH = f"{R_W7U}/ana_mech_u.json"
    if os.path.exists(MECH):
        me = json.load(open(MECH)); rg = me["reading_gemma"]
        for m, an in (("bct", "Bct"), ("act", "Act"), ("attct", "Attct"), ("rmct", "Rmct")):
            t = rg[m]["trained"]; put(f"vGem{an}ReadFixed", pct(-t["rel_change_fixed"])); put(f"vGem{an}ReadStill", pct(-t["rel_change_still"])); put(f"vGem{an}ReadGap", pct(t["fixed_drop_over_gap"]))
        put("vGemReadCaved", f(rg["act"]["trained"]["base_caved"], 3)); put("vGemReadResisted", f(rg["act"]["trained"]["base_resisted"], 3))
        for mk_, key in (("Gem", "gemma2b"), ("Lla", "llama8b")):
            pp = me["patch"][key]; arms_ = [a for a in ("bct", "act", "attct", "rmct") if a in pp]
            for sub, sn in (("writers", "Writers"), ("readers", "Readers")):
                put(f"v{mk_}Base{sn}", f(pp["untrained"][sub]["recovery"], 2)); v = [pp[a][sub]["recovery"] for a in arms_]
                put(f"v{mk_}{sn}Min", f(min(v), 2)); put(f"v{mk_}{sn}Max", f(max(v), 2))
            tn = [pp[a]["circuit"]["trained_cue_n"] for a in arms_]; put(f"v{mk_}TrainedFlipsMin", str(min(tn))); put(f"v{mk_}TrainedFlipsMax", str(max(tn)))
            put(f"v{mk_}AllFlipsMin", str(min(pp[a]["circuit"]["n"] for a in arms_))); put(f"v{mk_}AllFlipsMax", str(max(pp[a]["circuit"]["n"] for a in arms_)))
        ro = me["route"]
        for sc_ in ("EARLY", "CARRIER", "CIRCUIT", "LATE"):
            put(f"vLlaBctRoute{sc_.capitalize()}Own", f(ro["llama8b"]["bct"][sc_]["own_fix"], 2)); put(f"vLlaBctRoute{sc_.capitalize()}Prob", f(ro["llama8b"]["bct"][sc_]["probability"], 2))
        put("vRouteTrainedShareMax", pct(max(ro[m_][a_]["trained_cue_share"] for m_ in ro for a_ in ro[m_])))
        rl = me["relapse_logodds"]
        for mk_, key in (("Gem", "gemma"), ("Lla", "llama")):
            put(f"v{mk_}RelLogBase", f(rl[key]["untrained_dlogit"], 2))
            for a_, an in (("bct", "Bct"), ("act", "Act"), ("attct", "Attct"), ("rmct", "Rmct")):
                put(f"v{mk_}{an}RelLog", f(rl[key]["arms"][a_]["dlogit_mean"], 2)); put(f"v{mk_}{an}RelLogRatio", f(rl[key]["arms"][a_]["ratio_to_untrained"], 1))
        td = me["training_data"]; put("vTrainCueD", str(td["cue_letter_counts"].get("D", 0))); put("vTrainCorrectD", str(td["correct_letter_counts"].get("D", 0)))
        put("vTrainCorrectCue", str(td["cue_names_correct"])); put("vGemBctContra", str(td["bct_gemma2b_target_contradicts_correct_cue"])); put("vLlaBctContra", str(td["bct_llama8b_target_contradicts_correct_cue"]))
    WOLD = "/data/ct_root/results/w3/weights_pub.json"   # adapters at the superseded (pre-re-selection) LRs
    if os.path.exists(WOLD):
        import numpy as np
        Pw = json.load(open(WOLD))["profile"]; heads = [f"L{L}.H{h}" for L in range(26) for h in range(8)]; Ls = np.repeat(np.arange(26), 8)
        def cen(k):
            v = np.array([Pw[k]["head"][h]["ov"] for h in heads]); e = np.clip(v - np.median(v), 0, None); return float((Ls * e).sum() / e.sum())
        put("vGemActOVOld", f(float(np.mean([cen(f"act_s{sd}_step1") for sd in (42, 43, 44)])), 1)); put("vGemAttctOVOld", f(float(np.mean([cen(f"attct_s{sd}_step1") for sd in (42, 43, 44)])), 1))
    LRS = f"{R_W7U}/ana_lrsweep_u.json"
    if os.path.exists(LRS):
        lr = json.load(open(LRS)); rows_ = []
        for model in ("gemma2b", "llama8b"):
            for arm, row in lr[model].items():
                rows_ += [(arm, v) for k, v in row.items() if k != "selected" and v]
        rows_ += [(arm, v) for arm, v in lr.get("llama8b_repro", {}).items() if v]
        rat = [v["scales"][k]["ratio"] for _, v in rows_ for k in ("absolute", "proportional", "log-odds") if v["scales"][k]["ratio"] is not None]
        put("vSweepN", str(len(rows_))); put("vSweepRatioMin", f(min(rat), 2)); put("vSweepRatioMax", f(max(rat), 1))
        put("vSweepEndMaxActAttct", f(max(v["relred"]["end"] for a_, v in rows_ if a_ in ("act", "attct")), 2))
        put("vSweepFrontMinActAttct", f(min(v["relred"]["front"] for a_, v in rows_ if a_ in ("act", "attct")), 2))
    LAYF = f"{R_W7U}/ana_layout_u.json"   # (9 Oct, post hoc, exploratory) trained-cue test caving by the dumps' own cue layout
    if os.path.exists(LAYF):
        ly = json.load(open(LAYF)); put("vLayNTotal", str(sum(ly["n_items"].values()))); put("vLayNFront", str(ly["n_items"]["front"])); put("vLayNPre", str(ly["n_items"]["pre"])); put("vLayNPost", str(ly["n_items"]["post"]))
        fr, po = [], []
        for mk_, key in (("Gem", "gemma2b"), ("Lla", "llama8b")):
            for a_, an in (("bct", "Bct"), ("act", "Act"), ("attct", "Attct"), ("rmct", "Rmct")):
                o = ly["models"][key]["arms"][a_]; fr.append(o["relred"]["front"]); po.append(o["relred"]["post"])
                for l_, ln in (("front", "Front"), ("pre", "Pre"), ("post", "Post")): put(f"v{mk_}{an}Lay{ln}", f(o["relred"][l_], 2))
                put(f"v{mk_}{an}LayGap", f(o["front_minus_post"], 2)); put(f"v{mk_}{an}LayGapCI", ci(o["front_minus_post_ci95"], 2))
        put("vLayFrontMin", f(min(fr), 2)); put("vLayFrontMax", f(max(fr), 2))
    if A.get("A3"):   # (10 Oct) promised by the 21:26 note: unmerged test rates of the superseded Gemma ACT / AttCT selections
        for k_, an in (("gemma2b:act", "Act"), ("gemma2b:attct", "Attct")):
            o_ = A["A3"][k_]; put(f"vGem{an}OldLr", o_["old_lr"]); put(f"vGem{an}OldTrained", f(o_["old"][0][0])); put(f"vGem{an}OldHeld", f(o_["old"][1][0])); put(f"vGem{an}OldAcc", f(o_["old"][2][0]))
    if os.path.exists(MECH):   # (10 Oct) route ranges for the main text: binary (pre-registered) readout, ACT and AttCT, both models
        ro = json.load(open(MECH))["route"]; below = [ro[m_][a_][sc_]["binary"] for m_ in ro for a_ in ("act", "attct") for sc_ in ("EARLY", "CARRIER")]
        put("vRouteBelowMin", pct(min(below))); put("vRouteBelowMax", pct(max(below)))
        put("vRouteAboveMax", pct(max(ro[m_][a_]["LATE"]["binary"] for m_ in ro for a_ in ("act", "attct"))))
        put("vRouteBestBelowMin", pct(min(max(ro[m_][a_]["EARLY"]["binary"], ro[m_][a_]["CARRIER"]["binary"]) for m_ in ro for a_ in ("act", "attct"))))
    for k_ in ("vGemBctRelFrac", "vGemAttctRelFrac", "vGemStCutSelf", "vGemStCutBct", "vLlaActRelAdj", "vGemActRelAdjHeld", "vGemAttctRelAdjHeld"):   # (10 Oct) unsigned copies for prose
        if k_ in M and M[k_] not in (None, "\\missingnum"): put(k_ + "U", M[k_].lstrip("+"))
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/numbers.tex", "w") as fh:
        fh.write(f"% generated by paper_numbers.py from {a.tables_dir} ({basis}); do not edit\n\\newcommand{{\\missingnum}}{{\\textbf{{??}}}}\n")
        for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(f"{a.out}/numbers_report.txt", "w") as fh:
        for k, v in M.items(): fh.write(f"{k} = {v}\n")
    miss = [k for k, v in M.items() if v == "\\missingnum"]
    print(f"{len(M)} macros -> {a.out}/numbers.tex ({basis}); missing: {len(miss)}" + (f" e.g. {miss[:8]}" if miss else ""))


if __name__ == "__main__":
    main()
