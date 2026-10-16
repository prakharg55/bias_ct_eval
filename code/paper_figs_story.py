#!/usr/bin/env python3
"""
paper_figs_story.py (10 Oct 2026): main-text figures of the CT paper, one per step of the story, each with a takeaway title.
CPU only, seconds. Helpers (logged data access, lint) come from paper_figs.py.

  F0_hero      concept figure: what a good fix would do vs what all four methods do, and the three shared findings, each with a
               small inset drawn from the data (same-question cue following; relative reduction by position; circuit numbers)
  F5_sameq     same questions, cue before the options: following of a cue naming the correct answer and of a cue naming another
               wrong option, original -> after CT (arrows), per method and model
  F6_position  relative reduction of harmful-cue following at the four cue positions (lines with 95% CIs), per model
  F7_circuit   (a) circuit patch recovery vs the best random head set; (b, c) layer swap: share of the trained model's fixes
               that each block of attention layers reproduces (pre-registered binary readout), per model
  F8_variants  (a) Gemma BCT variants: helpful vs harmful drop; (b) relative reduction at front and end for BCT, position-diverse
               and response-only BCT
  F4_relapse   trained-cue harmful-cue following at steps 0 and 125 of benign fine-tuning (seed mean, band = seed SD)

Every plotted value is read through a paper_figs.Source (one per JSON file), which logs its JSON pointer; at the end every logged
value is re-read from disk and compared. Usage: paper_figs_story.py [--out_dir DIR] [--only F0,F5]
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_figs import Source, add_ax, lint, fmt_tick, RC, COLOR, MARK, NAME, INK, INK2, AXIS, GRID, WASH, C_UNTRAINED, DASH, W_FULL, W_COL

R = "/data/ct_root/results"
PT, SCJ, MEJ = f"{R}/paper_u/paper_tables.json", f"{R}/w7u/ana_scale_u.json", f"{R}/w7u/ana_mech_u.json"
MK = {"gemma": "gemma2b", "llama": "llama8b"}; MODELS = ["gemma", "llama"]; MNAME = {"gemma": "Gemma-2-2B-it", "llama": "Llama-3.1-8B-Instruct"}
METHODS = ["bct", "act", "attct", "rmct"]; POS = ["front", "pre", "post", "end"]
C_RIGHT, C_WRONG = "#1e7b3c", "#c43a2f"          # cue names the correct answer / a wrong answer
C_AFTER, C_BEFORE = "#33414d", "#bfc4c9"          # after CT / original (hero bars)
FS_T, FS_L, FS_A = 9.0, 7.8, 7.4                  # takeaway title, labels, in-panel annotations (pt on the page)
RCS = dict(RC, **{"font.size": FS_A, "axes.labelsize": FS_L, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": FS_A,
                  "axes.titlesize": FS_L, "lines.markersize": 4.8})


# ------------------------------------------------------------------------------------------------------------------- helpers
class Srcs:
    """The three logged sources; .begin(fig) starts a figure in all of them."""
    def __init__(self): self.P, self.C, self.M = Source(PT), Source(SCJ), Source(MEJ)
    def begin(self, name):
        for s in (self.P, self.C, self.M): s.begin(name)
    def values(self, name): return sum((s.log.get(name, []) for s in (self.P, self.C, self.M)), [])
    def audit(self): return sum(s.audit() for s in (self.P, self.C, self.M))


def wrap_to(fig, s, width_in, fs, weight="bold"):
    """Greedy word wrap measured with the renderer (inches)."""
    r = fig.canvas.get_renderer(); lines, cur = [], ""
    for w in s.split():
        t = (cur + " " + w).strip(); tt = fig.text(0, 0, t, fontsize=fs, fontweight=weight); wd = tt.get_window_extent(r).width / fig.dpi; tt.remove()
        if wd <= width_in or not cur: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]


def title_block(fig, s, x_in, top_in, width_in, fs=FS_T, color=INK, weight="bold"):
    """Takeaway title (wrapped) with its top at top_in; returns its height in inches."""
    W, H = fig.get_size_inches(); lines = wrap_to(fig, s, width_in, fs, weight)
    fig.text(x_in / W, top_in / H, "\n".join(lines), ha="left", va="top", fontsize=fs, fontweight=weight, color=color, linespacing=1.18)
    return len(lines) * fs * 1.18 / 72.0


def n_lines(s, width_in, fs=FS_T):
    f = plt.figure(figsize=(W_FULL, 1)); n = len(wrap_to(f, s, width_in, fs)); plt.close(f); return n


def legend_row(fig, handles, labels, x_in, y_in, loc="upper left", ncol=None, fs=FS_A, **kw):
    W, H = fig.get_size_inches()
    return fig.legend(handles, labels, loc=loc, bbox_to_anchor=(x_in / W, y_in / H), ncol=ncol or len(handles), fontsize=fs, frameon=False,
                      handlelength=kw.pop("handlelength", 1.6), columnspacing=kw.pop("columnspacing", 1.1), handletextpad=0.45, borderaxespad=0, **kw)


def finish(fig, name, out_dir, S, man):
    bad = lint(fig, name); files = [f"{out_dir}/{name}.pdf", f"{out_dir}/{name}.png"]
    fig.savefig(files[0]); fig.savefig(files[1], dpi=300); w, h = fig.get_size_inches(); plt.close(fig); vals = S.values(name)
    man[name] = {"files": files, "size_in": [round(float(w), 3), round(float(h), 3)], "n_values": len(vals), "lint": bad, "values": vals}
    print(f"{name}: {w:.2f} x {h:.2f} in, {len(vals)} values, {len(bad)} lint problem(s) -> {files[0]}")


def ax_style(ax, ygrid=True):
    if ygrid: ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)


# --------------------------------------------------------------------------------------------------------------- data blocks
def same_item(S, panel):
    """{(model, method): (Tb, Tc, Xb, Xc)} on the same questions, cue before the options (front + pre)."""
    out = {}
    for m in MODELS:
        for a in METHODS:
            b = (MK[m], a, "same_item", "before")
            out[(m, a)] = tuple(S.C.val(panel, f"{m}:{a}", q, *b, k) for k, q in (("Tb", "original, cue names T: P(answer T)"), ("Tc", "after CT, cue names T: P(answer T)"),
                                                                                  ("Xb", "original, cue names X: P(answer X)"), ("Xc", "after CT, cue names X: P(answer X)")))
    return out


def relred(S, panel, m, a, p, ci=False):
    base = (m, "gen", "arms", a, "relred", p)
    v = S.P.val(panel, f"{m}:{a}", f"relative reduction of harmful-cue following at {p}, seed mean", *base, "mean")
    return (v, tuple(S.P.val(panel, f"{m}:{a}", f"relative reduction at {p}, 95% CI", *base, "ci95"))) if ci else v


# ------------------------------------------------------------------------------------------------------------------- F0 hero
def fig_hero(S, out_dir, man):
    """Three plain panels in the style of the other figures: (a) same-question cue following, original vs after CT, correct vs
    wrong cue (mean over 4 methods x 2 models, whiskers = range); (b) share of harmful-cue following removed at the four positions,
    one line per method (mean over the 2 models, whiskers = range), because this finding separates RMCT from the others; (c) circuit patch recovery, original vs after CT (mean, range),
    best random set."""
    name = "F0_hero"; S.begin(name)
    cells = [(m, a) for m in MODELS for a in METHODS]
    V = {k: [S.C.val("(a) same questions", f"{m}:{a}", f"cue before options: {k}", MK[m], a, "same_item", "before", k) for m, a in cells] for k in ("Tb", "Tc", "Xb", "Xc")}
    RR = {a: [[relred(S, "(b) position", m, a, p) for m in MODELS] for p in POS] for a in METHODS}   # one line per method, mean over models
    pu = [S.P.val("(c) circuit", f"{m}:untrained", "circuit patch recovery", m, "causal", "untrained", "circuit") for m in MODELS]
    pa = [S.P.val("(c) circuit", f"{m}:{a}", "circuit patch recovery, pooled", m, "causal", "arms", a, "pooled", "circuit") for m, a in cells]
    rnd = [S.P.val("(c) circuit", f"{m}:untrained", "best random head set", m, "causal", "untrained", "best_random") for m in MODELS] + \
          [S.P.val("(c) circuit", f"{m}:{a}", "best random head set (max over seeds)", m, "causal", "arms", a, "best_random_max") for m, a in cells]
    st = lambda v: (float(np.mean(v)), float(np.min(v)), float(np.max(v)))
    TA, TB, TC = "(a) Correct cues lose their effect too", "(b) The fix depends on the cue position", "(c) The same heads still carry the cue errors"
    with plt.rc_context(RCS):
        W = W_FULL; botm, ax_h, leg_h = 0.34, 1.18, 0.22; left, gap = 0.44, 0.66; pw = (W - left - 2 * gap - 0.06) / 3
        X0 = [left + k * (pw + gap) for k in range(3)]; TW = [min(pw + gap - 0.12, W - (x - 0.38) - 0.03) for x in X0]   # titles may span the gap, not the edge
        nt = max(n_lines(t, w) for t, w in zip((TA, TB, TC), TW)); H = botm + ax_h + leg_h + nt * FS_T * 1.18 / 72 + 0.10
        fig = plt.figure(figsize=(W, H)); X = [left + k * (pw + gap) for k in range(3)]
        for x, t, w in zip(X, (TA, TB, TC), TW): title_block(fig, t, x - 0.38, H - 0.04, w)
        # (a) original vs after CT, correct cue (green) vs wrong cue (red)
        ax = add_ax(fig, X[0], botm, pw, ax_h); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 0.75); ax_style(ax)
        ax.set_yticks([0, 0.25, 0.5, 0.75]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75"]); ax.set_ylabel("model follows the cue", labelpad=1.5)
        for gi, (kt, kx) in enumerate((("Tb", "Xb"), ("Tc", "Xc"))):
            for dx, k, c in ((-0.18, kt, C_RIGHT), (0.18, kx, C_WRONG)):
                mu, lo, hi = st(V[k]); ax.bar(gi + dx, mu, width=0.34, color=c, lw=0, zorder=2); ax.plot([gi + dx] * 2, [lo, hi], color=INK, lw=0.8, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["original", "after CT"])
        legend_row(fig, [Patch(fc=C_RIGHT, lw=0), Patch(fc=C_WRONG, lw=0)], ["correct-answer cue", "wrong-answer cue"], X[0] - 0.38, botm + ax_h + leg_h - 0.02, ncol=2, handlelength=0.9, columnspacing=0.7)
        # (b) share of harmful-cue following removed at the four positions, one line per method (mean over models, whiskers = range)
        ylo = min(-0.3, min(v for a in METHODS for vs in RR[a] for v in vs) - 0.06)
        ax = add_ax(fig, X[1], botm, pw, ax_h); ax.set_xlim(-0.4, 3.4); ax.set_ylim(ylo, 1.12); ax_style(ax)
        ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"]); ax.set_ylabel("harmful-cue following\nremoved", labelpad=0.8, linespacing=1.0)
        ax.axvspan(-0.32, 0.32, color=WASH, lw=0, zorder=0); ax.axhline(0, color=AXIS, lw=0.8, zorder=1); ax.axvline(1.5, color=AXIS, lw=0.8, ls=DASH, zorder=1)
        ax.annotate("trained", (0, 1.12), xytext=(0, -1), textcoords="offset points", ha="center", va="top", fontsize=7.0, color=INK2)
        ax.set_xticks(range(4)); ax.set_xticklabels(["front", "pre", "post", "end"])
        for xc, lab in ((0.5, "before options"), (2.5, "after options")):
            ax.annotate(lab, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -12), textcoords="offset points", ha="center", va="top", fontsize=7.0, color=INK)
        for k, a in enumerate(METHODS):
            xs = np.arange(4) + (k - 1.5) * 0.07; sb = [st(vs) for vs in RR[a]]
            for x, (_, lo, hi) in zip(xs, sb): ax.plot([x, x], [lo, hi], color=COLOR[a], lw=0.8, zorder=2)
            ax.plot(xs, [mu for mu, _, _ in sb], color=COLOR[a], lw=1.3, marker=MARK[a], ms=4.2, mfc=COLOR[a], mec="white", mew=0.6, zorder=3 + 0.01 * k)
        hs = [Line2D([], [], color=COLOR[a], lw=1.3, marker=MARK[a], ms=4.2, mfc=COLOR[a], mec="white", mew=0.6) for a in METHODS]
        legend_row(fig, hs, [NAME[a] for a in METHODS], X[1] - 0.06, botm + ax_h + leg_h - 0.02, ncol=4, handlelength=0.9, columnspacing=0.45)
        # (c) circuit patch recovery, original vs after CT, with the best random head set
        ax = add_ax(fig, X[2], botm, pw, ax_h); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 1.0); ax_style(ax)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.set_ylabel("cue errors corrected\nby patching the circuit", labelpad=0.8, linespacing=1.0)
        rg = (rnd[:len(MODELS)], rnd[len(MODELS):])   # best random head set: original model, and the methods after CT
        for gi, (v, c) in enumerate(((pu, C_BEFORE), (pa, C_AFTER))):
            mu, lo, hi = st(v); ax.bar(gi, mu, width=0.5, color=c, lw=0, zorder=2); ax.plot([gi, gi], [lo, hi], color=INK, lw=0.8, zorder=3)
            ax.plot([gi - 0.2, gi + 0.2], [max(rg[gi])] * 2, color=INK, lw=1.6, solid_capstyle="butt", zorder=4)
        legend_row(fig, [Line2D([], [], color=INK, lw=1.6)], ["patching random heads"], X[2] - 0.38, botm + ax_h + leg_h - 0.02, ncol=1, handlelength=1.2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["original", "after CT"])
        finish(fig, name, out_dir, S, man)


# ------------------------------------------------------------------------------------------------------- F5 same questions
def fig_sameq(S, out_dir, man):
    name = "F5_sameq"; S.begin(name); D = same_item(S, "same questions")
    TITLE = "A cue that names the correct answer loses its effect as much as a wrong one"
    with plt.rc_context(RCS):
        W = W_COL; nt = n_lines(TITLE, W - 0.08); row = 0.165; head = 0.17; nrow = 8
        ax_h = nrow * row + 2 * head; botm = 0.40; leg_h = 0.22; H = botm + ax_h + leg_h + nt * FS_T * 1.18 / 72 + 0.10
        fig = plt.figure(figsize=(W, H)); title_block(fig, TITLE, 0.04, H - 0.04, W - 0.08)
        lab_w = 0.56; ax = add_ax(fig, lab_w, botm, W - lab_w - 0.16, ax_h); ax.set_xlim(0, 0.75); ax.set_ylim(ax_h, 0)
        ys, labels = [], []
        for mi, m in enumerate(MODELS):
            y0 = mi * (4 * row + head) + head
            ax.annotate({"gemma": "Gemma", "llama": "Llama"}[m], (0, y0 - 0.03), xycoords=("axes fraction", "data"), xytext=(-lab_w * 72 + 3, 0), textcoords="offset points", ha="left", va="bottom", fontsize=7.4, color=INK, fontweight="bold")
            for ai, a in enumerate(METHODS):
                y = y0 + (ai + 0.5) * row; ys.append(y); labels.append(NAME[a])
                if ai % 2 == 1: ax.axhspan(y - row / 2, y + row / 2, color=WASH, lw=0, zorder=0)
                Tb, Tc, Xb, Xc = D[(m, a)]
                for (b, c, col, dy) in ((Tb, Tc, C_RIGHT, -0.045), (Xb, Xc, C_WRONG, 0.045)):
                    ax.annotate("", (c, y + dy), xytext=(b, y + dy), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, mutation_scale=6.5, shrinkA=2.2, shrinkB=0.5), zorder=3)
                    ax.plot([b], [y + dy], ls="none", marker="o", ms=3.8, mfc="white", mec=col, mew=1.0, zorder=4)
                if mi == 0 and ai == 0:   # direct labels on the first pair of arrows
                    ax.annotate("original", (Tb, y - 0.045), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.0, color=INK2)
                    ax.annotate("after CT", (Tc, y - 0.045), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.0, color=INK2)
        ax.set_yticks(ys); ax.set_yticklabels(labels, color=INK); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.grid(axis="x")
        ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75"]); ax.set_xlabel("model gives the letter the cue names", labelpad=2)
        hr = Line2D([], [], color=C_RIGHT, lw=1.3); hw = Line2D([], [], color=C_WRONG, lw=1.3)
        legend_row(fig, [hr, hw], ["correct-answer cue", "wrong-answer cue"], 0.04, botm + ax_h + leg_h - 0.02, ncol=2, handlelength=1.4, columnspacing=1.0)
        finish(fig, name, out_dir, S, man)


# --------------------------------------------------------------------------------------------------------------- F6 position
def fig_position(S, out_dir, man):
    name = "F6_position"; S.begin(name)
    D = {m: {a: [relred(S, f"{m}", m, a, p, ci=True) for p in POS] for a in METHODS} for m in MODELS}
    TITLE = "The fix fades for cues placed after the answer options"
    allv = [x for m in MODELS for a in METHODS for v, c in D[m][a] for x in (v, *c)]; ylo = min(-0.3, min(allv) - 0.04)
    with plt.rc_context(RCS):
        W = W_COL; nt = n_lines(TITLE, W - 0.08); botm, ax_h, leg_h = 0.46, 1.30, 0.24; sub_h = 0.17
        H = botm + ax_h + sub_h + leg_h + nt * FS_T * 1.18 / 72 + 0.08; fig = plt.figure(figsize=(W, H)); title_block(fig, TITLE, 0.04, H - 0.04, W - 0.08)
        left, gap = 0.70, 0.10; pw = (W - left - gap - 0.06) / 2
        hs = [Line2D([], [], color=COLOR[a], lw=1.3, marker=MARK[a], ms=4.2, mfc=COLOR[a], mec="white", mew=0.6) for a in METHODS]
        legend_row(fig, hs, [NAME[a] for a in METHODS], 0.04, botm + ax_h + sub_h + leg_h - 0.03, ncol=4, columnspacing=0.9, handlelength=1.5)
        for mi, m in enumerate(MODELS):
            ax = add_ax(fig, left + mi * (pw + gap), botm, pw, ax_h); ax.set_xlim(-0.4, 3.4); ax.set_ylim(ylo, 1.16)
            tk = [v for v in (-0.25, 0, 0.25, 0.5, 0.75, 1.0) if v >= ylo - 1e-9]; ax.set_yticks(tk); ax.set_yticklabels([fmt_tick(v) for v in tk] if mi == 0 else [""] * len(tk)); ax_style(ax)
            ax.axvspan(-0.32, 0.32, color=WASH, lw=0, zorder=0); ax.axhline(0, color=AXIS, lw=0.8, zorder=1); ax.axvline(1.5, color=AXIS, lw=0.8, ls=DASH, zorder=1)
            ax.annotate("trained", (0, 1.16), xytext=(0, -1), textcoords="offset points", ha="center", va="top", fontsize=7.0, color=INK2)
            ax.set_xticks(range(4)); ax.set_xticklabels(["front", "pre", "post", "end"])
            for xc, lab in ((0.5, "before\noptions"), (2.5, "after\noptions")):
                ax.annotate(lab, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -14), textcoords="offset points", ha="center", va="top", fontsize=7.0, color=INK, linespacing=1.0)
            if mi == 0: ax.set_ylabel("harmful-cue\nfollowing removed", labelpad=1.5, linespacing=1.0)
            fig.text((left + mi * (pw + gap)) / W, (botm + ax_h + 0.03) / H, MNAME[m], fontsize=7.4, color=INK2, va="bottom", style="italic")
            for k, a in enumerate(METHODS):
                xs = np.arange(4) + (k - 1.5) * 0.06; ys = [v for v, _ in D[m][a]]
                for x, (v, c) in zip(xs, D[m][a]): ax.plot([x, x], c, color=COLOR[a], lw=0.8, zorder=2)
                ax.plot(xs, ys, color=COLOR[a], lw=1.3, marker=MARK[a], ms=4.2, mfc=COLOR[a], mec="white", mew=0.6, zorder=3 + 0.01 * k)
        finish(fig, name, out_dir, S, man)


# --------------------------------------------------------------------------------------------------------------- F7 circuit
SCOPE = {"gemma": [("EARLY", "0–6"), ("CARRIER", "7–12"), ("CIRCUIT", "13–18"), ("LATE", "19–25")],
         "llama": [("EARLY", "0–8"), ("CARRIER", "9–14"), ("CIRCUIT", "15–18"), ("LATE", "19–31")]}


def fig_circuit(S, out_dir, man):
    name = "F7_circuit"; S.begin(name); rows = ["untrained"] + METHODS
    Pc, Rc = {}, {}
    for m in MODELS:
        for a in rows:
            b = (m, "causal", "untrained") if a == "untrained" else (m, "causal", "arms", a, "pooled")
            Pc[(m, a)] = S.P.val("(a) patching", f"{m}:{a}", "circuit patch recovery" + ("" if a == "untrained" else ", pooled over seeds"), *b, "circuit")
            Rc[(m, a)] = S.P.val("(a) patching", f"{m}:{a}", "best random head set" + ("" if a == "untrained" else " (max over seeds)"), *((m, "causal", "untrained", "best_random") if a == "untrained" else (m, "causal", "arms", a, "best_random_max")))
    swm = ["bct", "act", "attct"]
    Sw = {(m, a, sc): S.M.val(f"layer swap {m}", f"{m}:{a}", f"share of fixes reproduced by swapping {sc} (binary readout)", "route", MK[m], a, sc, "binary") for m in MODELS for a in swm for sc, _ in SCOPE[m]}
    TA, TB = "(a) Patching the circuit still corrects most leftover cue errors", "(b, c) The fix sits mainly in layers before the circuit"
    with plt.rc_context(RCS):
        W = W_FULL; botm, ax_h, sub_h, leg_h = 0.46, 1.22, 0.17, 0.24
        wa = 2.18; left_a = 0.60; gap_ab = 0.46; left_b = left_a + wa + gap_ab; gap_bc = 0.12; wb = (W - left_b - gap_bc - 0.06) / 2
        nt = max(n_lines(TA, wa + 0.2), n_lines(TB, 2 * wb + gap_bc)); H = botm + ax_h + sub_h + leg_h + nt * FS_T * 1.18 / 72 + 0.08
        fig = plt.figure(figsize=(W, H)); top = H - 0.04
        title_block(fig, TA, left_a - 0.36, top, wa + 0.30); title_block(fig, TB, left_b - 0.36, top, 2 * wb + gap_bc + 0.3)
        # (a) grouped bars: methods on x, Gemma dark / Llama light, best random set as a black tick
        ax = add_ax(fig, left_a, botm, wa, ax_h); ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_ylim(0, 1.0); ax_style(ax)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.set_ylabel("cue errors corrected\nby patching the circuit", labelpad=1.5, linespacing=1.0)
        for i, a in enumerate(rows):
            c = C_UNTRAINED if a == "untrained" else COLOR[a]
            for dx, m, alpha in ((-0.19, "gemma", 1.0), (0.19, "llama", 0.45)):
                ax.bar(i + dx, Pc[(m, a)], width=0.36, color=c, alpha=alpha, lw=0, zorder=2)
                ax.plot([i + dx - 0.15, i + dx + 0.15], [Rc[(m, a)]] * 2, color=INK, lw=1.4, solid_capstyle="butt", zorder=4)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([NAME[a] for a in rows])
        hs = [Patch(fc=INK2, lw=0), Patch(fc=INK2, alpha=0.45, lw=0), Line2D([], [], color=INK, lw=1.4)]
        legend_row(fig, hs, ["Gemma", "Llama", "patching random heads"], left_a - 0.36, botm + ax_h + sub_h + leg_h - 0.05, ncol=3, handlelength=1.1, columnspacing=0.9)
        # (b, c) layer swap per model
        hs = [Patch(fc=COLOR[a], lw=0) for a in swm]
        legend_row(fig, hs, [NAME[a] for a in swm], left_b, botm + ax_h + sub_h + leg_h - 0.05, ncol=3, handlelength=1.1, columnspacing=0.9)
        for mi, m in enumerate(MODELS):
            ax = add_ax(fig, left_b + mi * (wb + gap_bc), botm, wb, ax_h); ax.set_xlim(-0.55, 3.55); ax.set_ylim(0, 1.0); ax_style(ax)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"] if mi == 0 else [""] * 5)
            if mi == 0: ax.set_ylabel("share of the fix reproduced", labelpad=1.5)
            ax.axvspan(1.5, 2.5, color="#efe9f8", lw=0, zorder=0)
            for j, (sc, lay) in enumerate(SCOPE[m]):
                for k, a in enumerate(swm): ax.bar(j + (k - 1) * 0.26, Sw[(m, a, sc)], width=0.24, color=COLOR[a], lw=0, zorder=2)
            ax.set_xticks(range(4)); ax.set_xticklabels([lay + ("\ncircuit" if sc == "CIRCUIT" else "") for sc, lay in SCOPE[m]], linespacing=1.0)
            fig.text((left_b + mi * (wb + gap_bc)) / W, (botm + ax_h + 0.03) / H, MNAME[m], fontsize=7.4, color=INK2, va="bottom", style="italic")
        fig.text((left_b + wb + gap_bc / 2) / W, 0.04 / H, "attention layers copied from the trained model", fontsize=FS_L, color=INK, ha="center", va="bottom")
        finish(fig, name, out_dir, S, man)


# -------------------------------------------------------------------------------------------------------------- F8 variants
VARS = [("bct", "BCT"), ("bayes_r0.5", "Bayes-BCT, r=.5"), ("bayes_r0.8", "Bayes-BCT, r=.8"), ("goldgated", "gold-gated BCT"),
        ("bctro", "response-only BCT"), ("posdiv", "position-diverse BCT"), ("selftrain", "self-training control")]


def fig_variants(S, out_dir, man):
    name = "F8_variants"; S.begin(name); D = {}
    for a, _ in VARS:
        b = ("selftrain", "gemma2b", "gen_paired", "selftrain") if a == "selftrain" else ("gemma", "fix6", "arms", a); n = len(S.P.raw(*b, "helpful_drop"))
        D[a] = tuple(S.P.seedstat("(a) gemma", a, f"paired {k}, seed mean", [b + (k, i) for i in range(n)])[0] for k in ("helpful_drop", "harmful_drop"))
    E = []
    for m, a, lab in (("gemma", "bct", "BCT"), ("gemma", "posdiv", "position-\ndiverse"), ("llama", "bct", "BCT"), ("llama", "posdiv", "position-\ndiverse")):
        b = (m, "fix6", "arms", a, "relred_mean")
        E.append((m, lab, S.P.val("(b) position", f"{m}:{a}", "relative reduction at front, seed mean", *b, "front"), S.P.val("(b) position", f"{m}:{a}", "relative reduction at end, seed mean", *b, "end")))
    TA, TB = "(a) Every variant loses helpful and harmful cue following together", "(b) Training cues at all four positions extend the fix to the end position"
    LH = FS_T * 1.18 / 72
    with plt.rc_context(RCS):
        W = W_COL; nta, ntb = n_lines(TA, W - 0.08), n_lines(TB, W - 0.08); row = 0.142; ha = len(VARS) * row; hb = 0.66; lab_w = 1.16
        b_bot = 0.52; b_leg = b_bot + hb + 0.08; b_tt = b_leg + 0.20 + 0.04 + ntb * LH
        a_bot = b_tt + 0.08 + 0.38; a_leg = a_bot + ha + 0.06; H = a_leg + 0.20 + 0.04 + nta * LH + 0.04
        fig = plt.figure(figsize=(W, H))
        title_block(fig, TA, 0.04, H - 0.04, W - 0.08)
        legend_row(fig, [Patch(fc=C_RIGHT, lw=0), Patch(fc=C_WRONG, lw=0)], ["helpful-cue following lost", "harmful-cue following lost"], 0.04, a_leg, loc="lower left", ncol=2, handlelength=0.9, columnspacing=0.7)
        ax = add_ax(fig, lab_w, a_bot, W - lab_w - 0.16, ha); ax.set_ylim(len(VARS) - 0.5, -0.5); ax.set_xlim(0, 0.75)
        for i, (a, lab) in enumerate(VARS):
            if i % 2 == 1: ax.axhspan(i - 0.5, i + 0.5, color=WASH, lw=0, zorder=0)
            hd, hm = D[a]; ax.barh(i - 0.17, hd, height=0.32, color=C_RIGHT, lw=0, zorder=2); ax.barh(i + 0.17, hm, height=0.32, color=C_WRONG, lw=0, zorder=2)
        ax.set_yticks(range(len(VARS))); ax.set_yticklabels([lab for _, lab in VARS], color=INK); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.grid(axis="x")
        ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75"]); ax.set_xlabel("drop in cue following (Gemma)", labelpad=1.5)
        title_block(fig, TB, 0.04, b_tt, W - 0.08)
        legend_row(fig, [Patch(fc=C_BEFORE, lw=0), Patch(fc=C_AFTER, lw=0)], ["cue at front", "cue at the end"], 0.56, b_leg, loc="lower left", ncol=2, handlelength=0.9, columnspacing=0.7)
        ax = add_ax(fig, 0.52, b_bot, W - 0.60, hb); ax.set_xlim(-0.55, len(E) - 0.45); ax.set_ylim(-0.3, 1.0); ax_style(ax); ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"]); ax.set_ylabel("harmful-cue\nfollowing removed", labelpad=1.5, linespacing=1.0)
        for i, (m, lab, fr, en) in enumerate(E):
            ax.bar(i - 0.18, fr, width=0.34, color=C_BEFORE, lw=0, zorder=2); ax.bar(i + 0.18, en, width=0.34, color=C_AFTER, lw=0, zorder=2)
        ax.axvline(1.5, color=AXIS, lw=0.8, zorder=1)
        ax.set_xticks(range(len(E))); ax.set_xticklabels([e[1] for e in E], linespacing=1.0)
        for xc, m in ((0.5, "gemma"), (2.5, "llama")):
            ax.annotate(MNAME[m], (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -24), textcoords="offset points", ha="center", va="top", fontsize=7.2, color=INK, fontweight="bold")
        finish(fig, name, out_dir, S, man)


# ------------------------------------------------------------------------------------------------------------- F9 efficacy
def fig_efficacy(S, out_dir, man):
    name = "F9_efficacy"; S.begin(name); rows = ["untrained"] + METHODS; D = {}
    for m in MODELS:
        for a in rows:
            for st in ("trained", "held"):
                if a == "untrained": D[(m, a, st)] = (S.P.val(m, a, f"{st} flip|knew", m, "T1", "untrained", st, "value"), None)
                else:
                    b = (m, "T1", "arms", a, st); ck = "ci95_qcluster" if S.P.has(*b, "ci95_qcluster") else "ci95"
                    D[(m, a, st)] = (S.P.val(m, a, f"{st} flip|knew, seed mean", *b, "mean"), tuple(S.P.val(m, a, f"{st} flip|knew, 95% CI", *b, ck)))
    TITLE = "Every method cuts the trained cue, but transfer to new cue types is weaker"
    with plt.rc_context(RCS):
        W = W_COL; nt = n_lines(TITLE, W - 0.08); row = 0.215; ax_h = len(rows) * row; botm, sub_h, leg_h = 0.40, 0.17, 0.22
        H = botm + ax_h + sub_h + leg_h + nt * FS_T * 1.18 / 72 + 0.08; fig = plt.figure(figsize=(W, H)); title_block(fig, TITLE, 0.04, H - 0.04, W - 0.08)
        lab_w, gap = 0.58, 0.10; pw = (W - lab_w - gap - 0.08) / 2
        legend_row(fig, [Patch(fc=INK2, lw=0), Patch(fc=INK2, alpha=0.35, lw=0)], ["trained cue", "five held-out cue types (mean)"], 0.04, botm + ax_h + sub_h + leg_h - 0.02, ncol=2, handlelength=0.9, columnspacing=0.8)
        for mi, m in enumerate(MODELS):
            ax = add_ax(fig, lab_w + mi * (pw + gap), botm, pw, ax_h); ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xlim(0, 0.7)
            for i, a in enumerate(rows):
                c = C_UNTRAINED if a == "untrained" else COLOR[a]
                if i % 2 == 1: ax.axhspan(i - 0.5, i + 0.5, color=WASH, lw=0, zorder=0)
                for dy, st, al in ((-0.18, "trained", 1.0), (0.18, "held", 0.35)):
                    v, ci = D[(m, a, st)]; ax.barh(i + dy, v, height=0.34, color=c, alpha=al, lw=0, zorder=2)
                    if ci: ax.plot(ci, [i + dy] * 2, color=INK, lw=0.7, zorder=3)
            for st, ls, c in (("trained", (0, (2, 1.5)), INK2), ("held", (0, (2, 1.5)), "#a9a8a3")):
                ax.plot([D[(m, "untrained", st)][0]] * 2, [-0.5, len(rows) - 0.5], color=c, lw=0.9, ls=ls, zorder=4)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAME[a] for a in rows] if mi == 0 else [""] * len(rows), color=INK)
            ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.grid(axis="x")
            ax.set_xticks([0, 0.25, 0.5]); ax.set_xticklabels(["0", "0.25", "0.5"])
            fig.text((lab_w + mi * (pw + gap)) / W, (botm + ax_h + 0.03) / H, MNAME[m], fontsize=7.4, color=INK2, va="bottom", style="italic")
        fig.text((lab_w + pw + gap / 2) / W, 0.04 / H, "harmful-cue following on the test set", fontsize=FS_L, color=INK, ha="center", va="bottom")
        finish(fig, name, out_dir, S, man)


# --------------------------------------------------------------------------------------------------------------- F4 relapse
def fig_relapse(S, out_dir, man):
    name = "F4_robustness_mechanism"; S.begin(name); steps = ["0", "125"]; D = {}
    for m in MODELS:
        D[m] = {"untrained": [(S.P.val(m, "untrained", f"trained-cue flip|knew (generated), step {st}", m, "relapse_u", "untrained_gen", st, "trained"), None) for st in steps]}
        for a in METHODS:
            n = len(S.P.raw(m, "relapse_u", "arms", a, "per_seed"))
            D[m][a] = [S.P.seedstat(m, a, f"trained-cue flip|knew (generated), step {st}", [(m, "relapse_u", "arms", a, "per_seed", i, "gen", st, "trained") for i in range(n)]) for st in steps]
    TITLE = "Benign fine-tuning raises cue following in every model, the original one included"
    with plt.rc_context(RCS):
        W = W_COL; nt = n_lines(TITLE, W - 0.08); botm, ax_h, sub_h, leg_h = 0.38, 1.12, 0.17, 0.22
        H = botm + ax_h + sub_h + leg_h + nt * FS_T * 1.18 / 72 + 0.08; fig = plt.figure(figsize=(W, H)); title_block(fig, TITLE, 0.04, H - 0.04, W - 0.08)
        left, gap = 0.40, 0.14; pw = (W - left - gap - 0.06) / 2; rows = ["untrained"] + METHODS
        hs = [Line2D([], [], color=(C_UNTRAINED if a == "untrained" else COLOR[a]), lw=1.4 if a != "untrained" else 1.8, marker="o" if a == "untrained" else MARK[a], ms=4.0,
                     mfc=(C_UNTRAINED if a == "untrained" else COLOR[a]), mec="white", mew=0.6) for a in rows]
        legend_row(fig, hs, [NAME[a] for a in rows], 0.04, botm + ax_h + sub_h + leg_h - 0.03, ncol=5, columnspacing=0.55, handlelength=1.0)
        ymax = max(v + (sd or 0) for m in MODELS for a in rows for v, sd in D[m][a]); ytop = max(0.8, np.ceil((ymax + 0.02) / 0.2) * 0.2)
        for mi, m in enumerate(MODELS):
            ax = add_ax(fig, left + mi * (pw + gap), botm, pw, ax_h); ax.set_xlim(-12, 137); ax.set_ylim(0, ytop); ax.grid(axis="y")
            tk = np.arange(0, ytop + 1e-9, 0.2); ax.set_yticks(tk); ax.set_yticklabels([fmt_tick(v) for v in tk] if mi == 0 else [""] * len(tk))
            ax.set_xticks([0, 125]); ax.set_xlabel("fine-tuning steps", labelpad=1.5)
            if mi == 0: ax.set_ylabel("follows the trained cue", labelpad=1.5)
            fig.text((left + mi * (pw + gap)) / W, (botm + ax_h + 0.03) / H, MNAME[m], fontsize=7.4, color=INK2, va="bottom", style="italic")
            for k, a in enumerate(rows):
                c = C_UNTRAINED if a == "untrained" else COLOR[a]; mu = np.array([v for v, _ in D[m][a]]); sd = [s for _, s in D[m][a]]
                if all(s is not None for s in sd): ax.fill_between([0, 125], mu - np.array(sd), mu + np.array(sd), color=c, alpha=0.14, lw=0, zorder=2)
                ax.plot([0, 125], mu, color=c, lw=1.8 if a == "untrained" else 1.3, marker="o" if a == "untrained" else MARK[a], ms=4.2, mfc=c, mec="white", mew=0.6, zorder=3 + 0.01 * k)
        finish(fig, name, out_dir, S, man)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out_dir", default=f"{R}/paper_u/figs"); ap.add_argument("--only", default="")
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out_dir, exist_ok=True); S = Srcs(); man = {}
    only = {x.strip().upper() for x in a.only.split(",") if x.strip()}
    for key, fn in (("F0", fig_hero), ("F9", fig_efficacy), ("F5", fig_sameq), ("F6", fig_position), ("F7", fig_circuit), ("F8", fig_variants), ("F4", fig_relapse)):
        if not only or key in only: fn(S, a.out_dir, man)
    n = S.audit()
    json.dump({"sources": [PT, SCJ, MEJ], "script": os.path.abspath(__file__), "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "figures": man},
              open(f"{a.out_dir}/figs_story_manifest.json", "w"), indent=1)
    print(f"audit: {n} logged values re-read and equal; {sum(len(v['lint']) for v in man.values())} lint problem(s); {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
