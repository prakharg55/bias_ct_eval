"""An earlier set of figures (F0 to F4), drawn from paper_tables.json alone.

figures.py reuses its helpers (logged data access, plot styles and checks). analysis/run_all.py still
runs this script and writes its figures to $CT_ROOT/results/paper_u/figs, the default folder of
figures.py.

Usage: python -m ctcue.paper.figures_v1 --tables <dir>/paper_tables.json --out_dir <dir>
Output: <out_dir>/<figure>.pdf and .png, and <out_dir>/figs_manifest.json
"""

import sys

sys.dont_write_bytecode = True  # nothing may be written under /home (quota)
import argparse, json, os, textwrap, time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from ctcue.paths import CT_ROOT

PAPER = f"{CT_ROOT}/results/paper"
W_FULL, W_COL = 6.3, 3.03  # ACL/ARR: figure* and single column widths (inches)
MIN_TEXT, MIN_ANNOT, TITLE_WRAP = 7.0, 6.5, 28

# -------------------------------------------------------------------------------------------------------------
# display registry
MODEL = {"gemma": "Gemma-2-2B-it", "llama": "Llama-3.1-8B-Instruct"}
MODEL_ORDER = ["gemma", "llama"]
# used where the full name does not fit a narrow panel (F4 c)
MODEL_SHORT = {"llama": "Llama-3.1-8B"}
NAME = {
    "untrained": "original",
    "bct": "BCT",
    "act": "ACT",
    "attct": "AttCT",
    "rmct": "RMCT",
    "rmcta": "RMCT+anchor (expl.)",
    "bayes_r0.5": "Bayes r=.5",
    "bayes_r0.8": "Bayes r=.8",
    "goldgated": "gold-gated",
    "posdiv": "position-diverse",
    "bctro": "response-only BCT",
    "selftrain": "self-training control",
}
SHORT = {"rmcta": "RMCT+anchor"}  # point labels (F2)
ORDER = ["bct", "act", "attct", "rmct", "rmcta"]  # display order of CT methods
VAR_ORDER = ["bayes_r0.5", "bayes_r0.8", "goldgated", "posdiv"]  # display order of training-data variants
# variants shown in F3 as contrasting dashed lines
POS_VARIANTS = ["posdiv"]
TRAINED_POS = "front"  # hint position used in training
HERO_EXCLUDE = ["rmcta"]
# (10 Oct) the exploratory RMCT+anchor arm is reported in the appendix tables only
F4_EXCLUDE = ["rmcta"]
# (10 Oct) the captions carry the takeaway; no in-figure title                                           # exploratory arms are not in the hero figure (they are in F1-F4 and the tables)
NO_TAKEAWAY = {"F0_hero", "F3_position_bound", "F4_robustness_mechanism"}
# main-text F2 shows the training-data variants (the methods' panel is the hero's panel a)
F2_VARIANTS_ONLY = True
# main-text F4 shows relapse only (patch recovery is the hero's panel c)
F4_RELAPSE_ONLY = True
# (9 Oct) F4 at single-column width, no takeaway title (the caption states it)
F4_COLUMN = True
# takeaway title: wrap width (chars; 8 pt bold DejaVu fits ~95 in 6.3 in) and line height (in)
TAKE_WRAP, TAKE_LINE = 86, 0.145
TAKEAWAY = {  # one sentence per figure, stating what the figure shows (not what it plots)
    "F0_hero": "Four objectives, one signature: helpful cues are suppressed about as much as harmful ones (a), the fix works mainly before the answer options, where it was trained (b), and the circuit that carries residual cue-following stays in place (c).",
    "F1_caving": "Every method cuts following of the trained cue; transfer to held-out cues is partial, largest for RMCT, and absent for BCT on Llama.",
    "F2_content_blind": "Changing BCT's training data trades efficacy for a smaller helpful drop along the same line: no variant reaches the selective region.",
    "F3_position_bound": "The fix is largest before the answer options (front, the trained position, and pre) and fades after them; position-diverse data flattens it on Gemma and reverses it on Llama.",
    "F4_robustness_mechanism": "Benign fine-tuning raises trained-cue caving in every model; for BCT, ACT and AttCT roughly as much as in the untrained model.",
}


def take_height(name):
    if name in NO_TAKEAWAY:
        return 0.0
    return TAKE_LINE * len(textwrap.wrap(TAKEAWAY.get(name, ""), TAKE_WRAP, break_long_words=False)) + (
        0.06 if name in TAKEAWAY else 0.0
    )


def draw_takeaway(fig, name):
    """Bold takeaway sentence across the top of the figure (8 pt, left-aligned, wrapped); the space was reserved by take_height()."""
    if name not in TAKEAWAY or name in NO_TAKEAWAY:
        return
    W, H = fig.get_size_inches()
    lines = textwrap.wrap(TAKEAWAY[name], TAKE_WRAP, break_long_words=False)
    fig.text(
        0.02 / W,
        (H - 0.02) / H,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        linespacing=1.25,
    )


# Categorical hues from the dataviz reference palette; this five-hue subset is the best all-pairs subset of
# the eight documented hues (validate_palette.py --pairs all --surface "#ffffff": CVD dE 13.0 >= 8, normal dE
# 16.3 >= 15). Sibling methods get neighbouring hues. secondary encoding: methods differ by marker as well as
# colour
MARK = {"bct": "o", "act": "s", "attct": "^", "rmct": "v", "rmcta": "P"}
COLOR = {"bct": "#2a78d6", "act": "#4a3aa7", "attct": "#e87ba4", "rmct": "#008300", "rmcta": "#eda100"}
# unknown future methods: spare hue + own marker (secondary encoding)
SPARE = [("#e34948", "s"), ("#1baf7a", "^"), ("#eb6834", "v")]
C_UNTRAINED, C_VARIANT = "#6f6e69", "#1a1a19"
INK, INK2, AXIS, GRID, WASH = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f3f2ef"
DASH = (0, (3.2, 1.8))

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.6,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "axes.labelpad": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "lines.linewidth": 1.3,
    "lines.markersize": 4.6,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.handlelength": 1.7,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.5,
    "legend.borderaxespad": 0.0,
    "legend.borderpad": 0.0,
    "legend.labelspacing": 0.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def ordered(keys, order):
    keys = list(dict.fromkeys(keys))
    return [k for k in order if k in keys] + sorted(k for k in keys if k not in order)


# ----------------------------------------------------------------------------------------------------------------
# data access
class Source:
    """The only door to the numbers. val / ci / seedstat log (figure, panel, series, quantity, value, JSON pointer)."""

    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path))
        self.fig = None
        self.log = {}

    @staticmethod
    def ptr(keys):
        return "".join("/" + str(k).replace("~", "~0").replace("/", "~1") for k in keys)

    def begin(self, fig):
        self.fig = fig
        self.log[fig] = []

    def raw(self, *keys):
        """Unlogged lookup (structure discovery, labels). Raises KeyError if the path is missing or null."""
        x = self.d
        for k in keys:
            if isinstance(x, list):
                if not (isinstance(k, int) and -len(x) <= k < len(x)):
                    raise KeyError(self.ptr(keys))
                x = x[k]
            elif isinstance(x, dict) and k in x:
                x = x[k]
            else:
                raise KeyError(self.ptr(keys))
        if x is None:
            raise KeyError(self.ptr(keys))
        return x

    def has(self, *keys):
        try:
            self.raw(*keys)
            return True
        except KeyError:
            return False

    def val(self, panel, series, quantity, *keys):
        v = self.raw(*keys)
        self.log[self.fig].append(
            {"panel": panel, "series": series, "quantity": quantity, "value": v, "json_path": self.ptr(keys)}
        )
        return v

    def ci(self, panel, series, quantity, base, names):
        """First available interval among `names` under `base` (e.g. question-cluster CI, else item CI). Returns (lo, hi) or None."""
        for n in names:
            if self.has(*base, n):
                lo, hi = self.val(panel, series, quantity + " [" + n + "]", *base, n)
                return float(lo), float(hi)
        return None

    def seedstat(self, panel, series, quantity, keylists):
        """Seed mean and SD (ddof 1, None for one seed) of JSON leaves; the JSON stores the seeds, not these two numbers."""
        vals = [float(self.raw(*k)) for k in keylists]
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        self.log[self.fig].append(
            {
                "panel": panel,
                "series": series,
                "quantity": quantity,
                "value": mean,
                "sd": sd,
                "derived": f"mean and SD (ddof 1) of {len(vals)} seed values",
                "inputs": vals,
                "json_paths": [self.ptr(k) for k in keylists],
            }
        )
        return mean, sd

    def audit(self):
        """Re-read the file from disk and check every logged value against its pointer(s)."""
        d = json.load(open(self.path))
        n = 0

        def get(p):
            x = d
            for k in p.split("/")[1:]:
                k = k.replace("~1", "/").replace("~0", "~")
                x = x[int(k)] if isinstance(x, list) else x[k]
            return x

        for fig, rows in self.log.items():
            for r in rows:
                if "json_path" in r:
                    assert get(r["json_path"]) == r["value"], (fig, r)
                else:
                    v = [float(get(p)) for p in r["json_paths"]]
                    assert v == r["inputs"] and abs(float(np.mean(v)) - r["value"]) < 1e-15, (fig, r)
                n += 1
        return n


class Arms:
    """Methods / variants discovered from the JSON, with stable colours: colour follows the arm, never its rank in a panel."""

    def __init__(self, S):
        self.S = S
        sec = lambda m, k: list(S.raw(m, k, "arms")) if S.has(m, k, "arms") else []
        self.models = ordered(
            [
                m
                for m, v in S.d.items()
                if isinstance(v, dict) and any(k in v for k in ("T1", "gen", "causal"))
            ],
            MODEL_ORDER,
        )
        self.methods = {
            m: ordered(
                (
                    sec(m, "T1") + sec(m, "causal") + sec(m, "relapse")
                    if S.has(m, "T1", "arms")
                    else sec(m, "gen") + sec(m, "causal") + sec(m, "relapse")
                ),
                ORDER,
            )
            for m in self.models
        }
        self.variants = {
            m: ordered([a for a in sec(m, "gen") if a not in self.methods[m]], VAR_ORDER) for m in self.models
        }
        allm = ordered([a for m in self.models for a in self.methods[m]], ORDER)
        self.allvar = ordered([a for m in self.models for a in self.variants[m]], VAR_ORDER)
        self.style = {"untrained": (C_UNTRAINED, "o")}
        unknown = [a for a in allm if a not in COLOR]
        for a in allm:
            if a in COLOR:
                self.style[a] = (COLOR[a], MARK.get(a, "o"))
            else:
                i = unknown.index(a)
                self.style[a] = SPARE[i] if i < len(SPARE) else (INK2, "PXhp*"[(i - len(SPARE)) % 5])
                print(
                    f"NOTE: method '{a}' is not in COLOR; using spare style {self.style[a]} - add it to COLOR / NAME / ORDER"
                )
        for a in self.allvar:
            self.style[a] = (C_VARIANT, "D")
        self.all_methods = allm

    def color(self, a):
        return self.style[a][0]

    def marker(self, a):
        return self.style[a][1]

    def label(self, a, models=None):
        """Display name; if the arm exists for only some of the plotted models, say so."""
        s = NAME.get(a, a)
        if models and a != "untrained":
            have = [m for m in models if a in self.methods[m] + self.variants[m]]
            if 0 < len(have) < len(models):
                s += " (" + ", ".join(MODEL.get(m, m).split("-it")[0] for m in have) + " only)"
        return s


# ------------------------------------------------------------------------------------------------------------
# drawing helpers
def add_ax(fig, x, y, w, h):
    """Axes from a rectangle in INCHES (origin bottom-left of the figure)."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def dot(ax, x, y, color, marker="o", filled=True, ms=4.6, z=5):
    """Filled marker with a surface-colour ring, or a hollow marker with a surface-colour face."""
    if filled:
        ax.plot([x], [y], ls="none", marker=marker, ms=ms, mfc=color, mec="white", mew=0.6, zorder=z)
    else:
        ax.plot([x], [y], ls="none", marker=marker, ms=ms - 0.6, mfc="white", mec=color, mew=1.0, zorder=z)


def span(ax, xs, ys, color, lw=0.9, z=3):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=z)


def title_height(sub_lines):
    return (3.5 + 9.0 * sub_lines + 11.0) / 72.0


def wrap_title(s):
    return textwrap.wrap(s, TITLE_WRAP, break_long_words=False) if s else []


def panel_title(ax, head, sub=None):
    """Left-aligned two-level title: bold head (8 pt), optional wrapped descriptor (7.5 pt). `head` may be a list of alternatives
    (full name first): the first one that fits the panel width (plus 0.1 in of the gap to its right) is used, else the last.
    """
    lines = wrap_title(sub)
    if not isinstance(head, str):
        r = ax.figure.canvas.get_renderer()
        room = ax.get_window_extent(r).width + 0.1 * ax.figure.dpi
        cands = list(dict.fromkeys(head))
        for head in cands:
            t = ax.text(0, 0, head, fontsize=8.0, fontweight="bold")
            w = t.get_window_extent(r).width
            t.remove()
            if w <= room:
                break
    if lines:
        ax.annotate(
            "\n".join(lines),
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, 3.5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=INK2,
            linespacing=1.15,
        )
    ax.annotate(
        head,
        (0, 1),
        xycoords="axes fraction",
        xytext=(0, 3.5 + 9.0 * len(lines)),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
    )


def flow_legend(fig, handles, labels, max_w, loc="upper center", **kw):
    """Figure legend in as few rows as fit into max_w inches. Returns (legend, height in inches)."""
    r = fig.canvas.get_renderer()
    for ncol in range(len(handles), 0, -1):
        leg = fig.legend(handles, labels, ncol=ncol, loc=loc, bbox_to_anchor=(0.5, 0.5), **kw)
        bb = leg.get_window_extent(r)
        if bb.width / fig.dpi <= max_w or ncol == 1:
            return leg, bb.height / fig.dpi
        leg.remove()


def anchor(fig, leg, x_in, y_in):
    W, H = fig.get_size_inches()
    leg.set_bbox_to_anchor((x_in / W, y_in / H), transform=fig.transFigure)


def h_line(color, ls="-", marker=None, **kw):
    return Line2D(
        [],
        [],
        color=color,
        ls=ls,
        lw=kw.pop("lw", 1.3),
        marker=marker,
        ms=kw.pop("ms", 4.6),
        mfc=kw.pop("mfc", color),
        mec=kw.pop("mec", "white"),
        mew=kw.pop("mew", 0.6),
        **kw,
    )


def h_dot(color, marker="o", filled=True, ms=4.6):
    return Line2D(
        [],
        [],
        ls="none",
        marker=marker,
        ms=ms if filled else ms - 0.6,
        mfc=color if filled else "white",
        mec="white" if filled else color,
        mew=0.6 if filled else 1.0,
    )


def nice_ceiling(v, step):
    return float(np.ceil((v - 1e-9) / step) * step)


def nice_floor(v, step):
    return float(np.floor((v + 1e-9) / step) * step)


def fmt_tick(v, signed=False):
    return "0" if abs(v) < 1e-12 else (f"{v:+g}" if signed else f"{v:g}").replace("-", "−")


# -----------------------------------------------------------------------------------------------------
# geometry: marks, text, lint
def mark_boxes(ax, r, kinds=False):
    """[n, 4] display-space boxes (x0, y0, x1, y1) covering the data marks of an axes: every Line2D path (sampled) and every marker.
    With kinds=True also returns an array that is 0 for line samples and 1 for markers."""
    px = ax.figure.dpi / 72.0
    out, kd = [], []
    c = ax.get_window_extent(r)
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        xy = ln.get_transform().transform(
            np.column_stack(
                [np.asarray(ln.get_xdata(orig=False), float), np.asarray(ln.get_ydata(orig=False), float)]
            )
        )
        xy = xy[np.isfinite(xy).all(1)]
        if len(xy) == 0:
            continue
        if ln.get_linestyle() not in ("None", "none", " ", ""):
            hw = max(ln.get_linewidth() * px / 2, 0.5)
            for p, q in zip(xy[:-1], xy[1:]):
                # boxes tile the segment
                pts = p + np.linspace(0, 1, int(np.ceil(np.hypot(*(q - p)) / (2 * hw))) + 1)[:, None] * (
                    q - p
                )
                out.append(np.column_stack([pts - hw, pts + hw]))
                kd.append(np.zeros(len(pts), int))
        if ln.get_marker() not in ("None", "none", " ", "", None):
            hm = ln.get_markersize() * px / 2
            out.append(np.column_stack([xy - hm, xy + hm]))
            kd.append(np.ones(len(xy), int))
    B = np.vstack(out) if out else np.zeros((0, 4))
    K = np.concatenate(kd) if kd else np.zeros(0, int)
    keep = (B[:, 2] > c.x0) & (B[:, 0] < c.x1) & (B[:, 3] > c.y0) & (B[:, 1] < c.y1)
    return (B[keep], K[keep]) if kinds else B[keep]


def text_boxes(t, r, shrink=0.0):
    """Display-space boxes of a text: its bounding box, or squares along its baseline direction if it is rotated off-axis."""
    bb = t.get_window_extent(r)
    rot = t.get_rotation() % 180
    if min(rot % 90, 90 - rot % 90) < 1:
        return np.array([[bb.x0 + shrink, bb.y0 + shrink, bb.x1 - shrink, bb.y1 - shrink]])
    t.set_rotation(0)
    b0 = t.get_window_extent(r)
    t.set_rotation(rot)
    w, h = b0.width, b0.height
    cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
    a = np.radians(rot)
    hh = h / 2 - shrink
    ss = np.linspace(-(w - h) / 2, (w - h) / 2, max(int(w / (h / 2)), 2))
    px_, py_ = cx + ss * np.cos(a), cy + ss * np.sin(a)
    return np.column_stack([px_ - hh, py_ - hh, px_ + hh, py_ + hh])


def hits(A, B):
    """True if any box of A overlaps any box of B."""
    if len(A) == 0 or len(B) == 0:
        return False
    return bool(
        (
            (A[:, None, 0] < B[None, :, 2])
            & (A[:, None, 2] > B[None, :, 0])
            & (A[:, None, 1] < B[None, :, 3])
            & (A[:, None, 3] > B[None, :, 1])
        ).any()
    )


def figure_texts(fig):
    """(text, kind, axes) for every text that is actually drawn; kind 'annot' = free text inside / attached to an axes."""
    out = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
                    out += [(t, "tick", ax) for t in (tk.label1, tk.label2)]
            out += [(axis.label, "label", ax), (axis.offsetText, "label", ax)]
        out += [(t, "title", ax) for t in (ax.title, ax._left_title, ax._right_title)] + [
            (t, "annot", ax) for t in ax.texts
        ]
        if ax.get_legend() is not None:
            out += [(t, "legend", ax) for t in ax.get_legend().get_texts()]
    for leg in fig.legends:
        out += [(t, "legend", None) for t in leg.get_texts()]
    out += [(t, "figtext", None) for t in fig.texts]
    return [(t, k, ax) for t, k, ax in out if t.get_visible() and t.get_text().strip()]


def lint(fig, name):
    """Measure the drawn figure: font sizes, text outside the canvas, text on text, in-panel text on a marker or across a line."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    px = fig.dpi / 72.0
    bad = []
    T = figure_texts(fig)
    boxes = [text_boxes(t, r, shrink=0.5 * px) for t, _, _ in T]
    q = lambda t: repr(t.get_text().replace("\n", " / "))
    for (t, kind, ax), b in zip(T, boxes):
        if t.get_fontsize() < (MIN_ANNOT if kind == "annot" else MIN_TEXT) - 1e-6:
            bad.append(f"font {t.get_fontsize():.1f} pt < minimum: {q(t)} ({kind})")
        bb = t.get_window_extent(r)
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > W or bb.y1 > H:
            bad.append(f"text outside the canvas: {q(t)} ({kind})")
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            if hits(boxes[i], boxes[j]):
                bad.append(f"text overlaps text: {q(T[i][0])} and {q(T[j][0])}")
    marks = {ax: mark_boxes(ax, r) for ax in fig.axes}
    for (t, kind, ax), b in zip(T, boxes):
        if ax is not None and kind in ("annot", "legend") and hits(b, marks[ax]):
            bad.append(f"text on a marker / across a line: {q(t)}")
    for ax in fig.axes:
        for t in ax.texts:
            if (
                any(len(l) > TITLE_WRAP + 4 for l in t.get_text().split("\n"))
                and t.get_fontweight() == "bold"
            ):
                bad.append(f"title line longer than {TITLE_WRAP + 4} characters: {q(t)}")
    for b in bad:
        print(f"  LINT {name}: {b}")
    return bad


def place_point_labels(ax, pts, fs=MIN_ANNOT, avoid=None):
    """Label scatter points without overlaps. pts = [{'x', 'y', 'label', 'ms', 'leader': colour}] in data coordinates.
    Every point gets candidate boxes on rings around it (12 directions x 7 radii; a two-word label may also wrap) and a leader from
    its marker to the box, drawn in the point's colour: where markers overlap the leader colour carries identity, the text stays
    in ink. A candidate pays for distance, for pointing towards its neighbours, for touching a mark (markers, CI lines, reference
    lines) or a fixed label, for `avoid`, and for a leader that runs over another point's marker or across / along a line. Labels
    are then placed greedily, never on another label or across another label's leader; all orders are tried for <= 5 points (six
    heuristic + 200 seeded random orders otherwise) and the cheapest layout is drawn."""
    if not pts:
        return
    fig = ax.figure
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    c = ax.get_window_extent(r)
    n = len(pts)
    marks, kind = mark_boxes(ax, r, kinds=True)
    lines, mk = marks[kind == 0], marks[kind == 1]
    inv = ax.transData.inverted()
    P = ax.transData.transform([(p["x"], p["y"]) for p in pts])
    axbox = np.array([[c.x0, c.y0, c.x1, c.y1]])
    fixed = [b for b in (text_boxes(t, r) for t in ax.texts) if hits(b, axbox)]
    fixed = np.vstack(fixed) if fixed else np.zeros((0, 4))
    count = lambda A_, B_: (
        int(
            (
                (A_[:, None, 0] < B_[None, :, 2])
                & (A_[:, None, 2] > B_[None, :, 0])
                & (A_[:, None, 1] < B_[None, :, 3])
                & (A_[:, None, 3] > B_[None, :, 1])
            )
            .any(0)
            .sum()
        )
        if len(A_) and len(B_)
        else 0
    )

    def measure(s):
        t = ax.text(0, 0, s, fontsize=fs, linespacing=1.05)
        bb = t.get_window_extent(r)
        t.remove()
        return bb.width, bb.height

    cands = []
    for i, p in enumerate(pts):
        forms = [(p["label"],) + measure(p["label"]) + (0.0,)]
        sw = 0.0
        if " " in p["label"]:
            s2 = p["label"].replace(" ", "\n", 1)
            forms.append((s2,) + measure(s2) + (4.0,))
        d = P - P[i]
        near = (np.hypot(*d.T) < 40 * px) & (np.arange(n) != i)
        ctr = d[near].mean(0) if near.any() else np.zeros(2)
        away = float(np.degrees(np.arctan2(-ctr[1], -ctr[0]))) if np.hypot(*ctr) > 1e-6 else None
        others = mk[
            np.hypot((mk[:, 0] + mk[:, 2]) / 2 - P[i, 0], (mk[:, 1] + mk[:, 3]) / 2 - P[i, 1]) > 1.5 * px
        ]
        out = []
        for s, w, h, extra in forms:
            for rad in (11, 14, 18, 23, 29, 36, 44):
                # never axis-aligned: such a leader would read as a longer CI bar
                for ang in (a_ for a_ in np.arange(0, 360, 22.5) if a_ % 90):
                    co, si = np.cos(np.radians(ang)), np.sin(np.radians(ang))
                    Wd = w + sw
                    cx = P[i, 0] + rad * px * co + (np.sign(co) * Wd / 2 if abs(co) > 0.3 else 0)
                    cy = P[i, 1] + rad * px * si + (np.sign(si) * h / 2 if abs(si) > 0.3 else 0)
                    box = np.array([[cx - Wd / 2, cy - h / 2, cx + Wd / 2, cy + h / 2]])
                    pad = box + np.array([-1.2, -1.2, 1.2, 1.2]) * px
                    if (
                        box[0, 0] < c.x0 + px
                        or box[0, 1] < c.y0 + px
                        or box[0, 2] > c.x1 - px
                        or box[0, 3] > c.y1 - px
                    ):
                        continue
                    cost = (
                        rad
                        + extra
                        + (0.03 * abs((ang - away + 180) % 360 - 180) if away is not None else 0.0)
                    )
                    cost += (
                        60 * min(count(pad, marks), 10)
                        + 1e3 * hits(pad, fixed)
                        + (avoid(inv.transform(box.reshape(2, 2))) if avoid is not None else 0.0)
                    )
                    seg = np.zeros((0, 4))
                    ends = None
                    if True:  # leader: marker edge -> nearest point of the label box, stopping short of both
                        tgt = np.array(
                            [np.clip(P[i, 0], box[0, 0], box[0, 2]), np.clip(P[i, 1], box[0, 1], box[0, 3])]
                        )
                        v = tgt - P[i]
                        L = np.hypot(*v)
                        a_, b_ = P[i] + v / L * (pts[i].get("ms", 4.6) / 2 + 0.8) * px, P[i] + v / L * (
                            L - 1.5 * px
                        )
                        if np.hypot(*(b_ - a_)) > 2 * px:
                            q_ = a_ + np.linspace(0, 1, max(int(np.hypot(*(b_ - a_)) / (2 * px)), 1) + 1)[
                                :, None
                            ] * (b_ - a_)
                            seg = np.column_stack([q_ - 0.6 * px, q_ + 0.6 * px])
                            ends = (a_, b_)
                            cost += (
                                400 * hits(seg, others)
                                + 250 * min(count(lines, seg), 4)
                                + 1e3 * hits(seg, fixed)
                            )
                    out.append((cost, box, pad, seg, ends, s, w, sw, rad, ang))
        cands.append(sorted(out, key=lambda t: t[0]))

    def solve(order):
        total, chosen, boxes, segs = 0.0, {}, np.zeros((0, 4)), np.zeros((0, 4))
        for i in order:
            best = None
            for cd in cands[i]:
                if best is not None and cd[0] >= best[0]:
                    break
                cost = (
                    cd[0]
                    + 1e3 * (hits(cd[2], boxes) + hits(cd[2], segs) + hits(cd[3], boxes))
                    + 100 * hits(cd[3], segs)
                )
                if best is None or cost < best[0]:
                    best = (cost, cd)
            if best is None:
                continue
            total += best[0]
            chosen[i] = best[1]
            boxes = np.vstack([boxes, best[1][1]])
            segs = np.vstack([segs, best[1][3]])
        return total, chosen

    crowd = [int((np.hypot(*(P - q_).T) < 30 * px).sum()) for q_ in P]
    idx = list(range(n))
    if n <= 5:
        import itertools

        orders = list(itertools.permutations(idx))
    else:
        rng = np.random.default_rng(0)
        orders = [
            sorted(idx, key=k)
            for k in (
                lambda i: (-crowd[i], i),
                lambda i: (crowd[i], i),
                lambda i: P[i, 0],
                lambda i: -P[i, 0],
                lambda i: P[i, 1],
                lambda i: -P[i, 1],
            )
        ] + [list(rng.permutation(n)) for _ in range(200)]
    total, chosen = min((solve(o) for o in orders), key=lambda t: t[0])
    for i, (cost, box, pad, seg, ends, s, w, sw, rad, ang) in chosen.items():
        if os.environ.get("PAPER_FIGS_DEBUG"):
            print(
                f"    label {pts[i]['label']!r}: static cost {cost:.1f} (r {rad} pt, {ang:.1f} deg); cheapest statics: "
                + ", ".join(f"{t[0]:.0f}@{t[8]}/{t[9]:.0f}" for t in cands[i][:6])
            )
        xd, yd = inv.transform((box[0, 0] + sw + w / 2, (box[0, 1] + box[0, 3]) / 2))
        ax.text(xd, yd, s, ha="center", va="center", fontsize=fs, color=INK, zorder=7, linespacing=1.05)
        if ends is not None:
            (xa, ya), (xb, yb) = inv.transform(ends[0]), inv.transform(ends[1])
            ax.plot(
                [xa, xb], [ya, yb], color=pts[i].get("leader", AXIS), lw=0.7, zorder=4, solid_capstyle="butt"
            )
    return total


def finish(fig, name, out_dir, S, man):
    bad = lint(fig, name)
    files = [f"{out_dir}/{name}.pdf", f"{out_dir}/{name}.png"]
    fig.savefig(files[0])
    fig.savefig(files[1], dpi=300)
    w, h = fig.get_size_inches()
    plt.close(fig)
    man[name] = {
        "files": files,
        "size_in": [round(float(w), 3), round(float(h), 3)],
        "n_values": len(S.log[name]),
        "lint": bad,
        "values": S.log[name],
    }
    print(
        f"{name}: {w:.2f} x {h:.2f} in, {len(S.log[name])} values, {len(bad)} lint problem(s) -> {files[0]}, .png"
    )


def skip(name, why):
    print(f"{name}: SKIPPED ({why})")


# ------------------------------------------------------------------------------------------------------------------------
# F0 hero
def fig0(S, A, out_dir, man):
    """Hero figure: the three shared properties in one row. (a) paired helpful-uptake drop vs harmful-caving drop (content-blind diagonal,
    content-aware wedge); (b) relative reduction of harmful caving at the trained position vs the never-trained end position; (c) circuit
    patch recovery of residual flips, untrained vs each arm, with the best random head set. Colour = method; filled = first model, hollow = second.
    """
    name = "F0_hero"
    S.begin(name)
    models = [m for m in A.models if S.has(m, "gen", "arms") and S.has(m, "causal", "untrained")][:2]
    if len(models) < 1:
        return skip(name, "no gen / causal section")
    meths = ordered([a for m in models for a in A.methods[m] if a not in HERO_EXCLUDE], ORDER)
    fill = {m: (i == 0) for i, m in enumerate(models)}
    okg = lambda m, a: S.has(m, "gen", "arms", a, "harmful_drop", "mean") and S.has(
        m, "gen", "arms", a, "helpful_drop", "mean"
    )
    pos = [
        p
        for p in ("front", "end")
        if all(S.has(m, "gen", "arms", a, "relred", p, "mean") for m in models for a in meths if okg(m, a))
    ]
    Da, Db, Dc = {}, {}, {}
    for m in models:
        for a in meths:
            if okg(m, a):
                Da[(m, a)] = (
                    S.val(
                        "(a) content",
                        f"{m}:{a}",
                        "paired harmful-caving drop (x), seed mean",
                        m,
                        "gen",
                        "arms",
                        a,
                        "harmful_drop",
                        "mean",
                    ),
                    S.val(
                        "(a) content",
                        f"{m}:{a}",
                        "paired helpful-uptake drop (y), seed mean",
                        m,
                        "gen",
                        "arms",
                        a,
                        "helpful_drop",
                        "mean",
                    ),
                )
                Db[(m, a)] = [
                    S.val(
                        "(b) position",
                        f"{m}:{a}",
                        f"relative reduction of harmful caving at {p}, seed mean",
                        m,
                        "gen",
                        "arms",
                        a,
                        "relred",
                        p,
                        "mean",
                    )
                    for p in pos
                ]
        u = (m, "causal", "untrained")
        Dc[(m, "untrained")] = (
            S.val("(c) circuit", f"{m}:untrained", "circuit patch recovery", *u, "circuit"),
            S.val("(c) circuit", f"{m}:untrained", "best random head set", *u, "best_random"),
        )
        for a in meths:
            b = (m, "causal", "arms", a)
            if S.has(*b, "pooled", "circuit"):
                Dc[(m, a)] = (
                    S.val(
                        "(c) circuit",
                        f"{m}:{a}",
                        "circuit patch recovery (pooled over seeds)",
                        *b,
                        "pooled",
                        "circuit",
                    ),
                    S.val(
                        "(c) circuit",
                        f"{m}:{a}",
                        "best random head set (max over seeds)",
                        *b,
                        "best_random_max",
                    ),
                )
            elif S.has(*b, "per_seed", 0, "circuit"):
                Dc[(m, a)] = (
                    S.val(
                        "(c) circuit",
                        f"{m}:{a}",
                        "circuit patch recovery (first seed)",
                        *b,
                        "per_seed",
                        0,
                        "circuit",
                    ),
                    S.val(
                        "(c) circuit",
                        f"{m}:{a}",
                        "best random head set (max over seeds)",
                        *b,
                        "best_random_max",
                    ),
                )
    lim_a = max(nice_ceiling(max(max(v) for v in Da.values()) + 0.03, 0.25), 0.5)
    lo_a = (
        0.0
        if min(min(v) for v in Da.values()) >= 0
        else nice_floor(min(min(v) for v in Da.values()) - 0.02, 0.05)
    )
    allb = [x for v in Db.values() for x in v]
    ylo_b, yhi_b = min(0.0, nice_floor(min(allb) - 0.02, 0.1)), max(1.0, nice_ceiling(max(allb), 0.1)) + 0.05
    rows_c = ["untrained"] + meths
    nr = len(rows_c)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W_FULL, 3.0))
        # methods: colour + marker only (line style encodes the model in b)
        hl = [(h_dot(A.color(a), A.marker(a)), NAME.get(a, a)) for a in meths]
        if len(models) == 2:
            hl += [
                (h_line(INK2, ls="-"), MODEL.get(models[0], models[0]) + " (filled markers)"),
                (h_line(INK2, ls=DASH), MODEL.get(models[1], models[1]) + " (hollow markers)"),
            ]
        hl += [(h_dot(C_UNTRAINED, "D", filled=False, ms=4.4), "best random head set")]
        leg, leg_h = flow_legend(
            fig, [h for h, _ in hl], [l for _, l in hl], W_FULL - 0.3, loc="lower center"
        )
        left, right, gap_ab, gap_bc, botm, ax_h = 0.46, 0.08, 0.62, 0.20, 0.05 + leg_h + 0.12 + 0.34, 1.55
        H = botm + ax_h + title_height(0) + 0.04 + take_height(name)
        fig.set_size_inches(W_FULL, H)
        anchor(fig, leg, W_FULL / 2, 0.05)
        draw_takeaway(fig, name)
        # (a) is square; (c) = label column + axis
        lab_c, wc = 0.62, 1.60
        wa = ax_h
        wb = W_FULL - left - right - gap_ab - gap_bc - wa - lab_c - wc
        # (a) content-blindness
        ax = add_ax(fig, left, botm, wa, ax_h)
        ax.set_xlim(lo_a, lim_a)
        ax.set_ylim(lo_a, lim_a)
        ax.set_aspect("equal", adjustable="box")
        tk = np.arange(0, lim_a + 1e-9, 0.25)
        ax.set_xticks(tk)
        ax.set_yticks(tk)
        ax.set_xticklabels([fmt_tick(v) for v in tk])
        ax.set_yticklabels([fmt_tick(v) for v in tk])
        ax.grid(True)
        ax.add_patch(
            Polygon(
                (
                    [(0, 0), (lim_a, 0.5 * lim_a), (lim_a, lo_a), (0, lo_a)]
                    if lo_a < 0
                    else [(0, 0), (lim_a, 0.5 * lim_a), (lim_a, 0)]
                ),
                closed=True,
                fc="#e9e8e3",
                ec="none",
                zorder=0.5,
            )
        )
        ax.plot([lo_a, lim_a], [lo_a, lim_a], color=INK2, lw=0.8, zorder=1.5)
        ax.set_xlabel("drop in harmful-cue following")
        ax.set_ylabel("drop in helpful-cue following")
        p0, p1 = ax.transData.transform([(0, 0), (1, 1)])
        ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
        cpos = lo_a + 0.60 * (lim_a - lo_a)
        ax.annotate(
            "equal absolute loss (y = x)",
            (cpos, cpos),
            xytext=(4.2, -4.2),
            textcoords="offset points",
            rotation=ang,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=MIN_ANNOT,
            color=INK2,
        )
        ax.annotate(
            "selective\n(y ≤ 0.5 x)",
            (lim_a, lo_a),
            xytext=(-4, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=MIN_ANNOT,
            color=INK2,
            linespacing=1.15,
        )
        panel_title(ax, "(a) helpful vs harmful cues", None)
        for (m, a), (x, y) in Da.items():
            dot(ax, x, y, A.color(a), A.marker(a), fill[m], z=5)
        # (b) position-binding: trained position vs the never-trained end
        ax = add_ax(fig, left + wa + gap_ab, botm, wb, ax_h)
        ax.set_xlim(-0.45, len(pos) - 0.55)
        ax.set_ylim(ylo_b, yhi_b)
        tk = [v for v in np.arange(-1, 1.01, 0.25) if ylo_b - 1e-9 <= v <= yhi_b]
        ax.set_yticks(tk)
        ax.set_yticklabels([fmt_tick(v) for v in tk])
        ax.grid(axis="y")
        ax.set_xticks(range(len(pos)))
        ax.set_xticklabels([("front\n(trained)" if p == TRAINED_POS else p) for p in pos], linespacing=1.1)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("cue position", labelpad=1.0)
        ax.set_ylabel("relative reduction of\nharmful-cue following", linespacing=1.15)
        if ylo_b < 0:
            ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        panel_title(ax, "(b) effect by cue position", None)
        for k, (m, a) in enumerate(Db):
            col = A.color(a)
            xs = np.arange(len(pos)) + (k - (len(Db) - 1) / 2) * 0.03
            ys = Db[(m, a)]
            ax.plot(xs, ys, color=col, lw=1.3, ls="-" if fill[m] else DASH, zorder=3 + 0.01 * k)
            for x, v in zip(xs, ys):
                dot(ax, x, v, col, A.marker(a), fill[m], ms=4.4, z=5 + 0.01 * k)
        # (c) circuit patch recovery
        ax = add_ax(fig, left + wa + gap_ab + wb + gap_bc + lab_c, botm, wc, ax_h)
        ax.set_ylim(nr - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
        for i in range(1, nr, 2):
            ax.axhspan(i - 0.5, i + 0.5, color=WASH, lw=0, zorder=0)
        ax.set_yticks(range(nr))
        ax.set_yticklabels([NAME.get(a, a) for a in rows_c], color=INK)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel("remaining cue errors corrected")
        panel_title(ax, "(c) circuit patching", None)
        for i, a in enumerate(rows_c):
            for m in models:
                if (m, a) not in Dc:
                    continue
                v, rv = Dc[(m, a)]
                dy = (-0.17 if fill[m] else 0.17) if len(models) == 2 else 0.0
                col = C_UNTRAINED if a == "untrained" else A.color(a)
                dot(ax, rv, i + dy, C_UNTRAINED, "D", False, ms=4.0)
                dot(ax, v, i + dy, col, A.marker(a) if a != "untrained" else "o", fill[m], ms=4.4)
        finish(fig, name, out_dir, S, man)


# --------------------------------------------------------------------------------------------------------------------------
# F1
def fig1(S, A, out_dir, man):
    name = "F1_caving"
    S.begin(name)
    models = [m for m in A.models if S.has(m, "T1", "untrained") and S.has(m, "T1", "arms")]
    if not models:
        return skip(name, "no T1 section")
    arms = ordered([a for m in models for a in S.raw(m, "T1", "arms")], ORDER)
    rows = ["untrained"] + arms
    nr = len(rows)
    CI, DCI = ("ci95_qcluster", "ci95"), ("diff_ci95_qcluster", "diff_ci95")
    Q = {"trained": "trained-cue flip given knew", "held": "held-out mean flip given knew"}
    D = {}
    for m in models:  # gather (and log) every number first, so limits can be derived from the data
        D[m] = {
            "untrained": {
                st: (
                    S.val(m, "untrained", Q[st], m, "T1", "untrained", st, "value"),
                    S.ci(m, "untrained", Q[st] + ", 95% CI", (m, "T1", "untrained", st), CI),
                )
                for st in Q
            }
        }
        for a in S.raw(m, "T1", "arms"):
            D[m][a] = {
                st: (
                    S.val(m, a, Q[st] + ", seed mean", m, "T1", "arms", a, st, "mean"),
                    S.ci(m, a, Q[st] + ", 95% CI", (m, "T1", "arms", a, st), CI),
                )
                for st in Q
            }
            if S.has(m, "T1", "arms", a, "acc", "diff"):
                D[m][a]["dacc"] = (
                    S.val(m, a, "uncued accuracy, arm - untrained", m, "T1", "arms", a, "acc", "diff"),
                    S.ci(m, a, "uncued accuracy difference, 95% CI", (m, "T1", "arms", a, "acc"), DCI),
                )
    ends = lambda keys: [
        x
        for m in models
        for a in D[m]
        for k in keys
        if k in D[m][a]
        for x in ([D[m][a][k][0]] + list(D[m][a][k][1] or []))
    ]
    xmax = max(nice_ceiling(max(ends(Q)) + 0.02, 0.05), 0.5)
    alo, ahi = nice_floor(min(ends(["dacc"]) + [0]) - 0.02, 0.05), nice_ceiling(
        max(ends(["dacc"]) + [0]) + 0.02, 0.05
    )
    n_held = {len(S.raw(m, "T1", "n_items")) - 1 for m in models if S.has(m, "T1", "n_items")}
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W_FULL, 3.0))
        hl = [
            (h_dot(INK2), "trained cue"),
            (
                h_dot(INK2, filled=False),
                "held-out cues" + (f" (mean of {n_held.pop()})" if len(n_held) == 1 else " (mean)"),
            ),
            (Line2D([], [], color=C_UNTRAINED, lw=0.9), "untrained, trained cue"),
            (Line2D([], [], color=C_UNTRAINED, lw=0.9, ls=DASH), "untrained, held-out"),
        ]
        leg, leg_h = flow_legend(
            fig, [h for h, _ in hl], [l for _, l in hl], W_FULL - 0.4, loc="lower center"
        )
        lab_w, right, gap_b, gap_i, row_h = 0.80, 0.20, 0.30, 0.14, 0.27
        bot = 0.05 + leg_h + 0.11 + 0.47
        ax_h = nr * row_h
        H = bot + ax_h + title_height(0) + 0.02 + take_height(name)
        fig.set_size_inches(W_FULL, H)
        anchor(fig, leg, W_FULL / 2, 0.05)
        draw_takeaway(fig, name)
        block = (W_FULL - lab_w - right - (len(models) - 1) * gap_b) / len(models)
        acc_w = 0.31 * (block - gap_i)
        flip_w = block - gap_i - acc_w
        for mi, m in enumerate(models):
            x0 = lab_w + mi * (block + gap_b)
            axf = add_ax(fig, x0, bot, flip_w, ax_h)
            axa = add_ax(fig, x0 + flip_w + gap_i, bot, acc_w, ax_h)
            for ax in (axf, axa):
                ax.set_ylim(nr - 0.5, -0.5)
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0)
                ax.grid(axis="x")
                for i in range(1, nr, 2):
                    ax.axhspan(i - 0.5, i + 0.5, color=WASH, lw=0, zorder=0)
                ax.set_yticks(range(nr))
                ax.set_yticklabels(
                    (
                        ["\n".join(textwrap.wrap(NAME.get(a, a), 13, break_long_words=False)) for a in rows]
                        if (mi == 0 and ax is axf)
                        else [""] * nr
                    ),
                    color=INK,
                    linespacing=1.0,
                )
            axf.set_xlim(0, xmax)
            axf.set_xticks(np.arange(0, xmax + 1e-9, 0.25))
            axf.set_xticklabels([fmt_tick(v) for v in axf.get_xticks()])
            axf.set_xlabel("flip rate given knew")
            axa.set_xlim(alo, ahi)
            tk = [v for v in (-0.2, -0.1, 0.0, 0.1, 0.2) if alo - 1e-9 <= v <= ahi + 1e-9]
            axa.set_xticks(tk)
            axa.set_xticklabels([fmt_tick(v, signed=True) for v in tk])
            axa.set_xlabel("Δ uncued accuracy\n(arm − untrained)", linespacing=1.1)
            axa.axvline(0, color=C_UNTRAINED, lw=0.9, zorder=1.5)
            panel_title(axf, [MODEL.get(m, m), MODEL_SHORT.get(m, MODEL.get(m, m))])
            for st, ls in (("trained", "-"), ("held", DASH)):
                axf.axvline(D[m]["untrained"][st][0], color=C_UNTRAINED, lw=0.9, ls=ls, zorder=1.5)
            for i, a in enumerate(rows):
                if a not in D[m]:
                    axf.annotate(
                        "not run",
                        (0, i),
                        xycoords=("axes fraction", "data"),
                        xytext=(4, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=MIN_ANNOT,
                        color=INK2,
                        style="italic",
                    )
                    continue
                col = A.color(a)
                for st, dy, filled in (("trained", -0.19, True), ("held", 0.19, False)):
                    v, c = D[m][a][st]
                    if c:
                        span(axf, c, [i + dy, i + dy], col)
                    dot(axf, v, i + dy, col, A.marker(a), filled)
                if "dacc" in D[m][a]:
                    v, c = D[m][a]["dacc"]
                    if c:
                        span(axa, c, [i, i], col)
                    dot(axa, v, i, col, "s", True, ms=4.2)
        finish(fig, name, out_dir, S, man)


# --------------------------------------------------------------------------------------------------------------------------
# F2
def fig2(S, A, out_dir, man):
    name = "F2_content_blind"
    S.begin(name)
    models = [m for m in A.models if S.has(m, "gen", "arms")]
    ok = lambda m, a: S.has(m, "gen", "arms", a, "harmful_drop", "mean") and S.has(
        m, "gen", "arms", a, "helpful_drop", "mean"
    )
    panels = (
        [] if F2_VARIANTS_ONLY else [(m, "CT methods", [a for a in A.methods[m] if ok(m, a)]) for m in models]
    )
    for m in models:
        # BCT as the reference point on every model
        var = [a for a in A.variants[m] if ok(m, a)]
        ref = [a for a in ("bct",) if a in A.methods[m] and ok(m, a)]
        if var:
            panels.append((m, "BCT and BCT-pipeline variants", ref + var))
    panels = [p for p in panels if p[2]]
    if not panels:
        return skip(name, "no gen section")
    D = []
    for pi, (m, sub, arms) in enumerate(panels):
        pn = f"({'abcdefgh'[pi]}) {m}: {sub}"
        d = {}
        for a in arms:
            d[a] = {
                k: (
                    S.val(pn, a, q + ", seed mean", m, "gen", "arms", a, k, "mean"),
                    S.ci(pn, a, q + ", 95% CI", (m, "gen", "arms", a, k), ("ci95",)),
                )
                for k, q in (
                    ("harmful_drop", "paired harmful-caving drop (x)"),
                    ("helpful_drop", "paired helpful-uptake drop (y)"),
                )
            }
        D.append(d)
    allv = [x for d in D for a in d for k in d[a] for x in ([d[a][k][0]] + list(d[a][k][1] or []))]
    lim = max(nice_ceiling(max(allv) + 0.03, 0.25), 0.5)
    lo = 0.0 if min(allv) >= 0 else nice_floor(min(allv) - 0.02, 0.05)
    n = len(panels)
    with plt.rc_context(RC):
        left, right, gap, botm = 0.50, 0.16, 0.20, 0.40
        side = min((W_FULL - left - right - (n - 1) * gap) / n, 2.3)
        left += (W_FULL - left - right - n * side - (n - 1) * gap) / 2
        nsub = max(len(wrap_title(sub)) for _, sub, _ in panels)
        H = botm + side + title_height(nsub) + 0.02 + take_height(name)
        fig = plt.figure(figsize=(W_FULL, H))
        draw_takeaway(fig, name)
        for pi, ((m, sub, arms), d) in enumerate(zip(panels, D)):
            ax = add_ax(fig, left + pi * (side + gap), botm, side, side)
            ax.set_xlim(lo, lim)
            ax.set_ylim(lo, lim)
            ax.set_aspect("equal", adjustable="box")
            tk = np.arange(0, lim + 1e-9, 0.25)
            ax.set_xticks(tk)
            ax.set_yticks(tk)
            ax.set_xticklabels([fmt_tick(v) for v in tk])
            ax.set_yticklabels([fmt_tick(v) for v in tk] if pi == 0 else [""] * len(tk))
            ax.grid(True)
            ax.add_patch(
                Polygon(
                    (
                        [(0, 0), (lim, 0.5 * lim), (lim, lo), (0, lo)]
                        if lo < 0
                        else [(0, 0), (lim, 0.5 * lim), (lim, 0)]
                    ),
                    closed=True,
                    fc="#e9e8e3",
                    ec="none",
                    zorder=0.5,
                )
            )
            ax.plot([lo, lim], [lo, lim], color=INK2, lw=0.8, zorder=1.5)
            ax.set_xlabel("harmful-caving drop")
            if pi == 0:
                ax.set_ylabel("helpful-uptake drop")
                p0, p1 = ax.transData.transform([(0, 0), (1, 1)])
                ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
                cpos = lo + 0.74 * (lim - lo)
                ax.annotate(
                    "equal absolute loss (y = x)",
                    (cpos, cpos),
                    xytext=(4.2, -4.2),
                    textcoords="offset points",
                    rotation=ang,
                    rotation_mode="anchor",
                    ha="center",
                    va="center",
                    fontsize=MIN_ANNOT,
                    color=INK2,
                )
                ax.annotate(
                    "selective\n(y ≤ 0.5 x)",
                    (lim, lo),
                    xytext=(-4, 5),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=MIN_ANNOT,
                    color=INK2,
                    linespacing=1.15,
                )
            panel_title(
                ax,
                [f"({'abcdefgh'[pi]}) {nm}" for nm in (MODEL.get(m, m), MODEL_SHORT.get(m, MODEL.get(m, m)))],
                sub,
            )
            pts = []
            for a in arms:
                (x, cx), (y, cy) = d[a]["harmful_drop"], d[a]["helpful_drop"]
                col = A.color(a)
                if cx:
                    span(ax, cx, [y, y], col, lw=0.8)
                if cy:
                    span(ax, [x, x], cy, col, lw=0.8)
            for a in arms:
                x, y = d[a]["harmful_drop"][0], d[a]["helpful_drop"][0]
                ms = 4.6 if A.marker(a) == "o" else 4.2
                # variants share one ink: a grey leader cannot be mistaken for a CI bar
                dot(ax, x, y, A.color(a), A.marker(a), True, ms=ms, z=5)
                pts.append(
                    {
                        "x": x,
                        "y": y,
                        "label": SHORT.get(a, NAME.get(a, a)),
                        "ms": ms,
                        "leader": A.color(a) if a in A.methods[m] else AXIS,
                    }
                )
            # label corners inside the content-aware wedge
            place_point_labels(
                ax,
                pts,
                fs=MIN_ANNOT,
                avoid=lambda bx: 75.0 * sum(y_ <= 0.5 * x_ for x_ in bx[:, 0] for y_ in bx[:, 1]),
            )
        finish(fig, name, out_dir, S, man)


# --------------------------------------------------------------------------------------------------------------------------
# F3
def fig3(S, A, out_dir, man):
    name = "F3_position_bound"
    S.begin(name)
    models = [m for m in A.models if S.has(m, "gen", "arms") and S.has(m, "gen", "positions")]
    if not models:
        return skip(name, "no gen section")
    ok = lambda m, a, p: S.has(m, "gen", "arms", a, "relred", p, "mean")
    D = {}
    for m in models:
        pos = list(S.raw(m, "gen", "positions"))
        D[m] = {"pos": pos, "arms": {}}
        ref = [a for a in (S.raw(m, "fix", "arms") if S.has(m, "fix", "arms") else []) if a in A.methods[m]]
        for a in A.methods[m] + [v for v in A.variants[m] if v in POS_VARIANTS]:
            if not all(ok(m, a, p) for p in pos):
                continue
            D[m]["arms"][a] = [
                (
                    S.val(
                        m,
                        a,
                        f"relative reduction of harmful caving, {p}, seed mean",
                        m,
                        "gen",
                        "arms",
                        a,
                        "relred",
                        p,
                        "mean",
                    ),
                    S.ci(
                        m,
                        a,
                        f"relative reduction of harmful caving, {p}, 95% CI",
                        (m, "gen", "arms", a, "relred", p),
                        ("ci95",),
                    ),
                )
                for p in pos
            ]
        D[m]["ref"] = ref
    allv = [
        x for m in models for a in D[m]["arms"] for v, c in D[m]["arms"][a] for x in ([v] + list(c or []))
    ]
    ylo, yhi = min(0.0, nice_floor(min(allv) - 0.02, 0.1)), max(1.0, nice_ceiling(max(allv), 0.1)) + 0.05
    methods = ordered([a for m in models for a in D[m]["arms"] if a in A.methods[m]], ORDER)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W_FULL, 3.0))
        leg, leg_h = flow_legend(
            fig,
            [h_line(A.color(a), marker=A.marker(a)) for a in methods],
            [A.label(a, models) for a in methods],
            W_FULL - 0.3,
        )
        left, right, gap, botm, ax_h = 0.74, 0.08, 0.16, 0.44, 1.62
        pw = (W_FULL - left - right - (len(models) - 1) * gap) / len(models)
        H = botm + ax_h + title_height(0) + 0.10 + leg_h + 0.04 + take_height(name)
        fig.set_size_inches(W_FULL, H)
        anchor(fig, leg, W_FULL / 2, H - 0.04 - take_height(name))
        draw_takeaway(fig, name)
        for mi, m in enumerate(models):
            ax = add_ax(fig, left + mi * (pw + gap), botm, pw, ax_h)
            pos = D[m]["pos"]
            arms = list(D[m]["arms"])
            n = len(arms)
            ax.set_xlim(-0.5, len(pos) - 0.5)
            ax.set_ylim(ylo, yhi)
            tk = [v for v in np.arange(-1, 1.01, 0.25) if ylo - 1e-9 <= v <= yhi]
            ax.set_yticks(tk)
            ax.set_yticklabels([fmt_tick(v) for v in tk] if mi == 0 else [""] * len(tk))
            ax.grid(axis="y")
            ax.set_xticks(range(len(pos)))
            ax.set_xticklabels(
                [p + ("\n(trained)" if p == TRAINED_POS else "") for p in pos], linespacing=1.1
            )
            ax.tick_params(axis="x", length=0)
            ax.set_xlabel("cue position", labelpad=1.0)
            if mi == 0:
                ax.set_ylabel("relative reduction of\nharmful-cue following", linespacing=1.15)
            if TRAINED_POS in pos:
                ax.axvspan(
                    pos.index(TRAINED_POS) - 0.4, pos.index(TRAINED_POS) + 0.4, color=WASH, lw=0, zorder=0
                )
            if ylo < 0:
                ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
            panel_title(ax, [MODEL.get(m, m), MODEL_SHORT.get(m, MODEL.get(m, m))])
            for k, a in enumerate(arms):
                col = A.color(a)
                var = a in A.variants[m]
                xs = np.arange(len(pos)) + (k - (n - 1) / 2) * 0.055
                ys = [v for v, _ in D[m]["arms"][a]]
                ax.plot(xs, ys, color=col, lw=1.3, ls=DASH if var else "-", zorder=3 + 0.01 * k)
                for x, (v, c) in zip(xs, D[m]["arms"][a]):
                    if c:
                        span(ax, [x, x], c, col, lw=0.8, z=3 + 0.01 * k)
                    dot(ax, x, v, col, A.marker(a), True, ms=4.2 if var else 4.6, z=5 + 0.01 * k)
                if var:  # the one direct label: the contrasting dashed line, under its last segment
                    lab = NAME.get(a, a) + (
                        " " + " / ".join(NAME.get(r_, r_) for r_ in D[m]["ref"]) if D[m]["ref"] else ""
                    )
                    lows = [(c[0] if c else v) for v, c in D[m]["arms"][a][-2:]]
                    ax.annotate(
                        lab,
                        (xs[-1], min(lows)),
                        xytext=(3, -3.5),
                        textcoords="offset points",
                        ha="right",
                        va="top",
                        fontsize=MIN_ANNOT,
                        color=INK,
                    )
        finish(fig, name, out_dir, S, man)


# --------------------------------------------------------------------------------------------------------------------------
# F4
def fig4(S, A, out_dir, man):
    name = "F4_robustness_mechanism"
    S.begin(name)
    rel = [
        m
        for m in A.models
        if (S.has(m, "relapse", "untrained") and S.has(m, "relapse", "arms")) or S.has(m, "relapse_u", "arms")
    ]
    cau = [] if F4_RELAPSE_ONLY else [m for m in A.models if S.has(m, "causal", "untrained")]
    if not rel and not cau:
        return skip(name, "no relapse / causal section")
    letters = iter("abcdefgh")
    R, C = {}, {}
    # (a) relapse trajectories: untrained from the JSON leaf, arms as seed mean / SD of the per-seed leaves
    for m in rel:
        # tables --unmerged (relapse): GENERATED answers, steps 0 and 125
        if S.has(m, "relapse_u", "arms"):
            pn = f"relapse (generated answers, unmerged): {m}"
            steps = ["0", "125"]
            R[m] = {"steps": steps, "panel": pn, "series": {}, "generated": True}
            R[m]["series"]["untrained"] = [
                (
                    S.val(
                        pn,
                        "untrained",
                        f"trained-cue flip given knew (generated), step {st}",
                        m,
                        "relapse_u",
                        "untrained_gen",
                        st,
                        "trained",
                    ),
                    None,
                )
                for st in steps
            ]
            for a in [x for x in ordered(S.raw(m, "relapse_u", "arms"), ORDER) if x not in F4_EXCLUDE]:
                ps = S.raw(m, "relapse_u", "arms", a, "per_seed")
                if ps:
                    R[m]["series"][a] = [
                        S.seedstat(
                            pn,
                            a,
                            f"trained-cue flip given knew (generated), step {st}",
                            [
                                (m, "relapse_u", "arms", a, "per_seed", i, "gen", st, "trained")
                                for i in range(len(ps))
                            ],
                        )
                        for st in steps
                    ]
            continue
        # gemma: the pre-registered correction (T4d-v7): GENERATED answers, steps 0 and 125 only (the
        # first-position trajectory fails its gate)
        if S.has(m, "relapse7", "arms") and any(
            S.has(m, "relapse7", "arms", a, "per_seed", 0, "gen_rates") for a in S.raw(m, "relapse7", "arms")
        ):
            pn = f"relapse (generated answers): {m}"
            steps = ["0", "125"]
            R[m] = {"steps": steps, "panel": pn, "series": {}, "generated": True}
            R[m]["series"]["untrained"] = [
                (
                    S.val(
                        pn,
                        "untrained",
                        f"trained-cue flip given knew (generated), step {st}",
                        m,
                        "relapse7",
                        "untrained_gen",
                        st,
                        "trained",
                    ),
                    None,
                )
                for st in steps
            ]
            for a in ordered(S.raw(m, "relapse7", "arms"), ORDER):
                ps = S.raw(m, "relapse7", "arms", a, "per_seed")
                if ps:
                    R[m]["series"][a] = [
                        S.seedstat(
                            pn,
                            a,
                            f"trained-cue flip given knew (generated), step {st}",
                            [
                                (m, "relapse7", "arms", a, "per_seed", i, "gen_rates", st, "trained")
                                for i in range(len(ps))
                            ],
                        )
                        for st in steps
                    ]
            continue
        pn = f"relapse: {m}"
        steps = sorted(S.raw(m, "relapse", "untrained"), key=int)
        R[m] = {"steps": steps, "panel": pn, "series": {}}
        R[m]["series"]["untrained"] = [
            (
                S.val(
                    pn,
                    "untrained",
                    f"trained-cue flip given knew, step {st}",
                    m,
                    "relapse",
                    "untrained",
                    st,
                    "trained",
                ),
                None,
            )
            for st in steps
        ]
        for a in ordered(S.raw(m, "relapse", "arms"), ORDER):
            seeds = [
                i
                for i, p in enumerate(S.raw(m, "relapse", "arms", a, "per_seed"))
                if p is not None
                and all(S.has(m, "relapse", "arms", a, "per_seed", i, "traj", st, "trained") for st in steps)
            ]
            if seeds:
                R[m]["series"][a] = [
                    S.seedstat(
                        pn,
                        a,
                        f"trained-cue flip given knew, step {st}",
                        [(m, "relapse", "arms", a, "per_seed", i, "traj", st, "trained") for i in seeds],
                    )
                    for st in steps
                ]
    # (b) patch recovery: circuit (pooled over seeds) and the best random head set (the per-seed row holding
    # the maximum)
    for m in cau:
        pn = f"patch recovery: {m}"
        u = (m, "causal", "untrained")
        C[m] = {"panel": pn, "rows": {}}
        C[m]["rows"]["untrained"] = {
            "circuit": (
                S.val(pn, "untrained", "circuit patch recovery", *u, "circuit"),
                S.ci(pn, "untrained", "circuit patch recovery, Wilson 95%", u, ("wilson95",)),
            ),
            "random": (
                S.val(pn, "untrained", "best random head set", *u, "best_random"),
                S.ci(pn, "untrained", "best random head set, Wilson 95%", u, ("best_random_wilson95",)),
            ),
        }
        for a in ordered(S.raw(m, "causal", "arms") if S.has(m, "causal", "arms") else [], ORDER):
            b = (m, "causal", "arms", a)
            ps = S.raw(*b, "per_seed")
            i = int(np.argmax([p["best_random"] for p in ps]))
            src = "pooled" if S.has(*b, "pooled", "circuit") else ("per_seed", 0)
            src = (src,) if isinstance(src, str) else src
            C[m]["rows"][a] = {
                "circuit": (
                    S.val(
                        pn,
                        a,
                        f"circuit patch recovery ({'pooled over ' + str(len(ps)) + ' seed(s)' if src == ('pooled',) else 'first seed'})",
                        *b,
                        *src,
                        "circuit",
                    ),
                    S.ci(pn, a, "circuit patch recovery, Wilson 95%", b + src, ("wilson95",)),
                ),
                "random": (
                    S.val(
                        pn,
                        a,
                        f"best random head set (maximum over {len(ps)} seed(s))",
                        *b,
                        "per_seed",
                        i,
                        "best_random",
                    ),
                    S.ci(
                        pn,
                        a,
                        "best random head set, Wilson 95%",
                        b + ("per_seed", i),
                        ("best_random_wilson95",),
                    ),
                ),
            }
            if S.has(*b, "best_random_max"):
                assert S.raw(*b, "best_random_max") == C[m]["rows"][a]["random"][0]
    rows = ["untrained"] + ordered([a for m in cau for a in C[m]["rows"] if a != "untrained"], ORDER)
    nr = len(rows)
    lines = ["untrained"] + ordered([a for m in rel for a in R[m]["series"] if a != "untrained"], ORDER)
    WF = W_COL if (F4_COLUMN and not cau) else W_FULL
    GAP = 0.28 if WF == W_COL else 0.45
    if WF == W_COL:
        TAKEAWAY.pop(name, None)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(WF, 3.0))
        leg = leg2 = None
        leg_h = leg2_h = 0.0
        if rel:
            leg, leg_h = flow_legend(
                fig,
                [h_line(A.color(a), marker=A.marker(a)) for a in lines],
                [A.label(a) for a in lines],
                WF - 0.1,
            )
        if cau:
            leg2, leg2_h = flow_legend(
                fig,
                [h_dot(INK2), h_dot(C_UNTRAINED, "D", filled=False, ms=4.4)],
                ["circuit heads patched", "best random head set"],
                3.0,
                loc="lower center",
            )
        left, right, botm, ax_h, lab_w, gap_c = (
            (0.42 if WF == W_COL else 0.50),
            0.06 if WF == W_COL else 0.10,
            0.05 + leg2_h + 0.10 + 0.34,
            1.45 if WF == W_COL else 1.60,
            0.80,
            0.16,
        )
        avail = (
            WF
            - left
            - right
            - (lab_w if (rel and cau) else 0.30 if cau else 0)
            - (len(cau) - 1) * gap_c * bool(cau)
            - (len(rel) - 1) * GAP * bool(rel)
        )
        rw = (avail * (0.44 if cau else 1.0) / len(rel)) if rel else 0.0
        cw = ((avail - rw * len(rel)) / len(cau)) if cau else 0.0
        # short (narrow panels); the caption states the bases: gemma generated answers at steps 0 / 125, Llama
        # the v7.1 readout
        relsub = lambda m: "relapse, trained cue"
        subs = [relsub(m) for m in rel] + ["circuit patch recovery"] * len(cau)
        nsub = max(len(wrap_title(s)) for s in subs)
        H = botm + ax_h + title_height(nsub) + (0.10 + leg_h if leg else 0) + 0.04 + take_height(name)
        fig.set_size_inches(WF, H)
        draw_takeaway(fig, name)
        if leg:
            anchor(fig, leg, WF / 2, H - 0.04 - take_height(name))
        x = left
        for m in rel:
            first = x == left
            ax = add_ax(fig, x, botm, rw, ax_h)
            x += rw + GAP
            st = [int(s) for s in R[m]["steps"]]
            allv = [v + (sd or 0) for a in R[m]["series"] for v, sd in R[m]["series"][a]]
            ax.set_xlim(min(st) - 0.04 * (max(st) - min(st)), max(st) + 0.04 * (max(st) - min(st)))
            ax.set_ylim(0, max(nice_ceiling(max(allv) + 0.02, 0.1), 0.5))
            # "10" collides with "0" and "25" at this width
            ax.set_xticks(st)
            ax.set_xticklabels(["" if (v == 10 and 0 in st and 25 in st) else str(v) for v in st])
            ax.set_xlabel("fine-tuning steps" if WF == W_COL else "benign fine-tuning steps")
            ax.grid(axis="y")
            if first or WF != W_COL:
                ax.set_ylabel("trained-cue flip | knew")
            tk = np.arange(0, ax.get_ylim()[1] + 1e-9, 0.2)
            ax.set_yticks(tk)
            ax.set_yticklabels([fmt_tick(v) for v in tk])
            lt = next(letters)
            panel_title(
                ax,
                [f"({lt}) {nm}" for nm in (MODEL.get(m, m), MODEL_SHORT.get(m, MODEL.get(m, m)))],
                relsub(m),
            )
            for k, a in enumerate(ordered(R[m]["series"], ["untrained"] + ORDER)):
                col = A.color(a)
                mu = np.array([v for v, _ in R[m]["series"][a]])
                sd = [s for _, s in R[m]["series"][a]]
                if all(s is not None for s in sd):
                    ax.fill_between(
                        st, mu - np.array(sd), mu + np.array(sd), color=col, alpha=0.16, lw=0, zorder=2
                    )
                ax.plot(
                    st,
                    mu,
                    color=col,
                    lw=1.3,
                    marker=A.marker(a),
                    ms=4.2,
                    mfc=col,
                    mec="white",
                    mew=0.6,
                    zorder=3 + 0.01 * k,
                )
        x = x - GAP + lab_w if rel else left + 0.30
        cx0 = x
        for ci_, m in enumerate(cau):
            ax = add_ax(fig, x, botm, cw, ax_h)
            x += cw + gap_c
            ax.set_ylim(nr - 0.5, -0.5)
            ax.set_xlim(0, 1)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x")
            for i in range(1, nr, 2):
                ax.axhspan(i - 0.5, i + 0.5, color=WASH, lw=0, zorder=0)
            ax.set_yticks(range(nr))
            ax.set_yticklabels(
                (
                    ["\n".join(textwrap.wrap(NAME.get(a, a), 13, break_long_words=False)) for a in rows]
                    if ci_ == 0
                    else [""] * nr
                ),
                color=INK,
                linespacing=1.0,
            )
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "", "0.5", "", "1"])
            ax.set_xlabel("remaining cue errors corrected")
            lt = next(letters)
            panel_title(
                ax,
                [f"({lt}) {nm}" for nm in (MODEL.get(m, m), MODEL_SHORT.get(m, MODEL.get(m, m)))],
                "circuit patch recovery",
            )
            for i, a in enumerate(rows):
                if a not in C[m]["rows"]:
                    ax.annotate(
                        "not run",
                        (0.5, i),
                        xycoords=("axes fraction", "data"),
                        ha="center",
                        va="center",
                        fontsize=MIN_ANNOT,
                        color=INK2,
                        style="italic",
                    )
                    continue
                (v, c), (rv, rc_) = C[m]["rows"][a]["circuit"], C[m]["rows"][a]["random"]
                if rc_:
                    span(ax, rc_, [i, i], C_UNTRAINED)
                dot(ax, rv, i, C_UNTRAINED, "D", False, ms=4.4)
                if c:
                    span(ax, c, [i, i], A.color(a))
                dot(ax, v, i, A.color(a), A.marker(a), True)
        if leg2:
            anchor(fig, leg2, (cx0 + x - gap_c) / 2, 0.05)
        finish(fig, name, out_dir, S, man)


# ------------------------------------------------------------------------------------------------------------------------
# main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=f"{PAPER}/paper_tables.json")
    ap.add_argument("--out_dir", default=f"{PAPER}/figs")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    S = Source(a.tables)
    A = Arms(S)
    os.makedirs(a.out_dir, exist_ok=True)
    man = {}
    only = {x.strip().upper() for x in a.only.split(",") if x.strip()}
    for key, fn in (("F0", fig0), ("F1", fig1), ("F2", fig2), ("F3", fig3), ("F4", fig4)):
        if not only or key in only:
            fn(S, A, a.out_dir, man)
    n = S.audit()
    out = {
        "source": os.path.abspath(a.tables),
        "source_created": S.d.get("meta", {}).get("created"),
        "script": os.path.abspath(__file__),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "json_path_syntax": "RFC 6901 JSON pointer into `source`",
        "width_in": W_FULL,
        "min_font_pt": {"text": MIN_TEXT, "annotation": MIN_ANNOT},
        "colors": {**{k: A.color(k) for k in A.style}},
        "models": A.models,
        "methods": A.methods,
        "variants": A.variants,
        "figures": man,
    }
    if not only:
        json.dump(out, open(f"{a.out_dir}/figs_manifest.json", "w"), indent=1)
    else:
        print("--only given: figs_manifest.json not rewritten")
    print(
        f"audit: {n} logged values re-read from {a.tables} and equal; {sum(len(v['lint']) for v in man.values())} lint problem(s); {time.time() - t0:.1f} s"
    )


if __name__ == "__main__":
    main()
