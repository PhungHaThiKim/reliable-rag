"""Static trade-off figures (PNG) for the notebooks and the slides. Task A and Task B stay in separate figures;
nothing pools binary detection with set-valued or unanswerable questions.

Palette (validated: tools dataviz validate_palette, light, surface #F7F6F2, all pairs): blue #2a78d6, orange #eb6834,
aqua #1baf7a; orange and aqua are below 3:1 on the surface, so every colored mark carries a direct text label.
Other methods are drawn in neutral gray (emphasis form), magnitude tables use one sequential hue.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

SURFACE, INK, MUTED, GRID, NEUTRAL = "#F7F6F2", "#14213D", "#646D7C", "#E2DED5", "#9AA3AE"
HILITE = {"e2": "#2a78d6", "e3": "#eb6834", "e4": "#1baf7a"}
NAMES = {"e0": "E0", "e1": "E1", "e2": "E2", "e2_bm25": "E2 BM25", "e2_rerank": "E2 rerank", "e3": "E3", "e3a_exact": "E3a",
         "e3a": "E3b", "e4": "E4", "e4_filter": "E4 filter", "e4_cite": "E4 cite"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    return plt


def heatmap(rows: List[str], cols: List[str], values: List[List[Optional[float]]], out_png: str, title: str,
            col_note: str = "", vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Methods x groups, one sequential hue (more = darker), value printed in every cell. vmin/vmax set the color
    range (say so in col_note when it is not 0-1)."""
    plt = _plt()
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#EAF2FB", "#9CC3EE", "#2a78d6", "#123F7A"])
    v = np.array([[np.nan if x is None else x for x in r] for r in values], dtype=float)
    fig, ax = plt.subplots(figsize=(1.9 * len(cols) + 2.2, 0.62 * len(rows) + 1.6))
    ax.imshow(np.nan_to_num(v, nan=vmin), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), [NAMES.get(r, r) for r in rows])
    ax.tick_params(length=0, labelsize=13, colors=INK)
    for i in range(len(rows)):
        for j in range(len(cols)):
            x = v[i, j]
            ax.text(j, i, "-" if np.isnan(x) else f"{x:.2f}", ha="center", va="center", fontsize=13,
                    color="#FFFFFF" if (not np.isnan(x) and (x - vmin) / ((vmax - vmin) or 1) > 0.55) else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=15, pad=12)
    if col_note:
        fig.text(0.01, 0.01, col_note, color=MUTED, fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return out_png


def _clusters(pts: List[Tuple[float, float, str]], ax, tol: float = 0.035) -> List[Tuple[float, float, str]]:
    """Merge labels of points closer than `tol` (in axis-fraction units): one label 'E3, E4' instead of overlaps."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    sx, sy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    out: List[List] = []
    for x, y, name in sorted(pts, key=lambda p: (p[0], p[1])):
        for c in out:
            if abs((c[0] - x) / sx) < tol and abs((c[1] - y) / sy) < tol:
                c[2].append(name)
                break
        else:
            out.append([x, y, [name]])
    return [(x, y, ", ".join(n)) for x, y, n in out]


def _end_labels(ax, ends: List[List]):
    """Direct labels at line ends; ends that nearly coincide share one label."""
    for x, y, name in _clusters([tuple(e) for e in ends], ax):
        ax.annotate(name, (x, y), xytext=(8, 0), textcoords="offset points", color=INK, fontsize=13, va="center")


def risk_coverage(curves: Dict[str, Sequence[Sequence[float]]], out_png: str, title: str,
                  sweep: Optional[List[Dict]] = None) -> str:
    """Selective risk vs coverage (coverage over ALL answerable items, so abstaining ends a curve early)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ends = []
    for m, pts in curves.items():
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=HILITE.get(m, NEUTRAL), linewidth=2, label=NAMES.get(m, m))
        if not (sweep and m == "e4"):                    # E4's end sits among its tau circles: legend names it
            ends.append([xs[-1], ys[-1], NAMES.get(m, m)])
    _end_labels(ax, ends)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    if sweep:
        merged: Dict[Tuple[float, float], List[str]] = {}
        for s in sweep:
            key = (round(s["answer_coverage"], 3), round(1 - (s["answered_accuracy"] or 0), 3))
            merged.setdefault(key, []).append(str(s["tau"]))
        ax.scatter([k[0] for k in merged], [k[1] for k in merged], s=64, color=SURFACE, edgecolor=HILITE["e4"],
                   linewidth=2, zorder=3)
        taus = [str(s["tau"]) for s in sweep]
        fig.text(0.01, 0.01, f"Vòng tròn: E4 với ngưỡng tau {taus[0]} đến {taus[-1]} (tau cao hơn: từ chối nhiều hơn).",
                 color=MUTED, fontsize=11)
    ax.set_xlim(0, 1.08); ax.set_ylim(0, None)
    ax.set_xlabel("answer coverage (share of answerable questions answered)")
    ax.set_ylabel("risk (error rate among answered)")
    ax.set_title(title, loc="left", color=INK, fontsize=15, pad=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return out_png


def labeled_scatter(points: Dict[str, Tuple[float, float]], out_png: str, title: str, xlabel: str, ylabel: str,
                    arrow: Optional[Tuple[str, str]] = None, xlim=None, ylim=None) -> str:
    """Gray points with direct labels; E2/E3/E4 in their highlight colors; optional arrow (e.g. E3 -> E4)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    pts = [(x, y, NAMES.get(m, m)) for m, (x, y) in points.items() if x is not None and y is not None]
    for m, (x, y) in points.items():
        if x is not None and y is not None:
            ax.scatter([x], [y], s=90, color=HILITE.get(m, NEUTRAL), edgecolor=SURFACE, linewidth=2, zorder=3)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylim(*(ylim or (0, max([p[1] for p in pts] + [0.05]) * 1.25)))
    (x0, x1) = ax.get_xlim()
    for k, (x, y, name) in enumerate(_clusters(pts, ax, tol=0.07)):
        right = (x - x0) / ((x1 - x0) or 1.0) > 0.55          # labels on the right half grow leftwards
        ax.annotate(name, (x, y), xytext=(-8 if right else 8, 8 + 14 * (k % 2)), textcoords="offset points",
                    color=INK, fontsize=13, ha="right" if right else "left")
    if arrow and all(k in points and None not in points[k] for k in arrow):
        (x0, y0), (x1, y1) = points[arrow[0]], points[arrow[1]]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.5, "shrinkA": 8, "shrinkB": 8})
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=15, pad=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return out_png


def grounding_bars(rows: List[Tuple[str, float, float]], out_png: str, title: str, xlabel: str) -> str:
    """Horizontal bars: unsupported claims / original claims per method (E3, E4 highlighted), coverage printed beside."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(rows) + 1.8))
    ys = list(range(len(rows)))[::-1]
    for y, (m, v, cov) in zip(ys, rows):
        ax.barh(y, v, height=0.62, color=HILITE.get(m, NEUTRAL) if m in ("e3", "e4") else NEUTRAL)
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", color=INK, fontsize=12)
        ax.text(1.02, y, f"coverage {cov:.2f}", va="center", color=MUTED, fontsize=12, transform=ax.get_yaxis_transform())
    ax.set_yticks(ys, [NAMES.get(m, m) for m, _, _ in rows])
    ax.set_xlim(0, max([v for _, v, _ in rows] + [0.1]) * 1.18)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=13, colors=INK)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", color=INK, fontsize=15, pad=12)
    fig.tight_layout(); fig.subplots_adjust(right=0.80); fig.savefig(out_png, dpi=160); plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------------- from results.json
def taskb_figures(R: Dict, prefix: str, reader: str = "") -> List[str]:
    S = R["summaries"]
    ms = [m for m in ("e0", "e1", "e2", "e2_bm25", "e3", "e3a_exact", "e3a", "e4") if m in S]
    out = [heatmap(ms, ["G1 hiểu mã", "G2 policy", "G3 tác động", "G4 từ chối đúng"],
                   [[S[m]["G1"].get("exact_accuracy"), S[m]["G2"].get("exact_accuracy"), S[m]["G3"].get("exact_accuracy"),
                     S[m]["G4"].get("correct_abstention_rate")] for m in ms],
                   f"{prefix}_heatmap.png", f"Task B {reader}: exact accuracy (G1-G3), correct abstention (G4)",
                   "Mỗi cột một loại câu hỏi; không gộp thành một accuracy chung.")]
    out.append(risk_coverage({m: S[m]["answerable"].get("rc_curve") for m in ("e2", "e3", "e4") if m in S},
                             f"{prefix}_risk_coverage.png", f"Task B {reader}: risk-coverage, câu trả lời được (G1-G3)",
                             R.get("policy_sweep")))
    out.append(grounding_bars([(m, S[m]["grounding_all"].get("unsupported_over_original_claims") or 0.0,
                                S[m]["answerable"].get("answer_coverage") or 0.0) for m in ms],
                              f"{prefix}_grounding.png", f"Task B {reader}: claim không chứng cứ (cùng mẫu số)",
                              "unsupported claims / original claims, all questions"))
    out.append(labeled_scatter({m: (S[m]["grounding_all"].get("tokens_in_mean"), S[m]["answerable"].get("exact_accuracy"))
                                for m in ms}, f"{prefix}_cost.png", f"Task B {reader}: chi phí vs chất lượng",
                               "input tokens per question", "exact accuracy (G1-G3)"))
    return out


def taska_figure(runs: Dict[str, Dict], out_png: str) -> str:
    """Small multiples: one panel per (dataset, reader); x = strict malicious F1, y = unsupported over original claims."""
    plt = _plt()
    names = list(runs)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.8), squeeze=False)
    for ax, n in zip(axes[0], names):
        R = runs[n]; S = R["summaries"]; G = R.get("grounding_vs_e3") or {}
        for m in ("e0", "e1", "e2", "e3", "e3a_exact", "e3a", "e4"):
            if m not in S:
                continue
            y = (G.get(m) or {}).get("unsupported_over_original_claims", S[m].get("evidence/unsupported_over_original_claims",
                                                                                    S[m].get("evidence/unsupported_claim_rate")))
            x = S[m]["strict/f1_malicious"]
            ax.scatter([x], [y], s=80, color=HILITE.get(m, NEUTRAL), edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(NAMES[m], (x, y), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=12)
        ax.set_title(n, loc="left", color=INK, fontsize=13)
        ax.set_xlabel("strict malicious F1"); ax.set_ylabel("unsupported / original claims")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return out_png
