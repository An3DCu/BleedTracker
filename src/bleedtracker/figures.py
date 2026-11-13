"""The three figures, drawn from `reports/results.json` only.

The layout and colors follow my original figures. The text is rewritten. Every
number on a figure comes from the results file.
"""

import io
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure

RULE = "#22314F"
FOREST = "#1B4ED8"
BASE = "#7A8294"
ACCENT = "#2F6FED"
HIGHLIGHT = "#D8E2F8"
BELOW = "#B4243A"
GREYS = ["#6E7789", "#8C94A4", "#AEB4C0", "#C9CDD6"]
MINUS = "−"
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four"}
ACTION_ORDER = [
    "rewrite_title_and_refresh",
    "refresh_now",
    "expand_or_consolidate",
    "verify_then_review",
    "investigate_quiet_risk",
]
DISPOSITION_LABELS = {
    "editor_slots": "editor slots",
    "shadow_items": "shadow items",
    "rising": "rising, monitor only",
    "no_first_half_traffic": "no first-half traffic, monitor only",
    "eligible_not_sliding": "eligible, not sliding",
    "eligible_sliding_below_the_cut": "sliding, below the cut",
}


def signed(value: float) -> str:
    """A number with its sign. Negatives get a minus sign."""
    return f"{value:+.1f}".replace("-", MINUS)


def month_label(deployment: str) -> str:
    """'forward 2026-02 to 2026-03' becomes '2026-02 to 03'."""
    *_, start, _, end = deployment.split()
    return f"{start} to {end[-2:]}"


def months(count: int) -> str:
    return f"{WORDS.get(count, count)} month{'' if count == 1 else 's'}"


def draw_deployments(results: dict) -> Figure:
    """The first figure. It shows the four forward deployments and their margins."""
    rows = results["deployments"][1:]
    labels = [month_label(r["deployment"]) for r in rows]
    rule = [r["rule_precision_at_50"] for r in rows]
    forest = [r["forest_precision_at_50"] for r in rows]
    base = [r["base_rate"] for r in rows]
    margin = [r["margin_points"] for r in rows]
    floor = results["pre_registered_thresholds"]["queue_margin"] * 100
    x = np.arange(len(rows))
    sealed = len(rows) - 1

    fig = Figure(figsize=(7.4, 8.8))
    ax1, ax2 = fig.subplots(2, 1)
    ax1.axvspan(sealed - 0.5, sealed + 0.5, color=HIGHLIGHT, zorder=0)
    ax1.plot(x, rule, marker="o", ls="-", color=RULE, lw=2, zorder=3)
    ax1.plot(x, forest, marker="s", ls="--", color=FOREST, lw=2, zorder=3)
    ax1.plot(x, base, marker="^", ls=":", color=BASE, lw=1.8, zorder=3)
    for xi, (r, f, b) in enumerate(zip(rule, forest, base, strict=True)):
        up = r >= f
        ax1.annotate(
            f"{r:.2f}", (xi, r + (0.035 if up else -0.06)), ha="center", fontsize=10, color=RULE
        )
        ax1.annotate(
            f"{f:.2f}", (xi, f + (-0.06 if up else 0.035)), ha="center", fontsize=10, color=FOREST
        )
        ax1.annotate(f"{b:.2f}", (xi, b - 0.07), ha="center", fontsize=10, color=BASE)
    legend = [
        ("my rule (the queue that shipped)", RULE),
        ("random forest", FOREST),
        ("base rate of the month", BASE),
    ]
    for i, (text, color) in enumerate(legend):
        ax1.annotate(
            text,
            xy=(0.03, 0.955 - i * 0.072),
            xycoords="axes fraction",
            fontsize=10.5,
            color=color,
            weight="bold",
        )
    ax1.annotate(
        "sealed month,\nopened once",
        xy=(sealed, 1.22),
        ha="center",
        fontsize=9.5,
        color=RULE,
        weight="bold",
    )
    ax1.set_xticks(x, labels, fontsize=10)
    ax1.set_xlim(-0.4, len(rows) - 0.6)
    ax1.set_ylim(0.05, 1.32)
    ax1.set_ylabel("precision@50 (share of the 50 that declined)")
    ax1.set_xlabel("month the queue was released for")
    ax1.set_title(
        "a) Precision@50 in each forward deployment\n"
        "Trained on month t and judged on held-out clients in month t + 1",
        fontsize=11,
        loc="left",
    )
    ax1.grid(alpha=0.25, ls=":")

    colors = [ACCENT if i == sealed else RULE for i in range(len(rows))]
    ax2.bar(x, margin, color=colors, width=0.6)
    for xi, v in zip(x, margin, strict=True):
        ax2.text(
            xi,
            v + 1.6,
            signed(v),
            ha="center",
            fontsize=11,
            weight="bold",
            color=ACCENT if xi == sealed else RULE,
        )
    ax2.axhline(floor, color=FOREST, ls="--", lw=1.5, zorder=4)
    ax2.annotate(
        f"The dashed line is the {floor:.0f}-point margin I set before the sealed month.\n"
        "Below it, a queue isn't worth 50 editor slots.",
        xy=(0.03, 0.90),
        xycoords="axes fraction",
        fontsize=9.5,
        color=FOREST,
    )
    ax2.set_xticks(x, labels, fontsize=10)
    ax2.set_ylim(0, 96)
    ax2.set_ylabel("precision@50 minus the month's base rate (points)")
    ax2.set_xlabel("month the queue was released for")
    ax2.set_title(
        "b) Margin over the month's base rate\n"
        "The sealed month has the highest precision and the smallest margin",
        fontsize=11,
        loc="left",
    )
    ax2.grid(axis="y", alpha=0.25, ls=":")
    fig.suptitle("Four forward deployments, judged on held-out clients", fontsize=11.5, y=1.02)
    fig.tight_layout()
    return fig


def draw_queue_length(results: dict) -> Figure:
    """The second figure. It shows precision at every queue length and the shadow lift."""
    curve = results["sealed_precision_at_k"]
    k = [r["K"] for r in curve]
    base = curve[0]["base_rate"]
    sealed = results["deployments"][-1]
    decision = results["decision_envelope"]["decision_date"]
    lift_rows = results["shadow_queue"][1:]
    lift = [r["shadow_lift_points"] for r in lift_rows]
    interval = results["uncertainty"]["shadow_lift_interval_90"]
    mean = results["uncertainty"]["shadow_lift_mean_points"]

    fig = Figure(figsize=(7.4, 8.8))
    ax1, ax2 = fig.subplots(2, 1)
    ax1.axvspan(40, 60, color=HIGHLIGHT, zorder=0)
    ax1.plot(k, [r["rule"] for r in curve], marker="o", ls="-", color=RULE, lw=2, zorder=3)
    ax1.plot(k, [r["forest"] for r in curve], marker="s", ls="--", color=FOREST, lw=2, zorder=3)
    ax1.axhline(base, color=BASE, ls=":", lw=1.8, zorder=2)
    ax1.annotate(
        "review budget\nK = 50",
        xy=(50, 0.745),
        ha="center",
        fontsize=9.5,
        color=ACCENT,
        weight="bold",
    )
    ax1.annotate(
        f"base rate {base:.0%}. Random pages get this on average.",
        xy=(88, base + 0.012),
        fontsize=9.5,
        color=BASE,
    )
    ax1.annotate(
        "my rule",
        xy=(252, curve[-1]["rule"] - 0.030),
        ha="right",
        fontsize=10.5,
        color=RULE,
        weight="bold",
    )
    ax1.annotate(
        "random forest",
        xy=(252, curve[-1]["forest"] - 0.030),
        ha="right",
        fontsize=10.5,
        color=FOREST,
        weight="bold",
    )
    ax1.set_xlabel("K, the length of the queue (pages)")
    ax1.set_ylabel("precision@K (share that declined next month)")
    ax1.set_title(
        "a) The sealed month at every queue length\n"
        f"Decision on {decision}, {sealed['clients']} held-out clients",
        fontsize=11,
        loc="left",
    )
    ax1.set_ylim(0.70, 1.035)
    ax1.set_xlim(0, 262)
    ax1.grid(alpha=0.25, ls=":")

    x = np.arange(len(lift))
    ax2.bar(x, lift, color=[RULE if v > 0 else BELOW for v in lift], width=0.6)
    ax2.axhline(0, color=RULE, lw=1.2)
    ax2.axhline(mean, color=FOREST, ls="--", lw=1.5)
    holds_zero = "includes" if interval[0] <= 0 <= interval[1] else "doesn't include"
    ax2.annotate(
        f"Mean {signed(mean)} points. The 90% interval\n"
        f"from {signed(interval[0])} to {signed(interval[1])} {holds_zero} zero.",
        xy=(0.03, 0.90),
        xycoords="axes fraction",
        fontsize=9.5,
        color=FOREST,
    )
    for xi, v in zip(x, lift, strict=True):
        ax2.text(
            xi,
            v + (2.5 if v > 0 else -5.5),
            signed(v),
            ha="center",
            fontsize=11,
            weight="bold",
            color=RULE if v > 0 else BELOW,
        )
    ax2.set_xticks(x, [month_label(r["deployment"]) for r in lift_rows], fontsize=10)
    ax2.set_ylim(-22, 88)
    ax2.set_ylabel("shadow precision minus its pool's base rate (points)")
    ax2.set_xlabel("month the shadow queue was made for")
    above = sum(v > 0 for v in lift)
    below = sum(v < 0 for v in lift)
    sealed_note = ", including the sealed month." if lift[-1] < 0 else "."
    ax2.set_title(
        "b) The shadow queue against its own pool\n"
        f"Above its pool in {months(above)}. Below it in {WORDS.get(below, below)}"
        f"{sealed_note}",
        fontsize=11,
        loc="left",
    )
    ax2.grid(axis="y", alpha=0.25, ls=":")
    fig.suptitle(
        "Precision at every queue length and the shadow queue's lift", fontsize=11.5, y=1.02
    )
    fig.tight_layout()
    return fig


def draw_queue(results: dict) -> Figure:
    """The third figure. It shows the sealed queue and where every held-out page went."""
    queue = results["sealed_queue"]
    editor_mix = queue["action_mix_editor"]
    shadow_mix = queue["action_mix_shadow"]
    actions = [a for a in ACTION_ORDER if a in editor_mix or a in shadow_mix]
    disposition = queue["page_disposition"]
    pages = sum(disposition.values())
    queued = queue["editor_slots"] + queue["shadow_items"]
    decision = results["decision_envelope"]["decision_date"]

    fig = Figure(figsize=(7.4, 7.6))
    ax1, ax2 = fig.subplots(2, 1)
    y = np.arange(len(actions))
    editor = [int(editor_mix.get(a, 0)) for a in actions]
    shadow = [int(shadow_mix.get(a, 0)) for a in actions]
    ax1.barh(y, editor, color=RULE, label="editor slot")
    ax1.barh(
        y, shadow, left=editor, color=ACCENT, hatch="//", edgecolor="white", label="shadow item"
    )
    for yi, (e, s) in enumerate(zip(editor, shadow, strict=True)):
        if e:
            ax1.text(
                e / 2,
                yi,
                str(e),
                va="center",
                ha="center",
                color="white",
                fontsize=10.5,
                weight="bold",
            )
        if s:
            ax1.text(
                e + s / 2,
                yi,
                str(s),
                va="center",
                ha="center",
                color="white",
                fontsize=10.5,
                weight="bold",
            )
    ax1.set_yticks(y, [a.replace("_", " ") for a in actions], fontsize=10.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("items in the queue")
    ax1.set_xlim(0, max(e + s for e, s in zip(editor, shadow, strict=True)) * 1.08)
    ax1.set_title(
        "a) What the queue asks an editor to do\n"
        f"{queue['editor_slots']} editor slots and {queue['shadow_items']} shadow items "
        f"on {decision}",
        fontsize=11,
        loc="left",
    )
    ax1.legend(fontsize=10, frameon=False, loc="lower right")
    ax1.grid(axis="x", alpha=0.25, ls=":")

    values = list(disposition.values())
    yy = np.arange(len(values))
    ax2.barh(yy, values, color=[RULE, ACCENT, *GREYS])
    for yi, v in enumerate(values):
        ax2.text(v * 1.18, yi, f"{v:,}", va="center", fontsize=10)
    ax2.set_yticks(yy, [DISPOSITION_LABELS[k] for k in disposition], fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlim(0.7, pages * 4)
    ax2.set_xlabel(f"pages (log scale), {pages:,} held-out pages in total")
    ax2.set_title(
        "b) Where every held-out page went\n"
        f"The queue has {queued} places. The other {pages - queued:,} pages stay off it.",
        fontsize=11,
        loc="left",
    )
    ax2.grid(axis="x", alpha=0.25, ls=":")
    fig.suptitle(
        "The sealed month's queue and every held-out page behind it", fontsize=11.5, y=1.02
    )
    fig.tight_layout()
    return fig


FIGURES = {
    "figure_1_deployments": draw_deployments,
    "figure_2_queue_length": draw_queue_length,
    "figure_3_queue": draw_queue,
}


def draw_all(results: dict, folder: Path) -> list[Path]:
    """Draw every figure as PNG and SVG.

    The SVG carries no date and LF line endings. A redraw gives the same files.
    """
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    with matplotlib.rc_context({"svg.hashsalt": "bleedtracker"}):
        for name, draw in FIGURES.items():
            fig = draw(results)
            png = folder / f"{name}.png"
            svg = folder / f"{name}.svg"
            fig.savefig(
                png, dpi=150, bbox_inches="tight", facecolor="white", metadata={"Software": None}
            )
            buffer = io.BytesIO()
            fig.savefig(
                buffer,
                format="svg",
                bbox_inches="tight",
                facecolor="white",
                metadata={"Date": None},
            )
            svg.write_bytes(buffer.getvalue().replace(b"\r\n", b"\n"))
            written += [png, svg]
    return written
