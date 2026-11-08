"""Shared bar charts for the app (Altair, interactive) and notebook (matplotlib).

One job per chart: show the probability of each key, with the secret key
highlighted. Other keys are muted gray; the secret key is blue and also
labelled, so identity never depends on color alone.
"""

from __future__ import annotations

import numpy as np

KEY_COLOR = "#2a78d6"    # highlight: the secret key / measured results / Grover
OTHER_COLOR = "#898781"  # muted: every other key / ideal reference
CLASSICAL_COLOR = "#eb6834"  # classical brute force (second series)
INK = "#52514e"          # secondary text
GRID = "#e1e0d9"         # hairline grid


def _rows(values, n_bits: int, key: int) -> list[dict]:
    return [{
        "key": format(k, f"0{n_bits}b"),
        "probability": float(values[k]),
        "role": "secret key" if k == key else "other keys",
    } for k in range(2 ** n_bits)]


def counts_to_probs(counts: dict[str, int], n_bits: int) -> np.ndarray:
    """Convert a counts dict into an array of probabilities indexed by key."""
    total = sum(counts.values()) or 1
    probs = np.zeros(2 ** n_bits)
    for bits, c in counts.items():
        probs[int(bits, 2)] = c / total
    return probs


def altair_key_chart(values, n_bits: int, key: int, title: str = "", y_max: float | None = 1.0,
                     y_title: str = "Probability"):
    """Interactive bar chart of P(key) with the secret key highlighted (hover for values)."""
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(_rows(values, n_bits, key))
    y_scale = alt.Scale(domain=[0, y_max]) if y_max else alt.Scale()
    color = alt.Color("role:N", title=None,
                      scale=alt.Scale(domain=["secret key", "other keys"], range=[KEY_COLOR, OTHER_COLOR]),
                      legend=alt.Legend(orient="top"))
    base = alt.Chart(df).encode(x=alt.X("key:N", title="Key (bits)", sort=None, axis=alt.Axis(labelAngle=-45 if n_bits > 3 else 0)))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y("probability:Q", title=y_title, scale=y_scale, axis=alt.Axis(format="%")),
        color=color,
        tooltip=[alt.Tooltip("key:N", title="Key"),
                 alt.Tooltip("probability:Q", title="Probability", format=".1%"),
                 alt.Tooltip("role:N", title="")],
    )
    # Direct label on the secret key only (selective labelling).
    label = base.transform_filter(alt.datum.role == "secret key").mark_text(dy=-8, color=INK).encode(
        y=alt.Y("probability:Q", scale=y_scale), text=alt.Text("probability:Q", format=".0%"))
    return (bars + label).properties(title=title, height=280)


def mpl_key_chart(ax, values, n_bits: int, key: int, title: str = "", y_max: float | None = 1.0):
    """Same chart in matplotlib, drawn on an existing Axes (for the notebook)."""
    labels = [format(k, f"0{n_bits}b") for k in range(2 ** n_bits)]
    colors = [KEY_COLOR if k == key else OTHER_COLOR for k in range(2 ** n_bits)]
    ax.bar(labels, values, color=colors, width=0.7)
    ax.annotate(f"{values[key]:.0%}", (key, values[key]), textcoords="offset points",
                xytext=(0, 4), ha="center", color=INK, fontsize=9)
    if y_max:
        ax.set_ylim(0, y_max)
    ax.set_title(title)
    ax.set_xlabel("Key (bits)")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if n_bits > 3:
        ax.tick_params(axis="x", labelrotation=45)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=KEY_COLOR, label="secret key"), Patch(color=OTHER_COLOR, label="other keys")],
              frameon=False, loc="upper left")
    return ax


def altair_sweep_chart(sweep: list[dict], n_bits: int, measured_label: str):
    """P(secret key) after r = 0, 1, 2, ... iterations: measured vs. ideal, plus the random-guess line."""
    import altair as alt
    import pandas as pd

    rows = []
    for e in sweep:
        rows.append({"iterations": e["iterations"], "probability": e["correct_key_probability"],
                     "series": measured_label})
        rows.append({"iterations": e["iterations"], "probability": e["theoretical_success"],
                     "series": "ideal (noise-free)"})
    df = pd.DataFrame(rows)
    domain = [measured_label, "ideal (noise-free)"]
    color = alt.Color("series:N", title=None, scale=alt.Scale(domain=domain, range=[KEY_COLOR, OTHER_COLOR]),
                      legend=alt.Legend(orient="top", labelLimit=400))
    dash = alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=[[1, 0], [5, 4]]), legend=None)
    x = alt.X("iterations:O", title="Grover iterations", axis=alt.Axis(labelAngle=0))
    y = alt.Y("probability:Q", title="P(secret key)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"))
    tooltip = [alt.Tooltip("series:N", title=""), alt.Tooltip("iterations:O", title="Iterations"),
               alt.Tooltip("probability:Q", title="P(secret key)", format=".1%")]
    base = alt.Chart(df).encode(x=x, y=y, color=color)
    lines = base.mark_line(strokeWidth=2).encode(strokeDash=dash)
    points = base.mark_point(size=80, filled=True).encode(tooltip=tooltip)
    guess = 1 / 2 ** n_bits
    last_r = max(e["iterations"] for e in sweep)
    rule_df = pd.DataFrame({"probability": [guess], "iterations": [last_r], "label": [f"random guess {guess:.1%}"]})
    rule = alt.Chart(rule_df).mark_rule(color=OTHER_COLOR, strokeDash=[2, 2]).encode(y="probability:Q")
    # Label sits above the line, just left of the last iteration (clear of the data points).
    rule_text = alt.Chart(rule_df).mark_text(align="right", dx=-14, dy=-7, color=INK).encode(
        y="probability:Q", x=alt.X("iterations:O"), text="label:N")
    return (rule + rule_text + lines + points).properties(height=280)


def altair_compare_chart(rows: list[dict]):
    """Horizontal bars: P(secret key) for each run (ideal / saved runs), one row per run.

    rows: [{"run": label, "probability": p, "two_qubit_gates": g or None}, ...]
    """
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(rows)
    # The ideal row is a reference (muted gray); measured runs are the data (blue).
    df["kind"] = ["ideal" if r.startswith("ideal") else "measured" for r in df["run"]]
    y = alt.Y("run:N", title=None, sort=None, axis=alt.Axis(labelLimit=320))
    x = alt.X("probability:Q", title="P(secret key) at the optimal iteration count",
              scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5))
    color = alt.Color("kind:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["measured", "ideal"], range=[KEY_COLOR, OTHER_COLOR]))
    bars = alt.Chart(df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18).encode(
        y=y, x=x, color=color,
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("probability:Q", title="P(secret key)", format=".1%"),
                 alt.Tooltip("two_qubit_gates:Q", title="Transpiled 2-qubit gates")])
    labels = alt.Chart(df).mark_text(align="left", dx=4, color=INK).encode(
        y=y, x=x, text=alt.Text("probability:Q", format=".0%"))
    return (bars + labels).properties(height=max(90, 34 * len(df)))


def altair_scaling_chart(max_bits: int = 128, highlight: dict[int, str] | None = None):
    """Key-search cost vs. key size: classical (2^n / 2) vs. Grover (pi/4 * sqrt(2^n)), on a log2 scale."""
    import math

    import altair as alt
    import pandas as pd

    highlight = highlight or {4: "this project (n = 4)", 56: "DES (56-bit)", 128: "AES-128"}
    rows = []
    for n in range(2, max_bits + 1, 2) if max_bits > 8 else range(2, max_bits + 1):
        rows.append({"bits": n, "log2_ops": (n - 1), "method": "classical brute force (avg 2ⁿ/2)"})
        rows.append({"bits": n, "log2_ops": math.log2(math.pi / 4) + n / 2, "method": "Grover (≈ π/4 · √2ⁿ)"})
    df = pd.DataFrame(rows)
    domain = ["classical brute force (avg 2ⁿ/2)", "Grover (≈ π/4 · √2ⁿ)"]
    color = alt.Color("method:N", title=None, scale=alt.Scale(domain=domain, range=[CLASSICAL_COLOR, KEY_COLOR]),
                      legend=alt.Legend(orient="top"))
    x = alt.X("bits:Q", title="Key length (bits)", scale=alt.Scale(domain=[0, max_bits]))
    y = alt.Y("log2_ops:Q", title="Key guesses / oracle calls (log scale)",
              axis=alt.Axis(labelExpr="'2^' + datum.value", tickMinStep=1))
    tooltip = [alt.Tooltip("method:N", title=""), alt.Tooltip("bits:Q", title="Key bits"),
               alt.Tooltip("log2_ops:Q", title="log₂(operations)", format=".1f")]
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(x=x, y=y, color=color, tooltip=tooltip)

    marks = []
    for n, label in highlight.items():
        g = math.log2(math.pi / 4) + n / 2
        small = n < 16  # both points nearly overlap here: one combined label
        marks.append({"bits": n, "log2_ops": n - 1, "method": domain[0],
                      "label": label if small else f"{label}: 2^{n - 1:g} guesses"})
        marks.append({"bits": n, "log2_ops": g, "method": domain[1],
                      "label": "" if small else f"Grover: 2^{g:.1f}"})
    mdf = pd.DataFrame(marks)
    points = alt.Chart(mdf).mark_point(size=70, filled=True).encode(x=x, y=y, color=color, tooltip=tooltip)
    # Labels left of the point near the right edge, right of it near the left edge.
    # The lines rise to the right, so labels go up-and-LEFT of each point (clear of the line);
    # the small-n label goes up-and-right instead, since there is no room on the left.
    text_big = alt.Chart(mdf[mdf.bits >= 16]).mark_text(align="right", dx=-10, dy=-12, color=INK).encode(
        x=x, y=y, text="label:N")
    text_small = alt.Chart(mdf[mdf.bits < 16]).mark_text(align="left", dx=6, dy=-16, color=INK).encode(
        x=x, y=y, text="label:N")
    return (lines + points + text_big + text_small).properties(height=320)
