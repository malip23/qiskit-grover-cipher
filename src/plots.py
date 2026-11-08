"""Shared bar charts for the app (Altair, interactive) and notebook (matplotlib).

One job per chart: show the probability of each key, with the secret key
highlighted. Other keys are muted gray; the secret key is blue and also
labelled, so identity never depends on color alone.
"""

from __future__ import annotations

import numpy as np

KEY_COLOR = "#2a78d6"    # highlight: the secret key
OTHER_COLOR = "#898781"  # muted: every other key
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
