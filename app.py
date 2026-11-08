"""Streamlit demo: Grover attack on a toy cipher C = S(P XOR K).

Run from the repo root:
    streamlit run app.py

The hardware section only READS saved results from results/*.json, so the
demo never waits on a live hardware queue.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src import cipher
from src.grover import (build_grover_circuit, grover_iteration, key_probabilities_per_iteration,
                        most_likely_key, optimal_iterations, run_on_aer, theoretical_success)
from src.metrics import circuit_metrics, get_fake_backend, transpile_for_backend
from src.oracle import build_oracle
from src.plots import (altair_compare_chart, altair_key_chart, altair_scaling_chart, altair_sweep_chart,
                       counts_to_probs)
from src.run_hardware import load_results
from src.sbox_circuit import sbox_circuit

st.set_page_config(page_title="Grover vs. a toy cipher", layout="wide")


# ---------------------------------------------------------------------------
# Cached computations (re-run only when inputs change)
# ---------------------------------------------------------------------------

@st.cache_data
def cached_probabilities(n_bits: int, plaintext: int, ciphertext: int, iterations: int):
    return key_probabilities_per_iteration(cipher.get_sbox(n_bits), n_bits, plaintext, ciphertext, iterations)


@st.cache_data
def cached_sim_counts(n_bits: int, plaintext: int, ciphertext: int, iterations: int, shots: int, seed: int,
                      relative_phase: bool):
    qc = build_grover_circuit(cipher.get_sbox(n_bits), n_bits, plaintext, ciphertext, iterations,
                              relative_phase=relative_phase)
    return run_on_aer(qc, shots=shots, seed=seed)


@st.cache_data(show_spinner="Transpiling for an IBM device snapshot...")
def cached_cost_table(n_bits: int, plaintext: int, ciphertext: int, iterations: int) -> pd.DataFrame:
    """Gate counts for the exact vs. relative-phase S-box, before and after transpiling (FakeTorino)."""
    backend = get_fake_backend("FakeTorino")
    sb = cipher.get_sbox(n_bits)
    rows = []
    for label, rel in [("exact S-box", False), ("relative-phase S-box", True)]:
        qc = build_grover_circuit(sb, n_bits, plaintext, ciphertext, iterations, measure=False,
                                  relative_phase=rel)
        before, after = circuit_metrics(qc), circuit_metrics(transpile_for_backend(qc, backend))
        rows.append({"circuit": label, "gates (logical)": before["gate_count"],
                     "2-qubit gates (on device)": after["two_qubit_gates"], "depth (on device)": after["depth"]})
    return pd.DataFrame(rows).set_index("circuit")


@st.cache_resource(show_spinner="Drawing circuit...")
def cached_drawing(what: str, n_bits: int, plaintext: int, ciphertext: int, relative_phase: bool):
    sb = cipher.get_sbox(n_bits)
    qc = {"sbox": lambda: sbox_circuit(sb, n_bits, relative_phase=relative_phase),
          "oracle": lambda: build_oracle(sb, n_bits, plaintext, ciphertext, relative_phase=relative_phase),
          "iteration": lambda: grover_iteration(sb, n_bits, plaintext, ciphertext, relative_phase)}[what]()
    return qc.draw("mpl", fold=28, scale=0.8)


def bits(x: int, n: int) -> str:
    return cipher.to_bits(x, n)


# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------

st.sidebar.header("Setup")
n = st.sidebar.radio("Block size n (bits)", [2, 3, 4], index=2, horizontal=True,
                     help="n = 4 uses the PRESENT S-box; n = 2 and 3 are small enough for real hardware.")
N = 2 ** n
fmt = lambda x: f"{bits(x, n)}  ({x})"
plaintext = st.sidebar.selectbox("Plaintext P", range(N), index=min(9, N - 1), format_func=fmt)
key = st.sidebar.selectbox("Secret key K", range(N), index=min(3, N - 1), format_func=fmt)
r_opt = optimal_iterations(n)
iterations = st.sidebar.slider("Grover iterations", 0, max(2 * r_opt + 1, 3), r_opt,
                               help=f"Optimal: floor(pi/4 * sqrt(2^{n})) = {r_opt}")
shots = st.sidebar.select_slider("Simulator shots", [256, 512, 1024, 2048, 4096], value=1024)
seed = st.sidebar.number_input("Simulator seed", value=7, step=1)
relative_phase = st.sidebar.toggle(
    "Relative-phase S-box (cheaper)", value=True,
    help="Uses relative-phase Toffolis inside the S-box. Their extra phases cancel inside the oracle, "
         "so results are identical, but the circuit needs far fewer 2-qubit gates on hardware.")

sbox = cipher.get_sbox(n)
ciphertext = cipher.encrypt(plaintext, key, sbox)

# ---------------------------------------------------------------------------
# 1. The cipher and the classical attack
# ---------------------------------------------------------------------------

st.title("Grover's algorithm vs. a toy block cipher")
st.caption("Known-plaintext attack on  C = S(P ⊕ K). The attacker knows P, C and the S-box, and searches for K.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Plaintext P", bits(plaintext, n))
c2.metric("Ciphertext C", bits(ciphertext, n))
found, guesses = cipher.brute_force(plaintext, ciphertext, sbox, n)
c3.metric("Classical guesses (this key)", f"{guesses} / {N}", help=f"Average: {(N + 1) / 2:.1f}")
c4.metric("Grover oracle calls", r_opt, help="floor(pi/4 * sqrt(2^n)), the same for every key")

with st.expander("S-box table"):
    st.dataframe(pd.DataFrame({"x": [bits(x, n) for x in range(N)], "S(x)": [bits(y, n) for y in sbox]}).T,
                 width="stretch")

# ---------------------------------------------------------------------------
# 2. Amplitude amplification, step by step
# ---------------------------------------------------------------------------

st.header("1 · Key probabilities after each Grover iteration")
st.caption("Exact statevector simulation. Each iteration = oracle (flip the phase of the right key) + diffusion "
           "(reflect about the mean), which moves probability onto the secret key.")

probs = cached_probabilities(n, plaintext, ciphertext, iterations)
chart_slot = st.empty()
col_a, col_b = st.columns([4, 1])
step = col_a.slider("Show iteration", 0, iterations, iterations, key="step") if iterations > 0 else 0
play = col_b.button("▶ Play", width="stretch", disabled=iterations == 0)


def draw(r: int) -> None:
    chart_slot.altair_chart(
        altair_key_chart(probs[r], n, key,
                         title=f"After {r} iteration(s): P(secret key) = {probs[r][key]:.1%}"),
        width="stretch")


if play:
    for r in range(iterations + 1):
        draw(r)
        time.sleep(0.8)
else:
    draw(step)

if iterations > r_opt:
    st.info(f"Past the optimum ({r_opt}), the probability goes back down: Grover 'overshoots'.")

# ---------------------------------------------------------------------------
# 3. Simulator
# ---------------------------------------------------------------------------

st.header("2 · Aer simulator (noise-free, sampled)")
counts = cached_sim_counts(n, plaintext, ciphertext, iterations, shots, int(seed), relative_phase)
sim_probs = counts_to_probs(counts, n)
left, right = st.columns([3, 1])
left.altair_chart(altair_key_chart(sim_probs, n, key, title=f"{shots} shots, {iterations} iteration(s)"),
                  width="stretch")
guess = most_likely_key(counts)
right.metric("Most frequent key", bits(guess, n), "correct" if guess == key else "wrong",
             delta_color="normal" if guess == key else "inverse")
right.metric("P(secret key)", f"{sim_probs[key]:.1%}", help=f"Ideal: {theoretical_success(n, iterations):.1%}")
right.metric("Random guess", f"1 in {N}", help=f"{1 / N:.2%}")

with st.expander("Circuit cost: exact vs. relative-phase S-box"):
    st.caption("On real hardware every gate must become 1- and 2-qubit gates on connected qubits. "
               "2-qubit gates are the noisiest, so fewer is better. Numbers are for an offline "
               "snapshot of IBM's Torino device (optimization level 3).")
    cost = cached_cost_table(n, plaintext, ciphertext, iterations)
    st.dataframe(cost, width="stretch")
    ex, rp = cost["2-qubit gates (on device)"]
    if ex > 0 and rp < ex:
        st.success(f"Relative-phase S-box: {ex - rp} fewer 2-qubit gates ({(ex - rp) / ex:.0%} less).")

with st.expander("See the circuits"):
    if st.toggle("Draw circuits", value=False, help="Drawing large circuits takes a few seconds."):
        tabs = st.tabs(["S-box", "Oracle", "One Grover iteration"])
        for tab, what in zip(tabs, ["sbox", "oracle", "iteration"]):
            tab.pyplot(cached_drawing(what, n, plaintext, ciphertext, relative_phase), width="stretch")

# ---------------------------------------------------------------------------
# 4. Hardware (saved results only)
# ---------------------------------------------------------------------------

st.header("3 · Under IBM device noise (saved runs)")
st.caption("Saved runs from results/. Runs on a *fake backend* use an offline noise model of a real IBM "
           "device; any real-hardware runs are labelled 'real hardware'.")
records = [r for r in load_results() if r.get("n_bits") == n]


def kind_label(r: dict) -> str:
    return "real hardware" if r["backend_kind"] == "hardware" else "fake backend (noisy sim)"


def sbox_label(r: dict) -> str:
    # Older files have no "relative_phase" field: they used the exact S-box.
    return "relative-phase S-box" if r.get("relative_phase", False) else "exact S-box"


if not records and n == 4:
    st.info("**n = 4 is a simulator-only demo.** Even with the cheaper S-box, the 4-bit attack needs about "
            "640 two-qubit gates after compiling for an IBM device, far too deep to survive today's noise. "
            "Switch to **n = 3** (P = 101, K = 110) or **n = 2** (P = 01, K = 10) in the sidebar to see "
            "the saved noisy runs.")
elif not records:
    st.info(f"No saved runs for n = {n} yet. Create a noisy local simulation with:\n\n"
            f"`python -m src.run_hardware --n {n} --plaintext {plaintext} --key {key} --sweep --fake`")
else:
    matching = [r for r in records if r["plaintext"] == plaintext and r["key"] == key]
    if not matching:
        st.info("No saved run matches this exact (P, K). Showing other saved runs for this n instead.")
    pool = matching or records
    label = lambda r: (f"{kind_label(r)} · {r['backend']} · {sbox_label(r)} · P={bits(r['plaintext'], n)} "
                       f"K={bits(r['key'], n)}{' · sweep' if r.get('sweep') else ''} · {r['timestamp']}")
    rec = pool[st.selectbox("Saved run", range(len(pool)), format_func=lambda i: label(pool[i]))]
    if rec["backend_kind"] != "hardware":
        st.caption("⚠️ This is a *fake backend*: a local simulation using a noise model of an IBM device, "
                   "not a real quantum computer.")

    hw_probs = counts_to_probs(rec["counts"], n)
    left, right = st.columns([3, 1])
    left.altair_chart(altair_key_chart(hw_probs, n, rec["key"],
                                       title=f"{kind_label(rec)}: {rec['backend']}, {rec['iterations']} "
                                             f"iteration(s), {rec['shots']} shots"),
                      width="stretch")
    right.metric("P(secret key)", f"{hw_probs[rec['key']]:.1%}",
                 help=f"Ideal: {rec['theoretical_success']:.1%}; random: {1 / N:.1%}")
    hw_guess = most_likely_key(rec["counts"])
    right.metric("Most frequent key", bits(hw_guess, n), "correct" if hw_guess == rec["key"] else "wrong",
                 delta_color="normal" if hw_guess == rec["key"] else "inverse")
    tm = rec.get("transpiled_metrics", {})
    right.metric("Transpiled 2-qubit gates", tm.get("two_qubit_gates", "?"))
    right.metric("Transpiled depth", tm.get("depth", "?"))

    # Sweep: amplitude growth measured on the device, all iteration counts from ONE job.
    if rec.get("sweep"):
        st.subheader("Amplitude growth under noise")
        st.caption("Every iteration count below ran in the same job. Noise pulls the measured curve below "
                   "the ideal one, and more iterations mean more gates and more noise.")
        st.altair_chart(altair_sweep_chart(rec["sweep"], n, f"measured ({kind_label(rec)})"), width="stretch")

    # Side by side: ideal vs. every saved run for this same (n, P, K).
    same = [r for r in records if r["plaintext"] == rec["plaintext"] and r["key"] == rec["key"]]
    if same:
        st.subheader("Side by side: ideal vs. saved noisy runs")
        rows = [{"run": f"ideal (noise-free), {rec['iterations']} iteration(s)",
                 "probability": theoretical_success(n, rec["iterations"]), "two_qubit_gates": None}]
        for r in sorted(same, key=lambda r: (r["backend_kind"] != "fake", r.get("relative_phase", False))):
            rows.append({"run": f"{kind_label(r)} · {sbox_label(r)} · {r['backend']}",
                         "probability": r["correct_key_probability"],
                         "two_qubit_gates": r.get("transpiled_metrics", {}).get("two_qubit_gates")})
        st.altair_chart(altair_compare_chart(rows), width="stretch")

    st.caption(f"File: results/{rec['_file']} · job {rec.get('job_id')} · "
               f"dynamical decoupling: {rec.get('dynamical_decoupling')}")

# ---------------------------------------------------------------------------
# 5. Why it matters
# ---------------------------------------------------------------------------

st.header("4 · Why it matters: scaling to real ciphers")
st.caption("Brute-forcing an n-bit key takes about 2ⁿ/2 guesses classically, while Grover needs about "
           "(π/4)·√2ⁿ oracle calls: a quadratic speed-up that effectively halves the key length.")
st.altair_chart(altair_scaling_chart(), width="stretch")
st.markdown(
    "- **This project (n = 4):** 8.5 guesses on average vs. 3 Grover iterations.\n"
    "- **AES-128:** about 2¹²⁷ guesses vs. about 2⁶⁴ Grover iterations, still far out of reach. Each "
    "iteration would need a full AES circuit on a large error-corrected quantum computer, and the "
    "iterations must run one after another.\n"
    "- **The defense is simple:** double the key length. Against Grover, AES-256 keeps about 128 bits of "
    "security.\n"
    "- **Caveat for this toy:** one (P, C) pair already reveals K = S⁻¹(C) ⊕ P. The point here is the "
    "mechanism and its hardware cost, not a break.")
