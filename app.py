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
from src.grover import (build_grover_circuit, key_probabilities_per_iteration, most_likely_key,
                        optimal_iterations, run_on_aer, theoretical_success)
from src.metrics import circuit_metrics
from src.plots import altair_key_chart, counts_to_probs
from src.run_hardware import load_results

st.set_page_config(page_title="Grover vs. a toy cipher", layout="wide")


# ---------------------------------------------------------------------------
# Cached computations (re-run only when inputs change)
# ---------------------------------------------------------------------------

@st.cache_data
def cached_probabilities(n_bits: int, plaintext: int, ciphertext: int, iterations: int):
    return key_probabilities_per_iteration(cipher.get_sbox(n_bits), n_bits, plaintext, ciphertext, iterations)


@st.cache_data
def cached_sim_counts(n_bits: int, plaintext: int, ciphertext: int, iterations: int, shots: int, seed: int):
    qc = build_grover_circuit(cipher.get_sbox(n_bits), n_bits, plaintext, ciphertext, iterations)
    return run_on_aer(qc, shots=shots, seed=seed)


@st.cache_data
def cached_metrics(n_bits: int, plaintext: int, ciphertext: int, iterations: int):
    qc = build_grover_circuit(cipher.get_sbox(n_bits), n_bits, plaintext, ciphertext, iterations, measure=False)
    return circuit_metrics(qc)


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
counts = cached_sim_counts(n, plaintext, ciphertext, iterations, shots, int(seed))
sim_probs = counts_to_probs(counts, n)
left, right = st.columns([3, 1])
left.altair_chart(altair_key_chart(sim_probs, n, key, title=f"{shots} shots, {iterations} iteration(s)"),
                  width="stretch")
guess = most_likely_key(counts)
right.metric("Most frequent key", bits(guess, n), "correct" if guess == key else "wrong",
             delta_color="normal" if guess == key else "inverse")
right.metric("P(secret key)", f"{sim_probs[key]:.1%}", help=f"Ideal: {theoretical_success(n, iterations):.1%}")
right.metric("Random guess", f"{1 / N:.1%}")

with st.expander("Circuit cost (before transpiling)"):
    m = cached_metrics(n, plaintext, ciphertext, iterations)
    st.write({k: v for k, v in m.items() if k != "ops"})
    st.write("Gate counts:", m["ops"])

# ---------------------------------------------------------------------------
# 4. Hardware (saved results only)
# ---------------------------------------------------------------------------

st.header("3 · Real IBM quantum hardware (saved results)")
records = [r for r in load_results() if r.get("n_bits") == n]

if not records:
    st.warning(f"No saved results for n = {n} in results/. Create one with:\n\n"
               f"`python -m src.run_hardware --n {n} --plaintext {plaintext} --key {key}`  (real hardware)\n\n"
               f"`python -m src.run_hardware --n {n} --plaintext {plaintext} --key {key} --fake`  (noisy local simulation)")
else:
    exact = [r for r in records if r["plaintext"] == plaintext and r["key"] == key]
    if not exact:
        st.info("No saved run matches this exact (P, K). Showing other saved runs for this n instead.")
    pool = exact or records
    label = lambda r: (f"{r['backend_kind']} · {r['backend']} · P={bits(r['plaintext'], n)} "
                       f"K={bits(r['key'], n)} · {r['timestamp']}")
    rec = pool[st.selectbox("Saved run", range(len(pool)), format_func=lambda i: label(pool[i]))]
    hw_probs = counts_to_probs(rec["counts"], n)
    kind = "Real hardware" if rec["backend_kind"] == "hardware" else "Fake backend (noisy local simulation)"
    left, right = st.columns([3, 1])
    left.altair_chart(altair_key_chart(hw_probs, n, rec["key"],
                                       title=f"{kind}: {rec['backend']}, {rec['shots']} shots"),
                      width="stretch")
    right.metric("P(secret key)", f"{hw_probs[rec['key']]:.1%}",
                 help=f"Ideal: {rec['theoretical_success']:.1%}; random: {1 / N:.1%}")
    hw_guess = most_likely_key(rec["counts"])
    right.metric("Most frequent key", bits(hw_guess, n), "correct" if hw_guess == rec["key"] else "wrong",
                 delta_color="normal" if hw_guess == rec["key"] else "inverse")
    tm = rec.get("transpiled_metrics", {})
    right.metric("Transpiled 2-qubit gates", tm.get("two_qubit_gates", "?"))
    right.metric("Transpiled depth", tm.get("depth", "?"))
    st.caption(f"File: results/{rec['_file']} · job {rec.get('job_id')} · "
               f"dynamical decoupling: {rec.get('dynamical_decoupling')}")
