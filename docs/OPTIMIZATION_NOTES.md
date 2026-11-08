# Optimization notes: the S-box circuit and circuit costs

The pipeline (oracle → Grover → simulator → hardware → demo) works end to end with a *baseline* S-box
circuit. The S-box is the most expensive part of the oracle, so it's the main lever for better results
on real hardware. This page records what the baseline costs and where to go next.

## The contract (keep it, or update the callers too)

```python
def sbox_circuit(sbox: list[int], n_bits: int) -> QuantumCircuit:
    """Returns a reversible circuit on n_bits qubits mapping |x> to |S(x)>."""
```

- Exactly `n_bits` qubits, in place, no ancillas. Adding ancillas means changing `src/oracle.py` too.
- Qubit `i` holds bit `i` of x (Qiskit little-endian). `x = 0b0110` means qubits 1 and 2 are `|1⟩`.
- It must work for *any* permutation table, not only the three in `src/cipher.py`.
- The oracle uncomputes with `sbox_circuit(...).inverse()`.

**Safety net:** `pytest tests/test_sbox.py` checks every input of every S-box, the exact unitary, and
10 random permutations for each n = 1–4. `pytest tests/test_grover.py` checks that the oracle marks
exactly the right key and that Grover finds it. Run both after any change.

## How the baseline works

Transformation-based synthesis (Miller, Maslov & Dueck, DAC 2003). It goes through the inputs in order,
0, 1, 2, …, and whenever the current output for input `i` isn't `i`, it adds multi-controlled NOT gates
on the output side that fix that row without disturbing the rows already fixed. Once every row is fixed,
the gates it added undo S, so the S-box circuit is those gates in reverse order. The output uses only
X, CNOT, Toffoli and larger MCX gates.

It's simple and always correct, but it ignores the S-box's structure and the hardware, and large MCX
gates turn into many CNOTs after transpiling.

## Measured costs

`python -m src.metrics` prints this table. All runs are offline on the `FakeTorino` snapshot
(IBM Heron, CZ gates) at optimization level 3. "Exact" is the original phase-free S-box;
"rel-phase" is the current default (see below).

| n | S-box 2q: exact → rel-phase | generic unitary 2q | Grover 2q: exact → rel-phase | Grover depth: exact → rel-phase |
| --- | --- | --- | --- | --- |
| 2 | 3 → 3 | 0 | 2 → 2 | 8 → 8 |
| 3 | 20 → **12** | 30 | 104 → **72** (−31%) | 383 → **273** |
| 4 | 170 → **90** | 139 | 1127 → **642** (−43%) | 4115 → **2315** |

"Generic unitary" means Qiskit synthesizing the permutation matrix with no structure, via
`sbox_reference_circuit`. What the table says:

1. **For n = 4 (PRESENT), the relative-phase S-box now beats generic synthesis** (90 vs. 139 two-qubit
   gates). The exact version did not (170). Published hand-built PRESENT S-box circuits use only a
   handful of Toffolis, so there is still room.
2. **n = 3 decides the hardware results.** On a noisy fake-backend sweep (FakeTorino, 4096 shots,
   2 iterations) the key was found with probability **65.4% with the exact S-box and 70.9% with the
   relative-phase S-box**, against 94.5% ideal and 12.5% for a random guess. Both are single runs, so
   expect a few points of run-to-run variation.
3. **n = 4 on hardware is still out of reach** (about 640 two-qubit gates, depth about 2300). It stays
   a simulator-only demo unless the S-box gets dramatically cheaper.
4. **n = 2 doesn't really test the S-box.** Every 2-bit permutation is affine (X, CNOT and SWAP only),
   so the transpiler folds it to almost nothing. Its hardware run mainly tests Grover itself.

## Done: relative-phase Toffolis (the current default)

If the S-box circuit implements `D·S`, where `D` is any *diagonal* phase (each basis state picks up its
own phase), the oracle is still exactly right. The marking step in the middle of the oracle is also
diagonal, so `D` and `D†` cancel around it. Each Toffoli therefore becomes `RCCXGate` (3 two-qubit gates
on hardware instead of 7), and each 3-control Toffoli becomes `RC3XGate` (6 instead of 18). Both are a
permutation times a phase, so the whole circuit stays `D·S`.

- `sbox_circuit(sbox, n)` uses this by default; `sbox_circuit(sbox, n, relative_phase=False)` gives
  the exact circuit.
- `tests/test_sbox.py` checks the default is "the right permutation up to phases", and the exact mode
  is exactly the permutation matrix.
- `tests/test_grover.py::test_oracle_exact_for_every_plaintext_and_key` checks the full oracle unitary
  for **every** (P, K) at n = 2, 3, 4. That's the guarantee that the phases really cancel.
- `python -m src.run_hardware ... --exact` runs the old circuit, for before/after comparisons.

## Further ideas, roughly from easiest to most involved

- **Structure-aware synthesis for PRESENT:** decompose the S-box, use known low-Toffoli circuits from
  the literature, or search over affine-equivalent forms.
- **A cheaper 3-bit S-box.** `SBOX_3` in `src/cipher.py` is an arbitrary permutation. A nonlinear 3-bit
  S-box with a much cheaper circuit would directly improve the n = 3 hardware run.
- **Hardware-aware layout:** three or four qubits interacting all-to-all need SWAPs on the device's
  connectivity graph. Inspect `transpile_for_backend(...)` output and the chosen layout.
- **Ancilla-assisted MCX:** real devices have plenty of idle qubits. Using them needs an interface change.

## metrics.py reference

| Function | What it gives |
| --- | --- |
| `circuit_metrics(qc)` | gate count, 2-qubit gates, 3+-qubit gates, depth, 2-qubit depth, op counts |
| `get_fake_backend("FakeTorino")` | offline IBM device snapshot (no account, no queue) |
| `transpile_for_backend(qc, backend, level)` | ISA circuit via `generate_preset_pass_manager` |
| `before_after_metrics(qc, backend, level)` | metrics before and after transpiling |
| `compare_optimization_levels(qc, backend)` | one row per optimization level 0–3 |
| `cost_report()` / `python -m src.metrics` | the table above |

Possible additions: estimated success probability from the backend's error rates, results across
several fake backends (Eagle/ECR vs. Heron/CZ), and a comparison of transpiler seeds (layout varies run
to run).
