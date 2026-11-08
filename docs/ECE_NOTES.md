# Notes for the S-box circuit and metrics work

You own `src/sbox_circuit.py` and `src/metrics.py`. Both have working baseline versions, so the whole
pipeline (oracle → Grover → simulator → hardware → demo) already runs end to end. Your job is to make
the S-box circuit cheaper on real hardware and to grow the metrics.

## The contract (please don't break it)

```python
def sbox_circuit(sbox: list[int], n_bits: int) -> QuantumCircuit:
    """Returns a reversible circuit on n_bits qubits mapping |x> to |S(x)>."""
```

- Exactly `n_bits` qubits, in place, no ancillas. Adding ancillas changes the oracle and the
  hardware runner, so talk to the CS side first (see "Ideas" below; it may be worth it).
- Qubit `i` holds bit `i` of x (Qiskit little-endian). `x = 0b0110` means qubits 1 and 2 are `|1⟩`.
- It must work for *any* permutation table passed in, not only the three in `src/cipher.py`.
- The oracle uncomputes with `sbox_circuit(...).inverse()`, so you don't write the inverse yourself.

**Safety net:** `pytest tests/test_sbox.py` checks every input of every S-box, the exact unitary, and
10 random permutations for each n = 1–4. `pytest tests/test_grover.py` checks that the oracle marks
exactly the right key and that Grover still finds it. Run both after every change.

## How the baseline works

Transformation-based synthesis (Miller, Maslov & Dueck, DAC 2003). It goes through the inputs in order,
0, 1, 2, …, and whenever the current output for input `i` isn't `i`, it adds multi-controlled NOT gates
on the output side that fix that row without disturbing the rows already fixed. Once every row is fixed,
the gates it added undo S, so the S-box circuit is those gates in reverse order. The output uses only
X, CNOT, Toffoli and larger MCX gates.

It's simple and always correct, but it ignores the S-box's structure and the hardware, and large MCX
gates turn into many CNOTs after transpiling.

## Where it stands (measured)

`python -m src.metrics` prints this table. All runs are offline on the `FakeTorino` snapshot
(IBM Heron, CZ gates) at optimization level 3.

| n | S-box 2q gates | S-box depth | generic unitary 2q gates | full Grover 2q gates | full Grover depth |
|---|---|---|---|---|---|
| 2 | 3 | 11 | 0 | 2 | 8 |
| 3 | 20 | 73 | 30 | 104 | 383 |
| 4 | 170 | 622 | **139** | 1127 | 4115 |

"Generic unitary" means Qiskit synthesizing the permutation matrix with no structure, via
`sbox_reference_circuit`. What the table says:

1. **For n = 4 (PRESENT), the baseline is worse than generic unitary synthesis** (170 vs. 139 two-qubit
   gates). This is the clearest target. Matching or beating 139 is the minimum; published
   hand-built quantum circuits for the PRESENT S-box use only a handful of Toffolis.
2. **n = 3 is the real hardware battleground.** The full Grover circuit has 104 two-qubit gates. On a
   noisy fake-backend run it found the key with probability 64.6%, against 94.5% ideal and 12.5% for a
   random guess. Every CNOT saved here should show up directly in the hardware result.
3. **n = 4 on hardware is out of reach** with the current approach (about 1100 two-qubit gates, depth
   about 4100). It stays a simulator-only demo unless the S-box gets dramatically cheaper.
4. **n = 2 doesn't really test the S-box.** Every 2-bit permutation is affine (X, CNOT and SWAP only),
   so the transpiler folds it to almost nothing. Its hardware run mainly tests Grover itself.

## Ideas, roughly from easiest to most involved

- **Relative phases are allowed, though the test currently forbids them.** I checked this numerically
  for every (P, K) at n = 2, 3, 4. If your circuit implements `D·S`, where `D` is any *diagonal* phase
  (each basis state picks up its own phase), the oracle is still exactly right. The reason is that the
  marking step in the middle of the oracle is also diagonal, so `D` and `D†` cancel around it. That
  means you can use relative-phase Toffolis (`RCCXGate`, `RC3XGate`), which cost noticeably fewer
  CNOTs than exact ones. To do this, relax `test_exact_unitary_no_phases` in `tests/test_sbox.py` to
  check "permutation up to diagonal phases". The oracle test in `tests/test_grover.py` still guards the
  part that matters.
- **Structure-aware synthesis for PRESENT:** decompose the S-box, use known low-Toffoli circuits from
  the literature, or search over affine-equivalent forms.
- **A cheaper 3-bit S-box.** `SBOX_3` in `src/cipher.py` is an arbitrary permutation I picked. If you
  find a nonlinear 3-bit S-box that is much cheaper, tell me and I'll switch to it; that file is on the
  CS side, so we keep the tables in one place.
- **Hardware-aware layout:** three or four qubits interacting all-to-all need SWAPs on the device's
  connectivity graph. Look at `transpile_for_backend(...)` output and the chosen layout.
- **Ancilla-assisted MCX:** real devices have plenty of idle qubits. This needs an interface change,
  so we should discuss it first.

## metrics.py

| Function | What it gives |
|---|---|
| `circuit_metrics(qc)` | gate count, 2-qubit gates, 3+-qubit gates, depth, 2-qubit depth, op counts |
| `get_fake_backend("FakeTorino")` | offline IBM device snapshot (no account, no queue) |
| `transpile_for_backend(qc, backend, level)` | ISA circuit via `generate_preset_pass_manager` |
| `before_after_metrics(qc, backend, level)` | metrics before and after transpiling |
| `compare_optimization_levels(qc, backend)` | one row per optimization level 0–3 |
| `cost_report()` / `python -m src.metrics` | the table above |

Possible additions: estimated success probability from the backend's error rates, results across
several fake backends (Eagle/ECR vs. Heron/CZ), and a comparison of transpiler seeds (layout varies run
to run). The demo and the hardware runner only use `circuit_metrics`, so you're free to add more.

## Quick commands (Windows, from the repo root)

```powershell
.\qenv\Scripts\Activate.ps1
pytest tests/test_sbox.py tests/test_grover.py      # correctness
python -m src.metrics                               # cost table
python -m src.run_hardware --n 3 --plaintext 5 --key 6 --fake   # noisy local run, saves to results/
```

Please don't run the real-hardware command without checking with me first; our free hardware time is
limited.
