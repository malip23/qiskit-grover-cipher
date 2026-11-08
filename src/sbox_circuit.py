"""Reversible S-box circuits:  |x>  ->  |S(x)>  on n_bits qubits.

=============================================================================
OWNER: ECE partner.
This file is a correct, working BASELINE so the whole pipeline runs end to end.
The ECE partner will replace or optimize the S-box synthesis (fewer CNOTs,
lower depth, hardware-aware layout, ancilla tricks, etc.).

CONTRACT - keep this exact signature, the rest of the project depends on it:

    def sbox_circuit(sbox: list[int], n_bits: int) -> QuantumCircuit

  * The returned circuit acts on exactly n_bits qubits (no ancillas), in place.
  * Qubit i holds bit i of x (Qiskit's little-endian convention).
  * It must map every basis state |x> to |S(x)> with NO extra phase,
    because the oracle uncomputes it with .inverse(). tests/test_sbox.py
    checks both conditions; run `pytest tests/test_sbox.py` after any change.
=============================================================================

Baseline algorithm: transformation-based synthesis
(Miller, Maslov, Dueck, "A Transformation Based Algorithm for Reversible
Logic Synthesis", DAC 2003).

Idea: walk through the inputs i = 0, 1, 2, ... in order. Whenever the
current output f(i) is not equal to i, add multi-controlled NOT gates on the *output
side* that turn f(i) into i, choosing controls so that the outputs already
fixed (all j < i) are not disturbed. When every f(i) == i, the gates we
added undo the permutation, so the S-box circuit is those gates in reverse
order (every MCX gate is its own inverse).

Cost: correct for ANY permutation, but not gate-optimal. Large
multi-controlled X gates are expensive once transpiled to 1- and 2-qubit
hardware gates. That is the main thing to optimize.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


def _bits_set(x: int, n_bits: int) -> list[int]:
    """Indices of the 1 bits of x (these become control qubits)."""
    return [b for b in range(n_bits) if (x >> b) & 1]


def _transformation_gates(sbox: list[int], n_bits: int) -> list[tuple[list[int], int]]:
    """Run the MMD algorithm. Returns gates (controls, target) in discovery order.

    Each gate flips bit `target` when all bits in `controls` are 1.
    """
    f = list(sbox)  # working copy of the permutation; we "fix" it gate by gate
    gates: list[tuple[list[int], int]] = []

    def apply(controls: list[int], target: int) -> None:
        # Apply the gate to the output side of f (i.e. to every value f[x]).
        mask = sum(1 << c for c in controls)
        for x in range(len(f)):
            if f[x] & mask == mask:
                f[x] ^= 1 << target
        gates.append((controls, target))

    for i in range(2 ** n_bits):
        if f[i] == i:
            continue
        # Step 1: set the bits that are 1 in i but 0 in f(i).
        #   Controls = current 1 bits of f(i). Any j < i with those bits set
        #   would satisfy j >= f(i) >= i, which is impossible, so fixed rows are safe.
        for b in range(n_bits):
            if (i >> b) & 1 and not (f[i] >> b) & 1:
                apply(_bits_set(f[i], n_bits), b)
        # Step 2: clear the bits that are 1 in f(i) but 0 in i.
        #   Controls = 1 bits of i (f(i) now contains all of them).
        for b in range(n_bits):
            if not (i >> b) & 1 and (f[i] >> b) & 1:
                apply(_bits_set(i, n_bits), b)

    assert f == list(range(2 ** n_bits)), "synthesis failed to reach identity"
    return gates


def sbox_circuit(sbox: list[int], n_bits: int) -> QuantumCircuit:
    """Returns a reversible circuit on n_bits qubits mapping |x> to |S(x)>.

    BASELINE (transformation-based synthesis); see module docstring.
    """
    if sorted(sbox) != list(range(2 ** n_bits)):
        raise ValueError("sbox must be a permutation of 0 .. 2**n_bits - 1")

    qc = QuantumCircuit(n_bits, name=f"S{n_bits}")
    # The gates found by the algorithm undo S, so S is those gates reversed.
    for controls, target in reversed(_transformation_gates(sbox, n_bits)):
        if len(controls) == 0:
            qc.x(target)
        elif len(controls) == 1:
            qc.cx(controls[0], target)
        else:
            qc.mcx(controls, target)
    return qc


def sbox_reference_circuit(sbox: list[int], n_bits: int) -> QuantumCircuit:
    """Reference implementation: the S-box as one permutation-matrix UnitaryGate.

    Useful for testing and as a benchmark: Qiskit's generic unitary synthesis
    is usually MUCH worse than a structured circuit, so a good S-box circuit
    should beat this one on transpiled 2-qubit gate count.
    """
    dim = 2 ** n_bits
    matrix = np.zeros((dim, dim))
    for x, y in enumerate(sbox):
        matrix[y, x] = 1.0  # column x has its 1 in row S(x)
    qc = QuantumCircuit(n_bits, name=f"S{n_bits}_ref")
    qc.append(UnitaryGate(matrix, label=f"S{n_bits}"), range(n_bits))
    return qc
