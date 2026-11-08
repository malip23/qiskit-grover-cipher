"""Reversible S-box circuits:  |x>  ->  |S(x)>  on n_bits qubits.

=============================================================================
CONTRACT - keep this signature, the rest of the project depends on it:

    def sbox_circuit(sbox: list[int], n_bits: int) -> QuantumCircuit

  * The returned circuit acts on exactly n_bits qubits (no ancillas), in place.
  * Qubit i holds bit i of x (Qiskit's little-endian convention).
  * Every basis state |x> must go to |S(x)>, possibly times a phase e^{i phi_x}
    (see "Relative phases" below). Run `pytest` after any change.

See docs/OPTIMIZATION_NOTES.md for measured costs and further ideas.
=============================================================================

Relative phases (the optimization used by default)
--------------------------------------------------
Inside the oracle the S-box is always used as  S^-1 . (mark) . S, and the
marking step only multiplies basis states by +1 or -1 (it is "diagonal").
If our circuit actually does |x> -> e^{i phi_x} |S(x)>, i.e. D.S with D
diagonal, the phases cancel:  S^-1 D^-1 (mark) D S = S^-1 (mark) S,
because diagonal matrices commute. So the S-box only needs to be right
"up to diagonal phases".

That lets us swap each Toffoli (CCX) for a relative-phase Toffoli (RCCX) and
each 3-control MCX for RC3X. Each of those is a permutation times a phase,
so the whole circuit is still D.S. After transpiling to IBM hardware:
    CCX  -> 7 two-qubit gates     RCCX -> 3
    C3X  -> 18 two-qubit gates    RC3X -> 6
Pass relative_phase=False to get the exact (phase-free) circuit.

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
from qiskit.circuit.library import RC3XGate, RCCXGate, UnitaryGate


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


def sbox_circuit(sbox: list[int], n_bits: int, *, relative_phase: bool = True) -> QuantumCircuit:
    """Returns a reversible circuit on n_bits qubits mapping |x> to |S(x)>.

    Transformation-based synthesis; see module docstring.

    Args:
        relative_phase: True (default) uses relative-phase Toffolis, so each |x>
            picks up a phase e^{i phi_x}. That is fine inside the oracle and much
            cheaper on hardware. False gives the exact, phase-free circuit.
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
        elif relative_phase and len(controls) == 2:
            qc.append(RCCXGate(), [*controls, target])   # Toffoli up to a phase
        elif relative_phase and len(controls) == 3:
            qc.append(RC3XGate(), [*controls, target])   # 3-control Toffoli up to a phase
        else:
            qc.mcx(controls, target)                      # exact multi-controlled X
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
