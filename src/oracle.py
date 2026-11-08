"""Grover oracle for the known-plaintext attack on  C = S(P XOR K).

The oracle acts in place on n key qubits and flips the phase of exactly the
key(s) K with S(P XOR K) == C:

        |K>  ->  -|K>   if K encrypts P to C
        |K>  ->   |K>   otherwise

Construction (each step is reversible, so we can undo it afterwards):
    1. X on qubit i wherever P has a 1 bit        |K>          -> |K XOR P>
    2. S-box circuit                              |K XOR P>    -> |S(K XOR P)>
    3. X on qubit i wherever C has a 0 bit        the right key is now |11...1>
    4. Multi-controlled Z                         phase -1 on |11...1> only
    5. Undo steps 3, 2, 1                         qubits return to |K>, phase stays

Note: S is a permutation, so exactly ONE key matches (K = S^-1(C) XOR P).
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from .sbox_circuit import sbox_circuit


def apply_mcz(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Multi-controlled Z on `qubits`: phase -1 only on |11...1>.

    Built as H . MCX . H on the last qubit (Z = HXH). For 2 qubits it's just CZ.
    """
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        *controls, target = qubits
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)


def _x_where(qc: QuantumCircuit, value: int, n_bits: int, bit: int) -> None:
    """Apply X to every qubit i whose bit in `value` equals `bit` (0 or 1)."""
    for i in range(n_bits):
        if ((value >> i) & 1) == bit:
            qc.x(i)


def build_oracle(sbox: list[int], n_bits: int, plaintext: int, ciphertext: int,
                 relative_phase: bool = True) -> QuantumCircuit:
    """Phase oracle marking the key K with S(plaintext XOR K) == ciphertext.

    relative_phase: use the cheaper relative-phase S-box (default). Its extra
    phases cancel between S and S^-1 because the marking step is diagonal.
    """
    sbox_qc = sbox_circuit(sbox, n_bits, relative_phase=relative_phase)
    qubits = list(range(n_bits))

    qc = QuantumCircuit(n_bits, name="Oracle")
    # Compute
    _x_where(qc, plaintext, n_bits, bit=1)           # 1. K -> K XOR P
    qc.compose(sbox_qc, qubits, inplace=True)        # 2. -> S(K XOR P)
    _x_where(qc, ciphertext, n_bits, bit=0)          # 3. match -> |11...1>
    # Mark
    apply_mcz(qc, qubits)                            # 4. phase flip
    # Uncompute (reverse order)
    _x_where(qc, ciphertext, n_bits, bit=0)          # undo 3
    qc.compose(sbox_qc.inverse(), qubits, inplace=True)  # undo 2
    _x_where(qc, plaintext, n_bits, bit=1)           # undo 1
    return qc
