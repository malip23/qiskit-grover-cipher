"""The quantum S-box circuit must match the classical table on every input.

Run after ANY change to src/sbox_circuit.py:   pytest
"""

import random

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from src import cipher
from src.sbox_circuit import sbox_circuit, sbox_reference_circuit

SBOX_CASES = [(n, cipher.get_sbox(n)) for n in (2, 3, 4)]
MODES = [True, False]  # relative_phase on (default) / off (exact)


def permutation_matrix(sbox, n_bits):
    dim = 2 ** n_bits
    m = np.zeros((dim, dim))
    for x, y in enumerate(sbox):
        m[y, x] = 1
    return m


def is_permutation_up_to_phases(u, sbox, n_bits):
    """True if u = D . P_S: the right permutation, each entry a unit-modulus phase."""
    p = permutation_matrix(sbox, n_bits)
    return np.allclose(np.abs(u), p)


def test_present_sbox_table():
    # PRESENT S-box, Bogdanov et al. CHES 2007, Table 1.
    assert cipher.SBOX_PRESENT == [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
                                   0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


@pytest.mark.parametrize("n_bits,sbox", SBOX_CASES)
def test_tables_are_permutations(n_bits, sbox):
    assert cipher.is_permutation(sbox, n_bits)


@pytest.mark.parametrize("relative_phase", MODES)
@pytest.mark.parametrize("n_bits,sbox", SBOX_CASES)
def test_every_input_maps_to_sbox_output(n_bits, sbox, relative_phase):
    qc = sbox_circuit(sbox, n_bits, relative_phase=relative_phase)
    assert qc.num_qubits == n_bits, "S-box circuit must use exactly n_bits qubits (no ancillas)"
    for x in range(2 ** n_bits):
        probs = Statevector.from_int(x, 2 ** n_bits).evolve(qc).probabilities()
        assert np.isclose(probs[sbox[x]], 1.0), f"|{x}> should map to |{sbox[x]}>"


@pytest.mark.parametrize("n_bits,sbox", SBOX_CASES)
def test_exact_mode_has_no_phases(n_bits, sbox):
    # relative_phase=False must give exactly the permutation matrix.
    u = Operator(sbox_circuit(sbox, n_bits, relative_phase=False)).data
    assert np.allclose(u, permutation_matrix(sbox, n_bits))


@pytest.mark.parametrize("n_bits,sbox", SBOX_CASES)
def test_default_mode_is_permutation_up_to_phases(n_bits, sbox):
    # The default may add a phase per basis state (D . S): allowed, see sbox_circuit.py docstring.
    u = Operator(sbox_circuit(sbox, n_bits)).data
    assert is_permutation_up_to_phases(u, sbox, n_bits)


@pytest.mark.parametrize("relative_phase", MODES)
@pytest.mark.parametrize("n_bits", [1, 2, 3, 4])
def test_random_permutations(n_bits, relative_phase):
    # Synthesis must work for ANY permutation, not just our three tables.
    rng = random.Random(1000 + n_bits)
    for _ in range(10):
        sbox = list(range(2 ** n_bits))
        rng.shuffle(sbox)
        u = Operator(sbox_circuit(sbox, n_bits, relative_phase=relative_phase)).data
        if relative_phase:
            assert is_permutation_up_to_phases(u, sbox, n_bits)
        else:
            assert np.allclose(u, permutation_matrix(sbox, n_bits))


def test_relative_phase_is_cheaper():
    # The whole point of relative_phase: fewer multi-qubit gates to decompose.
    from src.metrics import circuit_metrics
    sb = cipher.get_sbox(4)
    fast = sbox_circuit(sb, 4).decompose(reps=3)
    exact = sbox_circuit(sb, 4, relative_phase=False).decompose(reps=3)
    assert circuit_metrics(fast)["two_qubit_gates"] < circuit_metrics(exact)["two_qubit_gates"]


def test_identity_gives_empty_circuit():
    assert sbox_circuit([0, 1, 2, 3], 2).size() == 0


def test_rejects_non_permutation():
    with pytest.raises(ValueError):
        sbox_circuit([0, 0, 1, 2], 2)


@pytest.mark.parametrize("n_bits,sbox", SBOX_CASES)
def test_reference_circuit(n_bits, sbox):
    assert np.allclose(Operator(sbox_reference_circuit(sbox, n_bits)).data, permutation_matrix(sbox, n_bits))
