"""Grover must recover the secret key on the simulator, for n = 2, 3, 4."""

import random

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from src import cipher
from src.grover import (build_grover_circuit, key_probabilities_per_iteration, most_likely_key,
                        optimal_iterations, run_on_aer, theoretical_success)
from src.oracle import build_oracle


def random_pairs(n_bits, count, seed):
    rng = random.Random(seed)
    return [(rng.randrange(2 ** n_bits), rng.randrange(2 ** n_bits)) for _ in range(count)]


CASES = [(n, p, k) for n in (2, 3, 4) for (p, k) in random_pairs(n, 4, seed=42 + n)]


def test_classical_cipher_roundtrip():
    for n in (2, 3, 4):
        sbox = cipher.get_sbox(n)
        for p in range(2 ** n):
            for k in range(2 ** n):
                assert cipher.decrypt(cipher.encrypt(p, k, sbox), k, sbox) == p


def test_brute_force_counts_guesses():
    sbox = cipher.get_sbox(4)
    key, guesses = cipher.brute_force(9, cipher.encrypt(9, 5, sbox), sbox, 4)
    assert key == 5 and guesses == 6


@pytest.mark.parametrize("n_bits,plaintext,key", CASES)
def test_oracle_marks_only_the_key(n_bits, plaintext, key):
    sbox = cipher.get_sbox(n_bits)
    ct = cipher.encrypt(plaintext, key, sbox)
    u = Operator(build_oracle(sbox, n_bits, plaintext, ct)).data
    expected = np.ones(2 ** n_bits)
    expected[key] = -1
    assert np.allclose(u, np.diag(expected)), "oracle must be diag(+1, ..., -1 at K, ..., +1)"


@pytest.mark.parametrize("n_bits", [2, 3, 4])
def test_oracle_exact_for_every_plaintext_and_key(n_bits):
    # Exhaustive: guards the relative-phase S-box (its phases must cancel inside the oracle).
    sbox = cipher.get_sbox(n_bits)
    for plaintext in range(2 ** n_bits):
        for key in range(2 ** n_bits):
            ct = cipher.encrypt(plaintext, key, sbox)
            u = Operator(build_oracle(sbox, n_bits, plaintext, ct)).data
            expected = np.ones(2 ** n_bits)
            expected[key] = -1
            assert np.allclose(u, np.diag(expected)), f"P={plaintext} K={key}"


@pytest.mark.parametrize("n_bits", [2, 3, 4])
def test_optimal_iterations(n_bits):
    assert optimal_iterations(n_bits) == {2: 1, 3: 2, 4: 3}[n_bits]


@pytest.mark.parametrize("n_bits,plaintext,key", CASES)
def test_grover_recovers_key_on_aer(n_bits, plaintext, key):
    sbox = cipher.get_sbox(n_bits)
    ct = cipher.encrypt(plaintext, key, sbox)
    shots = 2048
    counts = run_on_aer(build_grover_circuit(sbox, n_bits, plaintext, ct), shots=shots, seed=1234)
    assert most_likely_key(counts) == key
    success = counts.get(cipher.to_bits(key, n_bits), 0) / shots
    assert success > 0.85, f"success rate {success:.3f} too low"


@pytest.mark.parametrize("n_bits,plaintext,key", CASES[::4])
def test_statevector_probabilities_match_theory(n_bits, plaintext, key):
    sbox = cipher.get_sbox(n_bits)
    ct = cipher.encrypt(plaintext, key, sbox)
    probs = key_probabilities_per_iteration(sbox, n_bits, plaintext, ct)
    assert probs.shape == (optimal_iterations(n_bits) + 1, 2 ** n_bits)
    assert np.allclose(probs[0], 1 / 2 ** n_bits)  # starts uniform
    for r, row in enumerate(probs):
        assert np.isclose(row.sum(), 1.0)
        assert np.isclose(row[key], theoretical_success(n_bits, r))
