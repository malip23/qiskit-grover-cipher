"""Full Grover key search, plus simulation helpers.

One Grover iteration = Oracle (flip the phase of the right key)
                     + Diffusion (reflect all amplitudes about their mean).
Each iteration moves amplitude onto the marked key. After about
(pi/4) * sqrt(2**n) iterations, measuring gives the key with high probability.

Run from the repo root:
    python -m src.grover --n 4 --plaintext 0x9 --key 0x3
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from . import cipher
from .oracle import apply_mcz, build_oracle


def optimal_iterations(n_bits: int, n_solutions: int = 1) -> int:
    """floor(pi/4 * sqrt(N / M)) for N = 2**n keys and M solutions."""
    return math.floor(math.pi / 4 * math.sqrt(2 ** n_bits / n_solutions))


def theoretical_success(n_bits: int, iterations: int) -> float:
    """Ideal (noise-free) probability of measuring the key: sin^2((2r+1)*theta)."""
    theta = math.asin(1 / math.sqrt(2 ** n_bits))
    return math.sin((2 * iterations + 1) * theta) ** 2


def diffusion(n_bits: int) -> QuantumCircuit:
    """Grover diffusion: 2|s><s| - I, where |s> is the uniform superposition.

    H^n maps |s> to |0...0>; X^n maps |0...0> to |1...1>; the MCZ flips the phase
    of that one state; then undo X and H. (Overall sign is a global phase.)
    """
    qubits = list(range(n_bits))
    qc = QuantumCircuit(n_bits, name="Diffusion")
    qc.h(qubits)
    qc.x(qubits)
    apply_mcz(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)
    return qc


def grover_iteration(sbox: list[int], n_bits: int, plaintext: int, ciphertext: int,
                     relative_phase: bool = True) -> QuantumCircuit:
    """One Grover iteration (oracle then diffusion)."""
    qc = build_oracle(sbox, n_bits, plaintext, ciphertext, relative_phase=relative_phase)
    qc.compose(diffusion(n_bits), inplace=True)
    qc.name = "Grover iteration"
    return qc


def build_grover_circuit(sbox: list[int], n_bits: int, plaintext: int, ciphertext: int,
                         iterations: int | None = None, measure: bool = True,
                         barriers: bool = False, relative_phase: bool = True) -> QuantumCircuit:
    """Full Grover circuit: H on all key qubits, `iterations` x (oracle + diffusion), measure.

    Args:
        iterations:     number of Grover iterations; default is optimal_iterations(n_bits).
        measure:        add measure_all() (classical register named "meas").
        barriers:       add barriers between stages (nicer drawings; slightly blocks
                        transpiler optimization, so leave off for hardware).
        relative_phase: use the cheaper relative-phase S-box (default; same results).
    """
    if iterations is None:
        iterations = optimal_iterations(n_bits)
    step = grover_iteration(sbox, n_bits, plaintext, ciphertext, relative_phase)

    qc = QuantumCircuit(n_bits, name=f"Grover n={n_bits} r={iterations}")
    qc.h(range(n_bits))  # start in an equal superposition of all 2**n keys
    for _ in range(iterations):
        if barriers:
            qc.barrier()
        qc.compose(step, inplace=True)
    if measure:
        qc.measure_all()
    return qc


def run_on_aer(circuit: QuantumCircuit, shots: int = 1024, seed: int | None = None) -> dict[str, int]:
    """Run a measured circuit on the (noise-free) Aer simulator. Returns counts.

    Counts keys are bitstrings with qubit 0 on the RIGHT, so int(bits, 2) == key.
    """
    sim = AerSimulator(seed_simulator=seed)
    pm = generate_preset_pass_manager(backend=sim, optimization_level=1)
    sampler = AerSampler(seed=seed)
    result = sampler.run([pm.run(circuit)], shots=shots).result()
    return result[0].data.meas.get_counts()


def most_likely_key(counts: dict[str, int]) -> int:
    """The key that was measured most often."""
    return int(max(counts, key=counts.get), 2)


def key_probabilities_per_iteration(sbox: list[int], n_bits: int, plaintext: int, ciphertext: int,
                                    iterations: int | None = None) -> np.ndarray:
    """Exact probability of every key after 0, 1, ..., `iterations` Grover iterations.

    Returns an array of shape (iterations + 1, 2**n_bits); row r is the
    distribution after r iterations (row 0 is uniform). Uses statevector
    simulation, so no sampling noise. Useful for animations.
    """
    if iterations is None:
        iterations = optimal_iterations(n_bits)
    step = grover_iteration(sbox, n_bits, plaintext, ciphertext)

    init = QuantumCircuit(n_bits)
    init.h(range(n_bits))
    state = Statevector(init)
    rows = [state.probabilities()]
    for _ in range(iterations):
        state = state.evolve(step)
        rows.append(state.probabilities())
    # probabilities() index i corresponds to basis state |i>, i.e. key i.
    return np.array(rows)


def attack(n_bits: int, plaintext: int, key: int, shots: int = 1024, seed: int | None = None,
           iterations: int | None = None) -> dict:
    """Encrypt classically, then recover the key with Grover on Aer."""
    sbox = cipher.get_sbox(n_bits)
    ciphertext = cipher.encrypt(plaintext, key, sbox)
    r = optimal_iterations(n_bits) if iterations is None else iterations
    counts = run_on_aer(build_grover_circuit(sbox, n_bits, plaintext, ciphertext, r), shots, seed)
    found = most_likely_key(counts)
    return {
        "n_bits": n_bits, "plaintext": plaintext, "key": key, "ciphertext": ciphertext,
        "iterations": r, "shots": shots, "counts": counts, "recovered_key": found,
        "success": found == key,
        "measured_success_rate": counts.get(cipher.to_bits(key, n_bits), 0) / shots,
        "theoretical_success_rate": theoretical_success(n_bits, r),
    }


def _main() -> None:
    p = argparse.ArgumentParser(description="Grover key recovery on the Aer simulator")
    p.add_argument("--n", type=int, default=4, choices=[2, 3, 4], help="block size in bits")
    p.add_argument("--plaintext", type=lambda s: int(s, 0), default=None, help="e.g. 0x9 or 9")
    p.add_argument("--key", type=lambda s: int(s, 0), default=None, help="secret key, e.g. 0x3")
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    pt = args.plaintext if args.plaintext is not None else int(rng.integers(2 ** args.n))
    k = args.key if args.key is not None else int(rng.integers(2 ** args.n))

    res = attack(args.n, pt, k, args.shots, args.seed)
    b = lambda x: cipher.to_bits(x, args.n)
    print(f"n={args.n}  P={b(pt)}  K={b(k)}  C={b(res['ciphertext'])}  iterations={res['iterations']}")
    for bits, c in sorted(res["counts"].items(), key=lambda kv: -kv[1])[:5]:
        print(f"  key {bits}: {c:5d}  {'<-- secret key' if int(bits, 2) == k else ''}")
    print(f"recovered={b(res['recovered_key'])}  success={res['success']}  "
          f"measured={res['measured_success_rate']:.3f}  theory={res['theoretical_success_rate']:.3f}")


if __name__ == "__main__":
    _main()
