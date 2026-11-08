"""Circuit cost metrics: gate count, 2-qubit gate count, depth.

=============================================================================
BASELINE version so the pipeline (and the demo) can report costs. Possible
extensions: error-rate-weighted cost, estimated fidelity, or layout and
routing analysis.
=============================================================================

Why metrics matter: on real hardware each 2-qubit gate (CZ / ECR) has roughly
10x the error of a 1-qubit gate, and decoherence grows with depth. Two numbers
predict whether Grover survives the noise: transpiled 2-qubit gate count and
transpiled depth.

"Before transpile" = the abstract circuit (MCX gates count as one gate).
"After transpile"  = rewritten into the backend's native gates and wired
                     to its physical qubit connectivity.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

# Instructions that are not real gates and should not be counted.
_NON_GATES = {"barrier", "measure", "reset", "delay"}


def _is_gate(instruction) -> bool:
    return instruction.operation.name not in _NON_GATES


def circuit_metrics(qc: QuantumCircuit) -> dict:
    """Gate counts and depth of a circuit, as-is (no transpiling)."""
    gates = [inst for inst in qc.data if _is_gate(inst)]
    return {
        "num_qubits": qc.num_qubits,
        "gate_count": len(gates),
        "two_qubit_gates": sum(1 for g in gates if g.operation.num_qubits == 2),
        "multi_qubit_gates": sum(1 for g in gates if g.operation.num_qubits >= 3),
        # Depth counting only real gates (barriers/measurements ignored).
        "depth": qc.depth(filter_function=_is_gate),
        # Depth counting only 2-qubit gates: a good proxy for noise on hardware.
        "two_qubit_depth": qc.depth(
            filter_function=lambda inst: _is_gate(inst) and inst.operation.num_qubits == 2
        ),
        "ops": dict(qc.count_ops()),
    }


def get_fake_backend(name: str = "FakeTorino"):
    """Return an offline snapshot of an IBM device (no account or queue needed).

    Examples: "FakeTorino" (Heron, CZ gates), "FakeBrisbane" (Eagle, ECR gates).
    """
    from qiskit_ibm_runtime import fake_provider

    return getattr(fake_provider, name)()


def transpile_for_backend(qc: QuantumCircuit, backend, optimization_level: int = 3,
                          seed: int | None = 1234) -> QuantumCircuit:
    """Transpile qc to the backend's native gates and connectivity (an ISA circuit)."""
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=optimization_level, seed_transpiler=seed
    )
    return pm.run(qc)


def before_after_metrics(qc: QuantumCircuit, backend, optimization_level: int = 3,
                         seed: int | None = 1234) -> dict:
    """Metrics of qc before and after transpiling to `backend`."""
    tqc = transpile_for_backend(qc, backend, optimization_level, seed)
    return {
        "backend": backend.name,
        "optimization_level": optimization_level,
        "before": circuit_metrics(qc),
        "after": circuit_metrics(tqc),
    }


def compare_optimization_levels(qc: QuantumCircuit, backend, levels=(0, 1, 2, 3),
                                seed: int | None = 1234) -> list[dict]:
    """Transpile at each optimization level and return one metrics row per level."""
    rows = []
    for level in levels:
        m = circuit_metrics(transpile_for_backend(qc, backend, level, seed))
        rows.append({
            "optimization_level": level,
            "gate_count": m["gate_count"],
            "two_qubit_gates": m["two_qubit_gates"],
            "depth": m["depth"],
            "two_qubit_depth": m["two_qubit_depth"],
        })
    return rows


def format_table(rows: list[dict]) -> str:
    """Simple fixed-width table for printing a list of dicts."""
    if not rows:
        return ""
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = ["  ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join([line, sep, *body])


def cost_report(backend_name: str = "FakeTorino", optimization_level: int = 3) -> list[dict]:
    """Transpiled cost of each S-box circuit and of the full Grover circuit, for n = 2, 3, 4.

    `ref_unitary_2q` is the cost of Qiskit's generic unitary synthesis of the
    same permutation: a good S-box circuit should beat it.
    """
    from . import cipher
    from .grover import build_grover_circuit
    from .sbox_circuit import sbox_circuit, sbox_reference_circuit

    backend = get_fake_backend(backend_name)
    rows = []
    for n in (2, 3, 4):
        sb = cipher.get_sbox(n)
        s_after = before_after_metrics(sbox_circuit(sb, n), backend, optimization_level)["after"]
        ref_after = before_after_metrics(sbox_reference_circuit(sb, n), backend, optimization_level)["after"]
        grover = build_grover_circuit(sb, n, 1, cipher.encrypt(1, 2, sb), measure=False)
        g_after = before_after_metrics(grover, backend, optimization_level)["after"]
        rows.append({
            "n": n,
            "sbox_2q": s_after["two_qubit_gates"],
            "sbox_depth": s_after["depth"],
            "ref_unitary_2q": ref_after["two_qubit_gates"],
            "grover_2q": g_after["two_qubit_gates"],
            "grover_depth": g_after["depth"],
        })
    return rows


if __name__ == "__main__":
    # python -m src.metrics   -> cost table on an offline fake backend (no account needed)
    print("Transpiled costs on FakeTorino, optimization_level=3")
    print(format_table(cost_report()))
