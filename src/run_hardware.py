"""Run the Grover attack on real IBM Quantum hardware (or a local fake backend).

Credentials: NEVER hard-code an API key. Save it once with
    QiskitRuntimeService.save_account(...)
(see README). This script only calls QiskitRuntimeService(), which loads the
saved account from your home directory.

Usage (from the repo root):
    # Free, offline dry run on a noisy snapshot of an IBM device:
    python -m src.run_hardware --n 2 --plaintext 1 --key 2 --fake

    # Real hardware (uses your IBM Quantum time; asks for confirmation):
    python -m src.run_hardware --n 2 --plaintext 1 --key 2

Results are saved to results/<name>.json; the Streamlit app reads them from there.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from . import cipher
from .grover import build_grover_circuit, optimal_iterations, theoretical_success
from .metrics import circuit_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def get_backend(n_qubits: int, fake: str | None = None, backend_name: str | None = None,
                instance: str | None = None):
    """Pick a backend.

    fake:         name of an offline fake backend (e.g. "FakeTorino"): no account, no queue.
    backend_name: a specific real device; otherwise the least busy one is used.
    """
    if fake:
        from qiskit_ibm_runtime import fake_provider
        return getattr(fake_provider, fake)()

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=instance) if instance else QiskitRuntimeService()
    if backend_name:
        return service.backend(backend_name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)


def run_circuit(circuit: QuantumCircuit, backend, shots: int = 4096,
                optimization_level: int = 3, dynamical_decoupling: bool = True) -> dict:
    """Transpile `circuit` for `backend` and run it with SamplerV2.

    Returns a dict with counts, job id, and transpiled-circuit metrics.
    """
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    # 1. Rewrite the circuit in the backend's native gates + qubit layout (an "ISA circuit").
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_circuit = pm.run(circuit)

    # 2. Configure the sampler. In "job mode" we pass the backend directly.
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    if dynamical_decoupling:
        # Fill idle periods with X-pulse pairs so idle qubits lose less coherence.
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"

    # 3. Submit and wait.
    job = sampler.run([isa_circuit])
    print(f"Submitted job {job.job_id()} to {backend.name}; waiting for results...")
    pub_result = job.result()[0]

    return {
        "job_id": job.job_id(),
        "counts": pub_result.data.meas.get_counts(),
        "transpiled_metrics": circuit_metrics(isa_circuit),
        "logical_metrics": circuit_metrics(circuit),
    }


def save_result(record: dict, results_dir: Path = RESULTS_DIR) -> Path:
    """Write a result record to results/ as JSON and return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    n = record["n_bits"]
    stamp = record["timestamp"].replace(":", "").replace("-", "")[:15]
    name = (f"{record['backend_kind']}_n{n}_p{record['plaintext']}_k{record['key']}"
            f"_{record['backend']}_{stamp}.json")
    path = results_dir / name
    path.write_text(json.dumps(record, indent=2))
    return path


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Load all saved result records (newest first). Used by the Streamlit app."""
    records = []
    for p in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            rec = json.loads(p.read_text())
            rec["_file"] = p.name
            records.append(rec)
        except (json.JSONDecodeError, OSError):
            continue
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Grover key search on IBM hardware")
    p.add_argument("--n", type=int, default=2, choices=[2, 3, 4], help="block size (2 or 3 recommended on hardware)")
    p.add_argument("--plaintext", type=lambda s: int(s, 0), required=True)
    p.add_argument("--key", type=lambda s: int(s, 0), required=True)
    p.add_argument("--iterations", type=int, default=None, help="default: optimal")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--optimization-level", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--no-dd", action="store_true", help="disable dynamical decoupling")
    p.add_argument("--backend", default=None, help="specific device name (default: least busy)")
    p.add_argument("--instance", default=None, help="IBM Cloud instance CRN/name (optional)")
    p.add_argument("--fake", nargs="?", const="FakeTorino", default=None,
                   help="run locally on a fake backend instead (default FakeTorino)")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt for real hardware")
    args = p.parse_args()

    sbox = cipher.get_sbox(args.n)
    ciphertext = cipher.encrypt(args.plaintext, args.key, sbox)
    iterations = optimal_iterations(args.n) if args.iterations is None else args.iterations
    circuit = build_grover_circuit(sbox, args.n, args.plaintext, ciphertext, iterations)

    backend = get_backend(args.n, fake=args.fake, backend_name=args.backend, instance=args.instance)
    kind = "fake" if args.fake else "hardware"
    print(f"Backend: {backend.name} ({kind}); n={args.n}, iterations={iterations}, shots={args.shots}")

    if kind == "hardware" and not args.yes:
        if input("This uses your IBM Quantum time. Submit? [y/N] ").strip().lower() != "y":
            print("Aborted; nothing submitted.")
            return

    out = run_circuit(circuit, backend, args.shots, args.optimization_level, not args.no_dd)
    key_bits = cipher.to_bits(args.key, args.n)
    record = {
        "backend_kind": kind,
        "backend": backend.name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_bits": args.n,
        "sbox": sbox,
        "plaintext": args.plaintext,
        "key": args.key,
        "ciphertext": ciphertext,
        "iterations": iterations,
        "shots": args.shots,
        "optimization_level": args.optimization_level,
        "dynamical_decoupling": not args.no_dd,
        "job_id": out["job_id"],
        "counts": out["counts"],
        "correct_key_probability": out["counts"].get(key_bits, 0) / args.shots,
        "theoretical_success": theoretical_success(args.n, iterations),
        "logical_metrics": out["logical_metrics"],
        "transpiled_metrics": out["transpiled_metrics"],
    }
    path = save_result(record)
    print(f"P(correct key) = {record['correct_key_probability']:.3f} "
          f"(ideal {record['theoretical_success']:.3f}, random guess {1 / 2 ** args.n:.3f})")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
