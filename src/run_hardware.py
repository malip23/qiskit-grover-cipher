"""Run the Grover attack on real IBM Quantum hardware (or a local fake backend).

Credentials: NEVER hard-code an API key. Save it once with
    QiskitRuntimeService.save_account(...)
(see README). This script only calls QiskitRuntimeService(), which loads the
saved account from your home directory.

Usage (from the repo root):
    # Free, offline dry run on a noisy snapshot of an IBM device:
    python -m src.run_hardware --n 2 --plaintext 1 --key 2 --fake

    # Sweep: 0, 1, ..., optimal+1 Grover iterations, all in ONE job (one queue wait):
    python -m src.run_hardware --n 3 --plaintext 5 --key 6 --sweep --fake

    # Real hardware (uses your IBM Quantum time; asks for confirmation):
    python -m src.run_hardware --n 3 --plaintext 5 --key 6 --sweep

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


def run_circuits(circuits: list[QuantumCircuit], backend, shots: int = 4096,
                 optimization_level: int = 3, dynamical_decoupling: bool = True) -> dict:
    """Transpile `circuits` for `backend` and run them ALL in a single SamplerV2 job.

    Each circuit becomes one "PUB" (primitive unified bloc) of the same job, so
    a sweep of several circuits costs only one trip through the queue.

    Returns {"job_id": ..., "pubs": [{"counts", "transpiled_metrics", "logical_metrics"}, ...]}.
    """
    # NOTE: SamplerV2 is deprecated as of qiskit-ibm-runtime 0.50 (it still works with the
    # pinned 0.51.0). Its replacement is qiskit_ibm_runtime.executor_sampler.Sampler.
    # Migrate before upgrading qiskit-ibm-runtime past the removal date.
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    # 1. Rewrite each circuit in the backend's native gates + qubit layout ("ISA circuits").
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_circuits = pm.run(circuits)

    # 2. Configure the sampler. In "job mode" we pass the backend directly.
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    if dynamical_decoupling:
        # Fill idle periods with X-pulse pairs so idle qubits lose less coherence.
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"

    # 3. Submit once, wait, and unpack one result per circuit.
    job = sampler.run(isa_circuits)
    print(f"Submitted job {job.job_id()} ({len(circuits)} circuit(s)) to {backend.name}; waiting...")
    result = job.result()

    return {
        "job_id": job.job_id(),
        "pubs": [{
            "counts": pub.data.meas.get_counts(),
            "transpiled_metrics": circuit_metrics(isa),
            "logical_metrics": circuit_metrics(logical),
        } for pub, isa, logical in zip(result, isa_circuits, circuits)],
    }


def save_result(record: dict, results_dir: Path = RESULTS_DIR) -> Path:
    """Write a result record to results/ as JSON and return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = record["timestamp"].replace(":", "").replace("-", "")[:15]
    tags = ("sweep_" if record.get("sweep") else "") + ("" if record.get("relative_phase", True) else "exact_")
    name = (f"{record['backend_kind']}_{tags}n{record['n_bits']}_p{record['plaintext']}_k{record['key']}"
            f"_{record['backend']}_{stamp}.json")
    path = results_dir / name
    path.write_text(json.dumps(record, indent=2))
    return path


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Load all saved result records (newest first). Used by the Streamlit app."""
    records = []
    for p in sorted(results_dir.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
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
    p.add_argument("--sweep", action="store_true",
                   help="run 0, 1, ..., optimal+1 iterations in one job (shows amplitude growth)")
    p.add_argument("--exact", action="store_true",
                   help="use the exact S-box instead of the cheaper relative-phase one (for comparison)")
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
    main_r = optimal_iterations(args.n) if args.iterations is None else args.iterations
    sweep_rs = list(range(0, optimal_iterations(args.n) + 2)) if args.sweep else [main_r]
    if main_r not in sweep_rs:
        sweep_rs.append(main_r)
    circuits = [build_grover_circuit(sbox, args.n, args.plaintext, ciphertext, r,
                                     relative_phase=not args.exact) for r in sweep_rs]

    backend = get_backend(args.n, fake=args.fake, backend_name=args.backend, instance=args.instance)
    kind = "fake" if args.fake else "hardware"
    print(f"Backend: {backend.name} ({kind}); n={args.n}, iterations={sweep_rs}, shots={args.shots}, "
          f"S-box={'exact' if args.exact else 'relative-phase'}")

    if kind == "hardware" and not args.yes:
        if input(f"This submits 1 job ({len(circuits)} circuit(s)) using your IBM Quantum time. "
                 f"Submit? [y/N] ").strip().lower() != "y":
            print("Aborted; nothing submitted.")
            return

    out = run_circuits(circuits, backend, args.shots, args.optimization_level, not args.no_dd)
    key_bits = cipher.to_bits(args.key, args.n)
    per_r = [{
        "iterations": r,
        "counts": pub["counts"],
        "correct_key_probability": pub["counts"].get(key_bits, 0) / args.shots,
        "theoretical_success": theoretical_success(args.n, r),
        "transpiled_metrics": pub["transpiled_metrics"],
    } for r, pub in zip(sweep_rs, out["pubs"])]
    main_pub = out["pubs"][sweep_rs.index(main_r)]
    main_entry = per_r[sweep_rs.index(main_r)]

    record = {
        "backend_kind": kind,
        "backend": backend.name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_bits": args.n,
        "sbox": sbox,
        "relative_phase": not args.exact,
        "plaintext": args.plaintext,
        "key": args.key,
        "ciphertext": ciphertext,
        "iterations": main_r,
        "shots": args.shots,
        "optimization_level": args.optimization_level,
        "dynamical_decoupling": not args.no_dd,
        "job_id": out["job_id"],
        # Top-level fields describe the main (optimal) iteration count.
        "counts": main_pub["counts"],
        "correct_key_probability": main_entry["correct_key_probability"],
        "theoretical_success": main_entry["theoretical_success"],
        "logical_metrics": main_pub["logical_metrics"],
        "transpiled_metrics": main_pub["transpiled_metrics"],
        # One entry per iteration count when --sweep is used (null otherwise).
        "sweep": sorted(per_r, key=lambda e: e["iterations"]) if args.sweep else None,
    }
    path = save_result(record)
    for e in record["sweep"] or [main_entry]:
        print(f"  r={e['iterations']}: P(correct key) = {e['correct_key_probability']:.3f} "
              f"(ideal {e['theoretical_success']:.3f}, 2q gates {e['transpiled_metrics']['two_qubit_gates']})")
    print(f"Random guess: {1 / 2 ** args.n:.3f}.  Saved {path}")


if __name__ == "__main__":
    main()
