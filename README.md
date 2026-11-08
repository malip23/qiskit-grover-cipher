# qiskit-grover-cipher

**A Grover attack on a toy block cipher.** We recover the secret key of a tiny cipher from one known
(plaintext, ciphertext) pair using Grover's algorithm, on the Qiskit Aer simulator and on real IBM
quantum hardware.

## The cipher

```
C = S(P ⊕ K)        P = plaintext, K = secret key, C = ciphertext (all n bits)
                    S = an n-bit S-box (a permutation / lookup table)
```

| n | S-box | Where it runs |
| --- | --- | --- |
| 4 | PRESENT cipher S-box `C 5 6 B 9 0 A D 3 E F 8 4 7 1 2` (Bogdanov et al., CHES 2007) | simulator (main version) |
| 3 | `6 4 1 7 2 0 3 5` | simulator + hardware |
| 2 | `2 0 3 1` | simulator + hardware |

Smaller S-boxes are used on hardware because deep circuits get wiped out by noise.

## The attack

The **oracle** works in place on n key qubits:

1. X gates where P has a 1 bit → `|K ⊕ P⟩`
2. Reversible S-box circuit → `|S(K ⊕ P)⟩`
3. X gates where C has a 0 bit → the correct key becomes `|11…1⟩`
4. Multi-controlled Z → phase-flip that state
5. Undo steps 3, 2, 1

Then the standard Grover **diffusion**. Repeat `floor(π/4 · √2ⁿ)` times (1, 2, 3 for n = 2, 3, 4) and
measure. The ideal success probability is 100% / 94.5% / 96.1%.

| n | keys | classical guesses (avg) | Grover oracle calls |
| --- | --- | --- | --- |
| 2 | 4 | 2.5 | 1 |
| 3 | 8 | 4.5 | 2 |
| 4 | 16 | 8.5 | 3 |

> **Honest caveat.** Because S is a permutation, one (P, C) pair determines the key exactly:
> K = S⁻¹(C) ⊕ P. A classical attacker who knows S can simply invert it. The toy cipher is a
> teaching example: it shows *how* Grover turns a key search into amplitude amplification, and
> what that costs in gates and noise on today's hardware. Real ciphers (AES) mix many rounds,
> so there's no shortcut, and Grover's √N speed-up roughly halves the effective key length.

## Repository layout

```
src/
  cipher.py        classical cipher, S-box tables, brute-force search
  sbox_circuit.py  reversible S-box synthesis (baseline, room to optimize)
  metrics.py       gate / 2-qubit gate / depth, before and after transpile
  oracle.py        the Grover oracle
  grover.py        full Grover circuit, Aer runs, per-iteration probabilities
  run_hardware.py  IBM Quantum runner (SamplerV2 + dynamical decoupling)
  plots.py         shared bar charts for the app and notebook
tests/             pytest suite
app.py             Streamlit demo
notebooks/         step-by-step walkthrough
results/           saved hardware runs (JSON), read by the demo
docs/OPTIMIZATION_NOTES.md   measured circuit costs and ideas for making the S-box cheaper
```

## Setup

Requires Python 3.11+ (developed on 3.14).

```bash
# 1. Create and activate the virtual environment
python -m venv qenv
# Windows (PowerShell):
.\qenv\Scripts\Activate.ps1
# macOS / Linux:
source qenv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Save your IBM Quantum credentials (once, only needed for real hardware)

Get an API key and your instance CRN from <https://quantum.cloud.ibm.com>. Then run this **once in a
Python shell, not in a file in this repo**:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="<YOUR_API_KEY>",
    instance="<YOUR_INSTANCE_CRN>",  # optional
    set_as_default=True,
)
```

This stores the credentials in `~/.qiskit/qiskit-ibm.json` in your home directory. The code only ever
calls `QiskitRuntimeService()`, which loads the saved account. **Never commit an API key.**

## Running

All commands are run from the repo root with `qenv` activated.

### Tests

```bash
pytest                  # everything
pytest tests/test_sbox.py   # just the S-box circuits (run after any sbox_circuit.py change)
```

### Simulator

```bash
python -m src.grover --n 4 --plaintext 0x9 --key 0x3
python -m src.grover --n 3 --seed 1        # random plaintext and key
```

### Hardware

```bash
# Free offline dry run on a noisy snapshot of an IBM device (no account, no queue):
python -m src.run_hardware --n 2 --plaintext 1 --key 2 --fake

# Real hardware: uses your IBM Quantum time and asks you to confirm before submitting
python -m src.run_hardware --n 2 --plaintext 1 --key 2
python -m src.run_hardware --n 3 --plaintext 5 --key 6 --shots 4096
```

This picks the least busy device, transpiles with `generate_preset_pass_manager(optimization_level=3)`,
runs with `SamplerV2` and dynamical decoupling, and saves counts and metadata to
`results/<kind>_n<n>_p<P>_k<K>_<backend>_<time>.json`. Commit these JSON files so the demo can show them.

### Demo

```bash
streamlit run app.py
```

Pick n, a plaintext and a secret key. The demo shows the ciphertext, the classical guess count, a
stepped / animated chart of key probabilities per Grover iteration, the simulator histogram, and a
hardware histogram **loaded from `results/`**, so it never depends on a live hardware queue.

### Circuit cost report

```bash
python -m src.metrics      # transpiled gate counts and depth on an offline fake IBM backend
```

### Notebook

```bash
jupyter lab notebooks/01_walkthrough.ipynb
```

## Author

Built solo for a Qiskit / IBM Quantum hackathon.
