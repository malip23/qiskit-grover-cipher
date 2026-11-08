"""Grover attack on a toy cipher C = S(P XOR K).

Modules:
    cipher        classical reference implementation (CS)
    sbox_circuit  reversible S-box synthesis (ECE)
    metrics       circuit cost metrics (ECE)
    oracle        Grover oracle for the known-plaintext attack (CS)
    grover        full Grover circuit + simulation helpers (CS)
    run_hardware  IBM Quantum hardware runner (CS)
"""
