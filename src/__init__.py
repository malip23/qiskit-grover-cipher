"""Grover attack on a toy cipher C = S(P XOR K).

Modules:
    cipher        classical reference implementation
    sbox_circuit  reversible S-box synthesis
    metrics       circuit cost metrics
    oracle        Grover oracle for the known-plaintext attack
    grover        full Grover circuit + simulation helpers
    run_hardware  IBM Quantum hardware runner
    plots         shared charts for the app and notebook
"""
