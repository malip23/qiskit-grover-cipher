"""Classical reference implementation of the toy cipher  C = S(P XOR K).

Everything quantum in this project is checked against the functions here.

Bit convention (used everywhere in the project):
    An n-bit value x is stored as an int 0 <= x < 2**n.
    Bit i of x is (x >> i) & 1, and lives on qubit i in the quantum circuits.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# S-box tables
# ---------------------------------------------------------------------------

# 2-bit S-box: any permutation of {0,1,2,3} with no fixed points (S(x) != x).
SBOX_2 = [0x2, 0x0, 0x3, 0x1]

# 3-bit S-box: a permutation of {0..7} with no fixed points.
SBOX_3 = [0x6, 0x4, 0x1, 0x7, 0x2, 0x0, 0x3, 0x5]

# 4-bit S-box from the PRESENT block cipher
# (Bogdanov et al., "PRESENT: An Ultra-Lightweight Block Cipher", CHES 2007, Table 1).
SBOX_PRESENT = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
                0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]

# Default S-box for each block size.
SBOXES: dict[int, list[int]] = {
    2: SBOX_2,
    3: SBOX_3,
    4: SBOX_PRESENT,
}


def get_sbox(n_bits: int) -> list[int]:
    """Return the default S-box table for an n-bit block (n = 2, 3 or 4)."""
    if n_bits not in SBOXES:
        raise ValueError(f"No S-box defined for n_bits={n_bits}; choose from {sorted(SBOXES)}")
    return SBOXES[n_bits]


def is_permutation(sbox: list[int], n_bits: int) -> bool:
    """True if sbox is a bijection on {0, ..., 2**n_bits - 1}."""
    return sorted(sbox) == list(range(2 ** n_bits))


def invert_sbox(sbox: list[int]) -> list[int]:
    """Return the inverse table S^-1, so that S^-1[S[x]] == x."""
    inv = [0] * len(sbox)
    for x, y in enumerate(sbox):
        inv[y] = x
    return inv


# ---------------------------------------------------------------------------
# Cipher
# ---------------------------------------------------------------------------

def encrypt(plaintext: int, key: int, sbox: list[int]) -> int:
    """C = S(P XOR K)."""
    return sbox[plaintext ^ key]


def decrypt(ciphertext: int, key: int, sbox: list[int]) -> int:
    """P = S^-1(C) XOR K."""
    return invert_sbox(sbox)[ciphertext] ^ key


# ---------------------------------------------------------------------------
# Classical attack: brute-force key search
# ---------------------------------------------------------------------------

def brute_force(plaintext: int, ciphertext: int, sbox: list[int],
                n_bits: int) -> tuple[int | None, int]:
    """Try keys 0, 1, 2, ... until one encrypts plaintext to ciphertext.

    Returns:
        (key, guesses): the recovered key (or None) and how many keys were tried.
        On average a classical search needs about 2**n / 2 guesses; Grover needs
        about (pi/4) * sqrt(2**n) oracle calls.
    """
    guesses = 0
    for k in range(2 ** n_bits):
        guesses += 1
        if encrypt(plaintext, k, sbox) == ciphertext:
            return k, guesses
    return None, guesses


def to_bits(x: int, n_bits: int) -> str:
    """Format x as an n-bit binary string, most significant bit first (like Qiskit counts)."""
    return format(x, f"0{n_bits}b")
