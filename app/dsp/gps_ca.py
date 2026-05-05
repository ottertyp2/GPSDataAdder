"""GPS L1 C/A code generation helpers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


G2_TAPS: dict[int, tuple[int, int]] = {
    1: (2, 6),
    2: (3, 7),
    3: (4, 8),
    4: (5, 9),
    5: (1, 9),
    6: (2, 10),
    7: (1, 8),
    8: (2, 9),
    9: (3, 10),
    10: (2, 3),
    11: (3, 4),
    12: (5, 6),
    13: (6, 7),
    14: (7, 8),
    15: (8, 9),
    16: (9, 10),
    17: (1, 4),
    18: (2, 5),
    19: (3, 6),
    20: (4, 7),
    21: (5, 8),
    22: (6, 9),
    23: (1, 3),
    24: (4, 6),
    25: (5, 7),
    26: (6, 8),
    27: (7, 9),
    28: (8, 10),
    29: (1, 6),
    30: (2, 7),
    31: (3, 8),
    32: (4, 9),
}

CA_CODE_LENGTH = 1023
CA_CODE_RATE_HZ = 1.023e6


@lru_cache(maxsize=32)
def generate_ca_code(prn: int) -> np.ndarray:
    """Generate one 1023-chip GPS L1 C/A PRN code as +/-1 values."""

    if prn not in G2_TAPS:
        raise ValueError(f"Unsupported PRN {prn}. Supported range is 1..32.")

    tap1, tap2 = G2_TAPS[prn]
    g1 = np.ones(10, dtype=np.int8)
    g2 = np.ones(10, dtype=np.int8)
    code = np.empty(CA_CODE_LENGTH, dtype=np.int8)

    for idx in range(CA_CODE_LENGTH):
        g1_out = g1[-1]
        g2_out = g2[tap1 - 1] ^ g2[tap2 - 1]
        code[idx] = 1 - 2 * (g1_out ^ g2_out)

        g1_feedback = g1[2] ^ g1[9]
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1[1:] = g1[:-1]
        g2[1:] = g2[:-1]
        g1[0] = g1_feedback
        g2[0] = g2_feedback

    return code.astype(np.float32)


def sample_ca_code(
    prn: int,
    sample_rate: float,
    num_samples: int,
    code_phase_chips: float = 0.0,
    code_rate_hz: float = CA_CODE_RATE_HZ,
) -> np.ndarray:
    """Sample a local C/A code replica at the requested sample rate."""

    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive for C/A code sampling.")
    if int(num_samples) < 0:
        raise ValueError("Number of C/A code samples must not be negative.")
    if not np.isfinite(code_rate_hz) or code_rate_hz <= 0:
        raise ValueError("Code rate must be positive for C/A code sampling.")

    base_code = generate_ca_code(prn)
    chip_positions = code_phase_chips + (np.arange(num_samples, dtype=np.float64) * code_rate_hz / sample_rate)
    chip_indices = np.floor(chip_positions).astype(np.int64) % CA_CODE_LENGTH
    return base_code[chip_indices]


def code_phase_samples_to_chips(code_phase_samples: int, sample_rate: float) -> float:
    """Convert a code phase expressed in samples to chips."""

    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive for code-phase conversion.")
    samples_per_ms = int(round(sample_rate * 1e-3))
    if samples_per_ms <= 0:
        raise ValueError("Sample rate is too low for code-phase conversion.")
    return float(code_phase_samples) * CA_CODE_LENGTH / float(samples_per_ms)
