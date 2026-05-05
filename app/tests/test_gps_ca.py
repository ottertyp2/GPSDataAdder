"""Tests for GPS C/A code generation."""

from __future__ import annotations

import numpy as np
import pytest

from app.dsp.gps_ca import G2_TAPS, generate_ca_code, sample_ca_code


REFERENCE_FIRST_10_CHIPS_OCTAL = {
    1: "1440",
    2: "1620",
    3: "1710",
    8: "1454",
    22: "1763",
    32: "1712",
}


def _first_10_chips_octal(prn: int) -> str:
    bits = "".join("1" if chip < 0 else "0" for chip in generate_ca_code(prn)[:10])
    return bits[0] + format(int(bits[1:], 2), "03o")


def test_generate_ca_code_length_and_values() -> None:
    code = generate_ca_code(1)

    assert code.shape == (1023,)
    assert set(np.unique(code)).issubset({-1.0, 1.0})


@pytest.mark.parametrize("prn, expected_octal", REFERENCE_FIRST_10_CHIPS_OCTAL.items())
def test_generate_ca_code_matches_reference_first_10_chips(prn: int, expected_octal: str) -> None:
    assert _first_10_chips_octal(prn) == expected_octal


def test_prn_3_and_8_use_distinct_reference_taps() -> None:
    assert G2_TAPS[3] == (4, 8)
    assert G2_TAPS[8] == (2, 9)


def test_sample_ca_code_size() -> None:
    sampled = sample_ca_code(3, 4_092_000.0, 4_092)

    assert sampled.shape == (4_092,)
