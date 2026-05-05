"""Tests for synthetic LNAV bit generation."""

from __future__ import annotations

import numpy as np

from app.dsp.lnav import (
    LNAV_SUBFRAME_BITS,
    LNAV_WORD_BITS,
    PREAMBLE,
    build_lnav_bit_stream,
    build_lnav_bit_stream_from_templates,
    build_synthetic_subframe,
    check_lnav_word,
    extract_lnav_data_bits,
    find_lnav_tow,
)


def test_lnav_stream_starts_with_visible_preamble() -> None:
    bits = build_lnav_bit_stream(600, seed=1)

    assert "".join(str(int(bit)) for bit in bits[:8]) == PREAMBLE
    assert "".join(str(int(bit)) for bit in bits[LNAV_SUBFRAME_BITS : LNAV_SUBFRAME_BITS + 8]) == PREAMBLE


def test_lnav_words_have_valid_parity() -> None:
    bits = build_lnav_bit_stream(900, seed=2).astype(int).tolist()
    previous_word: list[int] | None = None

    for start in range(0, len(bits) - LNAV_WORD_BITS + 1, LNAV_WORD_BITS):
        word = bits[start : start + LNAV_WORD_BITS]
        assert check_lnav_word(word, previous_word)
        previous_word = word


def test_how_subframe_id_cycles() -> None:
    bits = build_lnav_bit_stream(5 * LNAV_SUBFRAME_BITS, start_tow_count=42, seed=3).astype(int).tolist()
    previous_word: list[int] | None = None
    subframe_ids: list[int] = []

    for start in range(0, len(bits), LNAV_SUBFRAME_BITS):
        tlm = bits[start : start + LNAV_WORD_BITS]
        how = bits[start + LNAV_WORD_BITS : start + 2 * LNAV_WORD_BITS]
        assert check_lnav_word(tlm, previous_word)
        assert check_lnav_word(how, tlm)
        how_data = extract_lnav_data_bits(how, tlm)
        subframe_ids.append(int("".join(str(bit) for bit in how_data[19:22]), 2))
        previous_word = bits[start + 9 * LNAV_WORD_BITS : start + 10 * LNAV_WORD_BITS]

    assert subframe_ids == [1, 2, 3, 4, 5]


def test_lnav_tow_can_be_found_from_hard_bits() -> None:
    bits = build_lnav_bit_stream(600, start_tow_count=1234, start_subframe_id=3, seed=4)

    estimate = find_lnav_tow(bits)

    assert estimate is not None
    assert estimate.tow_count == 1234
    assert estimate.tow_seconds == 7404
    assert estimate.subframe_id == 3
    assert estimate.polarity == "normal"


def test_lnav_templates_rebuild_continuous_how_timing() -> None:
    rng = np.random.default_rng(12)
    previous = None
    templates = []
    for subframe_id in (1, 2, 3):
        words = build_synthetic_subframe(subframe_id, 500 + subframe_id, rng, previous)
        previous = words[-1]
        templates.append(
            {
                "subframe_id": subframe_id,
                "tow_count": 500 + subframe_id,
                "words": ["".join(str(bit) for bit in word) for word in words],
            }
        )

    bits = build_lnav_bit_stream_from_templates(
        LNAV_SUBFRAME_BITS * 3,
        templates,
        start_tow_count=1234,
        start_subframe_id=2,
    )
    estimate = find_lnav_tow(bits)

    assert estimate is not None
    assert estimate.tow_count == 1234
    assert estimate.subframe_id == 2


def test_lnav_generation_is_deterministic() -> None:
    first = build_lnav_bit_stream(1000, seed=7)
    second = build_lnav_bit_stream(1000, seed=7)

    np.testing.assert_array_equal(first, second)
