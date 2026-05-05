"""Synthetic LNAV-like navigation bit generation with GPS word parity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


PREAMBLE = "10001011"
LNAV_WORD_BITS = 30
LNAV_DATA_BITS = 24
LNAV_SUBFRAME_WORDS = 10
LNAV_SUBFRAME_BITS = LNAV_WORD_BITS * LNAV_SUBFRAME_WORDS
MAX_TOW_COUNT = 604_800 // 6


def _xor_selected(bits: Sequence[int], indices: tuple[int, ...]) -> int:
    value = 0
    for index in indices:
        value ^= int(bits[index - 1])
    return value


def int_to_bits(value: int, width: int) -> list[int]:
    """Convert an integer to an MSB-first 0/1 list."""

    if width <= 0:
        raise ValueError("Bit width must be positive.")
    if value < 0 or value >= (1 << width):
        raise ValueError(f"Value {value} does not fit in {width} bits.")
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def compute_lnav_parity(data_bits: Sequence[int], d29_star: int, d30_star: int) -> list[int]:
    """Compute GPS LNAV parity bits D25..D30 for a 24-bit data payload."""

    if len(data_bits) != LNAV_DATA_BITS:
        raise ValueError("LNAV parity needs exactly 24 data bits.")

    d = [int(value) for value in data_bits]
    p25 = int(d29_star) ^ _xor_selected(d, (1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23))
    p26 = int(d30_star) ^ _xor_selected(d, (2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24))
    p27 = int(d29_star) ^ _xor_selected(d, (1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22))
    p28 = int(d30_star) ^ _xor_selected(d, (2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23))
    p29 = int(d30_star) ^ _xor_selected(d, (1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24))
    p30 = int(d29_star) ^ _xor_selected(d, (3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24))
    return [p25, p26, p27, p28, p29, p30]


def make_lnav_word(data_bits: Sequence[int], previous_word: Sequence[int] | None = None) -> list[int]:
    """Create one transmitted 30-bit LNAV word from 24 data bits."""

    if len(data_bits) != LNAV_DATA_BITS:
        raise ValueError("LNAV words need exactly 24 data bits.")
    previous = list(previous_word) if previous_word is not None else [0] * LNAV_WORD_BITS
    if len(previous) != LNAV_WORD_BITS:
        raise ValueError("Previous LNAV word must have 30 bits.")

    d29_star = int(previous[28])
    d30_star = int(previous[29])
    data = [int(value) for value in data_bits]
    transmitted_data = [value ^ d30_star for value in data]
    parity = compute_lnav_parity(data, d29_star, d30_star)
    return transmitted_data + parity


def extract_lnav_data_bits(word: Sequence[int], previous_word: Sequence[int] | None = None) -> list[int]:
    """Recover the 24 data bits from a transmitted LNAV word."""

    if len(word) < LNAV_DATA_BITS:
        raise ValueError("LNAV word is too short.")
    previous = list(previous_word) if previous_word is not None else [0] * LNAV_WORD_BITS
    d30_star = int(previous[29]) if len(previous) >= LNAV_WORD_BITS else 0
    return [int(bit) ^ d30_star for bit in list(word)[:LNAV_DATA_BITS]]


def check_lnav_word(word: Sequence[int], previous_word: Sequence[int] | None = None) -> bool:
    """Check the parity bits of one transmitted 30-bit LNAV word."""

    if len(word) != LNAV_WORD_BITS:
        return False
    previous = list(previous_word) if previous_word is not None else [0] * LNAV_WORD_BITS
    d29_star = int(previous[28]) if len(previous) >= LNAV_WORD_BITS else 0
    d30_star = int(previous[29]) if len(previous) >= LNAV_WORD_BITS else 0
    data = extract_lnav_data_bits(word, previous)
    expected = compute_lnav_parity(data, d29_star, d30_star)
    return expected == [int(bit) for bit in word[LNAV_DATA_BITS:]]


def _random_payload(rng: np.random.Generator) -> list[int]:
    value = int(rng.integers(0, 1 << LNAV_DATA_BITS, endpoint=False))
    return int_to_bits(value, LNAV_DATA_BITS)


def _make_payload_word(
    data_bits: list[int],
    previous_word: Sequence[int] | None,
    force_d30_zero: bool,
    rng: np.random.Generator,
) -> list[int]:
    """Build a payload word, optionally retrying until D30 is zero."""

    word = make_lnav_word(data_bits, previous_word)
    if not force_d30_zero or word[29] == 0:
        return word

    for _attempt in range(128):
        candidate_data = _random_payload(rng)
        word = make_lnav_word(candidate_data, previous_word)
        if word[29] == 0:
            return word
    raise RuntimeError("Could not create a synthetic payload word with D30=0.")


def build_synthetic_subframe(
    subframe_id: int,
    tow_count: int,
    rng: np.random.Generator,
    previous_word: Sequence[int] | None = None,
) -> list[list[int]]:
    """Build one synthetic parity-valid LNAV subframe.

    Word 10 is selected with D30=0 so the next TLM preamble remains visible
    to simple offline decoders that scan hard bits for the raw preamble.
    """

    if subframe_id < 1 or subframe_id > 5:
        raise ValueError("Subframe ID must be in the range 1..5.")
    tow = int(tow_count) % MAX_TOW_COUNT
    tlm_data = [int(bit) for bit in PREAMBLE] + int_to_bits(0x22C0, 16)
    how_data = int_to_bits(tow, 17) + [0, 0] + int_to_bits(subframe_id, 3) + [0, 0]

    words: list[list[int]] = []
    previous = list(previous_word) if previous_word is not None else None
    for data_bits in (tlm_data, how_data):
        word = make_lnav_word(data_bits, previous)
        words.append(word)
        previous = word

    for word_number in range(3, LNAV_SUBFRAME_WORDS + 1):
        payload = _random_payload(rng)
        if word_number == 3 and subframe_id in (4, 5):
            page_id = ((tow + subframe_id) % 32) + 1
            payload = [0, 1] + int_to_bits(page_id, 6) + payload[8:]
        force_d30_zero = word_number == LNAV_SUBFRAME_WORDS
        word = _make_payload_word(payload, previous, force_d30_zero, rng)
        words.append(word)
        previous = word

    return words


def build_lnav_bit_stream(
    num_bits: int,
    start_tow_count: int = 100,
    seed: int = 20260505,
) -> np.ndarray:
    """Build a deterministic transmitted LNAV bit stream."""

    if num_bits < 0:
        raise ValueError("Number of navigation bits must not be negative.")
    rng = np.random.default_rng(seed)
    stream: list[int] = []
    previous_word: list[int] | None = None
    subframe_index = 0
    while len(stream) < num_bits:
        subframe_id = (subframe_index % 5) + 1
        tow_count = (int(start_tow_count) + subframe_index) % MAX_TOW_COUNT
        words = build_synthetic_subframe(subframe_id, tow_count, rng, previous_word)
        stream.extend(bit for word in words for bit in word)
        previous_word = words[-1]
        subframe_index += 1
    return np.asarray(stream[:num_bits], dtype=np.int8)
