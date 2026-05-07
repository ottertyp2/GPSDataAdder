"""Tests for multi-satellite position relocation overlay writing."""

from __future__ import annotations

import json

import numpy as np

from app.dsp.lnav import build_synthetic_subframe
from app.dsp.relocation_overlay import (
    GPS_L1_WAVELENGTH_M,
    RelocationChannelPlan,
    RelocationOverlayPlan,
    SPEED_OF_LIGHT_M_S,
    add_relocation_overlay_to_file,
    code_phase_for_range_delta,
    _generate_relocation_satellite_block,
    _nav_bit_indices,
    _nav_time_shift_samples,
    _synthetic_doppler_hz,
    _synthetic_doppler_rate_hz_s,
    _shift_code_phase_from_geometry,
    _shift_code_phase_samples_from_geometry,
    _synthetic_code_rate_hz,
)
from app.dsp.synthetic_satellite import SyntheticSatelliteConfig


def _templates() -> tuple[dict[str, object], ...]:
    rng = np.random.default_rng(22)
    previous = None
    templates = []
    for subframe_id in (1, 2, 3):
        words = build_synthetic_subframe(subframe_id, 700 + subframe_id, rng, previous)
        previous = words[-1]
        templates.append(
            {
                "subframe_id": subframe_id,
                "tow_count": 700 + subframe_id,
                "tow_seconds": (700 + subframe_id) * 6,
                "words": ["".join(str(bit) for bit in word) for word in words],
            }
        )
    return tuple(templates)


def test_relocation_overlay_adds_stronger_replica_without_replacing_samples(tmp_path) -> None:
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    metadata_path = tmp_path / "output.json"
    original = np.zeros(20_000, dtype=np.complex64)
    original.tofile(input_path)
    plan = RelocationOverlayPlan(
        input_path=str(input_path),
        sample_rate_hz=1_023_000.0,
        total_samples=int(original.size),
        duration_s=float(original.size) / 1_023_000.0,
        baseline_latitude_deg=50.0,
        baseline_longitude_deg=7.0,
        baseline_altitude_m=100.0,
        target_latitude_deg=50.0,
        target_longitude_deg=7.01,
        target_altitude_m=100.0,
        shift_east_m=700.0,
        shift_north_m=0.0,
        shift_up_m=0.0,
        target_cn0_dbhz=50.0,
        compute_backend="cpu",
        worker_count=1,
        in_flight_blocks=1,
        chunk_samples=4096,
        channels=(
            RelocationChannelPlan(
                prn=3,
                doppler_hz=500.0,
                code_rate_hz=1_023_000.0,
                original_code_phase_samples=100,
                code_phase_samples=103,
                range_delta_m=900.0,
                range_delta_samples=3.0,
                amplitude=0.2,
                start_tow_count=701,
                start_subframe_id=1,
                reference_bit_index=0,
                reference_sample=0,
                nav_subframes=_templates(),
            ),
        ),
        summary_lines=("test",),
    )

    result = add_relocation_overlay_to_file(input_path, output_path, plan, metadata_path=metadata_path)
    augmented = np.fromfile(output_path, dtype=np.complex64)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result.channel_count == 1
    assert augmented.shape == original.shape
    assert np.max(np.abs(augmented)) > 0.0
    assert metadata["mode"] == "position_relocation_overlay"


def test_code_phase_uses_fraunhofer_pseudorange_sign() -> None:
    assert code_phase_for_range_delta(1000, 12.4, 6061) == 988
    assert code_phase_for_range_delta(1000, -12.4, 6061) == 1012


def test_code_phase_rebased_from_reference_sample() -> None:
    sample_rate_hz = 6_061_000.0
    reference_sample = 394_357_576
    code_rate_hz = 1_023_002.0
    reference_chips = 770.25

    original, shifted = _shift_code_phase_samples_from_geometry(
        reference_chips,
        reference_sample,
        0.0,
        code_rate_hz,
        sample_rate_hz,
    )

    assert 0 <= original < int(round(sample_rate_hz * 1e-3))
    assert shifted == original


def test_fractional_code_phase_preserves_sub_sample_geometry() -> None:
    sample_rate_hz = 6_061_000.0
    source_code_rate_hz = 1_023_000.0
    range_delta_m = SPEED_OF_LIGHT_M_S * 0.05 / source_code_rate_hz

    shift = _shift_code_phase_from_geometry(
        reference_code_phase_chips=100.0,
        reference_sample=0,
        range_delta_m=range_delta_m,
        source_code_rate_hz=source_code_rate_hz,
        synthetic_code_rate_hz=source_code_rate_hz,
        sample_rate_hz=sample_rate_hz,
    )

    assert shift.original_code_phase_samples == shift.code_phase_samples
    assert np.isclose((shift.original_code_phase_chips - shift.code_phase_chips) % 1023.0, 0.05)


def test_range_rate_compensation_adjusts_code_rate_sign() -> None:
    source_code_rate_hz = 1_023_000.0

    assert _synthetic_code_rate_hz(source_code_rate_hz, 100.0) < source_code_rate_hz
    assert _synthetic_code_rate_hz(source_code_rate_hz, -100.0) > source_code_rate_hz


def test_global_range_delta_is_split_into_lnav_time_shift() -> None:
    sample_rate_hz = 1_000_000.0

    assert _nav_time_shift_samples(41_900.0, sample_rate_hz) == 42_000
    assert _nav_time_shift_samples(-41_900.0, sample_rate_hz) == -42_000

    indices = _nav_bit_indices(
        start_sample=0,
        sample_count=41,
        sample_rate_hz=1_000.0,
        reference_sample=20,
        reference_bit_index=1,
    )

    assert int(indices[0]) == 0
    assert int(indices[20]) == 1


def test_doppler_compensation_uses_l1_wavelength() -> None:
    source_doppler_hz = 1200.0

    assert np.isclose(_synthetic_doppler_hz(source_doppler_hz, GPS_L1_WAVELENGTH_M), source_doppler_hz - 1.0)
    assert np.isclose(_synthetic_doppler_rate_hz_s(GPS_L1_WAVELENGTH_M), -1.0)


def test_relocation_generator_uses_fractional_code_phase() -> None:
    config = SyntheticSatelliteConfig(
        sample_rate_hz=6_138_000.0,
        prn=3,
        doppler_hz=0.0,
        code_phase_samples=0,
        amplitude=1.0,
    )
    nav_bits = np.zeros(10, dtype=np.int8)

    rounded = _generate_relocation_satellite_block(
        config,
        start_sample=0,
        sample_count=128,
        nav_bits=nav_bits,
        reference_sample=0,
        reference_bit_index=0,
        code_rate_hz=1_023_000.0,
    )
    fractional = _generate_relocation_satellite_block(
        config,
        start_sample=0,
        sample_count=128,
        nav_bits=nav_bits,
        reference_sample=0,
        reference_bit_index=0,
        code_rate_hz=1_023_000.0,
        initial_code_phase_chips=0.5,
    )

    assert not np.array_equal(rounded, fractional)


def test_relocation_generator_applies_doppler_rate() -> None:
    config = SyntheticSatelliteConfig(
        sample_rate_hz=1_023_000.0,
        prn=3,
        doppler_hz=500.0,
        code_phase_samples=0,
        amplitude=1.0,
    )
    nav_bits = np.zeros(10, dtype=np.int8)

    constant = _generate_relocation_satellite_block(
        config,
        start_sample=0,
        sample_count=2046,
        nav_bits=nav_bits,
        reference_sample=0,
        reference_bit_index=0,
        code_rate_hz=1_023_000.0,
    )
    drifting = _generate_relocation_satellite_block(
        config,
        start_sample=0,
        sample_count=2046,
        nav_bits=nav_bits,
        reference_sample=0,
        reference_bit_index=0,
        code_rate_hz=1_023_000.0,
        doppler_rate_hz_s=200.0,
    )

    assert not np.allclose(constant, drifting)
