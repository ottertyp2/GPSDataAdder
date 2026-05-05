"""Tests for synthetic satellite generation and file augmentation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.dsp.lnav import build_lnav_bit_stream
from app.dsp.synthetic_satellite import (
    SyntheticSatelliteConfig,
    add_synthetic_satellite_to_file,
    detect_synthetic_signal_plan,
    estimate_realistic_amplitude,
    generate_synthetic_satellite_block,
)


def test_block_generation_is_continuous_across_chunks() -> None:
    config = SyntheticSatelliteConfig(sample_rate_hz=4_092_000.0, prn=3, doppler_hz=750.0, amplitude=0.2)
    nav_bits = build_lnav_bit_stream(1000, seed=config.nav_seed)

    full = generate_synthetic_satellite_block(config, 0, 12_000, nav_bits)
    split = np.concatenate(
        [
            generate_synthetic_satellite_block(config, 0, 5000, nav_bits),
            generate_synthetic_satellite_block(config, 5000, 7000, nav_bits),
        ]
    )

    np.testing.assert_allclose(full, split, rtol=1e-6, atol=1e-6)


def test_file_augmentation_writes_complex64_output_and_metadata(tmp_path) -> None:
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    metadata_path = tmp_path / "output.json"
    original = np.zeros(20_000, dtype=np.complex64)
    original.tofile(input_path)
    config = SyntheticSatelliteConfig(sample_rate_hz=1_023_000.0, prn=5, doppler_hz=250.0, amplitude=0.1)

    result = add_synthetic_satellite_to_file(
        input_path,
        output_path,
        config,
        chunk_samples=4096,
        metadata_path=metadata_path,
        compute_backend="cpu",
    )
    augmented = np.fromfile(output_path, dtype=np.complex64)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result.total_samples == original.size
    assert augmented.dtype == np.complex64
    assert augmented.shape == original.shape
    assert np.max(np.abs(augmented)) == pytest.approx(config.amplitude)
    assert metadata["config"]["prn"] == 5
    assert metadata["amplitude_mode"] == "manual"
    assert metadata["processing"]["compute_backend"] == "cpu"
    assert metadata["synthetic_signature_id"] == result.synthetic_signature_id


def test_auto_amplitude_estimate_uses_input_rms_and_cn0(tmp_path) -> None:
    input_path = tmp_path / "level.bin"
    np.full(20_000, 0.6 + 0.8j, dtype=np.complex64).tofile(input_path)

    estimate = estimate_realistic_amplitude(
        input_path,
        sample_rate_hz=1_000_000.0,
        target_cn0_dbhz=40.0,
        probe_samples=4096,
        probe_windows=3,
    )

    assert estimate.input_rms == pytest.approx(1.0)
    assert estimate.relative_db == pytest.approx(-20.0)
    assert estimate.amplitude == pytest.approx(0.1)


def test_file_augmentation_can_use_auto_amplitude(tmp_path) -> None:
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    metadata_path = tmp_path / "output.json"
    np.full(12_000, 1.0 + 0.0j, dtype=np.complex64).tofile(input_path)
    config = SyntheticSatelliteConfig(sample_rate_hz=1_000_000.0, prn=7, amplitude=999.0)

    result = add_synthetic_satellite_to_file(
        input_path,
        output_path,
        config,
        chunk_samples=3000,
        metadata_path=metadata_path,
        auto_amplitude=True,
        target_cn0_dbhz=40.0,
        compute_backend="cpu",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result.amplitude_mode == "auto"
    assert result.effective_amplitude == pytest.approx(0.1)
    assert result.amplitude_estimate is not None
    assert metadata["amplitude_mode"] == "auto"
    assert metadata["config"]["amplitude"] == pytest.approx(0.1)


def test_parallel_cpu_output_matches_single_worker(tmp_path) -> None:
    input_path = tmp_path / "input.bin"
    one_path = tmp_path / "one.bin"
    many_path = tmp_path / "many.bin"
    rng = np.random.default_rng(11)
    original = (
        rng.normal(0.0, 0.2, 30_000).astype(np.float32)
        + 1j * rng.normal(0.0, 0.2, 30_000).astype(np.float32)
    ).astype(np.complex64)
    original.tofile(input_path)
    config = SyntheticSatelliteConfig(sample_rate_hz=1_023_000.0, prn=11, doppler_hz=-750.0, amplitude=0.03)

    add_synthetic_satellite_to_file(
        input_path,
        one_path,
        config,
        chunk_samples=4096,
        compute_backend="cpu",
        worker_count=1,
        in_flight_blocks=1,
    )
    result = add_synthetic_satellite_to_file(
        input_path,
        many_path,
        config,
        chunk_samples=4096,
        compute_backend="cpu",
        worker_count=3,
        in_flight_blocks=2,
    )

    np.testing.assert_array_equal(np.fromfile(one_path, dtype=np.complex64), np.fromfile(many_path, dtype=np.complex64))
    assert result.compute_backend == "cpu"
    assert result.worker_count == 3
    assert result.in_flight_blocks == 2


def test_detect_signal_plan_is_deterministic_and_populates_parameters(tmp_path) -> None:
    input_path = tmp_path / "capture.bin"
    np.full(25_000, 0.6 + 0.8j, dtype=np.complex64).tofile(input_path)

    first = detect_synthetic_signal_plan(
        input_path,
        sample_rate_hz=1_000_000.0,
        mode="balanced",
        requested_backend="cpu",
        worker_count=2,
        in_flight_blocks=4,
        chunk_samples=5000,
    )
    second = detect_synthetic_signal_plan(
        input_path,
        sample_rate_hz=1_000_000.0,
        mode="balanced",
        requested_backend="cpu",
        worker_count=2,
        in_flight_blocks=4,
        chunk_samples=5000,
    )

    assert first == second
    assert 1 <= first.prn <= 32
    assert first.amplitude == pytest.approx(10 ** ((42.0 - 60.0) / 20.0))
    assert first.compute_backend == "cpu"
    assert first.worker_count == 2
    assert first.in_flight_blocks == 4
    assert first.start_tow_count == 100
    assert first.start_subframe_id == 1
    assert first.tow_source == "not detected"
    assert len(first.summary_lines) == 5


def test_detect_signal_plan_imitates_measurement_tow_from_sidecar(tmp_path) -> None:
    input_path = tmp_path / "capture.bin"
    np.full(25_000, 0.6 + 0.8j, dtype=np.complex64).tofile(input_path)
    sidecar = input_path.with_suffix(input_path.suffix + ".synthetic.json")
    sidecar.write_text(
        json.dumps(
            {
                "config": {
                    "start_tow_count": 3456,
                    "start_subframe_id": 4,
                    "prn": 9,
                    "doppler_hz": 1250.0,
                    "code_phase_samples": 99,
                }
            }
        ),
        encoding="utf-8",
    )

    plan = detect_synthetic_signal_plan(
        input_path,
        sample_rate_hz=1_000_000.0,
        mode="balanced",
        requested_backend="cpu",
        worker_count=2,
        in_flight_blocks=4,
        chunk_samples=5000,
    )

    assert plan.start_tow_count == 3456
    assert plan.start_subframe_id == 4
    assert plan.measurement_tow_count == 3456
    assert plan.measurement_tow_seconds == 20_736
    assert "metadata sidecar" in plan.tow_source


def test_input_and_output_must_differ(tmp_path) -> None:
    path = tmp_path / "same.bin"
    np.zeros(8, dtype=np.complex64).tofile(path)

    with pytest.raises(ValueError, match="different"):
        add_synthetic_satellite_to_file(path, path, SyntheticSatelliteConfig())
