"""Tests for multi-satellite position relocation overlay writing."""

from __future__ import annotations

import json

import numpy as np

from app.dsp.lnav import build_synthetic_subframe
from app.dsp.relocation_overlay import RelocationChannelPlan, RelocationOverlayPlan, add_relocation_overlay_to_file


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
                original_code_phase_samples=100,
                code_phase_samples=103,
                range_delta_m=900.0,
                range_delta_samples=3.0,
                amplitude=0.2,
                start_tow_count=701,
                start_subframe_id=1,
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
