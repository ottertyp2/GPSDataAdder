"""Offline PVT relocation overlay for local complex64 GPS captures."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np

from app.dsp.lnav import build_lnav_bit_stream_from_templates
from app.dsp.synthetic_satellite import (
    COMPLEX64_DTYPE,
    DEFAULT_CHUNK_SAMPLES,
    DEFAULT_WORKER_COUNT,
    NAV_BIT_RATE_BPS,
    ProcessingCancelled,
    SyntheticSatelliteConfig,
    _create_sized_file,
    _effective_in_flight_blocks,
    _effective_worker_count,
    _generate_synthetic_satellite_block_gpu,
    _load_cupy,
    count_complex64_samples,
    estimate_realistic_amplitude,
    generate_synthetic_satellite_block,
    resolve_compute_backend,
)
from app.dsp.tow_detect import _fraunhofer_project_path


SPEED_OF_LIGHT_M_S = 299_792_458.0
WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
DEFAULT_RELOCATION_CN0_DBHZ = 50.0
DEFAULT_RELOCATION_TRACKING_S = 42.0
DEFAULT_RELOCATION_MAX_SATELLITES = 6


@dataclass(frozen=True)
class RelocationChannelPlan:
    """One stronger synthetic replica of an already received PRN."""

    prn: int
    doppler_hz: float
    original_code_phase_samples: int
    code_phase_samples: int
    range_delta_m: float
    range_delta_samples: float
    amplitude: float
    start_tow_count: int
    start_subframe_id: int
    nav_subframes: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class RelocationOverlayPlan:
    """A complete offline position relocation overlay plan."""

    input_path: str
    sample_rate_hz: float
    total_samples: int
    duration_s: float
    baseline_latitude_deg: float
    baseline_longitude_deg: float
    baseline_altitude_m: float
    target_latitude_deg: float
    target_longitude_deg: float
    target_altitude_m: float
    shift_east_m: float
    shift_north_m: float
    shift_up_m: float
    target_cn0_dbhz: float
    compute_backend: str
    worker_count: int
    in_flight_blocks: int
    chunk_samples: int
    channels: tuple[RelocationChannelPlan, ...]
    summary_lines: tuple[str, ...]


@dataclass(frozen=True)
class RelocationAddResult:
    """Result from writing a relocation overlay file."""

    input_path: str
    output_path: str
    metadata_path: str | None
    total_samples: int
    channel_count: int
    compute_backend: str
    worker_count: int
    in_flight_blocks: int


def lla_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    lat = np.deg2rad(float(latitude_deg))
    lon = np.deg2rad(float(longitude_deg))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    normal = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.asarray(
        [
            (normal + altitude_m) * cos_lat * np.cos(lon),
            (normal + altitude_m) * cos_lat * np.sin(lon),
            (normal * (1.0 - WGS84_E2) + altitude_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def ecef_to_lla(ecef_m: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.asarray(ecef_m, dtype=np.float64)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = np.sin(lat)
        normal = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        lat = np.arctan2(z + WGS84_E2 * normal * sin_lat, p)
    sin_lat = np.sin(lat)
    normal = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / max(np.cos(lat), 1e-12) - normal
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


def offset_lla(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    east_m: float,
    north_m: float,
    up_m: float,
) -> tuple[float, float, float]:
    lat = np.deg2rad(float(latitude_deg))
    lon = np.deg2rad(float(longitude_deg))
    east = np.asarray([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.asarray([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=np.float64)
    up = np.asarray([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
    shifted = lla_to_ecef(latitude_deg, longitude_deg, altitude_m) + east * east_m + north * north_m + up * up_m
    return ecef_to_lla(shifted)


def _run_fraunhofer_relocation_analysis(
    input_path: Path,
    sample_rate_hz: float,
    compute_backend: str,
    max_workers: int,
    tracking_s: float = DEFAULT_RELOCATION_TRACKING_S,
    max_satellites: int = DEFAULT_RELOCATION_MAX_SATELLITES,
) -> dict[str, object]:
    project = _fraunhofer_project_path()
    if project is None:
        raise RuntimeError("Fraunhofer_FHR project not found. Set FRAUNHOFER_FHR_PATH or place it next to GPSDataAdder.")

    script = r"""
import json
import sys

from app.dsp.acquisition import acquisition_rank_key, acquisition_result_is_plausible, scan_prns_from_session
from app.dsp.io import Complex64FileSource
from app.dsp.navdecode import decode_navigation_from_tracking
from app.dsp.pvt import compute_pvt_from_navigation
from app.dsp.tracking import track_file
from app.models import SessionConfig

file_path = sys.argv[1]
sample_rate = float(sys.argv[2])
backend = sys.argv[3]
max_workers = int(sys.argv[4])
tracking_s = float(sys.argv[5])
max_satellites = int(sys.argv[6])

source = Complex64FileSource(file_path)
start_sample = int(round(60.0 * sample_rate))
sample_count = int(round(1.2 * sample_rate))
sample_count = min(sample_count, max(0, source.total_samples - start_sample))
if sample_count <= 0:
    raise RuntimeError("Relocation acquisition window is outside the file.")

samples = source.read_window(start_sample, sample_count)
session = SessionConfig(
    file_path=file_path,
    sample_rate=sample_rate,
    compute_backend=backend,
    max_workers=max_workers,
    gpu_enabled=(backend != "cpu"),
)
session.start_sample = start_sample
session.sample_count = sample_count
session.doppler_min = -12000
session.doppler_max = 12000
session.doppler_step = 500
session.integration_ms = 20
session.acquisition_segment_count = 3
session.spread_acquisition_blocks = False

results = scan_prns_from_session(samples, session, prns=list(range(1, 33)), log_callback=None)
ranked = sorted(results, key=acquisition_rank_key, reverse=True)
ranked = [result for result in ranked if acquisition_result_is_plausible(result)] or ranked
ranked = ranked[: max(1, max_satellites)]
acquisition_by_prn = {int(result.prn): result for result in ranked}

tracking_by_prn = {}
bit_by_prn = {}
nav_by_prn = {}
absolute_start_by_prn = {}
pvt_result = None
for acquisition in ranked:
    local = SessionConfig(
        file_path=file_path,
        sample_rate=sample_rate,
        compute_backend=backend,
        max_workers=max_workers,
        gpu_enabled=(backend != "cpu"),
    )
    local.prn = int(acquisition.prn)
    local.tracking_ms = int(round(max(24.0, tracking_s) * 1000.0))
    absolute_start = start_sample + int(acquisition.best_candidate.segment_start_sample)
    tracking = track_file(file_path, absolute_start, local, acquisition, log_callback=None)
    bit_result, nav_result = decode_navigation_from_tracking(tracking)
    tracking_by_prn[int(acquisition.prn)] = tracking
    bit_by_prn[int(acquisition.prn)] = bit_result
    nav_by_prn[int(acquisition.prn)] = nav_result
    absolute_start_by_prn[int(acquisition.prn)] = int(absolute_start)
    candidate = compute_pvt_from_navigation(tracking_by_prn, bit_by_prn, nav_by_prn)
    if candidate.solution is not None:
        pvt_result = candidate
        break

if pvt_result is None or pvt_result.solution is None:
    raise RuntimeError("Fraunhofer_FHR did not decode enough ephemerides for a PVT relocation plan.")

channels = []
used_prns = [int(obs.prn) for obs in pvt_result.observations]
for prn in used_prns:
    acquisition = acquisition_by_prn.get(prn)
    nav_result = nav_by_prn.get(prn)
    bit_result = bit_by_prn.get(prn)
    if acquisition is None or nav_result is None:
        continue
    templates = []
    for subframe in sorted(nav_result.subframes, key=lambda item: int(item.start_bit)):
        if not subframe.valid or subframe.subframe_id is None or subframe.tow_seconds is None:
            continue
        if any(int(item["subframe_id"]) == int(subframe.subframe_id) for item in templates):
            continue
        bit_ms = 0
        if bit_result is not None and 0 <= subframe.start_bit < bit_result.bit_start_ms.size:
            bit_ms = int(bit_result.bit_start_ms[subframe.start_bit])
        file_time_s = float(absolute_start_by_prn.get(prn, start_sample)) / float(sample_rate) + bit_ms * 1e-3
        templates.append(
            {
                "subframe_id": int(subframe.subframe_id),
                "tow_count": int(subframe.tow_seconds) // 6,
                "tow_seconds": int(subframe.tow_seconds),
                "file_time_s": float(file_time_s),
                "words": [word.bits for word in subframe.words],
            }
        )
    if not {1, 2, 3}.issubset({int(item["subframe_id"]) for item in templates}):
        continue
    obs = next((item for item in pvt_result.observations if int(item.prn) == prn), None)
    if obs is None:
        continue
    first_template = sorted(templates, key=lambda item: float(item["file_time_s"]))[0]
    subframe_offset = int(round(float(first_template["file_time_s"]) / 6.0))
    channels.append(
        {
            "prn": prn,
            "doppler_hz": float(acquisition.best_candidate.doppler_hz),
            "code_phase_samples": int(acquisition.best_candidate.code_phase_samples),
            "satellite_position_m": [float(value) for value in obs.satellite_position_m],
            "start_tow_count": int((int(first_template["tow_count"]) - subframe_offset) % (604800 // 6)),
            "start_subframe_id": int(((int(first_template["subframe_id"]) - 1 - subframe_offset) % 5) + 1),
            "nav_subframes": templates,
        }
    )

if len(channels) < 4:
    raise RuntimeError("PVT solved, but fewer than four usable overlay channels had ephemeris templates.")

solution = pvt_result.solution
print(
    json.dumps(
        {
            "baseline": {
                "latitude_deg": float(solution.latitude_deg),
                "longitude_deg": float(solution.longitude_deg),
                "altitude_m": float(solution.altitude_m),
            },
            "residual_rms_m": pvt_result.residual_rms_m,
            "channels": channels,
            "summary_lines": pvt_result.summary_lines,
        },
        sort_keys=True,
    )
)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(input_path),
            str(float(sample_rate_hz)),
            compute_backend,
            str(int(max_workers)),
            str(float(tracking_s)),
            str(int(max_satellites)),
        ],
        cwd=str(project),
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else completed.stdout.strip()
        raise RuntimeError(detail or "Fraunhofer_FHR relocation analysis failed.")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Fraunhofer_FHR relocation analysis returned no result.")
    return json.loads(lines[-1])


def plan_relocation_overlay(
    input_path: str | Path,
    sample_rate_hz: float,
    target_latitude_deg: float | None = None,
    target_longitude_deg: float | None = None,
    target_altitude_m: float | None = None,
    offset_east_m: float = 0.0,
    offset_north_m: float = 0.0,
    offset_up_m: float = 0.0,
    use_offsets: bool = False,
    target_cn0_dbhz: float = DEFAULT_RELOCATION_CN0_DBHZ,
    requested_backend: str = "auto",
    worker_count: int | None = None,
    in_flight_blocks: int | None = None,
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
) -> RelocationOverlayPlan:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if chunk_samples <= 0:
        raise ValueError("Chunk size must be positive.")

    total_samples = count_complex64_samples(source)
    backend = resolve_compute_backend(requested_backend)
    workers = _effective_worker_count(worker_count, backend)
    in_flight = _effective_in_flight_blocks(in_flight_blocks, workers)
    analysis_workers = DEFAULT_WORKER_COUNT if backend == "gpu" else workers
    analysis = _run_fraunhofer_relocation_analysis(
        source,
        sample_rate_hz=sample_rate_hz,
        compute_backend=backend,
        max_workers=analysis_workers,
    )
    baseline = dict(analysis["baseline"])
    baseline_ecef = lla_to_ecef(
        float(baseline["latitude_deg"]),
        float(baseline["longitude_deg"]),
        float(baseline["altitude_m"]),
    )
    if use_offsets:
        target_latitude_deg, target_longitude_deg, target_altitude_m = offset_lla(
            float(baseline["latitude_deg"]),
            float(baseline["longitude_deg"]),
            float(baseline["altitude_m"]),
            float(offset_east_m),
            float(offset_north_m),
            float(offset_up_m),
        )
    if target_latitude_deg is None or target_longitude_deg is None or target_altitude_m is None:
        raise ValueError("Target latitude, longitude, and altitude are required.")
    target_ecef = lla_to_ecef(target_latitude_deg, target_longitude_deg, target_altitude_m)
    amplitude_estimate = estimate_realistic_amplitude(
        source,
        sample_rate_hz=sample_rate_hz,
        target_cn0_dbhz=target_cn0_dbhz,
    )
    samples_per_ms = max(1, int(round(float(sample_rate_hz) * 1e-3)))
    channels: list[RelocationChannelPlan] = []
    for raw_channel in analysis["channels"]:
        satellite = np.asarray(raw_channel["satellite_position_m"], dtype=np.float64)
        range_delta_m = float(np.linalg.norm(satellite - target_ecef) - np.linalg.norm(satellite - baseline_ecef))
        range_delta_samples = range_delta_m / SPEED_OF_LIGHT_M_S * float(sample_rate_hz)
        original_code_phase = int(raw_channel["code_phase_samples"])
        shifted_code_phase = int(round(original_code_phase + range_delta_samples)) % samples_per_ms
        channels.append(
            RelocationChannelPlan(
                prn=int(raw_channel["prn"]),
                doppler_hz=float(raw_channel["doppler_hz"]),
                original_code_phase_samples=original_code_phase,
                code_phase_samples=shifted_code_phase,
                range_delta_m=range_delta_m,
                range_delta_samples=range_delta_samples,
                amplitude=float(amplitude_estimate.amplitude),
                start_tow_count=int(raw_channel["start_tow_count"]),
                start_subframe_id=int(raw_channel["start_subframe_id"]),
                nav_subframes=tuple(dict(item) for item in raw_channel["nav_subframes"]),
            )
        )
    shift_ecef = target_ecef - baseline_ecef
    lat = np.deg2rad(float(baseline["latitude_deg"]))
    lon = np.deg2rad(float(baseline["longitude_deg"]))
    east = np.asarray([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.asarray([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=np.float64)
    up = np.asarray([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
    summary = (
        f"Baseline PVT: {baseline['latitude_deg']:.6f}, {baseline['longitude_deg']:.6f}, {baseline['altitude_m']:.1f} m.",
        f"Target PVT: {target_latitude_deg:.6f}, {target_longitude_deg:.6f}, {target_altitude_m:.1f} m.",
        f"Offset: east {float(np.dot(shift_ecef, east)):.1f} m, north {float(np.dot(shift_ecef, north)):.1f} m, up {float(np.dot(shift_ecef, up)):.1f} m.",
        f"Overlay: {len(channels)} received PRNs, target {target_cn0_dbhz:.1f} dB-Hz, {backend} backend.",
    )
    return RelocationOverlayPlan(
        input_path=str(source),
        sample_rate_hz=float(sample_rate_hz),
        total_samples=int(total_samples),
        duration_s=float(total_samples) / float(sample_rate_hz),
        baseline_latitude_deg=float(baseline["latitude_deg"]),
        baseline_longitude_deg=float(baseline["longitude_deg"]),
        baseline_altitude_m=float(baseline["altitude_m"]),
        target_latitude_deg=float(target_latitude_deg),
        target_longitude_deg=float(target_longitude_deg),
        target_altitude_m=float(target_altitude_m),
        shift_east_m=float(np.dot(shift_ecef, east)),
        shift_north_m=float(np.dot(shift_ecef, north)),
        shift_up_m=float(np.dot(shift_ecef, up)),
        target_cn0_dbhz=float(target_cn0_dbhz),
        compute_backend=backend,
        worker_count=workers,
        in_flight_blocks=in_flight,
        chunk_samples=int(chunk_samples),
        channels=tuple(channels),
        summary_lines=summary,
    )


def _required_nav_bits(total_samples: int, sample_rate_hz: float) -> int:
    return max(1, int(np.ceil(float(total_samples) / float(sample_rate_hz) * NAV_BIT_RATE_BPS)) + 2)


def _mix_relocation_block(
    data: np.ndarray,
    plan: RelocationOverlayPlan,
    start_sample: int,
    nav_bits_by_prn: dict[int, np.ndarray],
) -> np.ndarray:
    if plan.compute_backend == "gpu":
        cp = _load_cupy()
        mixed_gpu = cp.asarray(data, dtype=cp.complex64)
        for channel in plan.channels:
            config = SyntheticSatelliteConfig(
                sample_rate_hz=plan.sample_rate_hz,
                prn=channel.prn,
                doppler_hz=channel.doppler_hz,
                code_phase_samples=channel.code_phase_samples,
                amplitude=channel.amplitude,
                start_tow_count=channel.start_tow_count,
                start_subframe_id=channel.start_subframe_id,
                nav_seed=channel.prn,
            )
            mixed_gpu += _generate_synthetic_satellite_block_gpu(
                config,
                start_sample,
                int(data.size),
                nav_bits_by_prn[channel.prn],
            )
        return cp.asnumpy(mixed_gpu).astype(np.complex64, copy=False)

    mixed = data.astype(np.complex64, copy=True)
    for channel in plan.channels:
        config = SyntheticSatelliteConfig(
            sample_rate_hz=plan.sample_rate_hz,
            prn=channel.prn,
            doppler_hz=channel.doppler_hz,
            code_phase_samples=channel.code_phase_samples,
            amplitude=channel.amplitude,
            start_tow_count=channel.start_tow_count,
            start_subframe_id=channel.start_subframe_id,
            nav_seed=channel.prn,
        )
        mixed += generate_synthetic_satellite_block(
            config,
            start_sample,
            int(data.size),
            nav_bits_by_prn[channel.prn],
        )
    return mixed.astype(np.complex64, copy=False)


def _process_relocation_block(
    input_map: np.memmap,
    output_map: np.memmap,
    plan: RelocationOverlayPlan,
    start_sample: int,
    sample_count: int,
    nav_bits_by_prn: dict[int, np.ndarray],
) -> int:
    stop_sample = start_sample + sample_count
    data = np.asarray(input_map[start_sample:stop_sample], dtype=np.complex64)
    output_map[start_sample:stop_sample] = _mix_relocation_block(data, plan, start_sample, nav_bits_by_prn)
    return int(sample_count)


def add_relocation_overlay_to_file(
    input_path: str | Path,
    output_path: str | Path,
    plan: RelocationOverlayPlan,
    metadata_path: str | Path | None = None,
    progress_callback: Callable[[float], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> RelocationAddResult:
    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must be different.")
    if not source.exists():
        raise FileNotFoundError(source)
    total_samples = count_complex64_samples(source)
    if total_samples != plan.total_samples:
        raise ValueError("Input file size no longer matches the relocation plan.")

    nav_bits = {
        channel.prn: build_lnav_bit_stream_from_templates(
            _required_nav_bits(total_samples, plan.sample_rate_hz),
            channel.nav_subframes,
            start_tow_count=channel.start_tow_count,
            start_subframe_id=channel.start_subframe_id,
        )
        for channel in plan.channels
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_name(destination.name + ".partial")
    if temporary_destination.exists():
        temporary_destination.unlink()
    _create_sized_file(temporary_destination, total_samples)

    input_map: np.memmap | None = np.memmap(source, dtype=COMPLEX64_DTYPE, mode="r", shape=(total_samples,))
    output_map: np.memmap | None = np.memmap(temporary_destination, dtype=COMPLEX64_DTYPE, mode="r+", shape=(total_samples,))
    processed = 0
    futures: set[Future[int]] = set()
    try:
        with ThreadPoolExecutor(max_workers=plan.worker_count) as executor:
            for start_sample in range(0, total_samples, plan.chunk_samples):
                if cancel_callback is not None and cancel_callback():
                    raise ProcessingCancelled("Processing was canceled.")
                while len(futures) >= plan.in_flight_blocks:
                    done, pending = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        processed += int(future.result())
                        if progress_callback is not None:
                            progress_callback(100.0 * processed / max(total_samples, 1))
                    futures = set(pending)
                sample_count = min(plan.chunk_samples, total_samples - start_sample)
                futures.add(
                    executor.submit(
                        _process_relocation_block,
                        input_map,
                        output_map,
                        plan,
                        int(start_sample),
                        int(sample_count),
                        nav_bits,
                    )
                )
            while futures:
                if cancel_callback is not None and cancel_callback():
                    raise ProcessingCancelled("Processing was canceled.")
                done, pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    processed += int(future.result())
                    if progress_callback is not None:
                        progress_callback(100.0 * processed / max(total_samples, 1))
                futures = set(pending)
        assert output_map is not None
        output_map.flush()
        del output_map
        output_map = None
        del input_map
        input_map = None
        temporary_destination.replace(destination)
    except Exception:
        if output_map is not None:
            del output_map
            output_map = None
        if input_map is not None:
            del input_map
            input_map = None
        if temporary_destination.exists():
            temporary_destination.unlink()
        raise
    finally:
        if output_map is not None:
            del output_map
        if input_map is not None:
            del input_map

    metadata_output = Path(metadata_path) if metadata_path is not None else None
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(
            json.dumps(
                {
                    "mode": "position_relocation_overlay",
                    "plan": asdict(plan),
                    "notes": [
                        "Original samples are preserved; stronger synthetic replicas are added.",
                        "LNAV payload words are replayed from decoded source satellites with regenerated TLM/HOW timing.",
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    return RelocationAddResult(
        input_path=str(source),
        output_path=str(destination),
        metadata_path=str(metadata_output) if metadata_output is not None else None,
        total_samples=int(total_samples),
        channel_count=len(plan.channels),
        compute_backend=plan.compute_backend,
        worker_count=plan.worker_count,
        in_flight_blocks=plan.in_flight_blocks,
    )
