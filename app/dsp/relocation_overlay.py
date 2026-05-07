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

from app.dsp.gps_ca import (
    CA_CODE_LENGTH,
    CA_CODE_RATE_HZ,
    code_phase_samples_to_chips,
    generate_ca_code,
)
from app.dsp.lnav import build_broadcast_ephemeris_templates, build_lnav_bit_stream_from_templates
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
    _load_cupy,
    count_complex64_samples,
    estimate_realistic_amplitude,
    resolve_compute_backend,
)
from app.dsp.tow_detect import _fraunhofer_project_path


SPEED_OF_LIGHT_M_S = 299_792_458.0
GPS_L1_FREQUENCY_HZ = 1_575_420_000.0
GPS_L1_WAVELENGTH_M = SPEED_OF_LIGHT_M_S / GPS_L1_FREQUENCY_HZ
GPS_WEEK_SECONDS = 604_800.0
GPS_MU_M3_S2 = 3.986005e14
GPS_EARTH_ROTATION_RAD_S = 7.2921151467e-5
WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
DEFAULT_RELOCATION_CN0_DBHZ = 56.0
DEFAULT_RELOCATION_TRACKING_S = 72.0
DEFAULT_RELOCATION_MAX_SATELLITES = 8
DEFAULT_TARGET_OVERLAY_SATELLITES = 8
TARGET_SYNTHETIC_AMPLITUDE_SCALE = 2.0
MIN_TARGET_ELEVATION_DEG = 5.0
NOMINAL_GPS_SIGNAL_TRANSIT_S = 0.075
NAV_BIT_GUARD_BITS = 64


@dataclass(frozen=True)
class RelocationChannelPlan:
    """One synthetic received GPS L1 C/A channel in the relocation overlay."""

    prn: int
    doppler_hz: float
    code_rate_hz: float
    original_code_phase_samples: int
    code_phase_samples: int
    range_delta_m: float
    range_delta_samples: float
    amplitude: float
    start_tow_count: int
    start_subframe_id: int
    reference_bit_index: int
    reference_sample: int
    nav_subframes: tuple[dict[str, object], ...]
    range_delta_rate_m_s: float = 0.0
    range_delta_acceleration_m_s2: float = 0.0
    nav_time_shift_samples: int = 0
    source_reference_sample: int | None = None
    source_doppler_hz: float | None = None
    doppler_rate_hz_s: float = 0.0
    code_rate_rate_hz_s: float = 0.0
    original_code_phase_chips: float | None = None
    code_phase_chips: float | None = None
    reference_code_phase_chips: float | None = None
    source_code_rate_hz: float | None = None
    synthetic_ephemeris: bool = False
    target_elevation_deg: float | None = None


@dataclass(frozen=True)
class RelocationCodePhaseShift:
    """File-start C/A phase for the source and relocated replicas."""

    original_code_phase_samples: int
    code_phase_samples: int
    original_code_phase_chips: float
    code_phase_chips: float
    reference_code_phase_chips: float


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

import numpy as np

from app.dsp.acquisition import acquisition_rank_key, acquisition_result_is_plausible, scan_prns_from_session
from app.dsp.ephemeris import (
    decode_ephemeris,
    rotate_ecef_for_transit,
    satellite_clock_correction_s,
    satellite_position_ecef_m,
)
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
CA_CODE_LENGTH = 1023.0
CA_CODE_RATE_HZ = 1_023_000.0


def unwrap_code_phase_chips(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    unwrapped = np.empty(values.size, dtype=np.float64)
    unwrapped[0] = values[0]
    previous_raw = values[0]
    for index in range(1, values.size):
        raw_delta = values[index] - previous_raw
        delta = ((raw_delta + CA_CODE_LENGTH * 0.5) % CA_CODE_LENGTH) - CA_CODE_LENGTH * 0.5
        unwrapped[index] = unwrapped[index - 1] + delta
        previous_raw = values[index]
    return unwrapped


def fit_code_phase_at_ms(tracking, reference_ms):
    phases = None if tracking is None else tracking.loop_states.get("code_phase_chips")
    if phases is None or not (0 <= int(reference_ms) < len(phases)):
        return None
    reference_ms = int(reference_ms)
    lo = max(0, reference_ms - 10000)
    hi = min(len(phases), reference_ms + 10000)
    indices = np.arange(lo, hi, dtype=np.float64)
    values = np.asarray(phases[lo:hi], dtype=np.float64)
    mask = np.isfinite(values)
    if int(np.count_nonzero(mask)) < 100:
        raw_phase = float(phases[reference_ms]) % CA_CODE_LENGTH
        return raw_phase, CA_CODE_RATE_HZ
    indices = indices[mask]
    values = values[mask]
    unwrapped = unwrap_code_phase_chips(values)
    centered_ms = indices - float(reference_ms)
    slope_chips_per_ms, phase_at_reference = np.polyfit(centered_ms, unwrapped, 1)
    code_rate_hz = CA_CODE_RATE_HZ + float(slope_chips_per_ms) * 1000.0
    if not np.isfinite(code_rate_hz) or not (1_022_500.0 <= code_rate_hz <= 1_023_500.0):
        freqs = None if tracking is None else tracking.loop_states.get("prompt_code_freq_hz")
        if freqs is not None and 0 <= reference_ms < len(freqs) and float(freqs[reference_ms]) > 0.0:
            code_rate_hz = float(freqs[reference_ms])
        else:
            code_rate_hz = CA_CODE_RATE_HZ
    return float(phase_at_reference % CA_CODE_LENGTH), float(code_rate_hz)

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

pvt_result = compute_pvt_from_navigation(tracking_by_prn, bit_by_prn, nav_by_prn)

if pvt_result is None or pvt_result.solution is None:
    raise RuntimeError("Fraunhofer_FHR did not decode enough ephemerides for a PVT relocation plan.")

channels = []
obs_by_prn = {int(obs.prn): obs for obs in pvt_result.observations}
target_transmit_time_s = float(sorted(obs.transmit_time_s for obs in pvt_result.observations)[len(pvt_result.observations) // 2])
for prn in sorted(nav_by_prn):
    acquisition = acquisition_by_prn.get(prn)
    nav_result = nav_by_prn.get(prn)
    bit_result = bit_by_prn.get(prn)
    if acquisition is None or nav_result is None:
        continue
    ephemeris = decode_ephemeris(prn, nav_result.subframes)
    if ephemeris is None:
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
                "bit_start_ms": int(bit_ms),
                "file_time_s": float(file_time_s),
                "words": [word.bits for word in subframe.words],
            }
        )
    if not {1, 2, 3}.issubset({int(item["subframe_id"]) for item in templates}):
        continue
    obs = obs_by_prn.get(prn)
    velocity_time_s = target_transmit_time_s
    if obs is not None:
        satellite_position_m = [float(value) for value in obs.satellite_position_m]
        velocity_time_s = float(obs.corrected_transmit_time_s)
    else:
        clock_s = satellite_clock_correction_s(ephemeris, target_transmit_time_s)
        corrected_transmit_s = target_transmit_time_s - clock_s
        satellite_position = satellite_position_ecef_m(ephemeris, corrected_transmit_s)
        transit_s = float(
            sum((float(a) - float(b)) ** 2 for a, b in zip(satellite_position, pvt_result.solution.ecef_m)) ** 0.5
        ) / 299792458.0
        satellite_position = rotate_ecef_for_transit(satellite_position, transit_s)
        satellite_position_m = [float(value) for value in satellite_position]
        velocity_time_s = corrected_transmit_s

    def apparent_satellite_position(transmit_time_s):
        position = satellite_position_ecef_m(ephemeris, float(transmit_time_s))
        transit_s = float(
            sum((float(a) - float(b)) ** 2 for a, b in zip(position, pvt_result.solution.ecef_m)) ** 0.5
        ) / 299792458.0
        return rotate_ecef_for_transit(position, transit_s)

    velocity_step_s = 1.0
    satellite_center = apparent_satellite_position(velocity_time_s)
    satellite_minus = apparent_satellite_position(velocity_time_s - velocity_step_s)
    satellite_plus = apparent_satellite_position(velocity_time_s + velocity_step_s)
    satellite_velocity = (
        satellite_plus - satellite_minus
    ) / (2.0 * velocity_step_s)
    satellite_acceleration = (
        satellite_plus - 2.0 * satellite_center + satellite_minus
    ) / (velocity_step_s * velocity_step_s)
    first_template = sorted(templates, key=lambda item: float(item["file_time_s"]))[0]
    reference_sample = int(round(float(first_template["file_time_s"]) * float(sample_rate)))
    tracking = tracking_by_prn.get(prn)
    reference_code_phase_chips = None
    code_rate_hz = CA_CODE_RATE_HZ
    if tracking is not None:
        reference_ms = int(first_template["bit_start_ms"])
        fit = fit_code_phase_at_ms(tracking, reference_ms)
        if fit is not None:
            reference_code_phase_chips, code_rate_hz = fit
    if reference_code_phase_chips is None:
        reference_code_phase_chips = float(acquisition.best_candidate.code_phase_samples) / float(sample_rate) * CA_CODE_RATE_HZ
    channels.append(
        {
            "prn": prn,
            "doppler_hz": float(acquisition.best_candidate.doppler_hz),
            "reference_code_phase_chips": float(reference_code_phase_chips),
            "code_rate_hz": float(code_rate_hz),
            "satellite_position_m": satellite_position_m,
            "satellite_velocity_m_s": [float(value) for value in satellite_velocity],
            "satellite_acceleration_m_s2": [float(value) for value in satellite_acceleration],
            "start_tow_count": int(first_template["tow_count"]),
            "start_subframe_id": int(first_template["subframe_id"]),
            "reference_file_time_s": float(first_template["file_time_s"]),
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
            "receiver_time_offset_s": None if pvt_result.receiver_time_offset_s is None else float(pvt_result.receiver_time_offset_s),
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
    channels: list[RelocationChannelPlan] = []
    skipped_below_horizon: list[int] = []
    used_prns: set[int] = set()
    for raw_channel in analysis["channels"]:
        used_prns.add(int(raw_channel["prn"]))
        satellite = np.asarray(raw_channel["satellite_position_m"], dtype=np.float64)
        satellite_velocity = np.asarray(raw_channel.get("satellite_velocity_m_s", (0.0, 0.0, 0.0)), dtype=np.float64)
        if satellite_velocity.shape != (3,) or not np.all(np.isfinite(satellite_velocity)):
            satellite_velocity = np.zeros(3, dtype=np.float64)
        satellite_acceleration = np.asarray(
            raw_channel.get("satellite_acceleration_m_s2", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        if satellite_acceleration.shape != (3,) or not np.all(np.isfinite(satellite_acceleration)):
            satellite_acceleration = np.zeros(3, dtype=np.float64)
        target_elevation_deg = _satellite_elevation_deg(
            satellite,
            float(target_latitude_deg),
            float(target_longitude_deg),
            target_ecef,
        )
        if target_elevation_deg < MIN_TARGET_ELEVATION_DEG:
            skipped_below_horizon.append(int(raw_channel["prn"]))
            continue
        range_delta_m = float(np.linalg.norm(satellite - target_ecef) - np.linalg.norm(satellite - baseline_ecef))
        range_delta_samples = range_delta_m / SPEED_OF_LIGHT_M_S * float(sample_rate_hz)
        range_delta_rate_m_s = _range_delta_rate_m_s(satellite, satellite_velocity, baseline_ecef, target_ecef)
        range_delta_acceleration_m_s2 = _range_delta_acceleration_m_s2(
            satellite,
            satellite_velocity,
            satellite_acceleration,
            baseline_ecef,
            target_ecef,
        )
        source_reference_sample = int(round(float(raw_channel["reference_file_time_s"]) * float(sample_rate_hz)))
        nav_time_shift_samples = _nav_time_shift_samples(range_delta_samples, sample_rate_hz)
        reference_sample = int(source_reference_sample + nav_time_shift_samples)
        reference_bit_index = int(
            np.ceil(source_reference_sample * NAV_BIT_RATE_BPS / float(sample_rate_hz))
            + NAV_BIT_GUARD_BITS
        )
        source_code_rate_hz = float(raw_channel.get("code_rate_hz", CA_CODE_RATE_HZ))
        code_rate_hz = _synthetic_code_rate_hz(source_code_rate_hz, range_delta_rate_m_s)
        code_rate_rate_hz_s = _synthetic_code_rate_rate_hz_s(
            source_code_rate_hz,
            range_delta_acceleration_m_s2,
        )
        source_doppler_hz = float(raw_channel["doppler_hz"])
        doppler_hz = _synthetic_doppler_hz(source_doppler_hz, range_delta_rate_m_s)
        doppler_rate_hz_s = _synthetic_doppler_rate_hz_s(range_delta_acceleration_m_s2)
        code_phase_shift = _shift_code_phase_from_geometry(
            float(raw_channel["reference_code_phase_chips"]),
            source_reference_sample,
            range_delta_m,
            source_code_rate_hz,
            code_rate_hz,
            sample_rate_hz,
        )
        channels.append(
            RelocationChannelPlan(
                prn=int(raw_channel["prn"]),
                doppler_hz=doppler_hz,
                code_rate_hz=code_rate_hz,
                original_code_phase_samples=code_phase_shift.original_code_phase_samples,
                code_phase_samples=code_phase_shift.code_phase_samples,
                range_delta_m=range_delta_m,
                range_delta_samples=range_delta_samples,
                amplitude=float(amplitude_estimate.amplitude),
                start_tow_count=int(raw_channel["start_tow_count"]),
                start_subframe_id=int(raw_channel["start_subframe_id"]),
                reference_bit_index=reference_bit_index,
                reference_sample=reference_sample,
                nav_subframes=tuple(dict(item) for item in raw_channel["nav_subframes"]),
                range_delta_rate_m_s=range_delta_rate_m_s,
                range_delta_acceleration_m_s2=range_delta_acceleration_m_s2,
                nav_time_shift_samples=nav_time_shift_samples,
                source_reference_sample=source_reference_sample,
                source_doppler_hz=source_doppler_hz,
                doppler_rate_hz_s=doppler_rate_hz_s,
                code_rate_rate_hz_s=code_rate_rate_hz_s,
                original_code_phase_chips=code_phase_shift.original_code_phase_chips,
                code_phase_chips=code_phase_shift.code_phase_chips,
                reference_code_phase_chips=code_phase_shift.reference_code_phase_chips,
                source_code_rate_hz=source_code_rate_hz,
                synthetic_ephemeris=False,
                target_elevation_deg=target_elevation_deg,
            )
        )
    synthetic_channels = _build_target_synthetic_channels(
        analysis_channels=list(analysis["channels"]),
        used_prns=used_prns,
        sample_rate_hz=float(sample_rate_hz),
        duration_s=float(total_samples) / float(sample_rate_hz),
        target_latitude_deg=float(target_latitude_deg),
        target_longitude_deg=float(target_longitude_deg),
        target_altitude_m=float(target_altitude_m),
        target_ecef_m=target_ecef,
        amplitude=float(amplitude_estimate.amplitude),
        existing_channels=channels,
        desired_channel_count=DEFAULT_TARGET_OVERLAY_SATELLITES,
        receiver_time_offset_s=analysis.get("receiver_time_offset_s"),
    )
    channels.extend(synthetic_channels)
    if len(channels) < 4:
        skipped = ", ".join(str(prn) for prn in skipped_below_horizon) or "none"
        raise RuntimeError(
            "Target geometry has fewer than four usable received PRNs after the elevation mask. "
            f"Skipped below {MIN_TARGET_ELEVATION_DEG:.1f} deg target elevation: {skipped}. "
            "Use a smaller offset or add target-visible synthetic ephemeris channels."
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
        f"Overlay: {len(channels)} target-visible PRNs ({len(synthetic_channels)} synthetic target ephemeris), LNAV time shifts, fractional code phase, Doppler/range-rate drift, target {target_cn0_dbhz:.1f} dB-Hz, {backend} backend.",
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
    return max(1, int(np.ceil(float(total_samples) / float(sample_rate_hz) * NAV_BIT_RATE_BPS)) + NAV_BIT_GUARD_BITS * 2)


def code_phase_for_range_delta(
    original_code_phase_samples: int,
    range_delta_samples: float,
    samples_per_ms: int,
) -> int:
    """Map a desired pseudorange delta to Fraunhofer-compatible code phase.

    Fraunhofer_FHR forms receive time as bit time minus tracked code-phase
    offset. A larger local code phase therefore reduces pseudorange, so the
    synthetic replica must move in the opposite direction of the range delta.
    """

    if samples_per_ms <= 0:
        raise ValueError("samples_per_ms must be positive.")
    return int(round(float(original_code_phase_samples) - float(range_delta_samples))) % int(samples_per_ms)


def _code_phase_chips_to_samples(code_phase_chips: float, sample_rate_hz: float) -> int:
    samples_per_ms = max(1, int(round(float(sample_rate_hz) * 1e-3)))
    wrapped_chips = float(code_phase_chips) % float(CA_CODE_LENGTH)
    return int(round(wrapped_chips * float(samples_per_ms) / float(CA_CODE_LENGTH))) % samples_per_ms


def _synthetic_code_rate_hz(source_code_rate_hz: float, range_delta_rate_m_s: float) -> float:
    if not np.isfinite(source_code_rate_hz) or source_code_rate_hz <= 0.0:
        raise ValueError("Source code rate must be positive.")
    if not np.isfinite(range_delta_rate_m_s):
        raise ValueError("Range-delta rate must be finite.")
    return float(source_code_rate_hz) * (1.0 - float(range_delta_rate_m_s) / SPEED_OF_LIGHT_M_S)


def _synthetic_code_rate_rate_hz_s(source_code_rate_hz: float, range_delta_acceleration_m_s2: float) -> float:
    if not np.isfinite(source_code_rate_hz) or source_code_rate_hz <= 0.0:
        raise ValueError("Source code rate must be positive.")
    if not np.isfinite(range_delta_acceleration_m_s2):
        raise ValueError("Range-delta acceleration must be finite.")
    return -float(source_code_rate_hz) * float(range_delta_acceleration_m_s2) / SPEED_OF_LIGHT_M_S


def _synthetic_doppler_hz(source_doppler_hz: float, range_delta_rate_m_s: float) -> float:
    if not np.isfinite(source_doppler_hz):
        raise ValueError("Source Doppler must be finite.")
    if not np.isfinite(range_delta_rate_m_s):
        raise ValueError("Range-delta rate must be finite.")
    return float(source_doppler_hz) - float(range_delta_rate_m_s) / GPS_L1_WAVELENGTH_M


def _synthetic_doppler_rate_hz_s(range_delta_acceleration_m_s2: float) -> float:
    if not np.isfinite(range_delta_acceleration_m_s2):
        raise ValueError("Range-delta acceleration must be finite.")
    return -float(range_delta_acceleration_m_s2) / GPS_L1_WAVELENGTH_M


def _satellite_elevation_deg(
    satellite_ecef_m: np.ndarray,
    receiver_latitude_deg: float,
    receiver_longitude_deg: float,
    receiver_ecef_m: np.ndarray,
) -> float:
    """Return apparent satellite elevation above the receiver local horizon."""

    lat = np.deg2rad(float(receiver_latitude_deg))
    lon = np.deg2rad(float(receiver_longitude_deg))
    up = np.asarray([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
    line = np.asarray(satellite_ecef_m, dtype=np.float64) - np.asarray(receiver_ecef_m, dtype=np.float64)
    distance = float(np.linalg.norm(line))
    if distance <= 0.0 or not np.isfinite(distance):
        return -90.0
    sin_elevation = float(np.dot(line / distance, up))
    return float(np.rad2deg(np.arcsin(np.clip(sin_elevation, -1.0, 1.0))))


def _gps_time_delta_seconds(delta_s: float) -> float:
    value = float(delta_s)
    while value > GPS_WEEK_SECONDS * 0.5:
        value -= GPS_WEEK_SECONDS
    while value < -GPS_WEEK_SECONDS * 0.5:
        value += GPS_WEEK_SECONDS
    return value


def _synthetic_broadcast_position_ecef_m(
    transmit_time_s: float,
    *,
    toe_s: float,
    sqrt_a_sqrt_m: float,
    eccentricity: float,
    i0_rad: float,
    omega0_rad: float,
    omega_rad: float,
    m0_rad: float,
    delta_n_rad_s: float,
    omega_dot_rad_s: float,
    idot_rad_s: float,
) -> np.ndarray:
    semi_major_axis_m = float(sqrt_a_sqrt_m) * float(sqrt_a_sqrt_m)
    mean_motion = np.sqrt(GPS_MU_M3_S2 / (semi_major_axis_m**3)) + float(delta_n_rad_s)
    tk = _gps_time_delta_seconds(float(transmit_time_s) - float(toe_s))
    mean_anomaly = float(m0_rad) + mean_motion * tk
    eccentric_anomaly = mean_anomaly
    for _ in range(12):
        eccentric_anomaly = mean_anomaly + float(eccentricity) * np.sin(eccentric_anomaly)

    true_anomaly = np.arctan2(
        np.sqrt(1.0 - float(eccentricity) * float(eccentricity)) * np.sin(eccentric_anomaly),
        np.cos(eccentric_anomaly) - float(eccentricity),
    )
    argument_of_latitude = true_anomaly + float(omega_rad)
    radius_m = semi_major_axis_m * (1.0 - float(eccentricity) * np.cos(eccentric_anomaly))
    inclination = float(i0_rad) + float(idot_rad_s) * tk
    x_orbital = radius_m * np.cos(argument_of_latitude)
    y_orbital = radius_m * np.sin(argument_of_latitude)
    omega = (
        float(omega0_rad)
        + (float(omega_dot_rad_s) - GPS_EARTH_ROTATION_RAD_S) * tk
        - GPS_EARTH_ROTATION_RAD_S * float(toe_s)
    )
    return np.asarray(
        [
            x_orbital * np.cos(omega) - y_orbital * np.cos(inclination) * np.sin(omega),
            x_orbital * np.sin(omega) + y_orbital * np.cos(inclination) * np.cos(omega),
            y_orbital * np.sin(inclination),
        ],
        dtype=np.float64,
    )


def _rotated_synthetic_position(
    elements: dict[str, float],
    transmit_time_s: float,
    receiver_ecef_m: np.ndarray,
) -> np.ndarray:
    position = _synthetic_broadcast_position_ecef_m(float(transmit_time_s), **elements)
    transit_s = float(np.linalg.norm(position - np.asarray(receiver_ecef_m, dtype=np.float64))) / SPEED_OF_LIGHT_M_S
    return _rotate_ecef_for_transit(position, transit_s)


def _rotate_ecef_for_transit(ecef_m: np.ndarray, travel_time_s: float) -> np.ndarray:
    angle = GPS_EARTH_ROTATION_RAD_S * float(travel_time_s)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x, y, z = np.asarray(ecef_m, dtype=np.float64)
    return np.asarray([cos_a * x + sin_a * y, -sin_a * x + cos_a * y, z], dtype=np.float64)


def _line_of_sight_unit(satellite_ecef_m: np.ndarray, receiver_ecef_m: np.ndarray) -> np.ndarray:
    line = np.asarray(satellite_ecef_m, dtype=np.float64) - np.asarray(receiver_ecef_m, dtype=np.float64)
    distance = float(np.linalg.norm(line))
    if distance <= 0.0 or not np.isfinite(distance):
        return np.zeros(3, dtype=np.float64)
    return line / distance


def _candidate_geometry_score(
    satellite_ecef_m: np.ndarray,
    receiver_ecef_m: np.ndarray,
    existing_satellites: list[np.ndarray],
    elevation_deg: float,
) -> float:
    if not existing_satellites:
        return float(elevation_deg)
    candidate_los = _line_of_sight_unit(satellite_ecef_m, receiver_ecef_m)
    separations = []
    for satellite in existing_satellites:
        other_los = _line_of_sight_unit(satellite, receiver_ecef_m)
        dot = float(np.clip(np.dot(candidate_los, other_los), -1.0, 1.0))
        separations.append(float(np.rad2deg(np.arccos(dot))))
    return float(min(separations) * 3.0 + elevation_deg)


def _synthetic_orbit_candidates_for_prn(
    prn: int,
    transmit_time_s: float,
    target_latitude_deg: float,
    target_longitude_deg: float,
    target_ecef_m: np.ndarray,
    existing_satellites: list[np.ndarray],
) -> list[tuple[float, float, dict[str, float], np.ndarray]]:
    sqrt_a = float(np.sqrt(26_560_000.0))
    toe_s = float(round(float(transmit_time_s) / 16.0) * 16.0)
    inclination = float(np.deg2rad(55.0))
    eccentricity = 0.006 + 0.0003 * float(int(prn) % 11)
    omega_dot = -8.3e-9
    arg_base = ((int(prn) * 137.507764) % 360.0) * np.pi / 180.0
    rows: list[tuple[float, float, dict[str, float], np.ndarray]] = []
    for raan_index in range(16):
        omega0 = -np.pi + 2.0 * np.pi * (raan_index / 16.0) + 0.017 * int(prn)
        for anomaly_index in range(24):
            m0 = -np.pi + 2.0 * np.pi * (anomaly_index / 24.0) + 0.031 * int(prn)
            elements = {
                "toe_s": toe_s,
                "sqrt_a_sqrt_m": sqrt_a,
                "eccentricity": eccentricity,
                "i0_rad": inclination,
                "omega0_rad": float(((omega0 + np.pi) % (2.0 * np.pi)) - np.pi),
                "omega_rad": float(((arg_base + np.pi) % (2.0 * np.pi)) - np.pi),
                "m0_rad": float(((m0 + np.pi) % (2.0 * np.pi)) - np.pi),
                "delta_n_rad_s": 0.0,
                "omega_dot_rad_s": omega_dot,
                "idot_rad_s": 0.0,
            }
            satellite = _rotated_synthetic_position(elements, transmit_time_s, target_ecef_m)
            elevation = _satellite_elevation_deg(
                satellite,
                target_latitude_deg,
                target_longitude_deg,
                target_ecef_m,
            )
            if elevation < MIN_TARGET_ELEVATION_DEG:
                continue
            score = _candidate_geometry_score(satellite, target_ecef_m, existing_satellites, elevation)
            rows.append((score, elevation, elements, satellite))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows


def _reference_context_from_analysis(analysis_channels: list[dict[str, object]]) -> tuple[int, int, float, float]:
    if not analysis_channels:
        raise RuntimeError("Cannot synthesize target satellites without source timing evidence.")
    first = min(analysis_channels, key=lambda item: float(item["reference_file_time_s"]))
    tow_count = int(first["start_tow_count"])
    subframe_id = int(first["start_subframe_id"])
    reference_file_time_s = float(first["reference_file_time_s"])
    transmit_time_s = float((tow_count * 6 - 6) % int(GPS_WEEK_SECONDS))
    return tow_count, subframe_id, reference_file_time_s, transmit_time_s


def _tow_count_for_transmit_time(transmit_time_s: float) -> int:
    max_tow_count = int(round(GPS_WEEK_SECONDS / 6.0))
    return int(round((float(transmit_time_s) + 6.0) / 6.0)) % max_tow_count


def _transmit_time_from_tow_count(tow_count: int) -> float:
    return float((int(tow_count) * 6 - 6) % int(GPS_WEEK_SECONDS))


def _tow_count_for_file_time(
    reference_file_time_s: float,
    receiver_time_offset_s: float,
    range_m: float,
) -> int:
    transmit_time_s = (
        float(reference_file_time_s)
        + float(receiver_time_offset_s)
        - float(range_m) / SPEED_OF_LIGHT_M_S
    )
    return _tow_count_for_transmit_time(transmit_time_s)


def _synthetic_channel_timing(
    satellite_ecef_m: np.ndarray,
    target_ecef_m: np.ndarray,
    sample_rate_hz: float,
    transmit_time_s: float,
    receiver_time_offset_s: float,
) -> tuple[int, int, float, float]:
    range_m = float(np.linalg.norm(np.asarray(satellite_ecef_m, dtype=np.float64) - np.asarray(target_ecef_m, dtype=np.float64)))
    desired_receive_time_s = float(transmit_time_s) - float(receiver_time_offset_s) + range_m / SPEED_OF_LIGHT_M_S
    bit_start_time_s = round(desired_receive_time_s * 1000.0) / 1000.0
    code_phase_s = bit_start_time_s - desired_receive_time_s
    reference_sample = int(round(bit_start_time_s * float(sample_rate_hz)))
    reference_bit_index = int(np.ceil(reference_sample * NAV_BIT_RATE_BPS / float(sample_rate_hz)) + NAV_BIT_GUARD_BITS)
    reference_code_phase_chips = (code_phase_s * CA_CODE_RATE_HZ) % float(CA_CODE_LENGTH)
    return reference_sample, reference_bit_index, reference_code_phase_chips, range_m


def _build_target_synthetic_channels(
    *,
    analysis_channels: list[dict[str, object]],
    used_prns: set[int],
    sample_rate_hz: float,
    target_latitude_deg: float,
    target_longitude_deg: float,
    target_altitude_m: float,
    target_ecef_m: np.ndarray,
    amplitude: float,
    existing_channels: list[RelocationChannelPlan],
    desired_channel_count: int,
    receiver_time_offset_s: object | None = None,
    duration_s: float | None = None,
) -> list[RelocationChannelPlan]:
    desired_synthetic_count = max(0, int(desired_channel_count))
    if desired_synthetic_count == 0:
        return []
    source_start_tow_count, source_start_subframe_id, source_reference_file_time_s, source_transmit_time_s = _reference_context_from_analysis(
        analysis_channels
    )
    effective_receiver_time_offset_s = float(receiver_time_offset_s) if receiver_time_offset_s is not None else np.nan
    centered_reference = (
        np.isfinite(effective_receiver_time_offset_s)
        and duration_s is not None
        and np.isfinite(float(duration_s))
        and float(duration_s) > 0.0
    )
    if centered_reference:
        reference_file_time_s = max(0.0, min(float(duration_s), float(duration_s) * 0.5))
        start_tow_count = _tow_count_for_file_time(
            reference_file_time_s,
            effective_receiver_time_offset_s,
            NOMINAL_GPS_SIGNAL_TRANSIT_S * SPEED_OF_LIGHT_M_S,
        )
        start_subframe_id = 1
        transmit_time_s = _transmit_time_from_tow_count(start_tow_count)
    else:
        start_tow_count = source_start_tow_count
        start_subframe_id = source_start_subframe_id
        reference_file_time_s = source_reference_file_time_s
        transmit_time_s = source_transmit_time_s
    existing_channel_prns = {int(channel.prn) for channel in existing_channels}
    base_existing_satellites: list[np.ndarray] = []
    for raw_channel in analysis_channels:
        prn = int(raw_channel["prn"])
        if prn in existing_channel_prns:
            base_existing_satellites.append(np.asarray(raw_channel["satellite_position_m"], dtype=np.float64))

    def select_channels(at_transmit_time_s: float) -> list[tuple[int, float, dict[str, float], np.ndarray]]:
        selected_rows: list[tuple[int, float, dict[str, float], np.ndarray]] = []
        existing_satellites = [np.asarray(satellite, dtype=np.float64) for satellite in base_existing_satellites]
        while len(selected_rows) < desired_synthetic_count:
            best: tuple[int, float, float, dict[str, float], np.ndarray] | None = None
            for prn in range(1, 33):
                if prn in used_prns or any(item[0] == prn for item in selected_rows):
                    continue
                candidates = _synthetic_orbit_candidates_for_prn(
                    prn,
                    at_transmit_time_s,
                    target_latitude_deg,
                    target_longitude_deg,
                    target_ecef_m,
                    existing_satellites,
                )
                if not candidates:
                    continue
                score, elevation, elements, satellite = candidates[0]
                if best is None or score > best[1]:
                    best = (prn, score, elevation, elements, satellite)
            if best is None:
                break
            prn, _score, elevation, elements, satellite = best
            selected_rows.append((prn, float(elevation), elements, satellite))
            existing_satellites.append(satellite)
        return selected_rows

    selected = select_channels(transmit_time_s)

    if not selected:
        return []

    def channel_ranges(rows: list[tuple[int, float, dict[str, float], np.ndarray]]) -> list[float]:
        ranges = [
            float(np.linalg.norm(np.asarray(satellite, dtype=np.float64) - np.asarray(target_ecef_m, dtype=np.float64)))
            for _prn, _elevation, _elements, satellite in rows
        ]
        ranges.extend(
            float(np.linalg.norm(np.asarray(raw["satellite_position_m"], dtype=np.float64) - np.asarray(target_ecef_m, dtype=np.float64)))
            for raw in analysis_channels
            if int(raw["prn"]) in existing_channel_prns
        )
        return ranges

    ranges = channel_ranges(selected)
    median_range_m = float(np.median(np.asarray(ranges, dtype=np.float64)))
    if centered_reference:
        refined_tow_count = _tow_count_for_file_time(
            reference_file_time_s,
            effective_receiver_time_offset_s,
            median_range_m,
        )
        if refined_tow_count != start_tow_count:
            start_tow_count = refined_tow_count
            transmit_time_s = _transmit_time_from_tow_count(start_tow_count)
            selected = select_channels(transmit_time_s)
            ranges = channel_ranges(selected)
            median_range_m = float(np.median(np.asarray(ranges, dtype=np.float64)))
    if not np.isfinite(effective_receiver_time_offset_s):
        effective_receiver_time_offset_s = float(transmit_time_s) + median_range_m / SPEED_OF_LIGHT_M_S - float(reference_file_time_s)
    channels: list[RelocationChannelPlan] = []
    for prn, elevation, elements, satellite in selected:
        satellite_minus = _rotated_synthetic_position(elements, transmit_time_s - 1.0, target_ecef_m)
        satellite_plus = _rotated_synthetic_position(elements, transmit_time_s + 1.0, target_ecef_m)
        satellite_velocity = (satellite_plus - satellite_minus) * 0.5
        satellite_acceleration = satellite_plus - 2.0 * satellite + satellite_minus
        range_rate_m_s = _range_rate_m_s(satellite, satellite_velocity, target_ecef_m)
        range_acceleration_m_s2 = _range_acceleration_m_s2(
            satellite,
            satellite_velocity,
            satellite_acceleration,
            target_ecef_m,
        )
        reference_sample, reference_bit_index, reference_code_phase_chips, range_m = _synthetic_channel_timing(
            satellite,
            target_ecef_m,
            sample_rate_hz,
            transmit_time_s,
            effective_receiver_time_offset_s,
        )
        code_rate_hz = _synthetic_code_rate_hz(CA_CODE_RATE_HZ, range_rate_m_s)
        code_rate_rate_hz_s = _synthetic_code_rate_rate_hz_s(CA_CODE_RATE_HZ, range_acceleration_m_s2)
        doppler_hz = _synthetic_doppler_hz(0.0, range_rate_m_s)
        doppler_rate_hz_s = _synthetic_doppler_rate_hz_s(range_acceleration_m_s2)
        phase_shift = _shift_code_phase_from_geometry(
            reference_code_phase_chips,
            reference_sample,
            0.0,
            CA_CODE_RATE_HZ,
            code_rate_hz,
            sample_rate_hz,
        )
        templates = build_broadcast_ephemeris_templates(
            week_number_mod1024=2400 % 1024,
            tow_count=start_tow_count,
            toe_s=float(elements["toe_s"]),
            toc_s=float(elements["toe_s"]),
            sqrt_a_sqrt_m=float(elements["sqrt_a_sqrt_m"]),
            eccentricity=float(elements["eccentricity"]),
            i0_rad=float(elements["i0_rad"]),
            omega0_rad=float(elements["omega0_rad"]),
            omega_rad=float(elements["omega_rad"]),
            m0_rad=float(elements["m0_rad"]),
            delta_n_rad_s=float(elements["delta_n_rad_s"]),
            omega_dot_rad_s=float(elements["omega_dot_rad_s"]),
            idot_rad_s=float(elements["idot_rad_s"]),
            iode=(int(prn) * 7) & 0xFF,
        )
        channels.append(
            RelocationChannelPlan(
                prn=int(prn),
                doppler_hz=doppler_hz,
                code_rate_hz=code_rate_hz,
                original_code_phase_samples=phase_shift.original_code_phase_samples,
                code_phase_samples=phase_shift.code_phase_samples,
                range_delta_m=range_m,
                range_delta_samples=range_m / SPEED_OF_LIGHT_M_S * float(sample_rate_hz),
                amplitude=float(amplitude) * TARGET_SYNTHETIC_AMPLITUDE_SCALE,
                start_tow_count=int(start_tow_count),
                start_subframe_id=int(start_subframe_id),
                reference_bit_index=reference_bit_index,
                reference_sample=reference_sample,
                nav_subframes=templates,
                range_delta_rate_m_s=range_rate_m_s,
                range_delta_acceleration_m_s2=range_acceleration_m_s2,
                nav_time_shift_samples=0,
                source_reference_sample=reference_sample,
                source_doppler_hz=None,
                doppler_rate_hz_s=doppler_rate_hz_s,
                code_rate_rate_hz_s=code_rate_rate_hz_s,
                original_code_phase_chips=phase_shift.original_code_phase_chips,
                code_phase_chips=phase_shift.code_phase_chips,
                reference_code_phase_chips=reference_code_phase_chips,
                source_code_rate_hz=CA_CODE_RATE_HZ,
                synthetic_ephemeris=True,
                target_elevation_deg=elevation,
            )
        )
    return channels


def _range_rate_m_s(satellite_ecef_m: np.ndarray, satellite_velocity_m_s: np.ndarray, receiver_ecef_m: np.ndarray) -> float:
    line = np.asarray(satellite_ecef_m, dtype=np.float64) - np.asarray(receiver_ecef_m, dtype=np.float64)
    distance = float(np.linalg.norm(line))
    if distance <= 0.0 or not np.isfinite(distance):
        return 0.0
    unit = line / distance
    return float(np.dot(unit, np.asarray(satellite_velocity_m_s, dtype=np.float64)))


def _range_acceleration_m_s2(
    satellite_ecef_m: np.ndarray,
    satellite_velocity_m_s: np.ndarray,
    satellite_acceleration_m_s2: np.ndarray,
    receiver_ecef_m: np.ndarray,
) -> float:
    line = np.asarray(satellite_ecef_m, dtype=np.float64) - np.asarray(receiver_ecef_m, dtype=np.float64)
    distance = float(np.linalg.norm(line))
    if distance <= 0.0 or not np.isfinite(distance):
        return 0.0
    unit = line / distance
    velocity = np.asarray(satellite_velocity_m_s, dtype=np.float64)
    acceleration = np.asarray(satellite_acceleration_m_s2, dtype=np.float64)
    radial_velocity = float(np.dot(unit, velocity))
    transverse_speed_sq = max(0.0, float(np.dot(velocity, velocity)) - radial_velocity * radial_velocity)
    return float(np.dot(unit, acceleration) + transverse_speed_sq / distance)


def _range_delta_rate_m_s(
    satellite_ecef_m: np.ndarray,
    satellite_velocity_m_s: np.ndarray,
    baseline_ecef_m: np.ndarray,
    target_ecef_m: np.ndarray,
) -> float:
    """Return d(target range - baseline range) / dt for a static relocation."""

    target_rate = _range_rate_m_s(satellite_ecef_m, satellite_velocity_m_s, target_ecef_m)
    baseline_rate = _range_rate_m_s(satellite_ecef_m, satellite_velocity_m_s, baseline_ecef_m)
    return float(target_rate - baseline_rate)


def _range_delta_acceleration_m_s2(
    satellite_ecef_m: np.ndarray,
    satellite_velocity_m_s: np.ndarray,
    satellite_acceleration_m_s2: np.ndarray,
    baseline_ecef_m: np.ndarray,
    target_ecef_m: np.ndarray,
) -> float:
    """Return d2(target range - baseline range) / dt2 for a static relocation."""

    target_acceleration = _range_acceleration_m_s2(
        satellite_ecef_m,
        satellite_velocity_m_s,
        satellite_acceleration_m_s2,
        target_ecef_m,
    )
    baseline_acceleration = _range_acceleration_m_s2(
        satellite_ecef_m,
        satellite_velocity_m_s,
        satellite_acceleration_m_s2,
        baseline_ecef_m,
    )
    return float(target_acceleration - baseline_acceleration)


def _nav_time_shift_samples(range_delta_samples: float, sample_rate_hz: float) -> int:
    samples_per_ms = max(1, int(round(float(sample_rate_hz) * 1e-3)))
    code_periods = int(round(float(range_delta_samples) / float(samples_per_ms)))
    return int(code_periods * samples_per_ms)


def _file_start_code_phase_chips(
    reference_code_phase_chips: float,
    reference_sample: int,
    code_rate_hz: float,
    sample_rate_hz: float,
) -> float:
    if not np.isfinite(code_rate_hz) or code_rate_hz <= 0.0:
        raise ValueError("Code rate must be positive.")
    return (
        float(reference_code_phase_chips) - float(reference_sample) * float(code_rate_hz) / float(sample_rate_hz)
    ) % float(CA_CODE_LENGTH)


def _shift_code_phase_from_geometry(
    reference_code_phase_chips: float,
    reference_sample: int,
    range_delta_m: float,
    source_code_rate_hz: float,
    synthetic_code_rate_hz: float,
    sample_rate_hz: float,
) -> RelocationCodePhaseShift:
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("Sample rate must be positive.")
    if not np.isfinite(source_code_rate_hz) or source_code_rate_hz <= 0.0:
        raise ValueError("Source code rate must be positive.")
    if not np.isfinite(synthetic_code_rate_hz) or synthetic_code_rate_hz <= 0.0:
        raise ValueError("Synthetic code rate must be positive.")

    baseline_chips = _file_start_code_phase_chips(
        reference_code_phase_chips,
        reference_sample,
        source_code_rate_hz,
        sample_rate_hz,
    )
    range_delta_chips = float(range_delta_m) / SPEED_OF_LIGHT_M_S * float(source_code_rate_hz)
    target_reference_chips = (float(reference_code_phase_chips) - range_delta_chips) % float(CA_CODE_LENGTH)
    shifted_chips = (
        target_reference_chips
        - float(reference_sample) * float(synthetic_code_rate_hz) / float(sample_rate_hz)
    ) % float(CA_CODE_LENGTH)
    return RelocationCodePhaseShift(
        original_code_phase_samples=_code_phase_chips_to_samples(baseline_chips, sample_rate_hz),
        code_phase_samples=_code_phase_chips_to_samples(shifted_chips, sample_rate_hz),
        original_code_phase_chips=float(baseline_chips),
        code_phase_chips=float(shifted_chips),
        reference_code_phase_chips=float(target_reference_chips),
    )


def _shift_code_phase_samples_from_geometry(
    reference_code_phase_chips: float,
    reference_sample: int,
    range_delta_m: float,
    code_rate_hz: float,
    sample_rate_hz: float,
) -> tuple[int, int]:
    shift = _shift_code_phase_from_geometry(
        reference_code_phase_chips,
        reference_sample,
        range_delta_m,
        code_rate_hz,
        code_rate_hz,
        sample_rate_hz,
    )
    return shift.original_code_phase_samples, shift.code_phase_samples


def _nav_bit_indices(
    start_sample: int,
    sample_count: int,
    sample_rate_hz: float,
    reference_sample: int,
    reference_bit_index: int,
) -> np.ndarray:
    absolute_samples = int(start_sample) + np.arange(int(sample_count), dtype=np.float64)
    shifted_samples = absolute_samples - float(reference_sample)
    return (
        np.floor(shifted_samples * NAV_BIT_RATE_BPS / float(sample_rate_hz)).astype(np.int64)
        + int(reference_bit_index)
    )


def _initial_code_phase_chips(channel: RelocationChannelPlan, sample_rate_hz: float) -> float:
    if channel.code_phase_chips is not None:
        value = float(channel.code_phase_chips)
        if np.isfinite(value):
            return value % float(CA_CODE_LENGTH)
    return code_phase_samples_to_chips(channel.code_phase_samples, sample_rate_hz)


def _reference_code_phase_chips(channel: RelocationChannelPlan) -> float | None:
    if channel.reference_code_phase_chips is None:
        return None
    value = float(channel.reference_code_phase_chips)
    return value % float(CA_CODE_LENGTH) if np.isfinite(value) else None


def _source_reference_sample(channel: RelocationChannelPlan) -> int:
    if channel.source_reference_sample is None:
        return 0
    return int(channel.source_reference_sample)


def _code_chip_positions(
    absolute_samples: np.ndarray,
    sample_rate_hz: float,
    code_rate_hz: float,
    code_rate_rate_hz_s: float,
    initial_code_phase_chips: float,
    reference_sample: int,
    reference_code_phase_chips: float | None,
) -> np.ndarray:
    if reference_code_phase_chips is None:
        time_s = absolute_samples / float(sample_rate_hz)
        return (
            float(initial_code_phase_chips)
            + float(code_rate_hz) * time_s
            + 0.5 * float(code_rate_rate_hz_s) * time_s * time_s
        )

    relative_time_s = (absolute_samples - float(reference_sample)) / float(sample_rate_hz)
    return (
        float(reference_code_phase_chips)
        + float(code_rate_hz) * relative_time_s
        + 0.5 * float(code_rate_rate_hz_s) * relative_time_s * relative_time_s
    )


def _generate_relocation_satellite_block(
    config: SyntheticSatelliteConfig,
    start_sample: int,
    sample_count: int,
    nav_bits: np.ndarray,
    reference_sample: int,
    reference_bit_index: int,
    code_rate_hz: float,
    initial_code_phase_chips: float | None = None,
    reference_code_phase_chips: float | None = None,
    code_phase_reference_sample: int = 0,
    code_rate_rate_hz_s: float = 0.0,
    doppler_rate_hz_s: float = 0.0,
    doppler_reference_sample: int = 0,
) -> np.ndarray:
    if sample_count == 0:
        return np.empty(0, dtype=np.complex64)

    absolute_samples = start_sample + np.arange(sample_count, dtype=np.float64)
    code_phase_chips = (
        code_phase_samples_to_chips(config.code_phase_samples, config.sample_rate_hz)
        if initial_code_phase_chips is None
        else float(initial_code_phase_chips)
    )
    chip_positions = _code_chip_positions(
        absolute_samples,
        config.sample_rate_hz,
        code_rate_hz,
        code_rate_rate_hz_s,
        code_phase_chips,
        code_phase_reference_sample,
        reference_code_phase_chips,
    )
    base_code = generate_ca_code(config.prn)
    chip_indices = np.floor(chip_positions).astype(np.int64) % CA_CODE_LENGTH
    code = base_code[chip_indices]
    bit_indices = _nav_bit_indices(
        start_sample,
        sample_count,
        config.sample_rate_hz,
        reference_sample,
        reference_bit_index,
    )
    if bit_indices.size and (int(bit_indices.min()) < 0 or int(bit_indices.max()) >= nav_bits.size):
        raise ValueError("Navigation bit stream is too short for requested relocation block.")
    nav_symbols = (1 - 2 * nav_bits[bit_indices]).astype(np.float32)

    phase_rad = np.deg2rad(config.carrier_phase_deg)
    carrier_time_s = (absolute_samples - float(doppler_reference_sample)) / config.sample_rate_hz
    phase = (
        2.0
        * np.pi
        * (config.doppler_hz * carrier_time_s + 0.5 * float(doppler_rate_hz_s) * carrier_time_s * carrier_time_s)
        + phase_rad
    )
    carrier = np.empty(sample_count, dtype=np.complex64)
    carrier.real = np.cos(phase).astype(np.float32)
    carrier.imag = np.sin(phase).astype(np.float32)
    carrier *= (float(config.amplitude) * code * nav_symbols).astype(np.float32)
    return carrier


def _generate_relocation_satellite_block_gpu(
    config: SyntheticSatelliteConfig,
    start_sample: int,
    sample_count: int,
    nav_bits: np.ndarray,
    reference_sample: int,
    reference_bit_index: int,
    code_rate_hz: float,
    initial_code_phase_chips: float | None = None,
    reference_code_phase_chips: float | None = None,
    code_phase_reference_sample: int = 0,
    code_rate_rate_hz_s: float = 0.0,
    doppler_rate_hz_s: float = 0.0,
    doppler_reference_sample: int = 0,
):
    cp = _load_cupy()
    if sample_count == 0:
        return cp.empty(0, dtype=cp.complex64)

    absolute_samples = start_sample + cp.arange(sample_count, dtype=cp.float64)
    code_phase_chips = (
        code_phase_samples_to_chips(config.code_phase_samples, config.sample_rate_hz)
        if initial_code_phase_chips is None
        else float(initial_code_phase_chips)
    )
    base_code = cp.asarray(generate_ca_code(config.prn), dtype=cp.float32)
    if reference_code_phase_chips is None:
        code_time_s = absolute_samples / config.sample_rate_hz
        chip_positions = (
            float(code_phase_chips)
            + float(code_rate_hz) * code_time_s
            + 0.5 * float(code_rate_rate_hz_s) * code_time_s * code_time_s
        )
    else:
        code_time_s = (absolute_samples - float(code_phase_reference_sample)) / config.sample_rate_hz
        chip_positions = (
            float(reference_code_phase_chips)
            + float(code_rate_hz) * code_time_s
            + 0.5 * float(code_rate_rate_hz_s) * code_time_s * code_time_s
        )
    chip_indices = cp.floor(chip_positions).astype(cp.int64) % 1023
    code = base_code[chip_indices]

    nav_gpu = cp.asarray(nav_bits, dtype=cp.int8)
    shifted_samples = absolute_samples - float(reference_sample)
    bit_indices = cp.floor(shifted_samples * NAV_BIT_RATE_BPS / config.sample_rate_hz).astype(cp.int64) + int(reference_bit_index)
    nav_symbols = (1 - 2 * nav_gpu[bit_indices]).astype(cp.float32)

    phase_rad = np.deg2rad(config.carrier_phase_deg)
    carrier_time_s = (absolute_samples - float(doppler_reference_sample)) / config.sample_rate_hz
    phase = (
        2.0
        * cp.pi
        * (config.doppler_hz * carrier_time_s + 0.5 * float(doppler_rate_hz_s) * carrier_time_s * carrier_time_s)
        + phase_rad
    )
    carrier = cp.empty(sample_count, dtype=cp.complex64)
    carrier.real = cp.cos(phase).astype(cp.float32)
    carrier.imag = cp.sin(phase).astype(cp.float32)
    carrier *= (float(config.amplitude) * code * nav_symbols).astype(cp.float32)
    return carrier


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
            initial_phase = _initial_code_phase_chips(channel, plan.sample_rate_hz)
            reference_phase = _reference_code_phase_chips(channel)
            source_reference_sample = _source_reference_sample(channel)
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
            mixed_gpu += _generate_relocation_satellite_block_gpu(
                config,
                start_sample,
                int(data.size),
                nav_bits_by_prn[channel.prn],
                channel.reference_sample,
                channel.reference_bit_index,
                channel.code_rate_hz,
                initial_phase,
                reference_phase,
                source_reference_sample,
                channel.code_rate_rate_hz_s,
                channel.doppler_rate_hz_s,
                source_reference_sample,
            )
        return cp.asnumpy(mixed_gpu).astype(np.complex64, copy=False)

    mixed = data.astype(np.complex64, copy=True)
    for channel in plan.channels:
        initial_phase = _initial_code_phase_chips(channel, plan.sample_rate_hz)
        reference_phase = _reference_code_phase_chips(channel)
        source_reference_sample = _source_reference_sample(channel)
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
        mixed += _generate_relocation_satellite_block(
            config,
            start_sample,
            int(data.size),
            nav_bits_by_prn[channel.prn],
            channel.reference_sample,
            channel.reference_bit_index,
            channel.code_rate_hz,
            initial_phase,
            reference_phase,
            source_reference_sample,
            channel.code_rate_rate_hz_s,
            channel.doppler_rate_hz_s,
            source_reference_sample,
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
            reference_bit_index=channel.reference_bit_index,
            subframe_cycle_ids=(1, 2, 3) if channel.synthetic_ephemeris else None,
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
                        "Per-PRN LNAV arrival time, fractional C/A phase, carrier Doppler, Doppler drift, and code-rate drift are fitted from target geometry.",
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
