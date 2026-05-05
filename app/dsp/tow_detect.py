"""Lightweight TOW detection from an offline GPS L1 C/A IQ recording."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import numpy as np

from app.dsp.gps_ca import CA_CODE_RATE_HZ, code_phase_samples_to_chips, sample_ca_code
from app.dsp.lnav import LnavTowEstimate, find_lnav_tow
from app.dsp.synthetic_satellite import COMPLEX64_DTYPE, count_complex64_samples


DEFAULT_TOW_PRNS = tuple(range(1, 33))
DEFAULT_TOW_DOPPLER_MIN_HZ = -5_000
DEFAULT_TOW_DOPPLER_MAX_HZ = 5_000
DEFAULT_TOW_DOPPLER_STEP_HZ = 500
DEFAULT_TOW_ACQUISITION_MS = 8
DEFAULT_TOW_TRACK_SECONDS = 12.5
FRAUNHOFER_FAST_START_TIME_S = 60.0
FRAUNHOFER_FAST_ACQUISITION_WINDOW_S = 1.2
FRAUNHOFER_FAST_TRACKING_S = 18.0
FRAUNHOFER_FAST_MAX_SATELLITES = 3
FRAUNHOFER_PVT_START_TIME_S = 60.0
FRAUNHOFER_PVT_ACQUISITION_WINDOW_S = 3.0
FRAUNHOFER_PVT_TRACKING_S = 60.0
FRAUNHOFER_PVT_MIN_DURATION_S = (
    FRAUNHOFER_PVT_START_TIME_S
    + FRAUNHOFER_PVT_ACQUISITION_WINDOW_S
    + FRAUNHOFER_PVT_TRACKING_S
)


@dataclass(frozen=True)
class TowDetectionResult:
    """One decoded measurement TOW estimate."""

    tow_count: int
    tow_seconds: int
    subframe_id: int
    prn: int
    doppler_hz: float
    code_phase_samples: int
    acquisition_metric: float
    bit_index: int
    polarity: str
    tracked_ms: int
    source: str
    synthetic_start_tow_count: int | None = None
    synthetic_start_subframe_id: int | None = None
    file_time_s: float | None = None


@dataclass(frozen=True)
class _AcquisitionCandidate:
    prn: int
    doppler_hz: float
    code_phase_samples: int
    metric: float


def _load_window(path: Path, start_sample: int, sample_count: int) -> np.ndarray:
    total_samples = count_complex64_samples(path)
    start = min(max(0, int(start_sample)), total_samples)
    count = min(max(0, int(sample_count)), total_samples - start)
    if count <= 0:
        return np.empty(0, dtype=np.complex64)
    with path.open("rb") as handle:
        handle.seek(start * COMPLEX64_DTYPE.itemsize)
        return np.fromfile(handle, dtype=COMPLEX64_DTYPE, count=count).astype(np.complex64, copy=False)


def _select_ms_blocks(samples: np.ndarray, sample_rate_hz: float, block_count: int) -> np.ndarray:
    samples_per_ms = int(round(sample_rate_hz * 1e-3))
    usable_blocks = min(int(block_count), samples.size // max(samples_per_ms, 1))
    if usable_blocks <= 0:
        return np.empty((0, samples_per_ms), dtype=np.complex64)
    blocks = samples[: usable_blocks * samples_per_ms].reshape(usable_blocks, samples_per_ms)
    blocks = blocks.astype(np.complex64, copy=True)
    blocks -= np.mean(blocks, axis=1, keepdims=True).astype(np.complex64)
    return blocks


def _acquire_one_prn(
    samples: np.ndarray,
    sample_rate_hz: float,
    prn: int,
    doppler_bins: np.ndarray,
    integration_ms: int,
) -> _AcquisitionCandidate | None:
    blocks = _select_ms_blocks(samples, sample_rate_hz, integration_ms)
    if blocks.size == 0:
        return None
    samples_per_ms = blocks.shape[1]
    local_code = sample_ca_code(prn, sample_rate_hz, samples_per_ms)
    code_fft = np.conj(np.fft.fft(local_code))
    time_vector = np.arange(samples_per_ms, dtype=np.float64) / sample_rate_hz
    best: _AcquisitionCandidate | None = None

    for doppler_hz in doppler_bins:
        carrier = np.exp(-1j * 2.0 * np.pi * float(doppler_hz) * time_vector).astype(np.complex64)
        wiped = blocks * carrier[np.newaxis, :]
        correlation = np.fft.ifft(np.fft.fft(wiped, axis=1) * code_fft[np.newaxis, :], axis=1)
        metrics = np.sum(np.abs(correlation) ** 2, axis=0)
        peak_col = int(np.argmax(metrics))
        metric = float(metrics[peak_col] / (float(np.mean(metrics)) + 1e-12))
        code_phase = int((samples_per_ms - peak_col) % samples_per_ms)
        if best is None or metric > best.metric:
            best = _AcquisitionCandidate(
                prn=int(prn),
                doppler_hz=float(doppler_hz),
                code_phase_samples=code_phase,
                metric=metric,
            )
    return best


def _acquire_tow_candidate(
    path: Path,
    sample_rate_hz: float,
    prns: Sequence[int],
    acquisition_ms: int,
    doppler_min_hz: int,
    doppler_max_hz: int,
    doppler_step_hz: int,
) -> _AcquisitionCandidate | None:
    samples_per_ms = int(round(sample_rate_hz * 1e-3))
    probe = _load_window(path, 0, max(samples_per_ms * int(acquisition_ms), samples_per_ms))
    if probe.size < samples_per_ms:
        return None
    doppler_bins = np.arange(doppler_min_hz, doppler_max_hz + doppler_step_hz, doppler_step_hz)
    best: _AcquisitionCandidate | None = None
    for prn in prns:
        candidate = _acquire_one_prn(
            probe,
            sample_rate_hz=sample_rate_hz,
            prn=int(prn),
            doppler_bins=doppler_bins,
            integration_ms=acquisition_ms,
        )
        if candidate is not None and (best is None or candidate.metric > best.metric):
            best = candidate
    return best


def _carrier_aligned_prompt_ms(prompt: np.ndarray) -> np.ndarray:
    if prompt.size < 5:
        return prompt.real.astype(np.float64)
    complex_prompt = prompt.astype(np.complex128, copy=False)
    magnitude = np.abs(complex_prompt)
    valid = magnitude > 1e-12
    if np.count_nonzero(valid) < 5:
        return prompt.real.astype(np.float64)
    unit_squared = np.zeros_like(complex_prompt)
    unit_squared[valid] = (complex_prompt[valid] / magnitude[valid]) ** 2
    window = min(41, max(5, (unit_squared.size // 100) | 1))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smooth_real = np.convolve(unit_squared.real, kernel, mode="same")
    smooth_imag = np.convolve(unit_squared.imag, kernel, mode="same")
    phase = 0.5 * np.unwrap(np.angle(smooth_real + 1j * smooth_imag))
    aligned = complex_prompt * np.exp(-1j * phase)
    return aligned.real.astype(np.float64)


def _form_navigation_bits(prompt: np.ndarray) -> np.ndarray:
    prompt_ms = _carrier_aligned_prompt_ms(prompt)
    if prompt_ms.size < 20:
        return np.empty(0, dtype=np.int8)
    best_score = -np.inf
    best_sums = np.empty(0, dtype=np.float64)
    for offset in range(20):
        usable = prompt_ms[offset:]
        usable = usable[: (usable.size // 20) * 20]
        if usable.size == 0:
            continue
        sums = usable.reshape(-1, 20).sum(axis=1)
        score = float(np.sum(np.abs(sums)))
        if score > best_score:
            best_score = score
            best_sums = sums
    return (best_sums >= 0.0).astype(np.int8)


def _prompt_integrations(
    path: Path,
    sample_rate_hz: float,
    candidate: _AcquisitionCandidate,
    max_seconds: float,
) -> np.ndarray:
    total_samples = count_complex64_samples(path)
    samples_per_ms = int(round(sample_rate_hz * 1e-3))
    max_ms = min(int(max_seconds * 1000.0), total_samples // max(samples_per_ms, 1))
    if max_ms <= 0:
        return np.empty(0, dtype=np.complex64)

    source = np.memmap(path, dtype=COMPLEX64_DTYPE, mode="r", shape=(total_samples,))
    prompt = np.empty(max_ms, dtype=np.complex64)
    initial_code_phase = code_phase_samples_to_chips(candidate.code_phase_samples, sample_rate_hz)
    sample_indices = np.arange(samples_per_ms, dtype=np.float64)
    try:
        for ms_index in range(max_ms):
            start = int(round(ms_index * sample_rate_hz * 1e-3))
            stop = start + samples_per_ms
            if stop > total_samples:
                return prompt[:ms_index]
            block = np.asarray(source[start:stop], dtype=np.complex64)
            absolute_samples = start + sample_indices
            carrier = np.exp(-1j * 2.0 * np.pi * candidate.doppler_hz * absolute_samples / sample_rate_hz).astype(np.complex64)
            code_phase = initial_code_phase + (float(start) * CA_CODE_RATE_HZ / sample_rate_hz)
            code = sample_ca_code(candidate.prn, sample_rate_hz, samples_per_ms, code_phase_chips=code_phase)
            prompt[ms_index] = np.vdot(code, block * carrier) / samples_per_ms
    finally:
        del source
    return prompt


def _load_sidecar_tow(path: Path) -> TowDetectionResult | None:
    sidecar = path.with_suffix(path.suffix + ".synthetic.json")
    if not sidecar.exists():
        return None
    try:
        import json

        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        tow_count = int(config["start_tow_count"])
        prn = int(config.get("prn", 0))
        subframe_id = int(config.get("start_subframe_id", 0))
    except Exception:
        return None
    return TowDetectionResult(
        tow_count=tow_count,
        tow_seconds=tow_count * 6,
        subframe_id=subframe_id,
        prn=prn,
        doppler_hz=float(config.get("doppler_hz", 0.0)),
        code_phase_samples=int(config.get("code_phase_samples", 0)),
        acquisition_metric=0.0,
        bit_index=0,
        polarity="metadata",
        tracked_ms=0,
        source="metadata sidecar",
        synthetic_start_tow_count=tow_count,
        synthetic_start_subframe_id=subframe_id if 1 <= subframe_id <= 5 else 1,
        file_time_s=0.0,
    )


def _fraunhofer_project_path() -> Path | None:
    configured = os.environ.get("FRAUNHOFER_FHR_PATH", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path(__file__).resolve().parents[3] / "Fraunhofer_FHR")
    for candidate in candidates:
        if (candidate / "app" / "dsp" / "pvt_pipeline.py").exists():
            return candidate
    return None


def _detect_with_fraunhofer_pvt_pipeline(
    path: Path,
    sample_rate_hz: float,
    compute_backend: str = "auto",
) -> TowDetectionResult | None:
    project = _fraunhofer_project_path()
    if project is None:
        return None

    script = r"""
import json
import sys

from app.dsp.pvt_pipeline import run_pvt_pipeline
from app.models import SessionConfig

file_path = sys.argv[1]
sample_rate = float(sys.argv[2])
backend = sys.argv[3]
start_time_s = float(sys.argv[4])
acquisition_window_s = float(sys.argv[5])
tracking_s = float(sys.argv[6])
session = SessionConfig(
    file_path=file_path,
    sample_rate=sample_rate,
    compute_backend=backend,
    max_workers=0,
    gpu_enabled=(backend != "cpu"),
)
result = run_pvt_pipeline(
    file_path,
    session,
    start_time_s=start_time_s,
    acquisition_window_s=acquisition_window_s,
    tracking_s=tracking_s,
    max_satellites=8,
    log_callback=lambda message: print(message, file=sys.stderr),
)
rows = []
for prn, nav in result.nav_results_by_prn.items():
    tracking = result.tracking_results_by_prn.get(prn)
    bits = result.bit_results_by_prn.get(prn)
    for subframe in nav.subframes:
        if not subframe.valid or subframe.tow_seconds is None or subframe.subframe_id is None:
            continue
        bit_ms = 0
        if bits is not None and 0 <= subframe.start_bit < bits.bit_start_ms.size:
            bit_ms = int(bits.bit_start_ms[subframe.start_bit])
        file_time_s = 0.0
        if tracking is not None and tracking.sample_rate_hz > 0.0:
            file_time_s = float(tracking.source_start_sample) / float(tracking.sample_rate_hz) + bit_ms * 1e-3
        tow_count = int(subframe.tow_seconds) // 6
        subframe_offset = int(round(file_time_s / 6.0))
        rows.append(
            {
                "tow_count": tow_count,
                "tow_seconds": int(subframe.tow_seconds),
                "subframe_id": int(subframe.subframe_id),
                "prn": int(prn),
                "file_time_s": file_time_s,
                "bit_index": int(subframe.start_bit),
                "tracked_ms": int(tracking.times_s.size) if tracking is not None else 0,
                "synthetic_start_tow_count": int((tow_count - subframe_offset) % (604800 // 6)),
                "synthetic_start_subframe_id": int(((int(subframe.subframe_id) - 1 - subframe_offset) % 5) + 1),
                "source": "Fraunhofer_FHR PVT pipeline",
            }
        )
if rows:
    rows.sort(key=lambda item: item["file_time_s"])
    print(json.dumps(rows[0], sort_keys=True))
elif result.pvt_result.gps_time_of_week_s is not None:
    gps_tow = float(result.pvt_result.gps_time_of_week_s)
    tow_count = int(round(gps_tow / 6.0)) % (604800 // 6)
    print(
        json.dumps(
            {
                "tow_count": tow_count,
                "tow_seconds": tow_count * 6,
                "subframe_id": 1,
                "prn": 0,
                "file_time_s": 0.0,
                "bit_index": 0,
                "tracked_ms": 0,
                "synthetic_start_tow_count": tow_count,
                "synthetic_start_subframe_id": 1,
                "source": "Fraunhofer_FHR PVT solution time",
            },
            sort_keys=True,
        )
    )
else:
    print("{}")
"""
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(path),
                str(float(sample_rate_hz)),
                compute_backend,
                str(FRAUNHOFER_PVT_START_TIME_S),
                str(FRAUNHOFER_PVT_ACQUISITION_WINDOW_S),
                str(FRAUNHOFER_PVT_TRACKING_S),
            ],
            cwd=str(project),
            text=True,
            capture_output=True,
            timeout=1800,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not payload:
        return None
    return TowDetectionResult(
        tow_count=int(payload["tow_count"]),
        tow_seconds=int(payload["tow_seconds"]),
        subframe_id=int(payload["subframe_id"]),
        prn=int(payload.get("prn", 0)),
        doppler_hz=0.0,
        code_phase_samples=0,
        acquisition_metric=0.0,
        bit_index=int(payload.get("bit_index", 0)),
        polarity="pvt",
        tracked_ms=int(payload.get("tracked_ms", 0)),
        source=str(payload.get("source", "Fraunhofer_FHR PVT pipeline")),
        synthetic_start_tow_count=int(payload.get("synthetic_start_tow_count", payload["tow_count"])),
        synthetic_start_subframe_id=int(payload.get("synthetic_start_subframe_id", payload["subframe_id"])),
        file_time_s=float(payload.get("file_time_s", 0.0)),
    )


def _detect_with_fraunhofer_fast_tow_pipeline(
    path: Path,
    sample_rate_hz: float,
    compute_backend: str = "auto",
    max_workers: int | None = None,
) -> TowDetectionResult | None:
    project = _fraunhofer_project_path()
    if project is None:
        return None

    script = r"""
import json
import sys

from app.dsp.acquisition import acquisition_rank_key, acquisition_result_is_plausible, scan_prns_from_session
from app.dsp.io import Complex64FileSource
from app.dsp.navdecode import decode_navigation_from_tracking
from app.dsp.tracking import track_file
from app.models import SessionConfig

file_path = sys.argv[1]
sample_rate = float(sys.argv[2])
backend = sys.argv[3]
max_workers = int(sys.argv[4])
start_time_s = float(sys.argv[5])
acquisition_window_s = float(sys.argv[6])
tracking_s = float(sys.argv[7])
max_satellites = int(sys.argv[8])

source = Complex64FileSource(file_path)
start_sample = int(round(max(0.0, start_time_s) * sample_rate))
sample_count = int(round(max(0.2, acquisition_window_s) * sample_rate))
sample_count = min(sample_count, max(0, source.total_samples - start_sample))
if sample_count <= 0:
    print("{}")
    raise SystemExit(0)

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

results = scan_prns_from_session(
    samples,
    session,
    prns=list(range(1, 33)),
    progress_callback=None,
    log_callback=None,
)
ranked = sorted(results, key=acquisition_rank_key, reverse=True)
plausible = [result for result in ranked if acquisition_result_is_plausible(result)]
candidates = (plausible or ranked)[: max(1, max_satellites)]

rows = []
for acquisition in candidates:
    tracking_session = SessionConfig(
        file_path=file_path,
        sample_rate=sample_rate,
        compute_backend=backend,
        max_workers=max_workers,
        gpu_enabled=(backend != "cpu"),
    )
    tracking_session.prn = int(acquisition.prn)
    tracking_session.tracking_ms = int(round(max(6.0, tracking_s) * 1000.0))
    absolute_start = start_sample + int(acquisition.best_candidate.segment_start_sample)
    tracking = track_file(
        file_path,
        absolute_start,
        tracking_session,
        acquisition,
        progress_callback=None,
        log_callback=None,
    )
    bit_result, nav_result = decode_navigation_from_tracking(tracking)
    for subframe in nav_result.subframes:
        if not subframe.valid or subframe.tow_seconds is None or subframe.subframe_id is None:
            continue
        bit_ms = 0
        if 0 <= subframe.start_bit < bit_result.bit_start_ms.size:
            bit_ms = int(bit_result.bit_start_ms[subframe.start_bit])
        file_time_s = float(absolute_start) / float(sample_rate) + bit_ms * 1e-3
        tow_count = int(subframe.tow_seconds) // 6
        subframe_offset = int(round(file_time_s / 6.0))
        rows.append(
            {
                "tow_count": tow_count,
                "tow_seconds": int(subframe.tow_seconds),
                "subframe_id": int(subframe.subframe_id),
                "prn": int(acquisition.prn),
                "doppler_hz": float(acquisition.best_candidate.doppler_hz),
                "code_phase_samples": int(acquisition.best_candidate.code_phase_samples),
                "acquisition_metric": float(acquisition.best_candidate.metric),
                "file_time_s": file_time_s,
                "bit_index": int(subframe.start_bit),
                "tracked_ms": int(tracking.times_s.size),
                "synthetic_start_tow_count": int((tow_count - subframe_offset) % (604800 // 6)),
                "synthetic_start_subframe_id": int(((int(subframe.subframe_id) - 1 - subframe_offset) % 5) + 1),
                "source": f"Fraunhofer_FHR fast TOW pipeline ({backend})",
            }
        )
        break
    if rows:
        break

if rows:
    rows.sort(key=lambda item: item["file_time_s"])
    print(json.dumps(rows[0], sort_keys=True))
else:
    print("{}")
"""
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(path),
                str(float(sample_rate_hz)),
                compute_backend,
                str(int(max_workers or 0)),
                str(FRAUNHOFER_FAST_START_TIME_S),
                str(FRAUNHOFER_FAST_ACQUISITION_WINDOW_S),
                str(FRAUNHOFER_FAST_TRACKING_S),
                str(FRAUNHOFER_FAST_MAX_SATELLITES),
            ],
            cwd=str(project),
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not payload:
        return None
    return TowDetectionResult(
        tow_count=int(payload["tow_count"]),
        tow_seconds=int(payload["tow_seconds"]),
        subframe_id=int(payload["subframe_id"]),
        prn=int(payload.get("prn", 0)),
        doppler_hz=float(payload.get("doppler_hz", 0.0)),
        code_phase_samples=int(payload.get("code_phase_samples", 0)),
        acquisition_metric=float(payload.get("acquisition_metric", 0.0)),
        bit_index=int(payload.get("bit_index", 0)),
        polarity="pvt",
        tracked_ms=int(payload.get("tracked_ms", 0)),
        source=str(payload.get("source", "Fraunhofer_FHR fast TOW pipeline")),
        synthetic_start_tow_count=int(payload.get("synthetic_start_tow_count", payload["tow_count"])),
        synthetic_start_subframe_id=int(payload.get("synthetic_start_subframe_id", payload["subframe_id"])),
        file_time_s=float(payload.get("file_time_s", 0.0)),
    )


def detect_measurement_tow(
    input_path: str | Path,
    sample_rate_hz: float,
    compute_backend: str = "auto",
    max_workers: int | None = None,
    prns: Sequence[int] = DEFAULT_TOW_PRNS,
    acquisition_ms: int = DEFAULT_TOW_ACQUISITION_MS,
    track_seconds: float = DEFAULT_TOW_TRACK_SECONDS,
    doppler_min_hz: int = DEFAULT_TOW_DOPPLER_MIN_HZ,
    doppler_max_hz: int = DEFAULT_TOW_DOPPLER_MAX_HZ,
    doppler_step_hz: int = DEFAULT_TOW_DOPPLER_STEP_HZ,
    min_metric: float = 5.0,
) -> TowDetectionResult | None:
    """Try to determine measurement TOW from one existing GPS LNAV signal."""

    path = Path(input_path)
    sidecar = _load_sidecar_tow(path)
    if sidecar is not None:
        return sidecar

    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive for TOW detection.")
    total_samples = count_complex64_samples(path)
    duration_s = float(total_samples) / float(sample_rate_hz)
    if duration_s >= (FRAUNHOFER_FAST_START_TIME_S + FRAUNHOFER_FAST_TRACKING_S + 1.0):
        fast_result = _detect_with_fraunhofer_fast_tow_pipeline(
            path,
            sample_rate_hz=sample_rate_hz,
            compute_backend=compute_backend,
            max_workers=max_workers,
        )
        if fast_result is not None:
            return fast_result

    if os.environ.get("GPSDATAADDER_FULL_PVT_FALLBACK", "").strip() == "1" and duration_s >= FRAUNHOFER_PVT_MIN_DURATION_S:
        pvt_result = _detect_with_fraunhofer_pvt_pipeline(
            path,
            sample_rate_hz=sample_rate_hz,
            compute_backend=compute_backend,
        )
        if pvt_result is not None:
            return pvt_result

    candidate = _acquire_tow_candidate(
        path,
        sample_rate_hz=sample_rate_hz,
        prns=prns,
        acquisition_ms=acquisition_ms,
        doppler_min_hz=doppler_min_hz,
        doppler_max_hz=doppler_max_hz,
        doppler_step_hz=doppler_step_hz,
    )
    if candidate is None or candidate.metric < min_metric:
        return None
    prompt = _prompt_integrations(path, sample_rate_hz, candidate, max_seconds=track_seconds)
    bits = _form_navigation_bits(prompt)
    tow: LnavTowEstimate | None = find_lnav_tow(bits)
    if tow is None:
        return None
    return TowDetectionResult(
        tow_count=tow.tow_count,
        tow_seconds=tow.tow_seconds,
        subframe_id=tow.subframe_id,
        prn=candidate.prn,
        doppler_hz=candidate.doppler_hz,
        code_phase_samples=candidate.code_phase_samples,
        acquisition_metric=candidate.metric,
        bit_index=tow.bit_index,
        polarity=tow.polarity,
        tracked_ms=int(prompt.size),
        source="iq lnav how",
    )
