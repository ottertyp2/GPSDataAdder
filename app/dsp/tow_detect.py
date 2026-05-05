"""Lightweight TOW detection from an offline GPS L1 C/A IQ recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    )


def detect_measurement_tow(
    input_path: str | Path,
    sample_rate_hz: float,
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
