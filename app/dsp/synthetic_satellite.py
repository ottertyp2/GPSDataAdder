"""Synthetic GPS L1 C/A channel generation and complex64 file augmentation."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from app.dsp.gps_ca import CA_CODE_RATE_HZ, code_phase_samples_to_chips, generate_ca_code, sample_ca_code
from app.dsp.lnav import build_lnav_bit_stream


DEFAULT_SAMPLE_RATE_HZ = 200_000_000.0 / 33.0
COMPLEX64_DTYPE = np.dtype("<c8")
NAV_BIT_RATE_BPS = 50.0
DEFAULT_TARGET_CN0_DBHZ = 42.0
DEFAULT_AMPLITUDE_PROBE_SAMPLES = 262_144
DEFAULT_AMPLITUDE_PROBE_WINDOWS = 7
DEFAULT_CHUNK_SAMPLES = 4_000_000
DEFAULT_WORKER_COUNT = max(1, min(8, (os.cpu_count() or 2) - 1))
DEFAULT_IN_FLIGHT_BLOCKS = DEFAULT_WORKER_COUNT * 2
COMPUTE_BACKENDS = ("auto", "cpu", "gpu")


@dataclass(frozen=True)
class SyntheticSatelliteConfig:
    """Parameters that define the synthetic GPS channel."""

    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    prn: int = 22
    doppler_hz: float = 1500.0
    code_phase_samples: int = 350
    amplitude: float = 0.05
    carrier_phase_deg: float = 0.0
    start_tow_count: int = 100
    nav_seed: int = 20260505


@dataclass(frozen=True)
class AddResult:
    """Summary of one file augmentation run."""

    input_path: str
    output_path: str
    metadata_path: str | None
    total_samples: int
    duration_s: float
    synthetic_signature_id: str
    effective_amplitude: float = 0.0
    amplitude_mode: str = "manual"
    amplitude_estimate: "AutoAmplitudeEstimate | None" = None
    compute_backend: str = "cpu"
    worker_count: int = 1
    in_flight_blocks: int = 1


@dataclass(frozen=True)
class AutoAmplitudeEstimate:
    """Robust input-level estimate used to choose a plausible GPS channel amplitude."""

    amplitude: float
    input_rms: float
    target_cn0_dbhz: float
    relative_db: float
    sample_rate_hz: float
    probed_windows: int
    probed_samples: int


@dataclass(frozen=True)
class DetectedSignalPlan:
    """Recommended synthetic signal settings for one input recording."""

    mode: str
    prn: int
    doppler_hz: float
    code_phase_samples: int
    carrier_phase_deg: float
    amplitude: float
    target_cn0_dbhz: float
    sample_rate_hz: float
    total_samples: int
    duration_s: float
    input_rms: float
    relative_db: float
    compute_backend: str
    worker_count: int
    in_flight_blocks: int
    chunk_samples: int
    summary_lines: tuple[str, ...]


class ProcessingCancelled(RuntimeError):
    """Raised when a long-running file augmentation is canceled."""


def _validate_config(config: SyntheticSatelliteConfig) -> None:
    if not np.isfinite(config.sample_rate_hz) or config.sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")
    if config.prn < 1 or config.prn > 32:
        raise ValueError("PRN must be in the range 1..32.")
    if config.code_phase_samples < 0:
        raise ValueError("Code phase in samples must not be negative.")
    if not np.isfinite(config.amplitude):
        raise ValueError("Amplitude must be finite.")
    if not np.isfinite(config.doppler_hz):
        raise ValueError("Doppler frequency must be finite.")
    if not np.isfinite(config.carrier_phase_deg):
        raise ValueError("Carrier phase must be finite.")


def default_output_path(input_path: str | Path, prn: int) -> Path:
    """Return the default augmented output path for an input file."""

    path = Path(input_path)
    suffix = path.suffix or ".bin"
    return path.with_name(f"{path.stem}.with_prn{int(prn):02d}{suffix}")


def count_complex64_samples(path: str | Path) -> int:
    """Return the number of complex64 samples in a raw IQ file."""

    file_path = Path(path)
    size_bytes = file_path.stat().st_size
    if size_bytes % COMPLEX64_DTYPE.itemsize != 0:
        raise ValueError(f"{file_path} size is not divisible by complex64 sample size.")
    return size_bytes // COMPLEX64_DTYPE.itemsize


def _load_cupy():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() <= 0:
            raise RuntimeError("No CUDA device reported by CuPy.")
    except Exception as exc:
        raise RuntimeError("CuPy/CUDA is not available for GPU processing.") from exc
    return cp


def resolve_compute_backend(requested_backend: str) -> str:
    """Resolve auto/cpu/gpu into the backend that will actually run."""

    requested = requested_backend.lower().strip()
    if requested not in COMPUTE_BACKENDS:
        raise ValueError(f"Unsupported compute backend {requested_backend!r}.")
    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        _load_cupy()
        return "gpu"
    try:
        _load_cupy()
    except RuntimeError:
        return "cpu"
    return "gpu"


def _effective_worker_count(worker_count: int | None, compute_backend: str) -> int:
    if compute_backend == "gpu":
        return 1
    if worker_count is None or worker_count <= 0:
        return DEFAULT_WORKER_COUNT
    return max(1, int(worker_count))


def _effective_in_flight_blocks(in_flight_blocks: int | None, worker_count: int) -> int:
    if in_flight_blocks is None or in_flight_blocks <= 0:
        return max(1, worker_count * 2)
    return max(1, int(in_flight_blocks))


def _window_starts(total_samples: int, probe_samples: int, probe_windows: int) -> list[int]:
    if total_samples <= 0:
        return []
    if total_samples <= probe_samples:
        return [0]
    count = max(1, int(probe_windows))
    starts = np.linspace(0, total_samples - probe_samples, num=count)
    return sorted({int(round(start)) for start in starts})


def estimate_realistic_amplitude(
    path: str | Path,
    sample_rate_hz: float,
    target_cn0_dbhz: float = DEFAULT_TARGET_CN0_DBHZ,
    probe_samples: int = DEFAULT_AMPLITUDE_PROBE_SAMPLES,
    probe_windows: int = DEFAULT_AMPLITUDE_PROBE_WINDOWS,
) -> AutoAmplitudeEstimate:
    """Estimate a realistic GPS channel amplitude from a local IQ recording.

    The estimator samples a few windows across the file, computes a clipped RMS
    for each window, then places the synthetic signal at the requested C/N0.
    """

    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive for auto amplitude.")
    if not np.isfinite(target_cn0_dbhz):
        raise ValueError("Target C/N0 must be finite.")
    if probe_samples <= 0:
        raise ValueError("Probe sample count must be positive.")
    if probe_windows <= 0:
        raise ValueError("Probe window count must be positive.")

    source = Path(path)
    total_samples = count_complex64_samples(source)
    starts = _window_starts(total_samples, int(probe_samples), int(probe_windows))
    if not starts:
        raise ValueError("Cannot estimate amplitude from an empty file.")

    rms_values: list[float] = []
    probed_total = 0
    with source.open("rb") as handle:
        for start in starts:
            count = min(int(probe_samples), total_samples - start)
            if count <= 0:
                continue
            handle.seek(start * COMPLEX64_DTYPE.itemsize)
            data = np.fromfile(handle, dtype=COMPLEX64_DTYPE, count=count)
            if data.size == 0:
                continue
            finite = data[np.isfinite(data.real) & np.isfinite(data.imag)]
            if finite.size == 0:
                continue
            power = np.abs(finite.astype(np.complex64, copy=False)) ** 2
            power = power[np.isfinite(power)]
            if power.size == 0:
                continue
            cutoff = float(np.quantile(power, 0.99))
            clipped = power[power <= cutoff]
            if clipped.size == 0:
                clipped = power
            rms = float(np.sqrt(np.mean(clipped, dtype=np.float64)))
            if rms > 0.0 and np.isfinite(rms):
                rms_values.append(rms)
                probed_total += int(data.size)

    if not rms_values:
        raise ValueError("Cannot estimate auto amplitude because the input appears silent or invalid.")

    input_rms = float(np.median(np.asarray(rms_values, dtype=np.float64)))
    relative_db = float(target_cn0_dbhz - 10.0 * np.log10(sample_rate_hz))
    amplitude = float(input_rms * (10.0 ** (relative_db / 20.0)))
    if amplitude <= 0.0 or not np.isfinite(amplitude):
        raise ValueError("Auto amplitude estimate did not produce a finite positive amplitude.")
    return AutoAmplitudeEstimate(
        amplitude=amplitude,
        input_rms=input_rms,
        target_cn0_dbhz=float(target_cn0_dbhz),
        relative_db=relative_db,
        sample_rate_hz=float(sample_rate_hz),
        probed_windows=len(rms_values),
        probed_samples=probed_total,
    )


def _signature_id(config: SyntheticSatelliteConfig, total_samples: int) -> str:
    payload = {
        "config": asdict(config),
        "total_samples": int(total_samples),
        "format": "little-endian complex64",
        "signal": "GPS L1 C/A synthetic offline channel",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _required_nav_bits(total_samples: int, sample_rate_hz: float) -> int:
    duration_s = float(total_samples) / float(sample_rate_hz)
    return max(1, int(np.ceil(duration_s * NAV_BIT_RATE_BPS)) + 2)


def generate_synthetic_satellite_block(
    config: SyntheticSatelliteConfig,
    start_sample: int,
    sample_count: int,
    nav_bits: np.ndarray,
) -> np.ndarray:
    """Generate one block of the synthetic complex GPS channel."""

    _validate_config(config)
    if start_sample < 0:
        raise ValueError("Start sample must not be negative.")
    if sample_count < 0:
        raise ValueError("Sample count must not be negative.")
    if sample_count == 0:
        return np.empty(0, dtype=np.complex64)

    absolute_samples = start_sample + np.arange(sample_count, dtype=np.float64)
    code_phase_chips = code_phase_samples_to_chips(config.code_phase_samples, config.sample_rate_hz)
    block_code_phase = code_phase_chips + (float(start_sample) * CA_CODE_RATE_HZ / config.sample_rate_hz)
    code = sample_ca_code(
        config.prn,
        config.sample_rate_hz,
        sample_count,
        code_phase_chips=block_code_phase,
        code_rate_hz=CA_CODE_RATE_HZ,
    )

    bit_indices = np.floor(absolute_samples * NAV_BIT_RATE_BPS / config.sample_rate_hz).astype(np.int64)
    if bit_indices.size and int(bit_indices.max()) >= nav_bits.size:
        raise ValueError("Navigation bit stream is too short for requested block.")
    nav_symbols = (1 - 2 * nav_bits[bit_indices]).astype(np.float32)

    phase_rad = np.deg2rad(config.carrier_phase_deg)
    phase = 2.0 * np.pi * config.doppler_hz * absolute_samples / config.sample_rate_hz + phase_rad
    carrier = np.empty(sample_count, dtype=np.complex64)
    carrier.real = np.cos(phase).astype(np.float32)
    carrier.imag = np.sin(phase).astype(np.float32)
    carrier *= (float(config.amplitude) * code * nav_symbols).astype(np.float32)
    return carrier


def _generate_synthetic_satellite_block_gpu(
    config: SyntheticSatelliteConfig,
    start_sample: int,
    sample_count: int,
    nav_bits: np.ndarray,
):
    cp = _load_cupy()
    if sample_count == 0:
        return np.empty(0, dtype=np.complex64)

    absolute_samples = start_sample + cp.arange(sample_count, dtype=cp.float64)
    code_phase_chips = code_phase_samples_to_chips(config.code_phase_samples, config.sample_rate_hz)
    block_code_phase = code_phase_chips + (float(start_sample) * CA_CODE_RATE_HZ / config.sample_rate_hz)
    base_code = cp.asarray(generate_ca_code(config.prn), dtype=cp.float32)
    chip_positions = block_code_phase + (cp.arange(sample_count, dtype=cp.float64) * CA_CODE_RATE_HZ / config.sample_rate_hz)
    chip_indices = cp.floor(chip_positions).astype(cp.int64) % 1023
    code = base_code[chip_indices]

    nav_gpu = cp.asarray(nav_bits, dtype=cp.int8)
    bit_indices = cp.floor(absolute_samples * NAV_BIT_RATE_BPS / config.sample_rate_hz).astype(cp.int64)
    nav_symbols = (1 - 2 * nav_gpu[bit_indices]).astype(cp.float32)

    phase_rad = np.deg2rad(config.carrier_phase_deg)
    phase = 2.0 * cp.pi * config.doppler_hz * absolute_samples / config.sample_rate_hz + phase_rad
    carrier = cp.empty(sample_count, dtype=cp.complex64)
    carrier.real = cp.cos(phase).astype(cp.float32)
    carrier.imag = cp.sin(phase).astype(cp.float32)
    carrier *= (float(config.amplitude) * code * nav_symbols).astype(cp.float32)
    return carrier


def _mix_block(
    data: np.ndarray,
    config: SyntheticSatelliteConfig,
    start_sample: int,
    nav_bits: np.ndarray,
    compute_backend: str,
) -> np.ndarray:
    if compute_backend == "gpu":
        cp = _load_cupy()
        synthetic = _generate_synthetic_satellite_block_gpu(config, start_sample, int(data.size), nav_bits)
        mixed = cp.asarray(data, dtype=cp.complex64)
        mixed += synthetic
        return cp.asnumpy(mixed).astype(np.complex64, copy=False)

    synthetic = generate_synthetic_satellite_block(config, start_sample, int(data.size), nav_bits)
    mixed = data.astype(np.complex64, copy=True)
    mixed += synthetic
    return mixed.astype(np.complex64, copy=False)


def _process_mapped_block(
    input_map: np.memmap,
    output_map: np.memmap,
    config: SyntheticSatelliteConfig,
    start_sample: int,
    sample_count: int,
    nav_bits: np.ndarray,
    compute_backend: str,
) -> int:
    stop_sample = start_sample + sample_count
    data = np.asarray(input_map[start_sample:stop_sample], dtype=np.complex64)
    output_map[start_sample:stop_sample] = _mix_block(data, config, start_sample, nav_bits, compute_backend)
    return int(sample_count)


def _create_sized_file(path: Path, total_samples: int) -> None:
    with path.open("wb") as handle:
        size_bytes = total_samples * COMPLEX64_DTYPE.itemsize
        if size_bytes > 0:
            handle.seek(size_bytes - 1)
            handle.write(b"\0")


def _complete_futures(
    futures: set[Future[int]],
    processed: int,
    total_samples: int,
    progress_callback: Callable[[float], None] | None,
) -> tuple[set[Future[int]], int]:
    done, pending = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        processed += int(future.result())
        if progress_callback is not None:
            progress_callback(100.0 * processed / max(total_samples, 1))
    return set(pending), processed


def _augment_file_parallel(
    source: Path,
    temporary_destination: Path,
    config: SyntheticSatelliteConfig,
    total_samples: int,
    nav_bits: np.ndarray,
    chunk_samples: int,
    worker_count: int,
    in_flight_blocks: int,
    compute_backend: str,
    progress_callback: Callable[[float], None] | None,
    cancel_callback: Callable[[], bool] | None,
) -> None:
    _create_sized_file(temporary_destination, total_samples)
    if total_samples == 0:
        return

    input_map = np.memmap(source, dtype=COMPLEX64_DTYPE, mode="r", shape=(total_samples,))
    output_map = np.memmap(temporary_destination, dtype=COMPLEX64_DTYPE, mode="r+", shape=(total_samples,))
    processed = 0
    futures: set[Future[int]] = set()
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for start_sample in range(0, total_samples, chunk_samples):
                if cancel_callback is not None and cancel_callback():
                    raise ProcessingCancelled("Processing was canceled.")
                while len(futures) >= in_flight_blocks:
                    futures, processed = _complete_futures(futures, processed, total_samples, progress_callback)
                    if cancel_callback is not None and cancel_callback():
                        raise ProcessingCancelled("Processing was canceled.")
                sample_count = min(chunk_samples, total_samples - start_sample)
                futures.add(
                    executor.submit(
                        _process_mapped_block,
                        input_map,
                        output_map,
                        config,
                        int(start_sample),
                        int(sample_count),
                        nav_bits,
                        compute_backend,
                    )
                )

            while futures:
                if cancel_callback is not None and cancel_callback():
                    raise ProcessingCancelled("Processing was canceled.")
                futures, processed = _complete_futures(futures, processed, total_samples, progress_callback)

        if processed != total_samples:
            raise IOError(f"Only processed {processed} of {total_samples} samples.")
        output_map.flush()
    finally:
        del output_map
        del input_map


def _target_cn0_for_mode(mode: str) -> float:
    normalized = mode.lower().strip()
    if normalized == "weak":
        return 38.0
    if normalized == "balanced":
        return 42.0
    if normalized == "strong":
        return 46.0
    raise ValueError("Detect mode must be weak, balanced, or strong.")


def _file_fingerprint(path: Path, total_samples: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(path.name.encode("utf-8", errors="replace"))
    hasher.update(str(total_samples).encode("ascii"))
    starts = _window_starts(total_samples, 4096, 5)
    with path.open("rb") as handle:
        for start in starts:
            handle.seek(start * COMPLEX64_DTYPE.itemsize)
            hasher.update(handle.read(4096 * COMPLEX64_DTYPE.itemsize))
    return hasher.digest()


def detect_synthetic_signal_plan(
    input_path: str | Path,
    sample_rate_hz: float,
    mode: str = "balanced",
    requested_backend: str = "auto",
    worker_count: int | None = None,
    in_flight_blocks: int | None = None,
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
) -> DetectedSignalPlan:
    """Detect practical synthetic-signal parameters before writing an output file."""

    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if chunk_samples <= 0:
        raise ValueError("Chunk size must be positive.")

    total_samples = count_complex64_samples(source)
    target_cn0 = _target_cn0_for_mode(mode)
    amplitude_estimate = estimate_realistic_amplitude(
        source,
        sample_rate_hz=sample_rate_hz,
        target_cn0_dbhz=target_cn0,
    )
    fingerprint = _file_fingerprint(source, total_samples)
    digest_value = int.from_bytes(fingerprint[:8], "big", signed=False)
    samples_per_ms = max(1, int(round(sample_rate_hz * 1e-3)))
    doppler_steps = [-3500, -3000, -2500, -2000, -1500, -1000, -500, 500, 1000, 1500, 2000, 2500, 3000, 3500]
    backend = resolve_compute_backend(requested_backend)
    workers = _effective_worker_count(worker_count, backend)
    in_flight = _effective_in_flight_blocks(in_flight_blocks, workers)

    prn = int(digest_value % 32) + 1
    doppler_hz = float(doppler_steps[(digest_value >> 8) % len(doppler_steps)])
    code_phase_samples = int((digest_value >> 16) % samples_per_ms)
    carrier_phase_deg = float((digest_value >> 32) % 360)
    duration_s = float(total_samples) / float(sample_rate_hz)
    summary = (
        f"Detect mode: {mode.lower().strip()} ({target_cn0:.1f} dB-Hz target).",
        f"Input: {total_samples:,} samples, {duration_s / 60.0:.2f} min, RMS {amplitude_estimate.input_rms:.6g}.",
        f"Signal plan: PRN {prn}, Doppler {doppler_hz:.0f} Hz, code phase {code_phase_samples} samples, amplitude {amplitude_estimate.amplitude:.6g}.",
        f"Performance plan: {backend} backend, {workers} worker(s), {in_flight} in-flight block(s), chunk {chunk_samples:,} samples.",
    )
    return DetectedSignalPlan(
        mode=mode.lower().strip(),
        prn=prn,
        doppler_hz=doppler_hz,
        code_phase_samples=code_phase_samples,
        carrier_phase_deg=carrier_phase_deg,
        amplitude=amplitude_estimate.amplitude,
        target_cn0_dbhz=target_cn0,
        sample_rate_hz=float(sample_rate_hz),
        total_samples=int(total_samples),
        duration_s=duration_s,
        input_rms=amplitude_estimate.input_rms,
        relative_db=amplitude_estimate.relative_db,
        compute_backend=backend,
        worker_count=workers,
        in_flight_blocks=in_flight,
        chunk_samples=int(chunk_samples),
        summary_lines=summary,
    )


def _metadata_payload(
    config: SyntheticSatelliteConfig,
    input_path: Path,
    output_path: Path,
    total_samples: int,
    signature_id: str,
    amplitude_mode: str,
    amplitude_estimate: AutoAmplitudeEstimate | None,
    compute_backend: str,
    worker_count: int,
    in_flight_blocks: int,
    chunk_samples: int,
) -> dict[str, object]:
    nav_bits = build_lnav_bit_stream(
        96,
        start_tow_count=config.start_tow_count,
        seed=config.nav_seed,
    )
    return {
        "synthetic_signature_id": signature_id,
        "signal": "GPS L1 C/A synthetic offline channel",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "format": "little-endian complex64",
        "total_samples": int(total_samples),
        "duration_s": float(total_samples) / float(config.sample_rate_hz),
        "config": asdict(config),
        "amplitude_mode": amplitude_mode,
        "amplitude_estimate": asdict(amplitude_estimate) if amplitude_estimate is not None else None,
        "processing": {
            "compute_backend": compute_backend,
            "worker_count": worker_count,
            "in_flight_blocks": in_flight_blocks,
            "chunk_samples": int(chunk_samples),
        },
        "first_96_lnav_bits": "".join(str(int(bit)) for bit in nav_bits),
        "notes": [
            "Payload bits are synthetic and deterministic.",
            "LNAV words include valid parity for decoder testing.",
            "No RF transmission or SDR control is performed.",
        ],
    }


def add_synthetic_satellite_to_file(
    input_path: str | Path,
    output_path: str | Path,
    config: SyntheticSatelliteConfig,
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
    metadata_path: str | Path | None = None,
    progress_callback: Callable[[float], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    auto_amplitude: bool = False,
    target_cn0_dbhz: float = DEFAULT_TARGET_CN0_DBHZ,
    compute_backend: str = "auto",
    worker_count: int | None = None,
    in_flight_blocks: int | None = None,
) -> AddResult:
    """Read a complex64 IQ file, add one synthetic satellite, and write a new file."""

    _validate_config(config)
    if chunk_samples <= 0:
        raise ValueError("Chunk size must be positive.")

    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must be different.")
    if not source.exists():
        raise FileNotFoundError(source)

    total_samples = count_complex64_samples(source)
    backend = resolve_compute_backend(compute_backend)
    workers = _effective_worker_count(worker_count, backend)
    in_flight = _effective_in_flight_blocks(in_flight_blocks, workers)
    amplitude_estimate = None
    amplitude_mode = "manual"
    effective_config = config
    if auto_amplitude:
        if cancel_callback is not None and cancel_callback():
            raise ProcessingCancelled("Processing was canceled.")
        amplitude_estimate = estimate_realistic_amplitude(
            source,
            sample_rate_hz=config.sample_rate_hz,
            target_cn0_dbhz=target_cn0_dbhz,
        )
        effective_config = replace(config, amplitude=amplitude_estimate.amplitude)
        amplitude_mode = "auto"
        _validate_config(effective_config)

    nav_bits = build_lnav_bit_stream(
        _required_nav_bits(total_samples, effective_config.sample_rate_hz),
        start_tow_count=effective_config.start_tow_count,
        seed=effective_config.nav_seed,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_name(destination.name + ".partial")
    if temporary_destination.exists():
        temporary_destination.unlink()

    try:
        _augment_file_parallel(
            source=source,
            temporary_destination=temporary_destination,
            config=effective_config,
            total_samples=total_samples,
            nav_bits=nav_bits,
            chunk_samples=int(chunk_samples),
            worker_count=workers,
            in_flight_blocks=in_flight,
            compute_backend=backend,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        temporary_destination.replace(destination)
    except Exception:
        if temporary_destination.exists():
            temporary_destination.unlink()
        raise

    signature_id = _signature_id(effective_config, total_samples)
    metadata_output = Path(metadata_path) if metadata_path is not None else None
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata = _metadata_payload(
            effective_config,
            source,
            destination,
            total_samples,
            signature_id,
            amplitude_mode,
            amplitude_estimate,
            backend,
            workers,
            in_flight,
            int(chunk_samples),
        )
        metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return AddResult(
        input_path=str(source),
        output_path=str(destination),
        metadata_path=str(metadata_output) if metadata_output is not None else None,
        total_samples=int(total_samples),
        duration_s=float(total_samples) / float(effective_config.sample_rate_hz),
        synthetic_signature_id=signature_id,
        effective_amplitude=float(effective_config.amplitude),
        amplitude_mode=amplitude_mode,
        amplitude_estimate=amplitude_estimate,
        compute_backend=backend,
        worker_count=workers,
        in_flight_blocks=in_flight,
    )
