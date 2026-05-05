"""Synthetic GPS L1 C/A channel generation and complex64 file augmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from app.dsp.gps_ca import CA_CODE_RATE_HZ, code_phase_samples_to_chips, sample_ca_code
from app.dsp.lnav import build_lnav_bit_stream


DEFAULT_SAMPLE_RATE_HZ = 200_000_000.0 / 33.0
COMPLEX64_DTYPE = np.dtype("<c8")
NAV_BIT_RATE_BPS = 50.0
DEFAULT_TARGET_CN0_DBHZ = 42.0
DEFAULT_AMPLITUDE_PROBE_SAMPLES = 262_144
DEFAULT_AMPLITUDE_PROBE_WINDOWS = 7


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


def _metadata_payload(
    config: SyntheticSatelliteConfig,
    input_path: Path,
    output_path: Path,
    total_samples: int,
    signature_id: str,
    amplitude_mode: str,
    amplitude_estimate: AutoAmplitudeEstimate | None,
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
    chunk_samples: int = 1_000_000,
    metadata_path: str | Path | None = None,
    progress_callback: Callable[[float], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    auto_amplitude: bool = False,
    target_cn0_dbhz: float = DEFAULT_TARGET_CN0_DBHZ,
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

    processed = 0
    try:
        with source.open("rb") as input_handle, temporary_destination.open("wb") as output_handle:
            while processed < total_samples:
                if cancel_callback is not None and cancel_callback():
                    raise ProcessingCancelled("Processing was canceled.")
                count = min(int(chunk_samples), total_samples - processed)
                data = np.fromfile(input_handle, dtype=COMPLEX64_DTYPE, count=count)
                if data.size == 0:
                    break
                synthetic = generate_synthetic_satellite_block(effective_config, processed, int(data.size), nav_bits)
                mixed = data.astype(np.complex64, copy=False)
                mixed += synthetic
                mixed.tofile(output_handle)
                processed += int(data.size)
                if progress_callback is not None:
                    progress_callback(100.0 * processed / max(total_samples, 1))

        if processed != total_samples:
            raise IOError(f"Only processed {processed} of {total_samples} samples.")
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
    )
