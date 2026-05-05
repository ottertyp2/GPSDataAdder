"""Command-line entry point for GPSDataAdder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from app.dsp.synthetic_satellite import (
    DEFAULT_CHUNK_SAMPLES,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_TARGET_CN0_DBHZ,
    SyntheticSatelliteConfig,
    add_synthetic_satellite_to_file,
    detect_synthetic_signal_plan,
    default_output_path,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="GPSDataAdder",
        description="Add one synthetic offline GPS L1 C/A satellite channel to a complex64 IQ file.",
    )
    parser.add_argument("input", help="Input little-endian complex64 IQ file.")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output complex64 IQ file. Defaults to '<input>.with_prnXX.bin'.",
    )
    parser.add_argument(
        "--sample-rate",
        type=_positive_float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="Sample rate in samples per second. Default: 200e6/33.",
    )
    parser.add_argument("--prn", type=int, default=22, help="GPS C/A PRN to synthesize, 1..32.")
    parser.add_argument(
        "--doppler-hz",
        type=float,
        default=1500.0,
        help="Synthetic carrier frequency relative to the IQ recording, in Hz.",
    )
    parser.add_argument(
        "--code-phase-samples",
        type=_non_negative_int,
        default=350,
        help="Initial C/A code phase offset in samples.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.05,
        help="Manual synthetic channel amplitude in complex sample units.",
    )
    parser.add_argument(
        "--auto-amplitude",
        action="store_true",
        help="Estimate input RMS and choose amplitude from target C/N0.",
    )
    parser.add_argument(
        "--target-cn0-dbhz",
        type=float,
        default=DEFAULT_TARGET_CN0_DBHZ,
        help="Target C/N0 for --auto-amplitude. Default: 42.0 dB-Hz.",
    )
    parser.add_argument(
        "--carrier-phase-deg",
        type=float,
        default=0.0,
        help="Initial carrier phase in degrees.",
    )
    parser.add_argument(
        "--start-tow-count",
        type=_non_negative_int,
        default=100,
        help="Initial LNAV HOW TOW count. One count is 6 seconds.",
    )
    parser.add_argument(
        "--start-subframe-id",
        type=int,
        default=1,
        choices=(1, 2, 3, 4, 5),
        help="Initial LNAV subframe ID for the synthetic stream.",
    )
    parser.add_argument(
        "--nav-seed",
        type=int,
        default=20260505,
        help="Seed for deterministic synthetic LNAV payload bits.",
    )
    parser.add_argument(
        "--chunk-samples",
        type=_positive_float,
        default=DEFAULT_CHUNK_SAMPLES,
        help="Processing chunk size in complex samples.",
    )
    parser.add_argument(
        "--compute-backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Compute backend. Auto uses GPU when CuPy/CUDA is available.",
    )
    parser.add_argument(
        "--workers",
        type=_non_negative_int,
        default=0,
        help="CPU worker count. 0 chooses an automatic value.",
    )
    parser.add_argument(
        "--in-flight-blocks",
        type=_non_negative_int,
        default=0,
        help="Maximum queued processing blocks. 0 chooses an automatic value.",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Inspect the input and print a signal plan without writing output.",
    )
    parser.add_argument(
        "--detect-mode",
        choices=("weak", "balanced", "strong"),
        default="balanced",
        help="Detect-only target strength mode.",
    )
    parser.add_argument(
        "--metadata-out",
        help="Optional metadata JSON path. Defaults to '<output>.synthetic.json'.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write a synthetic-signature metadata sidecar.",
    )
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path, int(args.prn))
    metadata_path = None
    if not args.no_metadata:
        metadata_path = Path(args.metadata_out) if args.metadata_out else output_path.with_suffix(output_path.suffix + ".synthetic.json")

    if args.detect_only:
        plan = detect_synthetic_signal_plan(
            input_path,
            sample_rate_hz=float(args.sample_rate),
            mode=str(args.detect_mode),
            requested_backend=str(args.compute_backend),
            worker_count=None if int(args.workers) <= 0 else int(args.workers),
            in_flight_blocks=None if int(args.in_flight_blocks) <= 0 else int(args.in_flight_blocks),
            chunk_samples=int(args.chunk_samples),
        )
        print("Detect plan:")
        for line in plan.summary_lines:
            print(f"  {line}")
        return 0

    config = SyntheticSatelliteConfig(
        sample_rate_hz=float(args.sample_rate),
        prn=int(args.prn),
        doppler_hz=float(args.doppler_hz),
        code_phase_samples=int(args.code_phase_samples),
        amplitude=float(args.amplitude),
        carrier_phase_deg=float(args.carrier_phase_deg),
        start_tow_count=int(args.start_tow_count),
        start_subframe_id=int(args.start_subframe_id),
        nav_seed=int(args.nav_seed),
    )

    def report(progress: float) -> None:
        print(f"{progress:6.2f}%")

    result = add_synthetic_satellite_to_file(
        input_path=input_path,
        output_path=output_path,
        config=config,
        chunk_samples=int(args.chunk_samples),
        metadata_path=metadata_path,
        progress_callback=report,
        auto_amplitude=bool(args.auto_amplitude),
        target_cn0_dbhz=float(args.target_cn0_dbhz),
        compute_backend=str(args.compute_backend),
        worker_count=None if int(args.workers) <= 0 else int(args.workers),
        in_flight_blocks=None if int(args.in_flight_blocks) <= 0 else int(args.in_flight_blocks),
    )

    print(f"Wrote {result.output_path}")
    print(f"Samples: {result.total_samples}")
    print(f"Duration: {result.duration_s:.3f} s")
    print(f"Amplitude: {result.effective_amplitude:.9g} ({result.amplitude_mode})")
    print(f"Compute: {result.compute_backend}, workers {result.worker_count}, in-flight {result.in_flight_blocks}")
    if result.amplitude_estimate is not None:
        estimate = result.amplitude_estimate
        print(f"Input RMS: {estimate.input_rms:.9g}")
        print(f"Relative level: {estimate.relative_db:.2f} dB")
    print(f"Synthetic signature: {result.synthetic_signature_id}")
    if result.metadata_path:
        print(f"Metadata: {result.metadata_path}")
    return 0


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return run_gui()
    if args[0] == "--gui":
        return run_gui()
    if args[0] == "--cli":
        return run_cli(args[1:])
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
