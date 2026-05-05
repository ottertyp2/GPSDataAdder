"""Command-line entry point for GPSDataAdder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from app.dsp.synthetic_satellite import (
    DEFAULT_SAMPLE_RATE_HZ,
    SyntheticSatelliteConfig,
    add_synthetic_satellite_to_file,
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
        help="Synthetic channel amplitude in complex sample units.",
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
        "--nav-seed",
        type=int,
        default=20260505,
        help="Seed for deterministic synthetic LNAV payload bits.",
    )
    parser.add_argument(
        "--chunk-samples",
        type=_positive_float,
        default=1_000_000,
        help="Processing chunk size in complex samples.",
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

    config = SyntheticSatelliteConfig(
        sample_rate_hz=float(args.sample_rate),
        prn=int(args.prn),
        doppler_hz=float(args.doppler_hz),
        code_phase_samples=int(args.code_phase_samples),
        amplitude=float(args.amplitude),
        carrier_phase_deg=float(args.carrier_phase_deg),
        start_tow_count=int(args.start_tow_count),
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
    )

    print(f"Wrote {result.output_path}")
    print(f"Samples: {result.total_samples}")
    print(f"Duration: {result.duration_s:.3f} s")
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
