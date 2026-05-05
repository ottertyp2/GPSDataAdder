"""Qt workers for long-running GPSDataAdder tasks."""

from __future__ import annotations

import traceback
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QThread, Signal

from app.dsp.relocation_overlay import (
    RelocationAddResult,
    RelocationOverlayPlan,
    add_relocation_overlay_to_file,
    plan_relocation_overlay,
)
from app.dsp.synthetic_satellite import (
    AddResult,
    DetectedSignalPlan,
    ProcessingCancelled,
    SyntheticSatelliteConfig,
    add_synthetic_satellite_to_file,
    detect_synthetic_signal_plan,
)


class AddSyntheticWorker(QThread):
    """Run file augmentation away from the GUI thread."""

    progress_changed = Signal(float)
    message = Signal(str)
    succeeded = Signal(object)
    canceled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        config: SyntheticSatelliteConfig,
        chunk_samples: int,
        metadata_path: Path | None,
        auto_amplitude: bool = False,
        target_cn0_dbhz: float = 42.0,
        compute_backend: str = "auto",
        worker_count: int | None = None,
        in_flight_blocks: int | None = None,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.chunk_samples = chunk_samples
        self.metadata_path = metadata_path
        self.auto_amplitude = auto_amplitude
        self.target_cn0_dbhz = target_cn0_dbhz
        self.compute_backend = compute_backend
        self.worker_count = worker_count
        self.in_flight_blocks = in_flight_blocks
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_canceled(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        try:
            if self.auto_amplitude:
                self.message.emit(f"Auto-Amplitude: schaetze Eingabepegel fuer {self.target_cn0_dbhz:.1f} dB-Hz.")
            self.message.emit("Starte blockweise Verarbeitung.")
            result: AddResult = add_synthetic_satellite_to_file(
                self.input_path,
                self.output_path,
                self.config,
                chunk_samples=self.chunk_samples,
                metadata_path=self.metadata_path,
                progress_callback=self.progress_changed.emit,
                cancel_callback=self._is_canceled,
                auto_amplitude=self.auto_amplitude,
                target_cn0_dbhz=self.target_cn0_dbhz,
                compute_backend=self.compute_backend,
                worker_count=self.worker_count,
                in_flight_blocks=self.in_flight_blocks,
            )
            self.message.emit(f"Verwendete Amplitude: {result.effective_amplitude:.6g} ({result.amplitude_mode}).")
            self.message.emit(
                f"Compute: {result.compute_backend}, workers {result.worker_count}, "
                f"in-flight {result.in_flight_blocks}."
            )
            self.progress_changed.emit(100.0)
            self.succeeded.emit(result)
        except ProcessingCancelled:
            self.message.emit("Verarbeitung abgebrochen.")
            self.canceled.emit()
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail)


class DetectPlanWorker(QThread):
    """Detect a synthetic signal plan without writing an output file."""

    message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        input_path: Path,
        sample_rate_hz: float,
        mode: str,
        requested_backend: str,
        worker_count: int | None,
        in_flight_blocks: int | None,
        chunk_samples: int,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.sample_rate_hz = sample_rate_hz
        self.mode = mode
        self.requested_backend = requested_backend
        self.worker_count = worker_count
        self.in_flight_blocks = in_flight_blocks
        self.chunk_samples = chunk_samples

    def run(self) -> None:
        try:
            self.message.emit(f"Detect: analysiere {self.input_path.name}.")
            self.message.emit(
                f"Detect: mode {self.mode}, backend {self.requested_backend}, "
                f"chunk {self.chunk_samples:,} samples."
            )
            start_time = perf_counter()
            plan: DetectedSignalPlan = detect_synthetic_signal_plan(
                self.input_path,
                sample_rate_hz=self.sample_rate_hz,
                mode=self.mode,
                requested_backend=self.requested_backend,
                worker_count=self.worker_count,
                in_flight_blocks=self.in_flight_blocks,
                chunk_samples=self.chunk_samples,
            )
            elapsed_s = perf_counter() - start_time
            self.message.emit(f"Detect: fertig in {elapsed_s:.1f} s.")
            self.succeeded.emit(plan)
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail)


class RelocationPlanWorker(QThread):
    """Plan a multi-satellite position relocation overlay."""

    message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        input_path: Path,
        sample_rate_hz: float,
        target_latitude_deg: float,
        target_longitude_deg: float,
        target_altitude_m: float,
        offset_east_m: float,
        offset_north_m: float,
        offset_up_m: float,
        use_offsets: bool,
        target_cn0_dbhz: float,
        requested_backend: str,
        worker_count: int | None,
        in_flight_blocks: int | None,
        chunk_samples: int,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.sample_rate_hz = sample_rate_hz
        self.target_latitude_deg = target_latitude_deg
        self.target_longitude_deg = target_longitude_deg
        self.target_altitude_m = target_altitude_m
        self.offset_east_m = offset_east_m
        self.offset_north_m = offset_north_m
        self.offset_up_m = offset_up_m
        self.use_offsets = use_offsets
        self.target_cn0_dbhz = target_cn0_dbhz
        self.requested_backend = requested_backend
        self.worker_count = worker_count
        self.in_flight_blocks = in_flight_blocks
        self.chunk_samples = chunk_samples

    def run(self) -> None:
        try:
            self.message.emit("Position overlay: decoding PVT and source LNAV ephemerides.")
            start_time = perf_counter()
            plan: RelocationOverlayPlan = plan_relocation_overlay(
                self.input_path,
                sample_rate_hz=self.sample_rate_hz,
                target_latitude_deg=self.target_latitude_deg,
                target_longitude_deg=self.target_longitude_deg,
                target_altitude_m=self.target_altitude_m,
                offset_east_m=self.offset_east_m,
                offset_north_m=self.offset_north_m,
                offset_up_m=self.offset_up_m,
                use_offsets=self.use_offsets,
                target_cn0_dbhz=self.target_cn0_dbhz,
                requested_backend=self.requested_backend,
                worker_count=self.worker_count,
                in_flight_blocks=self.in_flight_blocks,
                chunk_samples=self.chunk_samples,
            )
            elapsed_s = perf_counter() - start_time
            self.message.emit(f"Position overlay: plan ready in {elapsed_s:.1f} s.")
            self.succeeded.emit(plan)
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail)


class RelocationAddWorker(QThread):
    """Write a relocation overlay away from the GUI thread."""

    progress_changed = Signal(float)
    message = Signal(str)
    succeeded = Signal(object)
    canceled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        plan: RelocationOverlayPlan,
        metadata_path: Path | None,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.plan = plan
        self.metadata_path = metadata_path
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_canceled(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        try:
            self.message.emit(f"Position overlay: writing {len(self.plan.channels)} stronger received-PRN replicas.")
            result: RelocationAddResult = add_relocation_overlay_to_file(
                self.input_path,
                self.output_path,
                self.plan,
                metadata_path=self.metadata_path,
                progress_callback=self.progress_changed.emit,
                cancel_callback=self._is_canceled,
            )
            self.progress_changed.emit(100.0)
            self.succeeded.emit(result)
        except ProcessingCancelled:
            self.message.emit("Position overlay canceled.")
            self.canceled.emit()
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail)
