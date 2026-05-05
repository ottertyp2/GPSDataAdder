"""Qt workers for long-running GPSDataAdder tasks."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.dsp.synthetic_satellite import (
    AddResult,
    ProcessingCancelled,
    SyntheticSatelliteConfig,
    add_synthetic_satellite_to_file,
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
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.chunk_samples = chunk_samples
        self.metadata_path = metadata_path
        self.auto_amplitude = auto_amplitude
        self.target_cn0_dbhz = target_cn0_dbhz
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
            )
            self.message.emit(f"Verwendete Amplitude: {result.effective_amplitude:.6g} ({result.amplitude_mode}).")
            self.progress_changed.emit(100.0)
            self.succeeded.emit(result)
        except ProcessingCancelled:
            self.message.emit("Verarbeitung abgebrochen.")
            self.canceled.emit()
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail)
