"""PySide6 GUI for GPSDataAdder."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app.dsp.synthetic_satellite import (
    DEFAULT_SAMPLE_RATE_HZ,
    SyntheticSatelliteConfig,
    count_complex64_samples,
    default_output_path,
)
from app.gui.workers import AddSyntheticWorker


class MainWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GPSDataAdder")
        self.resize(920, 680)
        self.worker: AddSyntheticWorker | None = None
        self._output_auto = True

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(self._build_file_group())
        layout.addWidget(self._build_signal_group())
        layout.addWidget(self._build_run_group())
        layout.addStretch(1)

    def _standard_icon(self, standard_icon: QStyle.StandardPixmap):
        return QApplication.style().standardIcon(standard_icon)

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("Dateien")
        grid = QGridLayout(group)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Eingabe: complex64 .bin/.dat/.iq")
        self.input_edit.textChanged.connect(self._input_changed)
        input_button = QPushButton()
        input_button.setIcon(self._standard_icon(QStyle.SP_DialogOpenButton))
        input_button.setToolTip("Eingabedatei auswählen")
        input_button.clicked.connect(self._choose_input)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Ausgabe: neue augmented Datei")
        self.output_edit.textEdited.connect(self._output_edited)
        output_button = QPushButton()
        output_button.setIcon(self._standard_icon(QStyle.SP_DialogSaveButton))
        output_button.setToolTip("Ausgabedatei auswählen")
        output_button.clicked.connect(self._choose_output)

        self.file_info_label = QLabel("Keine Datei geladen.")
        self.file_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        grid.addWidget(QLabel("Input"), 0, 0)
        grid.addWidget(self.input_edit, 0, 1)
        grid.addWidget(input_button, 0, 2)
        grid.addWidget(QLabel("Output"), 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(output_button, 1, 2)
        grid.addWidget(self.file_info_label, 2, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _build_signal_group(self) -> QGroupBox:
        group = QGroupBox("Synthetischer Satellit")
        grid = QGridLayout(group)

        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(1.0, 200_000_000.0)
        self.sample_rate_spin.setDecimals(4)
        self.sample_rate_spin.setSingleStep(1000.0)
        self.sample_rate_spin.setValue(DEFAULT_SAMPLE_RATE_HZ)
        self.sample_rate_spin.valueChanged.connect(self._update_file_info)

        self.prn_spin = QSpinBox()
        self.prn_spin.setRange(1, 32)
        self.prn_spin.setValue(22)
        self.prn_spin.valueChanged.connect(self._maybe_update_output_path)

        self.doppler_spin = QDoubleSpinBox()
        self.doppler_spin.setRange(-250_000.0, 250_000.0)
        self.doppler_spin.setDecimals(2)
        self.doppler_spin.setSingleStep(250.0)
        self.doppler_spin.setValue(1500.0)

        self.code_phase_spin = QSpinBox()
        self.code_phase_spin.setRange(0, 10_000_000)
        self.code_phase_spin.setValue(350)

        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(-1000.0, 1000.0)
        self.amplitude_spin.setDecimals(6)
        self.amplitude_spin.setSingleStep(0.01)
        self.amplitude_spin.setValue(0.05)

        self.carrier_phase_spin = QDoubleSpinBox()
        self.carrier_phase_spin.setRange(-360.0, 360.0)
        self.carrier_phase_spin.setDecimals(2)
        self.carrier_phase_spin.setSingleStep(5.0)

        self.tow_spin = QSpinBox()
        self.tow_spin.setRange(0, 100_799)
        self.tow_spin.setValue(100)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(20260505)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(50_000, 20_000_000)
        self.chunk_spin.setSingleStep(250_000)
        self.chunk_spin.setValue(1_000_000)

        form_left = QFormLayout()
        form_left.addRow("Sample rate", self.sample_rate_spin)
        form_left.addRow("PRN", self.prn_spin)
        form_left.addRow("Doppler Hz", self.doppler_spin)
        form_left.addRow("Code phase samples", self.code_phase_spin)

        form_right = QFormLayout()
        form_right.addRow("Amplitude", self.amplitude_spin)
        form_right.addRow("Carrier phase deg", self.carrier_phase_spin)
        form_right.addRow("Start TOW count", self.tow_spin)
        form_right.addRow("Chunk samples", self.chunk_spin)
        form_right.addRow("Nav seed", self.seed_spin)

        grid.addLayout(form_left, 0, 0)
        grid.addLayout(form_right, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.metadata_check = QCheckBox("Metadata JSON")
        self.metadata_check.setChecked(True)
        self.start_button = QPushButton("Start")
        self.start_button.setIcon(self._standard_icon(QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setIcon(self._standard_icon(QStyle.SP_BrowserStop))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.metadata_check)
        controls.addStretch(1)
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.status_label = QLabel("Bereit.")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(170)

        layout.addLayout(controls)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log)
        return group

    def _choose_input(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Input auswählen",
            "",
            "IQ files (*.bin *.dat *.iq);;All files (*.*)",
        )
        if path:
            self.input_edit.setText(path)

    def _choose_output(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Output auswählen",
            "",
            "IQ files (*.bin *.dat *.iq);;All files (*.*)",
        )
        if path:
            self.output_edit.setText(path)
            self._output_auto = False

    def _input_changed(self) -> None:
        self._maybe_update_output_path()
        self._update_file_info()

    def _output_edited(self) -> None:
        self._output_auto = False

    def _maybe_update_output_path(self) -> None:
        text = self.input_edit.text().strip()
        if not text or not self._output_auto:
            return
        self.output_edit.setText(str(default_output_path(Path(text), self.prn_spin.value())))

    def _update_file_info(self) -> None:
        text = self.input_edit.text().strip()
        if not text:
            self.file_info_label.setText("Keine Datei geladen.")
            return
        path = Path(text)
        if not path.exists():
            self.file_info_label.setText("Datei nicht gefunden.")
            return
        try:
            total_samples = count_complex64_samples(path)
        except Exception as exc:
            self.file_info_label.setText(str(exc))
            return
        size_gib = path.stat().st_size / (1024**3)
        duration_s = total_samples / max(self.sample_rate_spin.value(), 1.0)
        self.file_info_label.setText(
            f"{total_samples:,} Samples | {size_gib:.2f} GiB | {duration_s / 60.0:.2f} min"
        )

    def _config(self) -> SyntheticSatelliteConfig:
        return SyntheticSatelliteConfig(
            sample_rate_hz=float(self.sample_rate_spin.value()),
            prn=int(self.prn_spin.value()),
            doppler_hz=float(self.doppler_spin.value()),
            code_phase_samples=int(self.code_phase_spin.value()),
            amplitude=float(self.amplitude_spin.value()),
            carrier_phase_deg=float(self.carrier_phase_spin.value()),
            start_tow_count=int(self.tow_spin.value()),
            nav_seed=int(self.seed_spin.value()),
        )

    def _start(self) -> None:
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not input_text:
            QMessageBox.warning(self, "Input", "Bitte eine Eingabedatei wählen.")
            return
        if not output_text:
            QMessageBox.warning(self, "Output", "Bitte eine Ausgabedatei wählen.")
            return
        input_path = Path(input_text)
        output_path = Path(output_text)
        if not input_path.exists():
            QMessageBox.warning(self, "Input", "Die Eingabedatei existiert nicht.")
            return
        if input_path.resolve() == output_path.resolve():
            QMessageBox.warning(self, "Output", "Input und Output müssen verschieden sein.")
            return
        if output_path.exists():
            choice = QMessageBox.question(
                self,
                "Output überschreiben",
                "Die Ausgabedatei existiert bereits. Überschreiben?",
            )
            if choice != QMessageBox.Yes:
                return

        metadata_path = output_path.with_suffix(output_path.suffix + ".synthetic.json") if self.metadata_check.isChecked() else None
        self.worker = AddSyntheticWorker(
            input_path,
            output_path,
            self._config(),
            self.chunk_spin.value(),
            metadata_path,
        )
        self.worker.progress_changed.connect(self._set_progress)
        self.worker.message.connect(self._append_log)
        self.worker.succeeded.connect(self._finished)
        self.worker.canceled.connect(self._canceled)
        self.worker.failed.connect(self._failed)

        self.progress.setValue(0)
        self.status_label.setText("Läuft.")
        self._append_log(f"Input: {input_path}")
        self._append_log(f"Output: {output_path}")
        self._set_running(True)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status_label.setText("Abbruch angefordert.")

    def _set_progress(self, value: float) -> None:
        self.progress.setValue(int(max(0.0, min(100.0, value)) * 10))
        self.status_label.setText(f"{value:.2f}%")

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _finished(self, result: object) -> None:
        self._set_running(False)
        self.status_label.setText("Fertig.")
        self._append_log(f"Fertig: {getattr(result, 'output_path', '')}")
        self._append_log(f"Samples: {getattr(result, 'total_samples', '')}")
        self._append_log(f"Signature: {getattr(result, 'synthetic_signature_id', '')}")
        metadata_path = getattr(result, "metadata_path", None)
        if metadata_path:
            self._append_log(f"Metadata: {metadata_path}")
        self.worker = None

    def _canceled(self) -> None:
        self._set_running(False)
        self.progress.setValue(0)
        self.status_label.setText("Abgebrochen.")
        self.worker = None

    def _failed(self, message: str) -> None:
        self._set_running(False)
        self.status_label.setText("Fehler.")
        self._append_log(message)
        QMessageBox.critical(self, "Fehler", message)
        self.worker = None

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        for widget in (
            self.input_edit,
            self.output_edit,
            self.sample_rate_spin,
            self.prn_spin,
            self.doppler_spin,
            self.code_phase_spin,
            self.amplitude_spin,
            self.carrier_phase_spin,
            self.tow_spin,
            self.seed_spin,
            self.chunk_spin,
            self.metadata_check,
        ):
            widget.setEnabled(not running)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            choice = QMessageBox.question(
                self,
                "Verarbeitung läuft",
                "Die Verarbeitung läuft noch. Abbrechen und schließen?",
            )
            if choice != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()
