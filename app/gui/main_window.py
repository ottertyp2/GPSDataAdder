"""PySide6 GUI for GPSDataAdder."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    DEFAULT_CHUNK_SAMPLES,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_WORKER_COUNT,
    SyntheticSatelliteConfig,
    count_complex64_samples,
    default_output_path,
)
from app.gui.workers import AddSyntheticWorker, DetectPlanWorker, RelocationAddWorker, RelocationPlanWorker


class MainWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GPSDataAdder")
        self.resize(1000, 780)
        self.setMinimumSize(960, 740)
        self.worker: AddSyntheticWorker | None = None
        self.detect_worker: DetectPlanWorker | None = None
        self.relocation_plan_worker: RelocationPlanWorker | None = None
        self.relocation_worker: RelocationAddWorker | None = None
        self.relocation_plan: object | None = None
        self._output_auto = True

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        layout.addWidget(self._build_header())
        layout.addWidget(self._build_file_group())
        layout.addWidget(self._build_signal_group())
        layout.addWidget(self._build_relocation_group())
        layout.addWidget(self._build_run_group())
        layout.addStretch(1)
        self._apply_style()

    def _standard_icon(self, standard_icon: QStyle.StandardPixmap):
        return QApplication.style().standardIcon(standard_icon)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("headerPanel")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 12)
        title = QLabel("GPSDataAdder")
        title.setObjectName("appTitle")
        self.header_state_label = QLabel("Ready")
        self.header_state_label.setObjectName("headerState")
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.header_state_label)
        return header

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget#root {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #eef1f5, stop:1 #dde3ec);
                color: #172033;
                font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
            }
            QWidget {
                font-size: 10pt;
            }
            QWidget#headerPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f172a, stop:1 #1e293b);
                border-radius: 8px;
            }
            QLabel#appTitle {
                color: #f8fafc;
                font-size: 20pt;
                font-weight: 800;
            }
            QLabel#headerState {
                color: #94a3b8;
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 22px;
                padding: 14px;
                font-weight: 700;
                font-size: 10pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 6px 8px;
                min-height: 24px;
                selection-background-color: #3b82f6;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #3b82f6;
                background: #ffffff;
            }
            QPushButton {
                background: #f1f5f9;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                color: #1e293b;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover:enabled {
                background: #e2e8f0;
                border-color: #94a3b8;
            }
            QPushButton:pressed:enabled {
                background: #cbd5e1;
            }
            QPushButton:disabled {
                color: #94a3b8;
                background: #f1f5f9;
                border-color: #e2e8f0;
            }
            QPushButton#detectButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border: 1px solid #1d4ed8;
                color: #ffffff;
                font-weight: 800;
                font-size: 11pt;
                padding: 10px 22px;
                border-radius: 7px;
            }
            QPushButton#detectButton:hover:enabled {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #60a5fa, stop:1 #3b82f6);
            }
            QPushButton#detectButton:pressed:enabled {
                background: #1d4ed8;
            }
            QPushButton#startButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #10b981, stop:1 #059669);
                border: 1px solid #047857;
                color: #ffffff;
                font-weight: 800;
                font-size: 11pt;
                padding: 10px 22px;
                border-radius: 7px;
            }
            QPushButton#startButton:hover:enabled {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #34d399, stop:1 #10b981);
            }
            QPushButton#startButton:pressed:enabled {
                background: #047857;
            }
            QPushButton#cancelButton {
                background: #fef2f2;
                border: 1px solid #fca5a5;
                color: #991b1b;
                font-weight: 600;
            }
            QPushButton#cancelButton:hover:enabled {
                background: #fee2e2;
                border-color: #f87171;
            }
            QLabel#subtleLabel {
                color: #64748b;
                font-size: 9pt;
            }
            QLabel#planSummary {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #eff6ff, stop:1 #f0f9ff);
                border: 1px solid #bfdbfe;
                border-left: 5px solid #3b82f6;
                border-radius: 6px;
                color: #1e3a5f;
                padding: 12px 14px;
                font-family: Consolas, "Cascadia Code", "Courier New", monospace;
                font-size: 9pt;
                line-height: 1.5;
            }
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: #f8fafc;
                height: 18px;
                text-align: center;
                font-weight: 600;
                font-size: 9pt;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #059669);
                border-radius: 5px;
            }
            QPlainTextEdit {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 6px;
                color: #e2e8f0;
                font-family: Consolas, "Cascadia Code", "Courier New", monospace;
                font-size: 9pt;
                padding: 10px;
                selection-background-color: #3b82f6;
            }
            QCheckBox {
                spacing: 6px;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #94a3b8;
                background: #f8fafc;
            }
            QCheckBox::indicator:checked {
                background: #3b82f6;
                border-color: #2563eb;
            }
            """
        )

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("Dateien")
        grid = QGridLayout(group)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Eingabe: complex64 .bin/.dat/.iq")
        self.input_edit.textChanged.connect(self._input_changed)
        input_button = QPushButton()
        input_button.setObjectName("iconButton")
        input_button.setIcon(self._standard_icon(QStyle.SP_DialogOpenButton))
        input_button.setToolTip("Eingabedatei waehlen")
        input_button.clicked.connect(self._choose_input)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Ausgabe: neue augmented Datei")
        self.output_edit.textEdited.connect(self._output_edited)
        output_button = QPushButton()
        output_button.setObjectName("iconButton")
        output_button.setIcon(self._standard_icon(QStyle.SP_DialogSaveButton))
        output_button.setToolTip("Ausgabedatei waehlen")
        output_button.clicked.connect(self._choose_output)

        self.file_info_label = QLabel("Keine Datei geladen.")
        self.file_info_label.setObjectName("subtleLabel")
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

        self.auto_amplitude_check = QCheckBox("Auto amplitude")
        self.auto_amplitude_check.setChecked(True)
        self.auto_amplitude_check.setToolTip("Estimate input RMS and place the synthetic GPS channel at the target C/N0.")
        self.auto_amplitude_check.toggled.connect(self._update_amplitude_controls)

        self.target_cn0_spin = QDoubleSpinBox()
        self.target_cn0_spin.setRange(25.0, 55.0)
        self.target_cn0_spin.setDecimals(1)
        self.target_cn0_spin.setSingleStep(1.0)
        self.target_cn0_spin.setValue(42.0)
        self.target_cn0_spin.setToolTip("Target carrier-to-noise density for automatic amplitude.")

        self.carrier_phase_spin = QDoubleSpinBox()
        self.carrier_phase_spin.setRange(-360.0, 360.0)
        self.carrier_phase_spin.setDecimals(2)
        self.carrier_phase_spin.setSingleStep(5.0)

        self.tow_spin = QSpinBox()
        self.tow_spin.setRange(0, 100_799)
        self.tow_spin.setValue(100)

        self.subframe_spin = QSpinBox()
        self.subframe_spin.setRange(1, 5)
        self.subframe_spin.setValue(1)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(20260505)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(50_000, 20_000_000)
        self.chunk_spin.setSingleStep(250_000)
        self.chunk_spin.setValue(DEFAULT_CHUNK_SAMPLES)

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Auto", "auto")
        self.backend_combo.addItem("CPU", "cpu")
        self.backend_combo.addItem("GPU", "gpu")
        self.backend_combo.setToolTip("Auto uses CuPy/CUDA when available and falls back to CPU.")

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setSpecialValueText("Auto")
        self.workers_spin.setValue(0)
        self.workers_spin.setToolTip(f"CPU worker count. Auto is currently {DEFAULT_WORKER_COUNT}.")

        self.inflight_spin = QSpinBox()
        self.inflight_spin.setRange(0, 128)
        self.inflight_spin.setSpecialValueText("Auto")
        self.inflight_spin.setValue(0)
        self.inflight_spin.setToolTip("Maximum queued processing blocks. Auto is 2x workers.")

        form_left = QFormLayout()
        form_left.addRow("Sample rate", self.sample_rate_spin)
        form_left.addRow("PRN", self.prn_spin)
        form_left.addRow("Doppler Hz", self.doppler_spin)
        form_left.addRow("Code phase samples", self.code_phase_spin)

        form_right = QFormLayout()
        form_right.addRow("Amplitude", self.amplitude_spin)
        form_right.addRow("", self.auto_amplitude_check)
        form_right.addRow("Target C/N0 dB-Hz", self.target_cn0_spin)
        form_right.addRow("Carrier phase deg", self.carrier_phase_spin)
        form_right.addRow("Start TOW count", self.tow_spin)
        form_right.addRow("Start subframe ID", self.subframe_spin)
        form_right.addRow("Chunk samples", self.chunk_spin)
        form_right.addRow("Compute backend", self.backend_combo)
        form_right.addRow("CPU workers", self.workers_spin)
        form_right.addRow("In-flight blocks", self.inflight_spin)
        form_right.addRow("Nav seed", self.seed_spin)

        grid.addLayout(form_left, 0, 0)
        grid.addLayout(form_right, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self._update_amplitude_controls()
        return group

    def _build_relocation_group(self) -> QGroupBox:
        group = QGroupBox("Position Overlay")
        grid = QGridLayout(group)

        self.overlay_offset_check = QCheckBox("Use offset from detected PVT")
        self.overlay_offset_check.setChecked(True)

        self.overlay_east_spin = QDoubleSpinBox()
        self.overlay_east_spin.setRange(-20_000.0, 20_000.0)
        self.overlay_east_spin.setDecimals(1)
        self.overlay_east_spin.setSingleStep(100.0)
        self.overlay_east_spin.setValue(1000.0)

        self.overlay_north_spin = QDoubleSpinBox()
        self.overlay_north_spin.setRange(-20_000.0, 20_000.0)
        self.overlay_north_spin.setDecimals(1)
        self.overlay_north_spin.setSingleStep(100.0)

        self.overlay_up_spin = QDoubleSpinBox()
        self.overlay_up_spin.setRange(-2000.0, 2000.0)
        self.overlay_up_spin.setDecimals(1)
        self.overlay_up_spin.setSingleStep(10.0)

        self.overlay_lat_spin = QDoubleSpinBox()
        self.overlay_lat_spin.setRange(-90.0, 90.0)
        self.overlay_lat_spin.setDecimals(7)
        self.overlay_lat_spin.setSingleStep(0.0001)
        self.overlay_lat_spin.setValue(50.6163)

        self.overlay_lon_spin = QDoubleSpinBox()
        self.overlay_lon_spin.setRange(-180.0, 180.0)
        self.overlay_lon_spin.setDecimals(7)
        self.overlay_lon_spin.setSingleStep(0.0001)
        self.overlay_lon_spin.setValue(7.1326)

        self.overlay_alt_spin = QDoubleSpinBox()
        self.overlay_alt_spin.setRange(-1000.0, 20_000.0)
        self.overlay_alt_spin.setDecimals(1)
        self.overlay_alt_spin.setSingleStep(10.0)
        self.overlay_alt_spin.setValue(350.0)

        self.overlay_cn0_spin = QDoubleSpinBox()
        self.overlay_cn0_spin.setRange(35.0, 65.0)
        self.overlay_cn0_spin.setDecimals(1)
        self.overlay_cn0_spin.setSingleStep(1.0)
        self.overlay_cn0_spin.setValue(56.0)

        self.overlay_plan_button = QPushButton("Plan Position Overlay")
        self.overlay_plan_button.setObjectName("detectButton")
        self.overlay_plan_button.setIcon(self._standard_icon(QStyle.SP_FileDialogDetailedView))
        self.overlay_plan_button.clicked.connect(self._plan_relocation)

        self.overlay_write_button = QPushButton("Write Position Overlay")
        self.overlay_write_button.setObjectName("startButton")
        self.overlay_write_button.setIcon(self._standard_icon(QStyle.SP_MediaPlay))
        self.overlay_write_button.setEnabled(False)
        self.overlay_write_button.clicked.connect(self._start_relocation)

        self.overlay_summary = QLabel("No position overlay plan yet.")
        self.overlay_summary.setObjectName("planSummary")
        self.overlay_summary.setWordWrap(True)
        self.overlay_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)

        left = QFormLayout()
        left.addRow("", self.overlay_offset_check)
        left.addRow("East m", self.overlay_east_spin)
        left.addRow("North m", self.overlay_north_spin)
        left.addRow("Up m", self.overlay_up_spin)
        left.addRow("Overlay C/N0", self.overlay_cn0_spin)

        right = QFormLayout()
        right.addRow("Target lat", self.overlay_lat_spin)
        right.addRow("Target lon", self.overlay_lon_spin)
        right.addRow("Target alt m", self.overlay_alt_spin)

        buttons = QHBoxLayout()
        buttons.addWidget(self.overlay_plan_button)
        buttons.addWidget(self.overlay_write_button)
        buttons.addStretch(1)

        grid.addLayout(left, 0, 0)
        grid.addLayout(right, 0, 1)
        grid.addLayout(buttons, 1, 0, 1, 2)
        grid.addWidget(self.overlay_summary, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.metadata_check = QCheckBox("Metadata JSON")
        self.metadata_check.setChecked(True)
        self.detect_mode_combo = QComboBox()
        self.detect_mode_combo.addItem("Balanced", "balanced")
        self.detect_mode_combo.addItem("Weak", "weak")
        self.detect_mode_combo.addItem("Strong", "strong")
        self.detect_mode_combo.setToolTip("Detect mode sets the target C/N0 before generating the plan.")
        self.detect_button = QPushButton("  Detect  ")
        self.detect_button.setObjectName("detectButton")
        self.detect_button.setIcon(self._standard_icon(QStyle.SP_FileDialogContentsView))
        self.detect_button.clicked.connect(self._detect)
        self.start_button = QPushButton("  Start  ")
        self.start_button.setObjectName("startButton")
        self.start_button.setIcon(self._standard_icon(QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setIcon(self._standard_icon(QStyle.SP_BrowserStop))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.metadata_check)
        controls.addWidget(QLabel("Mode"))
        controls.addWidget(self.detect_mode_combo)
        controls.addWidget(self.detect_button)
        controls.addStretch(1)
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)

        self.plan_summary = QLabel("Run Detect to analyse the input file and generate a signal plan.")
        self.plan_summary.setObjectName("planSummary")
        self.plan_summary.setWordWrap(True)
        self.plan_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.status_label = QLabel("Bereit.")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(140)

        layout.addLayout(controls)
        layout.addWidget(self.plan_summary)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log)
        return group

    def _choose_input(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Input waehlen",
            "",
            "IQ files (*.bin *.dat *.iq);;All files (*.*)",
        )
        if path:
            self.input_edit.setText(path)

    def _choose_output(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Output waehlen",
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
            start_subframe_id=int(self.subframe_spin.value()),
            nav_seed=int(self.seed_spin.value()),
        )

    def _requested_backend(self) -> str:
        return str(self.backend_combo.currentData())

    def _worker_count(self) -> int | None:
        value = self.workers_spin.value()
        return None if value <= 0 else int(value)

    def _in_flight_blocks(self) -> int | None:
        value = self.inflight_spin.value()
        return None if value <= 0 else int(value)

    def _detect(self) -> None:
        input_text = self.input_edit.text().strip()
        if not input_text:
            QMessageBox.warning(self, "Input", "Bitte eine Eingabedatei waehlen.")
            return
        input_path = Path(input_text)
        if not input_path.exists():
            QMessageBox.warning(self, "Input", "Die Eingabedatei existiert nicht.")
            return
        self.detect_worker = DetectPlanWorker(
            input_path=input_path,
            sample_rate_hz=float(self.sample_rate_spin.value()),
            mode=str(self.detect_mode_combo.currentData()),
            requested_backend=self._requested_backend(),
            worker_count=self._worker_count(),
            in_flight_blocks=self._in_flight_blocks(),
            chunk_samples=int(self.chunk_spin.value()),
        )
        self.detect_worker.message.connect(self._append_log)
        self.detect_worker.succeeded.connect(self._detect_finished)
        self.detect_worker.failed.connect(self._detect_failed)
        self._set_detecting(True)
        self.status_label.setText("Detect laeuft.")
        self.header_state_label.setText("Detecting")
        self.plan_summary.setText("Detecting: analysing input level, signal plan, and Fraunhofer_FHR TOW.")
        self.detect_worker.start()

    def _plan_relocation(self) -> None:
        input_text = self.input_edit.text().strip()
        if not input_text:
            QMessageBox.warning(self, "Input", "Bitte eine Eingabedatei waehlen.")
            return
        input_path = Path(input_text)
        if not input_path.exists():
            QMessageBox.warning(self, "Input", "Die Eingabedatei existiert nicht.")
            return
        self.relocation_plan = None
        self.overlay_write_button.setEnabled(False)
        self.relocation_plan_worker = RelocationPlanWorker(
            input_path=input_path,
            sample_rate_hz=float(self.sample_rate_spin.value()),
            target_latitude_deg=float(self.overlay_lat_spin.value()),
            target_longitude_deg=float(self.overlay_lon_spin.value()),
            target_altitude_m=float(self.overlay_alt_spin.value()),
            offset_east_m=float(self.overlay_east_spin.value()),
            offset_north_m=float(self.overlay_north_spin.value()),
            offset_up_m=float(self.overlay_up_spin.value()),
            use_offsets=self.overlay_offset_check.isChecked(),
            target_cn0_dbhz=float(self.overlay_cn0_spin.value()),
            requested_backend=self._requested_backend(),
            worker_count=self._worker_count(),
            in_flight_blocks=self._in_flight_blocks(),
            chunk_samples=int(self.chunk_spin.value()),
        )
        self.relocation_plan_worker.message.connect(self._append_log)
        self.relocation_plan_worker.succeeded.connect(self._relocation_plan_finished)
        self.relocation_plan_worker.failed.connect(self._relocation_plan_failed)
        self._set_overlay_busy(True, planning=True)
        self.status_label.setText("Position overlay planning.")
        self.header_state_label.setText("Planning overlay")
        self.overlay_summary.setText("Planning: decoding PVT, source LNAV, and received satellite geometry.")
        self.relocation_plan_worker.start()

    def _relocation_plan_finished(self, result: object) -> None:
        self._set_overlay_busy(False, planning=True)
        self.relocation_plan = result
        self.status_label.setText("Position overlay plan ready.")
        self.header_state_label.setText("Overlay ready")
        self.overlay_write_button.setEnabled(True)
        self.overlay_lat_spin.setValue(float(getattr(result, "target_latitude_deg")))
        self.overlay_lon_spin.setValue(float(getattr(result, "target_longitude_deg")))
        self.overlay_alt_spin.setValue(float(getattr(result, "target_altitude_m")))
        if self._output_auto:
            input_path = Path(str(getattr(result, "input_path")))
            suffix = input_path.suffix or ".bin"
            self.output_edit.setText(str(input_path.with_name(f"{input_path.stem}.position_overlay{suffix}")))
        summary_text = "\n".join(getattr(result, "summary_lines"))
        self.overlay_summary.setText(summary_text)
        self._append_log("Position overlay plan:")
        for line in getattr(result, "summary_lines"):
            self._append_log(f"  {line}")
        for channel in getattr(result, "channels"):
            self._append_log(
                f"  PRN {channel.prn}: range delta {channel.range_delta_m:.1f} m, "
                f"range-rate delta {channel.range_delta_rate_m_s:+.3f} m/s, "
                f"code phase {channel.original_code_phase_samples}->{channel.code_phase_samples}."
            )
        self.relocation_plan_worker = None

    def _relocation_plan_failed(self, message: str) -> None:
        self._set_overlay_busy(False, planning=True)
        self.status_label.setText("Position overlay plan failed.")
        self.header_state_label.setText("Overlay failed")
        self.overlay_summary.setText("Position overlay planning failed. Details are in the log.")
        self._append_log(message)
        QMessageBox.critical(self, "Position Overlay", message)
        self.relocation_plan_worker = None

    def _start_relocation(self) -> None:
        if self.relocation_plan is None:
            QMessageBox.warning(self, "Position Overlay", "Bitte zuerst einen Position Overlay Plan erstellen.")
            return
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not input_text:
            QMessageBox.warning(self, "Input", "Bitte eine Eingabedatei waehlen.")
            return
        if not output_text:
            QMessageBox.warning(self, "Output", "Bitte eine Ausgabedatei waehlen.")
            return
        input_path = Path(input_text)
        output_path = Path(output_text)
        if not input_path.exists():
            QMessageBox.warning(self, "Input", "Die Eingabedatei existiert nicht.")
            return
        if input_path.resolve() == output_path.resolve():
            QMessageBox.warning(self, "Output", "Input und Output muessen verschieden sein.")
            return
        if output_path.exists():
            choice = QMessageBox.question(
                self,
                "Output ueberschreiben",
                "Die Ausgabedatei existiert bereits. Ueberschreiben?",
            )
            if choice != QMessageBox.Yes:
                return
        metadata_path = output_path.with_suffix(output_path.suffix + ".relocation.json") if self.metadata_check.isChecked() else None
        self.relocation_worker = RelocationAddWorker(
            input_path=input_path,
            output_path=output_path,
            plan=self.relocation_plan,
            metadata_path=metadata_path,
        )
        self.relocation_worker.progress_changed.connect(self._set_progress)
        self.relocation_worker.message.connect(self._append_log)
        self.relocation_worker.succeeded.connect(self._relocation_finished)
        self.relocation_worker.canceled.connect(self._canceled)
        self.relocation_worker.failed.connect(self._failed)
        self.progress.setValue(0)
        self.status_label.setText("Position overlay writing.")
        self.header_state_label.setText("Processing overlay")
        self._set_running(True)
        self.relocation_worker.start()

    def _start(self) -> None:
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not input_text:
            QMessageBox.warning(self, "Input", "Bitte eine Eingabedatei waehlen.")
            return
        if not output_text:
            QMessageBox.warning(self, "Output", "Bitte eine Ausgabedatei waehlen.")
            return
        input_path = Path(input_text)
        output_path = Path(output_text)
        if not input_path.exists():
            QMessageBox.warning(self, "Input", "Die Eingabedatei existiert nicht.")
            return
        if input_path.resolve() == output_path.resolve():
            QMessageBox.warning(self, "Output", "Input und Output muessen verschieden sein.")
            return
        if output_path.exists():
            choice = QMessageBox.question(
                self,
                "Output ueberschreiben",
                "Die Ausgabedatei existiert bereits. Ueberschreiben?",
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
            self.auto_amplitude_check.isChecked(),
            self.target_cn0_spin.value(),
            self._requested_backend(),
            self._worker_count(),
            self._in_flight_blocks(),
        )
        self.worker.progress_changed.connect(self._set_progress)
        self.worker.message.connect(self._append_log)
        self.worker.succeeded.connect(self._finished)
        self.worker.canceled.connect(self._canceled)
        self.worker.failed.connect(self._failed)

        self.progress.setValue(0)
        self.status_label.setText("Laeuft.")
        self.header_state_label.setText("Processing")
        self._append_log(f"Input: {input_path}")
        self._append_log(f"Output: {output_path}")
        self._set_running(True)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status_label.setText("Abbruch angefordert.")
        if self.relocation_worker is not None:
            self.relocation_worker.cancel()
            self.status_label.setText("Abbruch angefordert.")

    def _set_progress(self, value: float) -> None:
        self.progress.setValue(int(max(0.0, min(100.0, value)) * 10))
        self.status_label.setText(f"{value:.2f}%")

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _detect_finished(self, result: object) -> None:
        self._set_detecting(False)
        self.status_label.setText("Detect fertig.")
        self.header_state_label.setText("Plan ready")
        plan = result
        self.prn_spin.setValue(int(getattr(plan, "prn")))
        self.doppler_spin.setValue(float(getattr(plan, "doppler_hz")))
        self.code_phase_spin.setValue(int(getattr(plan, "code_phase_samples")))
        self.carrier_phase_spin.setValue(float(getattr(plan, "carrier_phase_deg")))
        self.tow_spin.setValue(int(getattr(plan, "start_tow_count")))
        self.subframe_spin.setValue(int(getattr(plan, "start_subframe_id")))
        self.seed_spin.setValue(int(getattr(plan, "nav_seed")))
        self.target_cn0_spin.setValue(float(getattr(plan, "target_cn0_dbhz")))
        self.amplitude_spin.setValue(float(getattr(plan, "amplitude")))
        self.auto_amplitude_check.setChecked(False)
        self.chunk_spin.setValue(int(getattr(plan, "chunk_samples")))
        self.workers_spin.setValue(int(getattr(plan, "worker_count")))
        self.inflight_spin.setValue(int(getattr(plan, "in_flight_blocks")))
        backend = str(getattr(plan, "compute_backend"))
        for index in range(self.backend_combo.count()):
            if self.backend_combo.itemData(index) == backend:
                self.backend_combo.setCurrentIndex(index)
                break
        summary_text = "\n".join(getattr(plan, "summary_lines"))
        self.plan_summary.setText(summary_text)
        self._append_log("Detect plan:")
        for line in getattr(plan, "summary_lines"):
            self._append_log(f"  {line}")
        self._append_log("Start uses the visible fixed amplitude. Run Detect again after changing the input or sample rate.")
        self.detect_worker = None

    def _detect_failed(self, message: str) -> None:
        self._set_detecting(False)
        self.status_label.setText("Detect Fehler.")
        self.header_state_label.setText("Detect failed")
        self.plan_summary.setText("Detect failed. Details are in the log.")
        self._append_log(message)
        QMessageBox.critical(self, "Detect Fehler", message)
        self.detect_worker = None

    def _finished(self, result: object) -> None:
        self._set_running(False)
        self.status_label.setText("Fertig.")
        self.header_state_label.setText("Done")
        self._append_log(f"Fertig: {getattr(result, 'output_path', '')}")
        self._append_log(f"Samples: {getattr(result, 'total_samples', '')}")
        self._append_log(f"Signature: {getattr(result, 'synthetic_signature_id', '')}")
        self._append_log(f"Amplitude: {getattr(result, 'effective_amplitude', '')}")
        self._append_log(
            f"Compute: {getattr(result, 'compute_backend', '')}, "
            f"workers {getattr(result, 'worker_count', '')}, "
            f"in-flight {getattr(result, 'in_flight_blocks', '')}"
        )
        amplitude_estimate = getattr(result, "amplitude_estimate", None)
        if amplitude_estimate is not None:
            self._append_log(
                f"Auto level: input RMS {amplitude_estimate.input_rms:.6g}, "
                f"relative {amplitude_estimate.relative_db:.2f} dB"
            )
        metadata_path = getattr(result, "metadata_path", None)
        if metadata_path:
            self._append_log(f"Metadata: {metadata_path}")
        self.worker = None

    def _relocation_finished(self, result: object) -> None:
        self._set_running(False)
        self.status_label.setText("Position overlay fertig.")
        self.header_state_label.setText("Done")
        self._append_log(f"Fertig: {getattr(result, 'output_path', '')}")
        self._append_log(f"Samples: {getattr(result, 'total_samples', '')}")
        self._append_log(f"Overlay channels: {getattr(result, 'channel_count', '')}")
        self._append_log(
            f"Compute: {getattr(result, 'compute_backend', '')}, "
            f"workers {getattr(result, 'worker_count', '')}, "
            f"in-flight {getattr(result, 'in_flight_blocks', '')}"
        )
        metadata_path = getattr(result, "metadata_path", None)
        if metadata_path:
            self._append_log(f"Metadata: {metadata_path}")
        self.relocation_worker = None

    def _canceled(self) -> None:
        self._set_running(False)
        self.progress.setValue(0)
        self.status_label.setText("Abgebrochen.")
        self.header_state_label.setText("Cancelled")
        self.worker = None
        self.relocation_worker = None

    def _failed(self, message: str) -> None:
        self._set_running(False)
        self.status_label.setText("Fehler.")
        self.header_state_label.setText("Error")
        self._append_log(message)
        QMessageBox.critical(self, "Fehler", message)
        self.worker = None
        self.relocation_worker = None

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
            self.auto_amplitude_check,
            self.target_cn0_spin,
            self.carrier_phase_spin,
            self.tow_spin,
            self.subframe_spin,
            self.seed_spin,
            self.chunk_spin,
            self.backend_combo,
            self.workers_spin,
            self.inflight_spin,
            self.detect_mode_combo,
            self.detect_button,
            self.overlay_offset_check,
            self.overlay_east_spin,
            self.overlay_north_spin,
            self.overlay_up_spin,
            self.overlay_lat_spin,
            self.overlay_lon_spin,
            self.overlay_alt_spin,
            self.overlay_cn0_spin,
            self.overlay_plan_button,
            self.overlay_write_button,
            self.metadata_check,
        ):
            widget.setEnabled(not running)
        if not running:
            self.overlay_write_button.setEnabled(self.relocation_plan is not None)
        if not running:
            self._update_amplitude_controls()

    def _update_amplitude_controls(self) -> None:
        auto_enabled = self.auto_amplitude_check.isChecked()
        ui_enabled = not (self.worker is not None and self.worker.isRunning())
        ui_enabled = ui_enabled and not (self.detect_worker is not None and self.detect_worker.isRunning())
        ui_enabled = ui_enabled and not (self.relocation_plan_worker is not None and self.relocation_plan_worker.isRunning())
        ui_enabled = ui_enabled and not (self.relocation_worker is not None and self.relocation_worker.isRunning())
        self.amplitude_spin.setEnabled(ui_enabled and not auto_enabled)
        self.target_cn0_spin.setEnabled(ui_enabled and auto_enabled)

    def _set_overlay_busy(self, busy: bool, planning: bool = False) -> None:
        self.overlay_plan_button.setEnabled(not busy)
        self.overlay_write_button.setEnabled((not busy) and self.relocation_plan is not None)
        for widget in (
            self.input_edit,
            self.output_edit,
            self.sample_rate_spin,
            self.backend_combo,
            self.workers_spin,
            self.inflight_spin,
            self.chunk_spin,
            self.overlay_offset_check,
            self.overlay_east_spin,
            self.overlay_north_spin,
            self.overlay_up_spin,
            self.overlay_lat_spin,
            self.overlay_lon_spin,
            self.overlay_alt_spin,
            self.overlay_cn0_spin,
            self.detect_button,
            self.start_button,
            self.metadata_check,
        ):
            widget.setEnabled(not busy)
        if not busy and planning:
            self._update_amplitude_controls()

    def _set_detecting(self, detecting: bool) -> None:
        self.detect_button.setEnabled(not detecting)
        self.start_button.setEnabled(not detecting)
        self.cancel_button.setEnabled(False)
        for widget in (
            self.input_edit,
            self.output_edit,
            self.sample_rate_spin,
            self.prn_spin,
            self.doppler_spin,
            self.code_phase_spin,
            self.amplitude_spin,
            self.auto_amplitude_check,
            self.target_cn0_spin,
            self.carrier_phase_spin,
            self.tow_spin,
            self.subframe_spin,
            self.seed_spin,
            self.chunk_spin,
            self.backend_combo,
            self.workers_spin,
            self.inflight_spin,
            self.detect_mode_combo,
            self.overlay_offset_check,
            self.overlay_east_spin,
            self.overlay_north_spin,
            self.overlay_up_spin,
            self.overlay_lat_spin,
            self.overlay_lon_spin,
            self.overlay_alt_spin,
            self.overlay_cn0_spin,
            self.overlay_plan_button,
            self.overlay_write_button,
            self.metadata_check,
        ):
            widget.setEnabled(not detecting)
        if not detecting:
            self.overlay_write_button.setEnabled(self.relocation_plan is not None)
        if not detecting:
            self._update_amplitude_controls()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.detect_worker is not None and self.detect_worker.isRunning():
            self.detect_worker.wait(3000)
        if self.relocation_plan_worker is not None and self.relocation_plan_worker.isRunning():
            self.relocation_plan_worker.wait(3000)
        if self.worker is not None and self.worker.isRunning():
            choice = QMessageBox.question(
                self,
                "Verarbeitung laeuft",
                "Die Verarbeitung laeuft noch. Abbrechen und schliessen?",
            )
            if choice != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait(3000)
        if self.relocation_worker is not None and self.relocation_worker.isRunning():
            choice = QMessageBox.question(
                self,
                "Verarbeitung laeuft",
                "Die Position Overlay Verarbeitung laeuft noch. Abbrechen und schliessen?",
            )
            if choice != QMessageBox.Yes:
                event.ignore()
                return
            self.relocation_worker.cancel()
            self.relocation_worker.wait(3000)
        event.accept()
