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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app.dsp.synthetic_satellite import (
    DEFAULT_CHUNK_SAMPLES,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_WORKER_COUNT,
    count_complex64_samples,
)
from app.dsp.relocation_overlay import DEFAULT_RELOCATION_CN0_DBHZ
from app.gui.workers import RelocationAddWorker, RelocationPlanWorker


class MainWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GPSDataAdder")
        self.resize(1180, 820)
        self.setMinimumSize(900, 680)
        self.relocation_plan_worker: RelocationPlanWorker | None = None
        self.relocation_worker: RelocationAddWorker | None = None
        self.relocation_plan: object | None = None
        self._updating_relocation_fields = False
        self._output_auto = True

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        layout.addWidget(self._build_header())

        scroll = QScrollArea()
        scroll.setObjectName("contentScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)
        scroll_layout.addWidget(self._build_file_group())
        scroll_layout.addWidget(self._build_processing_group())
        scroll_layout.addWidget(self._build_relocation_group())
        scroll_layout.addWidget(self._build_run_group())
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)
        self._apply_style()

    def _standard_icon(self, standard_icon: QStyle.StandardPixmap):
        return QApplication.style().standardIcon(standard_icon)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("headerPanel")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 14, 18, 14)
        title = QLabel("GPSDataAdder")
        title.setObjectName("appTitle")
        self.header_state_label = QLabel("Ready")
        self.header_state_label.setObjectName("headerState")
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.header_state_label)
        return header

    def _configure_field(self, widget: QWidget, wide: bool = False) -> QWidget:
        widget.setMinimumHeight(34)
        if wide:
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            widget.setMinimumWidth(180)
            widget.setMaximumWidth(260)
            widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return widget

    def _configure_form(self, form: QFormLayout) -> QFormLayout:
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        return form

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget#root {
                background: #eef2f7;
                color: #172033;
                font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
            }
            QScrollArea#contentScroll {
                background: transparent;
            }
            QWidget#scrollContent {
                background: transparent;
            }
            QWidget {
                font-size: 10pt;
            }
            QWidget#headerPanel {
                background: #111827;
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
                border: 1px solid #d7e0ea;
                border-radius: 8px;
                margin-top: 20px;
                padding: 18px 16px 16px 16px;
                font-weight: 700;
                font-size: 10pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
            }
            QLabel#sectionLabel {
                color: #334155;
                font-size: 9pt;
                font-weight: 800;
                text-transform: uppercase;
                padding-bottom: 2px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #fbfdff;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 5px 8px;
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
                padding: 8px 13px;
                font-weight: 600;
            }
            QPushButton#iconButton {
                min-width: 38px;
                max-width: 38px;
                min-height: 32px;
                padding: 4px;
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
                background: #eef6ff;
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
                background: #111827;
                border: 1px solid #253044;
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
        grid.setContentsMargins(10, 14, 10, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        self.input_edit = self._configure_field(QLineEdit(), wide=True)
        self.input_edit.setPlaceholderText("Eingabe: complex64 .bin/.dat/.iq")
        self.input_edit.textChanged.connect(self._input_changed)
        input_button = QPushButton()
        input_button.setObjectName("iconButton")
        input_button.setIcon(self._standard_icon(QStyle.SP_DialogOpenButton))
        input_button.setToolTip("Eingabedatei waehlen")
        input_button.clicked.connect(self._choose_input)

        self.output_edit = self._configure_field(QLineEdit(), wide=True)
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

        grid.addWidget(QLabel("Input"), 0, 0, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.input_edit, 0, 1)
        grid.addWidget(input_button, 0, 2)
        grid.addWidget(QLabel("Output"), 1, 0, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(output_button, 1, 2)
        grid.addWidget(self.file_info_label, 2, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _build_processing_group(self) -> QGroupBox:
        group = QGroupBox("Aufnahme und Verarbeitung")
        grid = QGridLayout(group)
        grid.setContentsMargins(10, 14, 10, 10)
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(8)

        self.sample_rate_spin = self._configure_field(QDoubleSpinBox())
        self.sample_rate_spin.setRange(1.0, 200_000_000.0)
        self.sample_rate_spin.setDecimals(4)
        self.sample_rate_spin.setSingleStep(1000.0)
        self.sample_rate_spin.setValue(DEFAULT_SAMPLE_RATE_HZ)
        self.sample_rate_spin.valueChanged.connect(self._update_file_info)
        self.sample_rate_spin.valueChanged.connect(self._invalidate_relocation_plan)

        self.chunk_spin = self._configure_field(QSpinBox())
        self.chunk_spin.setRange(50_000, 20_000_000)
        self.chunk_spin.setSingleStep(250_000)
        self.chunk_spin.setValue(DEFAULT_CHUNK_SAMPLES)
        self.chunk_spin.valueChanged.connect(self._invalidate_relocation_plan)

        self.backend_combo = self._configure_field(QComboBox())
        self.backend_combo.addItem("Auto", "auto")
        self.backend_combo.addItem("CPU", "cpu")
        self.backend_combo.addItem("GPU", "gpu")
        self.backend_combo.setToolTip("Auto uses CuPy/CUDA when available and falls back to CPU.")
        self.backend_combo.currentIndexChanged.connect(self._invalidate_relocation_plan)

        self.workers_spin = self._configure_field(QSpinBox())
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setSpecialValueText("Auto")
        self.workers_spin.setValue(0)
        self.workers_spin.setToolTip(f"CPU worker count. Auto is currently {DEFAULT_WORKER_COUNT}.")
        self.workers_spin.valueChanged.connect(self._invalidate_relocation_plan)

        self.inflight_spin = self._configure_field(QSpinBox())
        self.inflight_spin.setRange(0, 128)
        self.inflight_spin.setSpecialValueText("Auto")
        self.inflight_spin.setValue(0)
        self.inflight_spin.setToolTip("Maximum queued processing blocks. Auto is 2x workers.")
        self.inflight_spin.valueChanged.connect(self._invalidate_relocation_plan)

        form_input = self._configure_form(QFormLayout())
        form_input.addRow("Sample rate", self.sample_rate_spin)

        form_processing = self._configure_form(QFormLayout())
        form_processing.addRow("Chunk samples", self.chunk_spin)
        form_processing.addRow("Compute backend", self.backend_combo)
        form_processing.addRow("CPU workers", self.workers_spin)
        form_processing.addRow("In-flight blocks", self.inflight_spin)

        input_column = QVBoxLayout()
        input_column.setSpacing(14)
        input_column.addWidget(self._section_label("Recording"))
        input_column.addLayout(form_input)
        input_column.addStretch(1)

        processing_column = QVBoxLayout()
        processing_column.setSpacing(8)
        processing_column.addWidget(self._section_label("Processing"))
        processing_column.addLayout(form_processing)
        processing_column.addStretch(1)

        grid.addLayout(input_column, 0, 0)
        grid.addLayout(processing_column, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return group

    def _build_relocation_group(self) -> QGroupBox:
        group = QGroupBox("Position Overlay")
        grid = QGridLayout(group)
        grid.setContentsMargins(10, 14, 10, 10)
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(10)

        self.overlay_offset_check = QCheckBox("Use east/north/up offset from detected PVT")
        self.overlay_offset_check.setChecked(True)
        self.overlay_offset_check.toggled.connect(self._update_overlay_mode_controls)
        self.overlay_offset_check.toggled.connect(self._invalidate_relocation_plan)

        self.overlay_east_spin = self._configure_field(QDoubleSpinBox())
        self.overlay_east_spin.setRange(-20_000.0, 20_000.0)
        self.overlay_east_spin.setDecimals(1)
        self.overlay_east_spin.setSingleStep(100.0)
        self.overlay_east_spin.setValue(1000.0)
        self.overlay_east_spin.valueChanged.connect(self._invalidate_relocation_plan)

        self.overlay_north_spin = self._configure_field(QDoubleSpinBox())
        self.overlay_north_spin.setRange(-20_000.0, 20_000.0)
        self.overlay_north_spin.setDecimals(1)
        self.overlay_north_spin.setSingleStep(100.0)
        self.overlay_north_spin.valueChanged.connect(self._invalidate_relocation_plan)

        self.overlay_up_spin = self._configure_field(QDoubleSpinBox())
        self.overlay_up_spin.setRange(-2000.0, 2000.0)
        self.overlay_up_spin.setDecimals(1)
        self.overlay_up_spin.setSingleStep(10.0)
        self.overlay_up_spin.valueChanged.connect(self._invalidate_relocation_plan)

        self.overlay_lat_spin = self._configure_field(QDoubleSpinBox())
        self.overlay_lat_spin.setRange(-90.0, 90.0)
        self.overlay_lat_spin.setDecimals(7)
        self.overlay_lat_spin.setSingleStep(0.0001)
        self.overlay_lat_spin.setValue(50.6163)
        self.overlay_lat_spin.valueChanged.connect(self._invalidate_relocation_plan)

        self.overlay_lon_spin = self._configure_field(QDoubleSpinBox())
        self.overlay_lon_spin.setRange(-180.0, 180.0)
        self.overlay_lon_spin.setDecimals(7)
        self.overlay_lon_spin.setSingleStep(0.0001)
        self.overlay_lon_spin.setValue(7.1326)
        self.overlay_lon_spin.valueChanged.connect(self._invalidate_relocation_plan)

        self.overlay_alt_spin = self._configure_field(QDoubleSpinBox())
        self.overlay_alt_spin.setRange(-1000.0, 20_000.0)
        self.overlay_alt_spin.setDecimals(1)
        self.overlay_alt_spin.setSingleStep(10.0)
        self.overlay_alt_spin.setValue(350.0)
        self.overlay_alt_spin.valueChanged.connect(self._invalidate_relocation_plan)

        self.overlay_cn0_spin = self._configure_field(QDoubleSpinBox())
        self.overlay_cn0_spin.setRange(35.0, 65.0)
        self.overlay_cn0_spin.setDecimals(1)
        self.overlay_cn0_spin.setSingleStep(1.0)
        self.overlay_cn0_spin.setValue(DEFAULT_RELOCATION_CN0_DBHZ)
        self.overlay_cn0_spin.valueChanged.connect(self._invalidate_relocation_plan)

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

        left = self._configure_form(QFormLayout())
        left.addRow("", self.overlay_offset_check)
        left.addRow("East m", self.overlay_east_spin)
        left.addRow("North m", self.overlay_north_spin)
        left.addRow("Up m", self.overlay_up_spin)
        left.addRow("Overlay C/N0", self.overlay_cn0_spin)

        right = self._configure_form(QFormLayout())
        right.addRow("Target lat", self.overlay_lat_spin)
        right.addRow("Target lon", self.overlay_lon_spin)
        right.addRow("Target alt m", self.overlay_alt_spin)

        buttons = QHBoxLayout()
        buttons.addWidget(self.overlay_plan_button)
        buttons.addWidget(self.overlay_write_button)
        buttons.addStretch(1)

        grid.addWidget(self._section_label("Offset"), 0, 0)
        grid.addWidget(self._section_label("Target position"), 0, 1)
        grid.addLayout(left, 1, 0)
        grid.addLayout(right, 1, 1)
        grid.addLayout(buttons, 2, 0, 1, 2)
        grid.addWidget(self.overlay_summary, 3, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self._update_overlay_mode_controls()
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.metadata_check = QCheckBox("Metadata JSON")
        self.metadata_check.setChecked(True)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setIcon(self._standard_icon(QStyle.SP_BrowserStop))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.metadata_check)
        controls.addStretch(1)
        controls.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.status_label = QLabel("Bereit.")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(150)
        self.log.setMaximumHeight(230)

        layout.addLayout(controls)
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
        self._invalidate_relocation_plan()

    def _output_edited(self) -> None:
        self._output_auto = False

    def _maybe_update_output_path(self) -> None:
        text = self.input_edit.text().strip()
        if not text or not self._output_auto:
            return
        input_path = Path(text)
        suffix = input_path.suffix or ".bin"
        self.output_edit.setText(str(input_path.with_name(f"{input_path.stem}.position_overlay{suffix}")))

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

    def _requested_backend(self) -> str:
        return str(self.backend_combo.currentData())

    def _worker_count(self) -> int | None:
        value = self.workers_spin.value()
        return None if value <= 0 else int(value)

    def _in_flight_blocks(self) -> int | None:
        value = self.inflight_spin.value()
        return None if value <= 0 else int(value)

    def _use_overlay_offsets(self) -> bool:
        return self.overlay_offset_check.isChecked()

    def _update_overlay_mode_controls(self) -> None:
        use_offsets = self._use_overlay_offsets()
        offset_enabled = use_offsets and not self._overlay_is_busy()
        target_enabled = (not use_offsets) and not self._overlay_is_busy()
        for widget in (self.overlay_east_spin, self.overlay_north_spin, self.overlay_up_spin):
            widget.setEnabled(offset_enabled)
        for widget in (self.overlay_lat_spin, self.overlay_lon_spin, self.overlay_alt_spin):
            widget.setEnabled(target_enabled)
        if self.relocation_plan is None and not self._overlay_is_busy():
            if use_offsets:
                self.overlay_summary.setText("Offset mode: target coordinates will be calculated from the detected baseline PVT plus east/north/up offsets.")
            else:
                self.overlay_summary.setText("Coordinate mode: east/north/up offsets are ignored and the target latitude, longitude, and altitude are used directly.")

    def _invalidate_relocation_plan(self, *_args: object) -> None:
        if self._updating_relocation_fields or self._overlay_is_busy() or self.relocation_plan is None:
            return
        self.relocation_plan = None
        self.overlay_write_button.setEnabled(False)
        self.overlay_summary.setText("Overlay settings changed. Create a new position overlay plan before writing.")
        if hasattr(self, "status_label"):
            self.status_label.setText("Position overlay plan needs update.")
        if hasattr(self, "header_state_label"):
            self.header_state_label.setText("Ready")

    def _overlay_is_busy(self) -> bool:
        return bool(
            (self.relocation_plan_worker is not None and self.relocation_plan_worker.isRunning())
            or (self.relocation_worker is not None and self.relocation_worker.isRunning())
        )

    def _relocation_plan_kwargs(self, input_path: Path) -> dict[str, object]:
        use_offsets = self._use_overlay_offsets()
        return {
            "input_path": input_path,
            "sample_rate_hz": float(self.sample_rate_spin.value()),
            "target_latitude_deg": float(self.overlay_lat_spin.value()),
            "target_longitude_deg": float(self.overlay_lon_spin.value()),
            "target_altitude_m": float(self.overlay_alt_spin.value()),
            "offset_east_m": float(self.overlay_east_spin.value()) if use_offsets else 0.0,
            "offset_north_m": float(self.overlay_north_spin.value()) if use_offsets else 0.0,
            "offset_up_m": float(self.overlay_up_spin.value()) if use_offsets else 0.0,
            "use_offsets": use_offsets,
            "target_cn0_dbhz": float(self.overlay_cn0_spin.value()),
            "requested_backend": self._requested_backend(),
            "worker_count": self._worker_count(),
            "in_flight_blocks": self._in_flight_blocks(),
            "chunk_samples": int(self.chunk_spin.value()),
        }

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
        plan_kwargs = self._relocation_plan_kwargs(input_path)
        self.relocation_plan_worker = RelocationPlanWorker(**plan_kwargs)
        self.relocation_plan_worker.message.connect(self._append_log)
        self.relocation_plan_worker.succeeded.connect(self._relocation_plan_finished)
        self.relocation_plan_worker.failed.connect(self._relocation_plan_failed)
        self._set_overlay_busy(True, planning=True)
        self.status_label.setText("Position overlay planning.")
        self.header_state_label.setText("Planning overlay")
        if bool(plan_kwargs["use_offsets"]):
            self.overlay_summary.setText("Planning offset mode: decoding PVT, applying east/north/up offsets, and fitting received satellite geometry.")
        else:
            self.overlay_summary.setText("Planning coordinate mode: using the custom target latitude, longitude, and altitude directly.")
        self.relocation_plan_worker.start()

    def _relocation_plan_finished(self, result: object) -> None:
        self._set_overlay_busy(False, planning=True)
        self.relocation_plan = result
        self.status_label.setText("Position overlay plan ready.")
        self.header_state_label.setText("Overlay ready")
        self.overlay_write_button.setEnabled(True)
        self._updating_relocation_fields = True
        try:
            self.overlay_lat_spin.setValue(float(getattr(result, "target_latitude_deg")))
            self.overlay_lon_spin.setValue(float(getattr(result, "target_longitude_deg")))
            self.overlay_alt_spin.setValue(float(getattr(result, "target_altitude_m")))
        finally:
            self._updating_relocation_fields = False
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
                f"nav shift {channel.nav_time_shift_samples:+d} samples, "
                f"Doppler {channel.source_doppler_hz:.1f}->{channel.doppler_hz:.1f} Hz, "
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

    def _cancel(self) -> None:
        if self.relocation_worker is not None:
            self.relocation_worker.cancel()
            self.status_label.setText("Abbruch angefordert.")

    def _set_progress(self, value: float) -> None:
        self.progress.setValue(int(max(0.0, min(100.0, value)) * 10))
        self.status_label.setText(f"{value:.2f}%")

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

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
        self.relocation_worker = None

    def _failed(self, message: str) -> None:
        self._set_running(False)
        self.status_label.setText("Fehler.")
        self.header_state_label.setText("Error")
        self._append_log(message)
        QMessageBox.critical(self, "Fehler", message)
        self.relocation_worker = None

    def _set_running(self, running: bool) -> None:
        self.cancel_button.setEnabled(running)
        for widget in (
            self.input_edit,
            self.output_edit,
            self.sample_rate_spin,
            self.chunk_spin,
            self.backend_combo,
            self.workers_spin,
            self.inflight_spin,
            self.overlay_offset_check,
            self.overlay_cn0_spin,
            self.overlay_plan_button,
            self.overlay_write_button,
            self.metadata_check,
        ):
            widget.setEnabled(not running)
        self._update_overlay_mode_controls()
        if not running:
            self.overlay_write_button.setEnabled(self.relocation_plan is not None)

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
            self.overlay_cn0_spin,
            self.metadata_check,
        ):
            widget.setEnabled(not busy)
        self._update_overlay_mode_controls()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.relocation_plan_worker is not None and self.relocation_plan_worker.isRunning():
            self.relocation_plan_worker.wait(3000)
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
