"""GUI smoke tests."""

from __future__ import annotations

import os

import pytest


def test_main_window_constructs() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.windowTitle() == "GPSDataAdder"
    assert window.prn_spin.value() == 22
    assert window.auto_amplitude_check.isChecked()
    assert not window.amplitude_spin.isEnabled()
    assert window.target_cn0_spin.value() == 42.0
    assert window.tow_spin.value() == 100
    assert window.subframe_spin.value() == 1
    assert window.backend_combo.currentData() == "auto"
    assert window.workers_spin.value() == 0
    assert window.inflight_spin.value() == 0
    assert window.detect_mode_combo.currentData() == "balanced"
    window.close()
    assert app is not None
