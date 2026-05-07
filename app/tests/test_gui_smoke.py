"""GUI smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_main_window_constructs() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.windowTitle() == "GPSDataAdder"
    assert not hasattr(window, "prn_spin")
    assert not hasattr(window, "detect_button")
    assert not hasattr(window, "start_button")
    assert not hasattr(window, "auto_amplitude_check")
    assert window.backend_combo.currentData() == "auto"
    assert window.workers_spin.value() == 0
    assert window.inflight_spin.value() == 0

    assert window.overlay_offset_check.isChecked()
    assert window.overlay_east_spin.isEnabled()
    assert window.overlay_north_spin.isEnabled()
    assert window.overlay_up_spin.isEnabled()
    assert not window.overlay_lat_spin.isEnabled()
    assert not window.overlay_lon_spin.isEnabled()
    assert not window.overlay_alt_spin.isEnabled()

    window.overlay_lat_spin.setValue(48.1)
    window.overlay_lon_spin.setValue(11.6)
    window.overlay_alt_spin.setValue(550.0)
    window.overlay_east_spin.setValue(123.0)
    window.overlay_north_spin.setValue(-45.0)
    window.overlay_up_spin.setValue(6.0)
    window.overlay_offset_check.setChecked(False)

    assert not window.overlay_east_spin.isEnabled()
    assert not window.overlay_north_spin.isEnabled()
    assert not window.overlay_up_spin.isEnabled()
    assert window.overlay_lat_spin.isEnabled()
    assert window.overlay_lon_spin.isEnabled()
    assert window.overlay_alt_spin.isEnabled()

    coordinate_kwargs = window._relocation_plan_kwargs(Path("capture.bin"))
    assert coordinate_kwargs["use_offsets"] is False
    assert coordinate_kwargs["offset_east_m"] == 0.0
    assert coordinate_kwargs["offset_north_m"] == 0.0
    assert coordinate_kwargs["offset_up_m"] == 0.0
    assert coordinate_kwargs["target_latitude_deg"] == pytest.approx(48.1)
    assert coordinate_kwargs["target_longitude_deg"] == pytest.approx(11.6)
    assert coordinate_kwargs["target_altitude_m"] == pytest.approx(550.0)

    window.relocation_plan = object()
    window.overlay_write_button.setEnabled(True)
    window.overlay_offset_check.setChecked(True)
    assert window.relocation_plan is None
    assert not window.overlay_write_button.isEnabled()

    offset_kwargs = window._relocation_plan_kwargs(Path("capture.bin"))
    assert offset_kwargs["use_offsets"] is True
    assert offset_kwargs["offset_east_m"] == pytest.approx(123.0)
    assert offset_kwargs["offset_north_m"] == pytest.approx(-45.0)
    assert offset_kwargs["offset_up_m"] == pytest.approx(6.0)
    window.close()
    assert app is not None
