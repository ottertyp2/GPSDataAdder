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
    window.close()
    assert app is not None
