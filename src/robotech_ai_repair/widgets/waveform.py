"""Waveform display widgets."""

from __future__ import annotations

import numpy as np

try:
    from PySide6.QtCore import QEvent, Qt, Signal
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
    import pyqtgraph as pg
except Exception:  # pragma: no cover - import checked by app startup
    Qt = None  # type: ignore[assignment]
    Signal = object  # type: ignore[assignment]
    QLabel = QVBoxLayout = QWidget = object  # type: ignore[assignment]
    pg = None


class WaveformView(QWidget):
    """Compact waveform envelope with a movable marker line."""

    markerChanged = Signal(float)
    rightClicked = Signal(float)
    selectionDragged = Signal(float, float)
    focused = Signal()
    keyPressed = Signal(object)
    fileDropped = Signal(str)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.duration_seconds = 1.0
        self.x_scale_seconds = 1.0
        self._active = False
        self._suppress_marker_signal = False
        self._drag_start_seconds: float | None = None
        self._title = QLabel(title)
        self._plot = pg.PlotWidget()
        self._plot.setAcceptDrops(True)
        self._plot.viewport().setAcceptDrops(True)
        self._plot.setFocusPolicy(Qt.StrongFocus)
        self._plot.setMenuEnabled(False)
        self._plot.hideAxis("left")
        self._plot.showGrid(x=True, y=False, alpha=0.2)
        self._plot.setYRange(-1.0, 1.0)
        self._plot.setMouseEnabled(x=False, y=False)
        self._curve_min = self._plot.plot(pen=pg.mkPen("#7a8cff", width=1))
        self._curve_max = self._plot.plot(pen=pg.mkPen("#7a8cff", width=1))
        self._selection = pg.LinearRegionItem(
            values=(0.0, 0.0),
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(255, 190, 70, 45),
            movable=False,
        )
        self._selection.setZValue(-5)
        self._plot.addItem(self._selection)
        self._marker = pg.InfiniteLine(pos=0.0, angle=90, movable=True, pen=pg.mkPen("#ffcc44", width=2))
        self._plot.addItem(self._marker)
        self._cut_start = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#ffb000", width=2))
        self._cut_end = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#ff4f6d", width=2))
        self._start_marker = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#aaaaaa", width=2))
        self._playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#55ff99", width=2))
        self._plot.addItem(self._cut_start)
        self._plot.addItem(self._cut_end)
        self._plot.addItem(self._start_marker)
        self._plot.addItem(self._playhead)
        self._start_marker.setVisible(False)
        self._marker.sigPositionChanged.connect(self._marker_changed)
        self._plot.scene().sigMouseClicked.connect(self._scene_clicked)
        self._plot.viewport().installEventFilter(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._plot)

    def set_active(self, active: bool) -> None:
        """Visually mark the waveform that receives keyboard focus."""

        self._active = active
        if active:
            self._title.setStyleSheet("font-weight: 700; color: #ffcc44;")
            self._plot.setStyleSheet("border: 2px solid #ffcc44;")
        else:
            self._title.setStyleSheet("")
            self._plot.setStyleSheet("")

    def focus_for_keys(self) -> None:
        """Move real Qt focus to the waveform without changing markers."""

        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self._plot.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_time_unit(self, unit: str) -> None:
        """Set x-axis display unit while keeping public marker values in seconds."""

        self.x_scale_seconds = 60.0 if unit == "minutes" else 1.0

    def set_waveform(self, mins: np.ndarray, maxs: np.ndarray, duration_seconds: float) -> None:
        self.duration_seconds = max(duration_seconds, 0.001)
        display_duration = self.duration_seconds / self.x_scale_seconds
        x = np.linspace(0.0, display_duration, len(mins), endpoint=False) if len(mins) else np.array([])
        self._curve_min.setData(x, mins)
        self._curve_max.setData(x, maxs)
        self._plot.setXRange(0.0, display_duration, padding=0.0)

    def set_marker(self, seconds: float) -> None:
        self._suppress_marker_signal = True
        try:
            clamped = max(min(seconds, self.duration_seconds), 0.0)
            self._marker.setValue(clamped / self.x_scale_seconds)
        finally:
            self._suppress_marker_signal = False

    def set_selection(self, start_seconds: float, end_seconds: float) -> None:
        """Draw a non-editable selected region in seconds."""

        start = max(min(start_seconds, self.duration_seconds), 0.0)
        end = max(min(end_seconds, self.duration_seconds), 0.0)
        if end < start:
            start, end = end, start
        display_start = start / self.x_scale_seconds
        display_end = end / self.x_scale_seconds
        self._selection.setRegion((display_start, display_end))
        self._cut_start.setValue(display_start)
        self._cut_end.setValue(display_end)

    def set_playhead(self, seconds: float) -> None:
        """Move the playback marker in seconds."""

        clamped = max(min(seconds, self.duration_seconds), 0.0)
        self._playhead.setValue(clamped / self.x_scale_seconds)

    def set_start_marker(self, seconds: float) -> None:
        """Move the replay start marker in seconds."""

        clamped = max(min(seconds, self.duration_seconds), 0.0)
        self._start_marker.setVisible(True)
        self._start_marker.setValue(clamped / self.x_scale_seconds)

    def _scene_clicked(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() in {Qt.LeftButton, Qt.RightButton}:
            was_active = self._active
            self.focused.emit()
            if not was_active:
                event.accept()
                return
            scene_pos = self._plot.plotItem.vb.mapSceneToView(event.scenePos())
            seconds = float(scene_pos.x()) * self.x_scale_seconds
            self.set_marker(seconds)
            seconds = max(min(seconds, self.duration_seconds), 0.0)
            if event.button() == Qt.LeftButton:
                self.markerChanged.emit(seconds)
            else:
                self.rightClicked.emit(seconds)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if watched is self._plot.viewport():
            if event.type() == QEvent.Type.DragEnter and event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
            if event.type() == QEvent.Type.Drop:
                for url in event.mimeData().urls():
                    path = url.toLocalFile()
                    if path:
                        self.fileDropped.emit(path)
                        event.acceptProposedAction()
                        return True
            if event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                was_active = self._active
                self.focused.emit()
                if not was_active:
                    return True
                seconds = self._event_seconds(event)
                self._drag_start_seconds = seconds
                self.set_selection(seconds, seconds)
                return True
            if event.type() == QEvent.Type.MouseMove and self._drag_start_seconds is not None:
                seconds = self._event_seconds(event)
                self.set_selection(self._drag_start_seconds, seconds)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and self._drag_start_seconds is not None:
                seconds = self._event_seconds(event)
                start = self._drag_start_seconds
                self._drag_start_seconds = None
                if seconds < start:
                    start, seconds = seconds, start
                self.set_selection(start, seconds)
                self.selectionDragged.emit(start, seconds)
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.keyPressed.emit(event)

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.fileDropped.emit(path)
                event.acceptProposedAction()
                return

    def _event_seconds(self, event) -> float:  # type: ignore[no-untyped-def]
        position = event.position().toPoint()
        scene_point = self._plot.mapToScene(position)
        view_point = self._plot.plotItem.vb.mapSceneToView(scene_point)
        seconds = float(view_point.x()) * self.x_scale_seconds
        return max(min(seconds, self.duration_seconds), 0.0)

    def _marker_changed(self) -> None:
        if not self._suppress_marker_signal:
            self.markerChanged.emit(float(self._marker.value()) * self.x_scale_seconds)
