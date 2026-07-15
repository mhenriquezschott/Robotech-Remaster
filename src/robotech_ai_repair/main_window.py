"""Main window for the Robotech audio repair tool."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QRectF, Qt, QTimer, QUrl
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QStyle,
)

from .audio_engine import (
    AudioBuffer,
    active_lanes,
    apply_fade,
    apply_selection_action,
    build_work_action_preview,
    crop,
    db_to_linear,
    fit_to_duration,
    load_audio,
    mix_into,
    prepare_lane_audio,
    save_audio,
    time_to_frame,
    waveform_overview,
)
from .models import ClipLane, RepairAction, RepairProject
from .playback import PlaybackEngine
from .project import load_project, save_project
from .recipe_export import export_recipe
from .widgets.waveform import WaveformView


DEFAULT_VOICE_REMOVAL_MODEL = "melband_roformer_instvoc_duality_v1.ckpt"


class PathEdit(QWidget):
    """Line edit with browse button and file drag/drop."""

    pathChanged = Signal()

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.edit = QLineEdit()
        self.edit.setAcceptDrops(False)
        self.button = QPushButton("Open")
        self.button.clicked.connect(self.browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label))
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def path(self) -> Path | None:
        value = self.edit.text().strip()
        return Path(value) if value else None

    def set_path(self, path: str | Path) -> None:
        value = str(path)
        if self.edit.text() == value:
            return
        self.edit.setText(value)
        self.pathChanged.emit()

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open audio file")
        if path:
            self.set_path(path)

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        urls = event.mimeData().urls()
        if urls:
            self.set_path(urls[0].toLocalFile())
            event.acceptProposedAction()


class ClipTimelineBar(QWidget):
    """Drag handle showing a clip inside the current work-window duration."""

    offsetChanged = Signal(float)
    focused = Signal()
    dragStarted = Signal()
    fileDropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(28)
        self.setFocusPolicy(Qt.StrongFocus)
        self.work_duration = 7.0
        self.cut_start = 0.0
        self.cut_duration = 1.0
        self.clip_duration = 1.0
        self.offset_seconds = 0.0
        self.locked = False
        self._dragging = False
        self._drag_delta_seconds = 0.0

    def configure(
        self,
        work_duration: float,
        cut_start: float,
        cut_duration: float,
        clip_duration: float,
        offset_seconds: float,
        locked: bool,
    ) -> None:
        self.work_duration = max(work_duration, 0.001)
        self.cut_start = max(cut_start, 0.0)
        self.cut_duration = max(cut_duration, 0.001)
        self.clip_duration = max(clip_duration, 0.001)
        self.offset_seconds = offset_seconds
        self.locked = locked
        self.update()

    @property
    def display_duration(self) -> float:
        return self.cut_duration if self.clip_duration <= 0 else self.clip_duration

    @property
    def display_start(self) -> float:
        return self.cut_start + self.offset_seconds

    def set_offset(self, offset_seconds: float, emit: bool = False) -> None:
        display_duration = min(self.display_duration, self.work_duration)
        max_start = max(self.work_duration - display_duration, 0.0)
        display_start = min(max(self.cut_start + offset_seconds, 0.0), max_start)
        self.offset_seconds = display_start - self.cut_start
        self.update()
        if emit:
            self.offsetChanged.emit(self.offset_seconds)

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        margin = 10
        track_h = 12
        y = (self.height() - track_h) / 2
        track = QRectF(margin, y, max(self.width() - margin * 2, 1), track_h)
        painter.setPen(QPen(QColor("#c9c9c9"), 1.5))
        painter.setBrush(QColor("#f2f2f2"))
        painter.drawRoundedRect(track, track_h / 2, track_h / 2)

        scale = track.width() / self.work_duration
        handle_start = track.left() + max(self.display_start, 0.0) * scale
        handle_width = max(min(self.display_duration, self.work_duration) * scale, 18.0)
        if handle_start + handle_width > track.right():
            handle_start = track.right() - handle_width
        handle = QRectF(handle_start, y - 3, handle_width, track_h + 6)
        color = QColor("#777777" if not self.locked else "#9a9a9a")
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(handle, (track_h + 6) / 2, (track_h + 6) / 2)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.focused.emit()
        if event.button() != Qt.LeftButton or self.locked:
            return
        self.dragStarted.emit()
        self._dragging = True
        self._drag_delta_seconds = self._seconds_at_x(event.position().x()) - self.display_start
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not self._dragging:
            return
        display_start = self._seconds_at_x(event.position().x()) - self._drag_delta_seconds
        self.set_offset(display_start - self.cut_start, emit=True)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._dragging = False
        event.accept()

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

    def _seconds_at_x(self, x: float) -> float:
        margin = 10
        width = max(self.width() - margin * 2, 1)
        ratio = min(max((x - margin) / width, 0.0), 1.0)
        return ratio * self.work_duration


class LaneWidget(QGroupBox):
    """Controls for one replacement/texture lane."""

    playRequested = Signal(object)
    stopRequested = Signal(object)
    alignRequested = Signal(object)
    fitLengthRequested = Signal(object)
    resetRequested = Signal(object)
    focused = Signal(object)
    timelineDragStarted = Signal(object)
    settingsChanged = Signal(object)

    def __init__(self, lane: ClipLane, parent: QWidget | None = None) -> None:
        super().__init__(lane.name, parent)
        self.setAcceptDrops(True)
        self.lane = lane
        self._syncing = False
        self.clip_selector = QComboBox()
        self.clip_selector.setMaximumWidth(220)
        self.clip_selector.currentIndexChanged.connect(self._clip_selection_changed)
        self.path = PathEdit("")
        self.path.button.clicked.disconnect()
        self.path.button.clicked.connect(self.browse_clip)
        self.path.pathChanged.connect(self._path_changed)
        self.path.edit.editingFinished.connect(lambda: self.settingsChanged.emit(self))
        self.gain = QDoubleSpinBox()
        self.gain.setRange(-60.0, 24.0)
        self.gain.setMaximumWidth(90)
        self.gain.setDecimals(2)
        self.gain.setSuffix(" dB")
        self.gain.valueChanged.connect(self._gain_changed)
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-60.0, 60.0)
        self.offset.setMaximumWidth(95)
        self.offset.setDecimals(3)
        self.offset.setSingleStep(0.01)
        self.offset.setSuffix(" s")
        self.offset.valueChanged.connect(self._offset_spin_changed)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(25.0, 400.0)
        self.speed.setMaximumWidth(90)
        self.speed.setDecimals(1)
        self.speed.setSuffix(" %")
        self.speed.setSingleStep(1.0)
        self.speed.valueChanged.connect(self._speed_changed)
        self.clip_length = QLabel("0.000 s")
        self.clip_length.setMinimumWidth(64)
        self.clip_fade_in = QDoubleSpinBox()
        self.clip_fade_in.setRange(0.0, 2.0)
        self.clip_fade_in.setMaximumWidth(90)
        self.clip_fade_in.setDecimals(3)
        self.clip_fade_in.setSingleStep(0.005)
        self.clip_fade_in.setSuffix(" s")
        self.clip_fade_in.valueChanged.connect(self._fade_in_changed)
        self.clip_fade_out = QDoubleSpinBox()
        self.clip_fade_out.setRange(0.0, 2.0)
        self.clip_fade_out.setMaximumWidth(90)
        self.clip_fade_out.setDecimals(3)
        self.clip_fade_out.setSingleStep(0.005)
        self.clip_fade_out.setSuffix(" s")
        self.clip_fade_out.valueChanged.connect(self._fade_out_changed)
        self.timeline = ClipTimelineBar()
        self.timeline.offsetChanged.connect(self._timeline_offset_changed)
        self.timeline.focused.connect(lambda: self.focused.emit(self))
        self.timeline.dragStarted.connect(lambda: self.timelineDragStarted.emit(self))
        self.timeline.fileDropped.connect(self._file_dropped)
        self.fit_to_cut = QCheckBox("Fit to cut")
        self.fit_to_cut.stateChanged.connect(lambda _state: self._fit_to_cut_changed())
        self.fit_length_button = QPushButton("Fit")
        self.fit_length_button.setFocusPolicy(Qt.NoFocus)
        self.fit_length_button.clicked.connect(lambda: self.fitLengthRequested.emit(self))
        self.muted = QCheckBox("Mute")
        self.muted.stateChanged.connect(lambda _state: self._muted_changed())
        self.locked = QCheckBox("Lock")
        self.locked.stateChanged.connect(lambda _state: self._locked_changed())
        self.align_button = QPushButton("Align")
        self.align_button.setFocusPolicy(Qt.NoFocus)
        self.align_button.clicked.connect(lambda: self.alignRequested.emit(self))
        self.play_button = QPushButton("Play")
        self.play_button.setFocusPolicy(Qt.NoFocus)
        self.play_button.clicked.connect(lambda: self.playRequested.emit(self))
        self.stop_button = QPushButton("Stop")
        self.stop_button.setFocusPolicy(Qt.NoFocus)
        self.stop_button.clicked.connect(lambda: self.stopRequested.emit(self))
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFocusPolicy(Qt.NoFocus)
        self.reset_button.clicked.connect(lambda: self.resetRequested.emit(self))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(4)
        top_row.addWidget(self.clip_selector)
        top_row.addWidget(self.path, 1)
        layout.addLayout(top_row)
        layout.addWidget(self.timeline)
        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(3)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(3)
        controls.addWidget(QLabel("Gain"), 0, 0)
        controls.addWidget(self.gain, 0, 1)
        controls.addWidget(QLabel("Speed"), 0, 2)
        controls.addWidget(self.speed, 0, 3)
        controls.addWidget(QLabel("Length"), 0, 4)
        controls.addWidget(self.clip_length, 0, 5)
        controls.addWidget(QLabel("Offset"), 0, 6)
        controls.addWidget(self.offset, 0, 7)
        controls.addWidget(self.muted, 1, 0)
        controls.addWidget(self.locked, 1, 1)
        controls.addWidget(QLabel("In"), 1, 2)
        controls.addWidget(self.clip_fade_in, 1, 3)
        controls.addWidget(QLabel("Out"), 1, 4)
        controls.addWidget(self.clip_fade_out, 1, 5)
        controls.addWidget(self.align_button, 1, 6)
        controls.addWidget(self.fit_length_button, 1, 7)
        controls.addWidget(self.play_button, 2, 0, 1, 2)
        controls.addWidget(self.stop_button, 2, 2, 1, 2)
        controls.addWidget(self.reset_button, 2, 4, 1, 2)
        layout.addLayout(controls)
        self.from_model()
        self.set_clip_playing(False)

    def from_model(self) -> None:
        self._syncing = True
        self.lane.ensure_clip_items()
        self.lane.apply_selected_clip_item()
        self._refresh_clip_selector()
        self.path.set_path(self.lane.path)
        self.gain.setValue(self.lane.gain_db)
        self.speed.setValue(self.lane.speed_percent)
        self.clip_fade_in.setValue(self.lane.fade_in_seconds)
        self.clip_fade_out.setValue(self.lane.fade_out_seconds)
        self.offset.setValue(self.lane.offset_seconds)
        self.fit_to_cut.setChecked(False)
        self.muted.setChecked(True if not self.lane.path else self.lane.muted)
        self.locked.setChecked(self.lane.locked)
        self._syncing = False

    def to_model(self) -> None:
        path = self.path.path()
        self.lane.path = str(path) if path else ""
        self.lane.gain_db = self.gain.value()
        self.lane.speed_percent = self.speed.value()
        self.lane.fade_in_seconds = self.clip_fade_in.value()
        self.lane.fade_out_seconds = self.clip_fade_out.value()
        self.lane.offset_seconds = self.offset.value()
        self.lane.fit_to_cut = False
        self.lane.muted = True if not self.lane.path else self.muted.isChecked()
        self.lane.solo = False
        self.lane.locked = self.locked.isChecked()
        if self.lane.clip_items:
            self.lane.store_selected_clip_item()

    def _refresh_clip_selector(self) -> None:
        self.clip_selector.blockSignals(True)
        self.clip_selector.clear()
        if not self.lane.clip_items:
            self.clip_selector.addItem("(empty)")
        else:
            for item in self.lane.clip_items:
                self.clip_selector.addItem(item.label or Path(item.path).name or "(unnamed)")
        self.clip_selector.setCurrentIndex(min(max(self.lane.selected_clip_index, 0), self.clip_selector.count() - 1))
        self.clip_selector.blockSignals(False)

    def _clip_selection_changed(self, index: int) -> None:
        if self._syncing:
            return
        if index < 0 or not self.lane.clip_items:
            return
        self.to_model()
        self.lane.selected_clip_index = index
        self.lane.apply_selected_clip_item()
        self.from_model()
        self.focused.emit(self)
        self.settingsChanged.emit(self)

    def _add_clip_path(self, path: str) -> None:
        if not path:
            return
        self.to_model()
        if not self.lane.clip_items and not self.lane.path:
            self.lane.path = path
            self.lane.muted = False
        self.lane.add_clip_item(path)
        self.from_model()
        self.muted.setChecked(False)
        self.focused.emit(self)
        self.settingsChanged.emit(self)

    def browse_clip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Add clip to lane")
        if path:
            self._add_clip_path(path)

    def set_active(self, active: bool) -> None:
        self.setStyleSheet("QGroupBox { border: 2px solid #ffcc44; }" if active else "")

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.focused.emit(self)
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self._file_dropped(path)
                event.acceptProposedAction()
                return

    def _file_dropped(self, path: str) -> None:
        self._add_clip_path(path)

    def _path_changed(self) -> None:
        if self._syncing:
            return
        self.to_model()
        self.muted.setChecked(not bool(self.path.path()))
        if self.lane.clip_items:
            self.lane.store_selected_clip_item()
            self._refresh_clip_selector()
        self.settingsChanged.emit(self)

    def _offset_spin_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.lane.offset_seconds = value
        self.timeline.set_offset(value)
        self.lane.store_selected_clip_item()
        self.settingsChanged.emit(self)

    def _timeline_offset_changed(self, value: float) -> None:
        if self._syncing:
            return
        if abs(self.offset.value() - value) > 0.0005:
            self.offset.setValue(value)
        self.lane.offset_seconds = value
        self.lane.store_selected_clip_item()
        self.settingsChanged.emit(self)

    def _speed_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.lane.speed_percent = value
        self.lane.store_selected_clip_item()
        self.settingsChanged.emit(self)

    def _gain_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.lane.gain_db = value
        self.lane.store_selected_clip_item()
        self.settingsChanged.emit(self)

    def _fade_in_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.lane.fade_in_seconds = value
        self.lane.store_selected_clip_item()
        self.settingsChanged.emit(self)

    def _fade_out_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.lane.fade_out_seconds = value
        self.lane.store_selected_clip_item()
        self.settingsChanged.emit(self)

    def _fit_to_cut_changed(self) -> None:
        if self._syncing:
            return
        self.lane.fit_to_cut = False
        self.lane.store_selected_clip_item()
        self.settingsChanged.emit(self)

    def _muted_changed(self) -> None:
        if self._syncing:
            return
        self.lane.muted = True if not self.lane.path else self.muted.isChecked()
        self.lane.store_selected_clip_item()
        self.settingsChanged.emit(self)

    def _locked_changed(self) -> None:
        if self._syncing:
            return
        self.lane.locked = self.locked.isChecked()
        self.timeline.locked = self.lane.locked
        self.lane.store_selected_clip_item()
        self.timeline.update()

    def configure_timeline(
        self,
        work_duration: float,
        cut_start: float,
        cut_duration: float,
        clip_duration: float,
    ) -> None:
        display_duration = clip_duration
        self.clip_length.setText(f"{display_duration:.3f} s")
        self.timeline.configure(
            work_duration=work_duration,
            cut_start=cut_start,
            cut_duration=cut_duration,
            clip_duration=display_duration,
            offset_seconds=self.offset.value(),
            locked=self.locked.isChecked(),
        )

    def set_clip_playing(self, playing: bool) -> None:
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_button.setIcon(self.style().standardIcon(icon))


class MainWindow(QMainWindow):
    """First-iteration repair workbench."""

    def __init__(self, project: RepairProject | None = None, project_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Robotech Audio Repair Tool")
        self.resize(1664, 980)
        self.project = project or RepairProject(episode="S01E03", title="Space Fold")
        self.project.ensure_default_lanes()
        self._loaded_project_template = deepcopy(self.project)
        self.project_path: Path | None = project_path
        self.main_audio: AudioBuffer | None = None
        self.bed_audio: AudioBuffer | None = None
        self.reference_audio: AudioBuffer | None = None
        self._main_loaded_path: Path | None = None
        self._bed_loaded_path: Path | None = None
        self._bed_loaded_mode = ""
        self._video_loaded_path: Path | None = None
        self._reference_loaded_path: Path | None = None
        self._main_overview: tuple | None = None
        self._bed_overview: tuple | None = None
        self._reference_overview: tuple | None = None
        self._work_audio: AudioBuffer | None = None
        self._bed_work_audio: AudioBuffer | None = None
        self.cut_click_target = "start"
        self._active_play_scope = ""
        self._active_play_start = 0.0
        self._active_play_elapsed_ms = 0
        self._paused_scope = ""
        self._paused_seconds = 0.0
        self._last_play_buffer: AudioBuffer | None = None
        self._last_playback_start_seconds = 0.0
        self._last_display_start_seconds = 0.0
        self._last_loop_display_start_seconds = 0.0
        self._last_play_kind = ""
        self._last_selected_signature: tuple[int, int, int] | None = None
        self._playback_paused = False
        self._active_lane_widget: LaneWidget | None = None
        self._work_start_marker_seconds = 0.0
        self._undo_stack: list[tuple[RepairProject, AudioBuffer | None, AudioBuffer | None, float]] = []
        self._redo_stack: list[tuple[RepairProject, AudioBuffer | None, AudioBuffer | None, float]] = []
        self._restoring_history = False
        self.active_focus = "main"
        self._syncing_ui = False
        self.playback = PlaybackEngine()
        self.playhead_timer = QTimer(self)
        self.playhead_timer.setInterval(50)
        self.playhead_timer.timeout.connect(self.update_playhead)
        self.lane_widgets: list[LaneWidget] = []
        self._build_ui()
        self._build_menu()
        self._load_project_into_ui()
        QTimer.singleShot(0, self.refresh_audio)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_project_action = QAction("Open Project", self)
        open_project_action.triggered.connect(self.open_project)
        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self.save_project)
        save_project_as_action = QAction("Save Project As", self)
        save_project_as_action.triggered.connect(self.save_project_as)
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)
        redo_action = QAction("Redo", self)
        redo_action.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        redo_action.triggered.connect(self.redo)
        file_menu.addActions([open_project_action, save_project_action, save_project_as_action])
        file_menu.addSeparator()
        file_menu.addActions([undo_action, redo_action])

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        self.main_path = PathEdit("Main")
        self.main_path.pathChanged.connect(self.path_field_changed)
        root.addWidget(self.main_path)

        self.timeline = WaveformView("Main Episode Timeline")
        self.timeline.set_time_unit("minutes")
        self.timeline.focused.connect(lambda: self.set_focus("main"))
        self.timeline.keyPressed.connect(self.handle_key_event)
        self.timeline.markerChanged.connect(self.marker_changed)
        self.timeline.selectionDragged.connect(lambda _start, _end: None)
        self.timeline.fileDropped.connect(lambda path: self.main_path.set_path(path))
        root.addWidget(self.timeline, 2)

        self.marker_lock = QCheckBox("Lock top marker")
        self.marker_lock.setChecked(True)
        main_transport = QHBoxLayout()
        main_transport.addWidget(QLabel("Main transport"))
        focus_main = QPushButton("Focus main")
        focus_main.setFocusPolicy(Qt.NoFocus)
        focus_main.clicked.connect(lambda: self.set_focus("main"))
        main_transport.addWidget(focus_main)
        for text, callback in [
            ("Play Main From Marker", self.play_full_main),
            ("Pause", self.pause_playback),
            ("Stop", self.stop_playback),
        ]:
            button = QPushButton(text)
            button.setFocusPolicy(Qt.NoFocus)
            self._decorate_transport_button(button, text)
            button.clicked.connect(callback)
            main_transport.addWidget(button)
        self.main_loop = QCheckBox("Loop main")
        self.main_loop.setChecked(True)
        main_transport.addWidget(self.main_loop)
        main_transport.addWidget(self.marker_lock)
        root.addLayout(main_transport)

        controls = QGridLayout()
        self.marker = self._seconds_spin()
        self.window_duration = self._seconds_spin(maximum=60.0, step=1.0)
        self.snap_ms = QSpinBox()
        self.snap_ms.setRange(1, 1000)
        self.snap_ms.setSuffix(" ms")
        self.work_range = QLabel("Work clip: -")
        self.focus_label = QLabel("Focus: Main episode marker")
        for row, (label, widget) in enumerate(
            [
                ("Marker", self.marker),
                ("Window", self.window_duration),
                ("Snap", self.snap_ms),
            ]
        ):
            controls.addWidget(QLabel(label), row // 4, (row % 4) * 2)
            controls.addWidget(widget, row // 4, (row % 4) * 2 + 1)
        controls.addWidget(self.work_range, 1, 0, 1, 8)
        controls.addWidget(self.focus_label, 2, 0, 1, 8)
        root.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        work_panel = QWidget()
        work_layout = QVBoxLayout(work_panel)
        self.work_waveform = WaveformView("Work Clip Preview")
        self.work_waveform.focused.connect(lambda: self.set_focus("work"))
        self.work_waveform.keyPressed.connect(self.handle_key_event)
        self.work_waveform.markerChanged.connect(self.work_marker_changed)
        self.work_waveform.rightClicked.connect(self.work_end_marker_changed)
        self.work_waveform.selectionDragged.connect(self.work_selection_dragged)
        work_layout.addWidget(self.work_waveform, 2)
        work_cut_group = QGroupBox("Work Clip Selection")
        work_cut_layout = QGridLayout(work_cut_group)
        work_cut_layout.setContentsMargins(8, 6, 8, 6)
        work_cut_layout.setHorizontalSpacing(6)
        work_cut_layout.setVerticalSpacing(4)
        self.cut_start = self._seconds_spin(maximum=60.0, step=0.01)
        self.cut_end = self._seconds_spin(maximum=60.0, step=0.01)
        self.fade_in = self._seconds_spin(maximum=1.0, step=0.005)
        self.fade_out = self._seconds_spin(maximum=1.0, step=0.005)
        self.cut_action = QComboBox()
        self.cut_action.addItems(
            [
                "Keep selected area (gain only / no cut)",
                "Silence selected area (keep timing)",
                "Delete selected area (shorten clip)",
                "Fade out/in selected area",
                "Rubberband bridge selected area",
                "Stretch pre-cut edge across selected area",
                "Repeat pre-cut edge across selected area",
                "Blend pre/post edges across selected area",
                "Interpolate tiny gap with ambience",
                "Remove voice from selected area (audio-separator)",
            ]
        )
        self.selected_gain = QDoubleSpinBox()
        self.selected_gain.setRange(-24.0, 24.0)
        self.selected_gain.setDecimals(1)
        self.selected_gain.setSuffix(" dB")
        self.selected_gain.setValue(0.0)
        self.voice_keep_original = QDoubleSpinBox()
        self.voice_keep_original.setRange(0.0, 100.0)
        self.voice_keep_original.setDecimals(1)
        self.voice_keep_original.setSingleStep(5.0)
        self.voice_keep_original.setSuffix(" %")
        self.voice_keep_original.setToolTip(
            "For voice removal only: blend this percent of the original selected audio back into the de-voiced result."
        )
        self.edge_source = self._seconds_spin(maximum=2.0, step=0.005)
        self.edge_source.setMinimum(0.005)
        self.cut_start_target = QPushButton("Arrow keys move START")
        self.cut_end_target = QPushButton("Arrow keys move END")
        self.cut_start_target.setFocusPolicy(Qt.NoFocus)
        self.cut_end_target.setFocusPolicy(Qt.NoFocus)
        self.cut_start_target.setCheckable(True)
        self.cut_end_target.setCheckable(True)
        self.cut_start_target.setChecked(True)
        self.cut_start_target.clicked.connect(lambda: self.set_cut_click_target("start"))
        self.cut_end_target.clicked.connect(lambda: self.set_cut_click_target("end"))
        self.cut_start.valueChanged.connect(lambda _value: self.sync_cut_selection())
        self.cut_end.valueChanged.connect(lambda _value: self.sync_cut_selection())
        self.window_duration.valueChanged.connect(lambda _value: self.window_duration_changed())
        self.cut_length = QLabel("0.000 s")
        for index, (label, widget) in enumerate(
            [
                ("Cut start", self.cut_start),
                ("Cut end", self.cut_end),
                ("Cut length", self.cut_length),
                ("Fade out", self.fade_out),
                ("Fade in", self.fade_in),
                ("Action", self.cut_action),
                ("Selected gain", self.selected_gain),
                ("Voice keep", self.voice_keep_original),
                ("Edge src", self.edge_source),
            ]
        ):
            work_cut_layout.addWidget(QLabel(label), index // 3, (index % 3) * 2)
            work_cut_layout.addWidget(widget, index // 3, (index % 3) * 2 + 1)
        work_cut_layout.addWidget(self.cut_start_target, 3, 0, 1, 2)
        work_cut_layout.addWidget(self.cut_end_target, 3, 2, 1, 2)
        self.play_selected_button = QPushButton("Play selected area")
        self.play_selected_button.setFocusPolicy(Qt.NoFocus)
        self._decorate_transport_button(self.play_selected_button, "Play Work Clip")
        self.play_selected_button.clicked.connect(self.preview_selected_action)
        work_cut_layout.addWidget(self.play_selected_button, 3, 4, 1, 2)
        apply_cut = QPushButton("Apply to preview")
        apply_cut.setFocusPolicy(Qt.NoFocus)
        apply_cut.clicked.connect(self.apply_selected_action_to_preview)
        work_cut_layout.addWidget(apply_cut, 4, 0, 1, 2)
        reset_preview = QPushButton("Reset preview")
        reset_preview.setFocusPolicy(Qt.NoFocus)
        reset_preview.clicked.connect(self.reset_work_preview)
        work_cut_layout.addWidget(reset_preview, 4, 2, 1, 2)
        self.lock_selection = QCheckBox("Lock selection")
        self.lock_selection.setChecked(True)
        work_cut_layout.addWidget(self.lock_selection, 4, 4, 1, 2)
        work_layout.addWidget(work_cut_group)
        work_transport = QHBoxLayout()
        work_transport.addWidget(QLabel("Work transport"))
        focus_work = QPushButton("Focus work")
        focus_work.setFocusPolicy(Qt.NoFocus)
        focus_work.clicked.connect(lambda: self.set_focus("work"))
        work_transport.addWidget(focus_work)
        for text, callback in [
            ("Play Work Clip", self.play_work_clip),
            ("Play Work Mix", self.play_work_mix),
            ("Pause", self.pause_playback),
            ("Stop", self.stop_playback),
        ]:
            button = QPushButton(text)
            button.setFocusPolicy(Qt.NoFocus)
            self._decorate_transport_button(button, text)
            button.clicked.connect(callback)
            work_transport.addWidget(button)
        self.work_loop = QCheckBox("Loop work")
        self.work_loop.setChecked(True)
        work_transport.addWidget(self.work_loop)
        work_layout.addLayout(work_transport)
        for index, lane in enumerate(self.project.active_repair.lanes):
            lane.name = self.clip_bank_name(index)
            lane_widget = LaneWidget(lane)
            lane_widget.playRequested.connect(self.play_lane)
            lane_widget.stopRequested.connect(self.stop_lane)
            lane_widget.alignRequested.connect(self.auto_align_lane)
            lane_widget.fitLengthRequested.connect(self.fit_lane_length)
            lane_widget.resetRequested.connect(self.reset_lane)
            lane_widget.timelineDragStarted.connect(lambda _widget: self.push_undo())
            lane_widget.settingsChanged.connect(lambda _widget: self.update_lane_timeline_controls())
            lane_widget.focused.connect(lambda widget: self.set_focus(widget.lane.name))
            self.lane_widgets.append(lane_widget)
            work_layout.addWidget(lane_widget)
        buttons = QHBoxLayout()
        for text, callback in [
            ("Load Main", self.open_main_track),
            ("Reload files", self.reload_files),
            ("Undo", self.undo),
            ("Redo", self.redo),
            ("Export Work WAV", self.export_work_mix),
            ("Export Bed+Work WAV", self.export_bed_work_mix),
            ("Export Full Main WAV", self.export_full_main_patch),
            ("Export Ready Patch", self.export_ready_patch),
            ("Export Recipe", self.export_recipe),
        ]:
            button = QPushButton(text)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        self.ready_patch_cut_only = QCheckBox("Ready patch cut only")
        self.ready_patch_cut_only.setChecked(False)
        self.ready_patch_cut_only.setToolTip(
            "Off: export the full visible work window. On: export only the selected cut span."
        )
        buttons.addWidget(self.ready_patch_cut_only)
        work_layout.addLayout(buttons)

        ref_panel = QWidget()
        ref_layout = QVBoxLayout(ref_panel)
        self.bed_path = PathEdit("De-voiced bed")
        self.bed_path.pathChanged.connect(self.path_field_changed)
        ref_layout.addWidget(self.bed_path)
        self.bed_preview_mode = QComboBox()
        self.bed_preview_mode.addItem("As loaded / center stem", "as_loaded")
        self.bed_preview_mode.currentIndexChanged.connect(self.path_field_changed)
        self.bed_preview_mode.setVisible(False)
        self.bed_info = QLabel("Bed: not loaded")
        ref_layout.addWidget(self.bed_info)
        self.bed_waveform = WaveformView("De-voiced 5.1 Bed")
        self.bed_waveform.setMaximumHeight(240)
        self.bed_waveform.focused.connect(lambda: self.set_focus("bed"))
        self.bed_waveform.keyPressed.connect(self.handle_key_event)
        self.bed_waveform.markerChanged.connect(self.bed_marker_changed)
        self.bed_waveform.rightClicked.connect(lambda _seconds: None)
        self.bed_waveform.fileDropped.connect(lambda path: self.bed_path.set_path(path))
        ref_layout.addWidget(self.bed_waveform)
        bed_controls = QGroupBox("Bed Selection")
        bed_layout = QGridLayout(bed_controls)
        self.bed_action = QComboBox()
        self.bed_action.addItems(
            [
                "Keep selected area (gain only / no cut)",
                "Silence selected area (keep timing)",
                "Delete selected area (shorten clip)",
                "Fade out/in selected area",
                "Stretch pre-cut edge across selected area",
                "Repeat pre-cut edge across selected area",
            ]
        )
        self.bed_gain = QDoubleSpinBox()
        self.bed_gain.setRange(-24.0, 24.0)
        self.bed_gain.setDecimals(1)
        self.bed_gain.setSuffix(" dB")
        bed_layout.addWidget(QLabel("Action"), 0, 0)
        bed_layout.addWidget(self.bed_action, 0, 1, 1, 3)
        bed_layout.addWidget(QLabel("Gain"), 1, 0)
        bed_layout.addWidget(self.bed_gain, 1, 1)
        apply_bed = QPushButton("Apply to bed preview")
        apply_bed.setFocusPolicy(Qt.NoFocus)
        apply_bed.clicked.connect(self.apply_bed_action_to_preview)
        bed_layout.addWidget(apply_bed, 1, 2)
        reset_bed = QPushButton("Reset bed preview")
        reset_bed.setFocusPolicy(Qt.NoFocus)
        reset_bed.clicked.connect(self.reset_bed_preview)
        bed_layout.addWidget(reset_bed, 1, 3)
        ref_layout.addWidget(bed_controls)
        bed_transport = QHBoxLayout()
        for text, callback in [
            ("Play Bed", self.play_bed_clip),
            ("Play Bed + Work Mix", self.play_bed_with_work_mix),
            ("Pause", self.pause_playback),
            ("Stop", self.stop_playback),
        ]:
            button = QPushButton(text)
            button.setFocusPolicy(Qt.NoFocus)
            self._decorate_transport_button(button, text)
            button.clicked.connect(callback)
            bed_transport.addWidget(button)
        self.bed_loop = QCheckBox("Loop bed")
        self.bed_loop.setChecked(True)
        bed_transport.addWidget(self.bed_loop)
        self.bed_marker_lock = QCheckBox("Lock bed marker")
        self.bed_marker_lock.setChecked(False)
        bed_transport.addWidget(self.bed_marker_lock)
        ref_layout.addLayout(bed_transport)

        self.reference_path = PathEdit("Reference")
        self.reference_path.pathChanged.connect(self.path_field_changed)
        ref_layout.addWidget(self.reference_path)
        self.reference_waveform = WaveformView("Reference")
        self.reference_waveform.setMaximumHeight(220)
        self.reference_waveform.focused.connect(lambda: self.set_focus("reference"))
        self.reference_waveform.keyPressed.connect(self.handle_key_event)
        self.reference_waveform.fileDropped.connect(lambda path: self.reference_path.set_path(path))
        ref_layout.addWidget(self.reference_waveform)
        ref_buttons = QHBoxLayout()
        focus_ref = QPushButton("Focus reference")
        focus_ref.setFocusPolicy(Qt.NoFocus)
        focus_ref.clicked.connect(lambda: self.set_focus("reference"))
        ref_buttons.addWidget(focus_ref)
        play_ref = QPushButton("Play Reference")
        play_ref.setFocusPolicy(Qt.NoFocus)
        self._decorate_transport_button(play_ref, "Play Reference")
        play_ref.clicked.connect(self.play_reference)
        ref_buttons.addWidget(play_ref)
        pause_ref = QPushButton("Pause")
        pause_ref.setFocusPolicy(Qt.NoFocus)
        self._decorate_transport_button(pause_ref, "Pause")
        pause_ref.clicked.connect(self.pause_playback)
        ref_buttons.addWidget(pause_ref)
        stop_ref = QPushButton("Stop")
        stop_ref.setFocusPolicy(Qt.NoFocus)
        self._decorate_transport_button(stop_ref, "Stop")
        stop_ref.clicked.connect(self.stop_playback)
        ref_buttons.addWidget(stop_ref)
        self.reference_loop = QCheckBox("Loop reference")
        self.reference_loop.setChecked(True)
        ref_buttons.addWidget(self.reference_loop)
        ref_layout.addLayout(ref_buttons)
        self.video_path = PathEdit("Video")
        self.video_path.pathChanged.connect(self.path_field_changed)
        ref_layout.addWidget(self.video_path)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(150)
        self.video_widget.setMaximumHeight(220)
        ref_layout.addWidget(self.video_widget)
        video_buttons = QHBoxLayout()
        self.video_follow = QCheckBox("Follow audio")
        self.video_follow.setChecked(True)
        video_buttons.addWidget(self.video_follow)
        self.video_status = QLabel("Video: not loaded")
        video_buttons.addWidget(self.video_status, 1)
        ref_layout.addLayout(video_buttons)
        self.video_audio = QAudioOutput(self)
        self.video_audio.setVolume(0.0)
        self.video_player = QMediaPlayer(self)
        self.video_player.setAudioOutput(self.video_audio)
        self.video_player.setVideoOutput(self.video_widget)
        ref_layout.addStretch(1)

        work_scroll = QScrollArea()
        work_scroll.setWidgetResizable(True)
        work_scroll.setFrameShape(QFrame.NoFrame)
        work_scroll.setWidget(work_panel)
        splitter.addWidget(work_scroll)
        splitter.addWidget(ref_panel)
        splitter.setSizes([760, 420])
        root.addWidget(splitter, 5)
        self.status = QLabel("Ready")
        self.status.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status)
        self.setCentralWidget(central)

    def _seconds_spin(self, maximum: float = 9999.0, step: float = 0.001) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setSuffix(" s")
        return spin

    def _decorate_transport_button(self, button: QPushButton, text: str) -> None:
        icon_name = None
        if "Play" in text or "Preview" in text:
            icon_name = QStyle.StandardPixmap.SP_MediaPlay
        elif "Pause" in text:
            icon_name = QStyle.StandardPixmap.SP_MediaPause
        elif "Stop" in text:
            icon_name = QStyle.StandardPixmap.SP_MediaStop
        if icon_name is not None:
            button.setIcon(self.style().standardIcon(icon_name))

    def clip_bank_name(self, index: int) -> str:
        return f"Clip Bank {index + 1:02d}"

    def update_play_selected_button_icon(self, playing: bool) -> None:
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_selected_button.setIcon(self.style().standardIcon(icon))

    def _load_project_into_ui(self) -> None:
        self._syncing_ui = True
        self.project.ensure_default_lanes()
        self.main_path.set_path(self.project.main_track)
        self.bed_path.set_path(self.project.bed_track)
        bed_mode_index = self.bed_preview_mode.findData(self.project.bed_preview_mode)
        self.bed_preview_mode.setCurrentIndex(max(bed_mode_index, 0))
        self.video_path.set_path(self.project.video_track)
        if self.project.reference_tracks:
            self.reference_path.set_path(self.project.reference_tracks[0])
        region = self.project.active_repair
        self.marker.setValue(region.marker_seconds)
        self.window_duration.setValue(region.work_window_seconds)
        work_start = region.work_start_seconds
        self.cut_start.setValue(max(region.cut_start_seconds - work_start, 0.0))
        self.cut_end.setValue(max(region.cut_end_seconds - work_start, 0.0))
        self.snap_ms.setValue(region.snap_ms)
        self.selected_gain.setValue(region.selected_gain_db)
        self.voice_keep_original.setValue(region.voice_keep_original_percent)
        self.edge_source.setValue(region.edge_source_seconds)
        self.fade_in.setValue(region.edge.fade_in_seconds)
        self.fade_out.setValue(region.edge.fade_out_seconds)
        for index, widget in enumerate(self.lane_widgets):
            if index < len(self.project.active_repair.lanes):
                widget.lane = self.project.active_repair.lanes[index]
                widget.lane.name = self.clip_bank_name(index)
                widget.setTitle(widget.lane.name)
            widget.from_model()
        self._syncing_ui = False
        self.update_cut_length_label()
        self.update_lane_timeline_controls()
        self.update_focus_visuals()

    def _ui_to_project(self) -> None:
        if self._syncing_ui:
            return
        self.project.main_track = str(self.main_path.path() or "")
        self.project.bed_track = str(self.bed_path.path() or "")
        self.project.bed_preview_mode = str(self.bed_preview_mode.currentData() or "as_loaded")
        self.project.video_track = str(self.video_path.path() or "")
        reference = self.reference_path.path()
        self.project.reference_tracks = [str(reference)] if reference else []
        region = self.project.active_repair
        region.marker_seconds = self.marker.value()
        region.work_window_seconds = self.window_duration.value()
        work_start = region.work_start_seconds
        region.cut_start_seconds = work_start + self.cut_start.value()
        region.cut_end_seconds = work_start + self.cut_end.value()
        region.snap_ms = self.snap_ms.value()
        region.selected_gain_db = self.selected_gain.value()
        region.voice_keep_original_percent = self.voice_keep_original.value()
        region.edge_source_seconds = self.edge_source.value()
        region.edge.fade_in_seconds = self.fade_in.value()
        region.edge.fade_out_seconds = self.fade_out.value()
        for widget in self.lane_widgets:
            widget.to_model()

    def apply_default_lane_locks(self) -> None:
        """Lock aligned clips by default when a real work selection is loaded."""

        has_selection = (self.cut_end.value() - self.cut_start.value()) > 0.001
        for widget in self.lane_widgets:
            has_clip = bool(widget.path.path())
            aligned_to_selection = abs(widget.offset.value()) < 0.0005
            widget.locked.setChecked(bool(has_selection and has_clip and aligned_to_selection))

    def snapshot_state(self) -> tuple[RepairProject, AudioBuffer | None, AudioBuffer | None, float]:
        work_audio = None
        if self._work_audio is not None:
            work_audio = AudioBuffer(samples=self._work_audio.samples.copy(), sample_rate=self._work_audio.sample_rate)
        bed_audio = None
        if self._bed_work_audio is not None:
            bed_audio = AudioBuffer(samples=self._bed_work_audio.samples.copy(), sample_rate=self._bed_work_audio.sample_rate)
        return deepcopy(self.project), work_audio, bed_audio, self._work_start_marker_seconds

    def push_undo(self) -> None:
        if self._syncing_ui or self._restoring_history:
            return
        self._ui_to_project()
        self._undo_stack.append(self.snapshot_state())
        self._redo_stack.clear()
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def restore_state(self, state: tuple[RepairProject, AudioBuffer | None, AudioBuffer | None, float]) -> None:
        self._restoring_history = True
        try:
            self.project = deepcopy(state[0])
            self._work_start_marker_seconds = state[3]
            self._load_project_into_ui()
            self.timeline.set_marker(self.project.active_repair.marker_seconds)
            if state[1] is not None:
                self._work_audio = AudioBuffer(samples=state[1].samples.copy(), sample_rate=state[1].sample_rate)
                wmin, wmax = waveform_overview(self._work_audio, points=900)
                self.work_waveform.set_waveform(wmin, wmax, self._work_audio.duration_seconds)
                self.work_waveform.set_selection(self.cut_start.value(), self.cut_end.value())
                self.work_waveform.set_marker(self.cut_start.value())
                self.work_waveform.set_start_marker(self._work_start_marker_seconds)
                self.work_waveform.set_playhead(0.0)
            else:
                self.update_work_clip_view()
                self._work_start_marker_seconds = state[3]
                self.work_waveform.set_start_marker(self._work_start_marker_seconds)
            if state[2] is not None:
                self._bed_work_audio = AudioBuffer(samples=state[2].samples.copy(), sample_rate=state[2].sample_rate)
                bmin, bmax = waveform_overview(self._bed_work_audio, points=900)
                self.bed_waveform.set_waveform(bmin, bmax, self._bed_work_audio.duration_seconds)
                self.bed_waveform.set_selection(self.cut_start.value(), self.cut_end.value())
                self.bed_waveform.set_marker(self._work_start_marker_seconds)
                self.bed_waveform.set_start_marker(self._work_start_marker_seconds)
            self.update_work_range_label()
        finally:
            self._restoring_history = False

    def undo(self) -> None:
        if not self._undo_stack:
            self.status.setText("Nothing to undo.")
            return
        self._redo_stack.append(self.snapshot_state())
        self.restore_state(self._undo_stack.pop())
        self.status.setText("Undo.")

    def redo(self) -> None:
        if not self._redo_stack:
            self.status.setText("Nothing to redo.")
            return
        self._undo_stack.append(self.snapshot_state())
        self.restore_state(self._redo_stack.pop())
        self.status.setText("Redo.")

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.handle_key_event(event)

    def handle_key_event(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key_Z:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.redo()
                else:
                    self.undo()
                event.accept()
                return
            if event.key() == Qt.Key_Y:
                self.redo()
                event.accept()
                return
        if event.key() == Qt.Key_Space:
            self.play_or_pause_focused()
            event.accept()
            return
        if event.key() in {Qt.Key_Left, Qt.Key_Right}:
            direction = -1 if event.key() == Qt.Key_Left else 1
            step = self.snap_ms.value() / 1000.0
            if self.active_focus == "main":
                if self.marker_lock.isChecked():
                    self.status.setText("Top marker is locked; unlock it before nudging the main marker.")
                    event.accept()
                    return
                self.marker.setValue(max(self.marker.value() + direction * step, 0.0))
                self.marker_changed(self.marker.value())
                event.accept()
                return
            for lane_widget in self.lane_widgets:
                if self.active_focus == lane_widget.lane.name:
                    self.push_undo()
                    lane_widget.offset.setValue(lane_widget.offset.value() + direction * step)
                    lane_widget.to_model()
                    self.status.setText(f"Moved {lane_widget.lane.name} offset to {lane_widget.offset.value():.3f}s")
                    event.accept()
                    return
            target = self.cut_start if self.cut_click_target == "start" else self.cut_end
            self.push_undo()
            target.setValue(max(target.value() + direction * step, 0.0))
            if self.cut_start.value() > self.cut_end.value():
                if target is self.cut_start:
                    self.cut_end.setValue(self.cut_start.value())
                else:
                    self.cut_start.setValue(self.cut_end.value())
            self.sync_cut_selection()
            event.accept()
            return
        event.ignore()

    def set_focus(self, focus: str) -> None:
        self.active_focus = focus
        if focus == "main":
            self.focus_label.setText("Focus: Main episode marker")
            self.timeline.focus_for_keys()
        elif focus == "work":
            self.focus_label.setText(f"Focus: Work cut {self.cut_click_target}")
            self.work_waveform.focus_for_keys()
        elif focus == "bed":
            self.focus_label.setText("Focus: De-voiced bed")
            self.bed_waveform.focus_for_keys()
        elif focus == "reference":
            self.focus_label.setText("Focus: Reference")
            self.reference_waveform.focus_for_keys()
        else:
            self.focus_label.setText(f"Focus: {focus}")
        self.update_focus_visuals()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open repair project", "work/repair_projects", "Repair JSON (*.json)")
        if not path:
            return
        self.project = load_project(Path(path))
        self._loaded_project_template = deepcopy(self.project)
        self.project_path = Path(path)
        self._load_project_into_ui()
        self.refresh_audio()

    def save_project(self) -> None:
        if self.project_path is None:
            self.save_project_as()
            return
        self._ui_to_project()
        save_project(self.project_path, self.project)
        self.status.setText(f"Saved project: {self.project_path}")

    def save_open_project_quietly(self) -> None:
        """Persist the open project after deterministic exports."""

        if self.project_path is None:
            return
        self._ui_to_project()
        save_project(self.project_path, self.project)

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save repair project", "work/repair_projects/repair.json", "Repair JSON (*.json)")
        if not path:
            return
        self.project_path = Path(path)
        self.save_project()

    def open_main_track(self) -> None:
        self.main_path.browse()
        self.refresh_audio()

    def open_bed_track(self) -> None:
        self.bed_path.browse()
        self.refresh_audio()

    def open_video_track(self) -> None:
        self.video_path.browse()
        self.refresh_audio()

    def open_reference_track(self) -> None:
        self.reference_path.browse()
        self.refresh_audio()

    def path_field_changed(self) -> None:
        if self._syncing_ui:
            return
        self.push_undo()
        self.refresh_audio()

    def refresh_audio(self) -> None:
        self._ui_to_project()
        main = self.main_path.path()
        if main and main.exists():
            if self.main_audio is None or self._main_loaded_path != main:
                self.status.setText(f"Loading main audio: {main}")
                self.main_audio = load_audio(main)
                self._main_loaded_path = main
                self._main_overview = waveform_overview(self.main_audio, points=2600)
            self.project.sample_rate = self.main_audio.sample_rate
            self.project.channels = self.main_audio.channels
            assert self._main_overview is not None
            mins, maxs = self._main_overview
            self.timeline.set_waveform(mins, maxs, self.main_audio.duration_seconds)
            self.timeline.set_marker(self.project.active_repair.marker_seconds)
            self.update_work_clip_view()
        bed = self.bed_path.path()
        if bed and bed.exists():
            bed_mode = "as_loaded"
            if self.bed_audio is None or self._bed_loaded_path != bed or self._bed_loaded_mode != bed_mode:
                source_info = sf.info(bed)
                self.bed_audio = load_audio(bed)
                self._bed_loaded_path = bed
                self._bed_loaded_mode = bed_mode
                self._bed_overview = waveform_overview(self.bed_audio, points=900)
                self.bed_info.setText(
                    f"Bed: source {source_info.channels} ch, preview {self.bed_audio.channels} ch."
                )
            self.update_bed_clip_view()
        video = self.video_path.path()
        if video and video.exists() and self._video_loaded_path != video:
            self.video_player.setSource(QUrl.fromLocalFile(str(video.resolve())))
            self._video_loaded_path = video
            self.video_status.setText(f"Video: {video.name}")
            self.sync_video_to_scope("main", self.marker.value(), play=False)
        ref = self.reference_path.path()
        if ref and ref.exists():
            if self.reference_audio is None or self._reference_loaded_path != ref:
                self.reference_audio = load_audio(ref)
                self._reference_loaded_path = ref
                self._reference_overview = waveform_overview(self.reference_audio, points=900)
            assert self._reference_overview is not None
            rmin, rmax = self._reference_overview
            self.reference_waveform.set_waveform(rmin, rmax, self.reference_audio.duration_seconds)
        self.status.setText("Audio refreshed")

    def update_work_clip_view(self) -> None:
        if self.main_audio is None:
            return
        self._syncing_ui = True
        region = self.project.active_repair
        self._work_audio = crop(self.main_audio, region.work_start_seconds, region.work_window_seconds)
        wmin, wmax = waveform_overview(self._work_audio, points=900)
        self.work_waveform.set_waveform(wmin, wmax, self._work_audio.duration_seconds)
        local_start = max(region.cut_start_seconds - region.work_start_seconds, 0.0)
        local_end = max(region.cut_end_seconds - region.work_start_seconds, local_start)
        self.cut_start.setValue(local_start)
        self.cut_end.setValue(local_end)
        self.work_waveform.set_marker(local_start)
        self.work_waveform.set_selection(local_start, local_end)
        self._work_start_marker_seconds = 0.0
        self.work_waveform.set_start_marker(0.0)
        self.work_waveform.set_playhead(0.0)
        self.update_cut_length_label()
        self.update_lane_timeline_controls()
        self._syncing_ui = False
        self.update_work_range_label()
        self.update_bed_clip_view()

    def update_bed_clip_view(self) -> None:
        if self.bed_audio is None:
            return
        region = self.project.active_repair
        self._bed_work_audio = crop(self.bed_audio, region.work_start_seconds, region.work_window_seconds)
        bmin, bmax = waveform_overview(self._bed_work_audio, points=900)
        self.bed_waveform.set_waveform(bmin, bmax, self._bed_work_audio.duration_seconds)
        self.bed_waveform.set_marker(self._work_start_marker_seconds)
        self.bed_waveform.set_start_marker(self._work_start_marker_seconds)
        self.bed_waveform.set_selection(self.cut_start.value(), self.cut_end.value())
        self.bed_waveform.set_playhead(0.0)

    def update_work_range_label(self) -> None:
        region = self.project.active_repair
        self.work_range.setText(
            "Work clip: "
            f"{region.work_start_seconds:.3f}s -> {region.work_end_seconds:.3f}s "
            "from full main. Cut controls below are LOCAL work-clip seconds."
        )

    def reload_files(self) -> None:
        """Reset the session to loaded defaults and reread audio from disk."""

        self.project = deepcopy(self._loaded_project_template)
        self._load_project_into_ui()
        self.main_audio = None
        self.bed_audio = None
        self.reference_audio = None
        self._main_loaded_path = None
        self._bed_loaded_path = None
        self._bed_loaded_mode = ""
        self._reference_loaded_path = None
        self._main_overview = None
        self._bed_overview = None
        self._reference_overview = None
        self._bed_work_audio = None
        self.refresh_audio()

    def marker_changed(self, seconds: float) -> None:
        if self._syncing_ui:
            return
        if self.marker_lock.isChecked():
            self.timeline.set_marker(self.marker.value())
            self.sync_video_to_scope("main", self.marker.value(), play=False)
            return
        self.set_focus("main")
        snapped = self._snap(seconds)
        self.push_undo()
        self.marker.setValue(snapped)
        old_start = self.project.active_repair.work_start_seconds
        self.project.active_repair.marker_seconds = snapped
        new_start = self.project.active_repair.work_start_seconds
        delta = new_start - old_start
        self.project.active_repair.cut_start_seconds += delta
        self.project.active_repair.cut_end_seconds += delta
        self.cut_start.setValue(max(self.project.active_repair.cut_start_seconds - new_start, 0.0))
        self.cut_end.setValue(max(self.project.active_repair.cut_end_seconds - new_start, 0.0))
        self.refresh_audio()
        self.sync_video_to_scope("main", snapped, play=False)

    def window_duration_changed(self) -> None:
        """Recrop the work clip immediately when the window size changes."""

        if self._syncing_ui:
            return
        self.push_undo()
        region = self.project.active_repair
        cut_start_abs = region.cut_start_seconds
        cut_end_abs = region.cut_end_seconds
        region.marker_seconds = self.marker.value()
        region.work_window_seconds = self.window_duration.value()
        work_start = region.work_start_seconds
        self._syncing_ui = True
        self.cut_start.setValue(max(cut_start_abs - work_start, 0.0))
        self.cut_end.setValue(max(cut_end_abs - work_start, 0.0))
        self._syncing_ui = False
        self._ui_to_project()
        self.update_work_clip_view()
        self.status.setText(f"Work clip window set to {self.window_duration.value():.3f}s.")

    def work_marker_changed(self, seconds: float) -> None:
        if self._syncing_ui:
            return
        self.set_focus("work")
        snapped = self._snap(seconds)
        if self.lock_selection.isChecked():
            self.push_undo()
            self._work_start_marker_seconds = snapped
            self.work_waveform.set_start_marker(snapped)
            self.work_waveform.set_marker(self.cut_start.value())
            self.status.setText(f"Set work start marker to {snapped:.3f}s")
            self.sync_video_to_scope("work", snapped, play=False)
            return
        self.push_undo()
        self.cut_start.setValue(snapped)
        if self.cut_end.value() < snapped:
            self.cut_end.setValue(snapped)
        self._ui_to_project()
        self.sync_cut_selection(marker=snapped)
        self.status.setText(
            f"Set work cut start to {snapped:.3f}s "
            f"(episode {self.project.active_repair.work_start_seconds + snapped:.3f}s)"
        )

    def work_end_marker_changed(self, seconds: float) -> None:
        if self._syncing_ui:
            return
        self.set_focus("work")
        if self.lock_selection.isChecked():
            self.status.setText("Selection is locked; right-click did not change the cut end.")
            self.work_waveform.set_marker(self.cut_start.value())
            return
        snapped = self._snap(seconds)
        self.push_undo()
        self.cut_end.setValue(snapped)
        if self.cut_start.value() > snapped:
            self.cut_start.setValue(snapped)
        self._ui_to_project()
        self.sync_cut_selection(marker=snapped)
        self.status.setText(
            f"Set work cut end to {snapped:.3f}s "
            f"(episode {self.project.active_repair.work_start_seconds + snapped:.3f}s)"
        )

    def work_selection_dragged(self, start: float, end: float) -> None:
        self.set_focus("work")
        if self.lock_selection.isChecked():
            self.status.setText("Selection is locked; drag did not change the selected region.")
            self.work_waveform.set_selection(self.cut_start.value(), self.cut_end.value())
            return
        start = self._snap(start)
        end = self._snap(end)
        if end < start:
            start, end = end, start
        self.push_undo()
        self.cut_start.setValue(start)
        self.cut_end.setValue(end)
        self.sync_cut_selection(marker=start)
        self.status.setText(
            f"Selected work clip region {start:.3f}s -> {end:.3f}s "
            f"(episode {self.project.active_repair.work_start_seconds + start:.3f}s"
            f" -> {self.project.active_repair.work_start_seconds + end:.3f}s)"
        )

    def set_cut_click_target(self, target: str) -> None:
        self.set_focus("work")
        self.cut_click_target = target
        self.cut_start_target.setChecked(target == "start")
        self.cut_end_target.setChecked(target == "end")
        self.focus_label.setText(f"Focus: Work cut {target}")

    def sync_cut_selection(self, marker: float | None = None) -> None:
        if self._syncing_ui:
            return
        self._ui_to_project()
        self.work_waveform.set_selection(self.cut_start.value(), self.cut_end.value())
        self.work_waveform.set_marker(self.cut_start.value() if marker is None else marker)
        self.update_cut_length_label()
        self.update_lane_timeline_controls()

    def update_cut_length_label(self) -> None:
        length = max(self.cut_end.value() - self.cut_start.value(), 0.0)
        self.cut_length.setText(f"{length:.3f} s")

    def update_lane_timeline_controls(self) -> None:
        work_duration = self._work_audio.duration_seconds if self._work_audio is not None else self.window_duration.value()
        cut_start = self.cut_start.value()
        cut_duration = max(self.cut_end.value() - self.cut_start.value(), 0.001)
        for lane_widget in self.lane_widgets:
            lane_widget.to_model()
            clip_duration = self.lane_clip_duration(lane_widget.lane)
            lane_widget.configure_timeline(
                work_duration=work_duration,
                cut_start=cut_start,
                cut_duration=cut_duration,
                clip_duration=clip_duration,
            )

    def lane_clip_duration(self, lane: ClipLane) -> float:
        if not lane.path:
            return max(self.cut_end.value() - self.cut_start.value(), 0.001)
        try:
            info = sf.info(lane.path)
            duration = info.frames / info.samplerate if info.samplerate else 0.0
            speed = max(lane.speed_percent, 1.0) / 100.0
            return max(duration / speed, 0.001)
        except Exception:
            return max(self.cut_end.value() - self.cut_start.value(), 0.001)

    def update_focus_visuals(self) -> None:
        self.timeline.set_active(self.active_focus == "main")
        self.work_waveform.set_active(self.active_focus == "work")
        self.bed_waveform.set_active(self.active_focus == "bed")
        self.reference_waveform.set_active(self.active_focus == "reference")
        for widget in self.lane_widgets:
            widget.set_active(self.active_focus == widget.lane.name)

    def _snap(self, seconds: float) -> float:
        step = self.snap_ms.value() / 1000.0
        return round(seconds / step) * step if step else seconds

    def selection_action_code(self) -> str:
        return self.action_code_from_text(self.cut_action.currentText())

    def bed_action_code(self) -> str:
        return self.action_code_from_text(self.bed_action.currentText())

    def action_code_from_text(self, text: str) -> str:
        if text.startswith("Delete"):
            return "delete"
        if text.startswith("Fade"):
            return "fade_silence"
        if text.startswith("Rubberband"):
            return "rubberband_bridge"
        if text.startswith("Stretch"):
            return "stretch_edge"
        if text.startswith("Repeat"):
            return "repeat_edge"
        if text.startswith("Blend"):
            return "blend_edges"
        if text.startswith("Interpolate"):
            return "interp_ambience"
        if text.startswith("Keep"):
            return "keep"
        if text.startswith("Remove voice"):
            return "remove_voice"
        return "silence"

    def current_work_audio(self) -> AudioBuffer | None:
        """Return the visible work clip, loading the source preview if needed."""

        if self._work_audio is None:
            self.refresh_audio()
        return self._work_audio

    def preview_selected_action(self) -> None:
        """Play only the currently selected work region from the visible preview.

        This deliberately does not render a fresh action. The selected action
        becomes audible here only after `Apply to preview` has changed the
        visible work clip.
        """

        self._ui_to_project()
        work = self.current_work_audio()
        if work is None:
            self._error("Load a main audio file first.")
            return
        try:
            start = min(max(self.cut_start.value(), 0.0), work.duration_seconds)
            end = min(max(self.cut_end.value(), start), work.duration_seconds)
            duration = end - start
            if duration <= 0:
                self._error("Select a non-empty work area first.")
                return
            start_frame = time_to_frame(start, work.sample_rate)
            end_frame = time_to_frame(end, work.sample_rate)
            signature = (id(work), start_frame, end_frame)
            if self._active_play_scope == "work" and self._last_play_kind == "selected":
                if self._last_selected_signature == signature and (self.playhead_timer.isActive() or self._playback_paused):
                    self.pause_playback()
                    return
                self.stop_playback()
            selected = crop(work, start, duration)
            self._last_selected_signature = signature
            self.play_buffer(selected, "work", display_start_seconds=start, playback_start_seconds=0.0, kind="selected")
        except Exception as exc:
            self._error(str(exc))

    def apply_selected_action_to_preview(self) -> None:
        """Apply the selected action visually to the work preview only."""

        self._ui_to_project()
        if self.main_audio is None:
            self.refresh_audio()
        if self.main_audio is None:
            self._error("Load a main audio file first.")
            return
        if self.selection_action_code() == "remove_voice":
            self.apply_voice_removal_to_preview()
            return
        try:
            self.push_undo()
            preview = build_work_action_preview(
                self.project,
                self.main_audio,
                self.selection_action_code(),
                selected_gain_db=self.selected_gain.value(),
            )
            self._work_audio = preview
            wmin, wmax = waveform_overview(preview, points=900)
            self.work_waveform.set_waveform(wmin, wmax, preview.duration_seconds)
            self.work_waveform.set_selection(self.cut_start.value(), self.cut_end.value())
            self.work_waveform.set_marker(self.cut_start.value())
            self._work_start_marker_seconds = min(self._work_start_marker_seconds, preview.duration_seconds)
            self.work_waveform.set_start_marker(self._work_start_marker_seconds)
            self.work_waveform.set_playhead(0.0)
            self.record_repair_action("work", self.selection_action_code(), self.selected_gain.value())
            self.status.setText("Applied selected action to visible preview only. Export/recipe still uses project settings.")
        except Exception as exc:
            self._error(str(exc))

    def apply_voice_removal_to_preview(self) -> None:
        """Run audio-separator on the selected work region and keep the Instrumental stem."""

        work = self.current_work_audio()
        if work is None:
            self._error("Load a main audio file first.")
            return
        start = min(max(self.cut_start.value(), 0.0), work.duration_seconds)
        end = min(max(self.cut_end.value(), start), work.duration_seconds)
        duration = end - start
        if duration <= 0:
            self._error("Select a non-empty work area first.")
            return
        separator = self.find_audio_separator()
        if separator is None:
            self._error(
                "audio-separator was not found. Install it in .venv-separation "
                "or launch the tool from an environment where audio-separator is on PATH."
            )
            return
        try:
            self.push_undo()
            out_dir = self.voice_extract_output_dir()
            input_wav = out_dir / "selected_region_input.wav"
            context_wav = out_dir / "separator_context_input.wav"
            selected = crop(work, start, duration)
            save_audio(input_wav, selected)
            separator_context = self.separator_context_audio(work)
            save_audio(context_wav, separator_context)
            sep_dir = out_dir / "separator"
            sep_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "audio_separator.log"
            cmd = [
                str(separator),
                str(context_wav),
                "--output_dir",
                str(sep_dir),
                "--output_format",
                "WAV",
                "--model_filename",
                DEFAULT_VOICE_REMOVAL_MODEL,
                "--model_file_dir",
                "work/models/audio-separator",
                "--single_stem",
                "Instrumental",
                "--sample_rate",
                str(work.sample_rate),
            ]
            self.status.setText("Running voice removal on selected area...")
            QApplication.processEvents()
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log_path.write_text(result.stdout or "", encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"audio-separator failed with exit code {result.returncode}. Log: {log_path}")
            if "Failed to process file" in (result.stdout or ""):
                raise RuntimeError(f"audio-separator reported an internal processing failure. Log: {log_path}")
            instrumental_path = self.find_separator_stem(sep_dir, "Instrumental", log_path)
            instrumental = load_audio(instrumental_path, target_sample_rate=work.sample_rate, channels=work.channels)
            replacement_audio = crop(instrumental, start, duration)
            start_frame = time_to_frame(start, work.sample_rate)
            end_frame = min(start_frame + time_to_frame(duration, work.sample_rate), work.frames)
            target_frames = end_frame - start_frame
            replacement = replacement_audio.samples[:, :target_frames]
            if replacement.shape[1] < target_frames:
                replacement = np.pad(replacement, ((0, 0), (0, target_frames - replacement.shape[1])))
            original = work.samples[:, start_frame:end_frame]
            keep_ratio = self.voice_keep_original.value() / 100.0
            if keep_ratio > 0.0:
                replacement = (
                    replacement[:, :target_frames] * (1.0 - keep_ratio)
                    + original[:, :target_frames] * keep_ratio
                )
            if self.selected_gain.value():
                replacement = replacement * db_to_linear(self.selected_gain.value())
            preview_samples = work.samples.copy()
            preview_samples[:, start_frame:end_frame] = replacement
            self._work_audio = AudioBuffer(samples=preview_samples, sample_rate=work.sample_rate)
            wmin, wmax = waveform_overview(self._work_audio, points=900)
            self.work_waveform.set_waveform(wmin, wmax, self._work_audio.duration_seconds)
            self.work_waveform.set_selection(self.cut_start.value(), self.cut_end.value())
            self.work_waveform.set_marker(self.cut_start.value())
            self.work_waveform.set_start_marker(self._work_start_marker_seconds)
            self.work_waveform.set_playhead(0.0)
            manifest = {
                "action": "remove_voice_selected_area",
                "model": DEFAULT_VOICE_REMOVAL_MODEL,
                "separator": str(separator),
                "command": cmd,
                "input_wav": str(input_wav),
                "separator_context_wav": str(context_wav),
                "separator_log": str(log_path),
                "instrumental_wav": str(instrumental_path),
                "separator_context_duration_seconds": separator_context.duration_seconds,
                "work_start_seconds": self.project.active_repair.work_start_seconds,
                "selection_local_start_seconds": start,
                "selection_local_end_seconds": end,
                "selection_episode_start_seconds": self.project.active_repair.work_start_seconds + start,
                "selection_episode_end_seconds": self.project.active_repair.work_start_seconds + end,
                "selected_gain_db": self.selected_gain.value(),
                "voice_keep_original_percent": self.voice_keep_original.value(),
            }
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.record_repair_action(
                "work",
                "remove_voice",
                self.selected_gain.value(),
                notes=f"voice removal outputs: {out_dir}",
            )
            self.status.setText(f"Voice removal applied to preview. Files: {out_dir}")
        except subprocess.CalledProcessError as exc:
            self._error(f"audio-separator failed with exit code {exc.returncode}.")
        except Exception as exc:
            self._error(str(exc))

    def separator_context_audio(self, work: AudioBuffer) -> AudioBuffer:
        """Return enough context for separator models that dislike tiny clips."""

        minimum_seconds = 20.0
        if work.duration_seconds >= minimum_seconds:
            return work
        target_frames = time_to_frame(minimum_seconds, work.sample_rate)
        padded = np.pad(work.samples, ((0, 0), (0, max(target_frames - work.frames, 0))))
        return AudioBuffer(samples=padded.astype(np.float32, copy=False), sample_rate=work.sample_rate)

    def find_audio_separator(self) -> Path | None:
        for candidate in [
            Path(".venv-separation/bin/audio-separator"),
            Path(".venv-repair/bin/audio-separator"),
        ]:
            if candidate.exists():
                return candidate
        found = shutil.which("audio-separator")
        return Path(found) if found else None

    def voice_extract_output_dir(self) -> Path:
        region = self.project.active_repair
        start_ms = int(round((region.work_start_seconds + self.cut_start.value()) * 1000))
        end_ms = int(round((region.work_start_seconds + self.cut_end.value()) * 1000))
        safe_repair_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in region.repair_id)
        folder = (
            Path("work/review/repair_tool/voice_extract")
            / (self.project.episode or "episode")
            / f"{safe_repair_id}_{start_ms:09d}_{end_ms:09d}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def find_separator_stem(self, folder: Path, stem: str, log_path: Path | None = None) -> Path:
        matches = sorted(folder.glob(f"*{stem}*.wav"))
        if len(matches) != 1:
            log_note = f" Log: {log_path}" if log_path else ""
            raise RuntimeError(f"Expected one {stem} WAV in {folder}, found {len(matches)}.{log_note}")
        return matches[0]

    def reset_work_preview(self) -> None:
        """Restore the visible work waveform to the unmodified source window."""

        self.push_undo()
        self.update_work_clip_view()
        self.status.setText("Reset visible work preview to the source work clip.")

    def play_work_mix(self) -> None:
        self._ui_to_project()
        work = self.current_work_audio()
        if work is None:
            self._error("Load a main audio file first.")
            return
        try:
            mix = self.build_visible_work_mix(work)
            start = min(max(self._work_start_marker_seconds, 0.0), mix.duration_seconds)
            self.play_buffer(mix, "work", start, kind="mix")
        except Exception as exc:
            self._error(str(exc))

    def play_full_main(self) -> None:
        self._ui_to_project()
        if self.main_audio is None:
            self.refresh_audio()
        if self.main_audio is None:
            self._error("Load a main audio file first.")
            return
        try:
            self.play_buffer(self.main_audio, "main", self.marker.value(), kind="main")
        except Exception as exc:
            self._error(str(exc))

    def play_work_clip(self) -> None:
        self._ui_to_project()
        work = self.current_work_audio()
        if work is None:
            self._error("Load a main audio file first.")
            return
        try:
            start = min(max(self._work_start_marker_seconds, 0.0), work.duration_seconds)
            self.play_buffer(work, "work", start, kind="work")
        except Exception as exc:
            self._error(str(exc))

    def current_bed_work_audio(self) -> AudioBuffer | None:
        if self._bed_work_audio is None and self.bed_audio is not None:
            self.update_bed_clip_view()
        return self._bed_work_audio

    def bed_marker_changed(self, seconds: float) -> None:
        if self._syncing_ui:
            return
        self.set_focus("bed")
        if self.bed_marker_lock.isChecked():
            self.bed_waveform.set_marker(self._work_start_marker_seconds)
            self.status.setText("Bed marker is locked; unlock it before moving the bed start marker.")
            return
        snapped = self._snap(seconds)
        self._work_start_marker_seconds = snapped
        self.bed_waveform.set_start_marker(snapped)
        self.work_waveform.set_start_marker(snapped)
        self.status.setText(f"Set shared work/bed start marker to {snapped:.3f}s")

    def play_bed_clip(self) -> None:
        self._ui_to_project()
        bed = self.current_bed_work_audio()
        if bed is None:
            self._error("Load a de-voiced bed audio file first.")
            return
        try:
            start = min(max(self._work_start_marker_seconds, 0.0), bed.duration_seconds)
            self.play_buffer(bed, "bed", start, kind="bed")
        except Exception as exc:
            self._error(str(exc))

    def play_bed_with_work_mix(self) -> None:
        self._ui_to_project()
        bed = self.current_bed_work_audio()
        work = self.current_work_audio()
        if bed is None or work is None:
            self._error("Load both main work audio and de-voiced bed audio first.")
            return
        try:
            combined = self.build_bed_plus_work_mix(bed, work)
            start = min(max(self._work_start_marker_seconds, 0.0), combined.duration_seconds)
            self.play_buffer(combined, "bed", start, kind="bed_mix")
        except Exception as exc:
            self._error(str(exc))

    def apply_bed_action_to_preview(self) -> None:
        bed = self.current_bed_work_audio()
        if bed is None:
            self._error("Load a de-voiced bed audio file first.")
            return
        try:
            self.push_undo()
            cut_start = time_to_frame(self.cut_start.value(), bed.sample_rate)
            cut_end = time_to_frame(self.cut_end.value(), bed.sample_rate)
            self._bed_work_audio = apply_selection_action(
                bed,
                cut_start,
                cut_end,
                self.bed_action_code(),
                selected_gain_db=self.bed_gain.value(),
                fade_in_seconds=self.fade_in.value(),
                fade_out_seconds=self.fade_out.value(),
                edge_source_seconds=self.edge_source.value(),
            )
            bmin, bmax = waveform_overview(self._bed_work_audio, points=900)
            self.bed_waveform.set_waveform(bmin, bmax, self._bed_work_audio.duration_seconds)
            self.bed_waveform.set_selection(self.cut_start.value(), self.cut_end.value())
            self.bed_waveform.set_start_marker(self._work_start_marker_seconds)
            self.bed_waveform.set_marker(self._work_start_marker_seconds)
            self.record_repair_action("bed", self.bed_action_code(), self.bed_gain.value())
            self.status.setText("Applied selected action to bed preview only.")
        except Exception as exc:
            self._error(str(exc))

    def reset_bed_preview(self) -> None:
        self.push_undo()
        self.update_bed_clip_view()
        self.status.setText("Reset bed preview to source work-window slice.")

    def record_repair_action(self, target: str, action: str, gain_db: float, notes: str = "") -> None:
        """Append a reproducible action entry for project save and recipe export."""

        region = self.project.active_repair
        local_start = self.cut_start.value()
        local_end = self.cut_end.value()
        region.actions.append(
            RepairAction(
                target=target,
                action=action,
                local_start_seconds=local_start,
                local_end_seconds=local_end,
                episode_start_seconds=region.work_start_seconds + local_start,
                episode_end_seconds=region.work_start_seconds + local_end,
                selected_gain_db=gain_db,
                fade_in_seconds=self.fade_in.value(),
                fade_out_seconds=self.fade_out.value(),
                edge_source_seconds=self.edge_source.value(),
                voice_keep_original_percent=self.voice_keep_original.value() if action == "remove_voice" else 0.0,
                notes=notes,
            )
        )

    def build_visible_work_mix(self, work: AudioBuffer) -> AudioBuffer:
        """Mix enabled clip lanes over the currently visible work preview."""

        region = self.project.active_repair
        mixed = work.samples.copy()
        for lane in active_lanes(region):
            lane_audio = prepare_lane_audio(lane, sample_rate=work.sample_rate, channels=work.channels)
            start_seconds = self.cut_start.value() + lane.offset_seconds
            start = int(round(start_seconds * work.sample_rate))
            lane_samples = lane_audio.samples * db_to_linear(lane.gain_db)
            lane_samples = apply_fade(lane_samples, work.sample_rate, lane.fade_in_seconds, lane.fade_out_seconds)
            mix_into(mixed, lane_samples, start)
        return AudioBuffer(samples=np.clip(mixed, -0.98, 0.98), sample_rate=work.sample_rate)

    def build_bed_plus_work_mix(self, bed: AudioBuffer, work: AudioBuffer) -> AudioBuffer:
        """Overlay the visible work mix on top of the visible bed preview."""

        work_mix = self.build_visible_work_mix(work)
        frames = max(bed.frames, work_mix.frames)
        channels = max(bed.channels, work_mix.channels)
        mixed = np.zeros((channels, frames), dtype=np.float32)
        mix_into(mixed, np.pad(bed.samples, ((0, channels - bed.channels), (0, 0))), 0)
        mix_into(mixed, np.pad(work_mix.samples, ((0, channels - work_mix.channels), (0, 0))), 0)
        return AudioBuffer(samples=np.clip(mixed, -0.98, 0.98), sample_rate=bed.sample_rate)

    def play_lane(self, lane_widget: LaneWidget) -> None:
        self.set_focus(lane_widget.lane.name)
        self.focus_label.setText(f"Focus: {lane_widget.lane.name}")
        lane_widget.to_model()
        if not lane_widget.lane.path:
            self._error("Load a file into this clip lane first.")
            return
        if self._active_lane_widget is lane_widget and self._last_play_kind == "lane":
            if self.playhead_timer.isActive() or self._playback_paused:
                self.pause_playback()
                return
        try:
            audio = prepare_lane_audio(
                lane_widget.lane,
                sample_rate=self.project.sample_rate,
                channels=self.project.channels,
                fit_duration_seconds=None,
            )
            audio.samples *= 10 ** (lane_widget.lane.gain_db / 20.0)
            audio.samples = apply_fade(
                audio.samples,
                audio.sample_rate,
                lane_widget.lane.fade_in_seconds,
                lane_widget.lane.fade_out_seconds,
            )
            self._active_lane_widget = lane_widget
            for widget in self.lane_widgets:
                widget.set_clip_playing(widget is lane_widget)
            self.play_buffer(audio, "lane", 0.0, kind="lane")
        except Exception as exc:
            self._error(str(exc))

    def stop_lane(self, lane_widget: LaneWidget) -> None:
        if self._active_lane_widget is lane_widget:
            self.stop_playback()
        lane_widget.set_clip_playing(False)

    def reset_lane(self, lane_widget: LaneWidget) -> None:
        """Restore one clip lane to the state it had when this project was loaded."""

        try:
            index = self.lane_widgets.index(lane_widget)
        except ValueError:
            return
        self.push_undo()
        template_lanes = self._loaded_project_template.active_repair.lanes
        if index < len(template_lanes):
            replacement = deepcopy(template_lanes[index])
        else:
            replacement = ClipLane(name=self.clip_bank_name(index))
        replacement.name = self.clip_bank_name(index)
        self.project.active_repair.lanes[index] = replacement
        lane_widget.lane = replacement
        lane_widget.setTitle(replacement.name)
        lane_widget.from_model()
        self.update_lane_timeline_controls()
        self.status.setText(f"Reset {replacement.name} to loaded settings.")

    def auto_align_lane(self, lane_widget: LaneWidget) -> None:
        self.push_undo()
        cut_duration = max(self.cut_end.value() - self.cut_start.value(), 0.0)
        has_selection = cut_duration > 0.001
        new_offset = 0.0 if has_selection else -self.cut_start.value()
        lane_widget.offset.setValue(new_offset)
        lane_widget.locked.setChecked(has_selection)
        lane_widget.to_model()
        self.update_lane_timeline_controls()
        lock_note = "locked to selection" if has_selection else "left unlocked"
        self.status.setText(f"Auto-aligned {lane_widget.lane.name}; {lock_note}.")

    def fit_lane_length(self, lane_widget: LaneWidget) -> None:
        if not lane_widget.lane.path:
            self._error("Load a file into this clip lane first.")
            return
        cut_duration = max(self.cut_end.value() - self.cut_start.value(), 0.001)
        try:
            info = sf.info(lane_widget.lane.path)
            source_duration = info.frames / info.samplerate if info.samplerate else 0.0
        except Exception as exc:
            self._error(str(exc))
            return
        if source_duration <= 0:
            self._error("Clip duration could not be read.")
            return
        self.push_undo()
        speed = max(min((source_duration / cut_duration) * 100.0, 400.0), 25.0)
        lane_widget.speed.setValue(speed)
        lane_widget.to_model()
        self.update_lane_timeline_controls()
        self.status.setText(f"Fit {lane_widget.lane.name} to {cut_duration:.3f}s using {speed:.1f}% speed.")

    def play_reference(self) -> None:
        self.set_focus("reference")
        if self.reference_audio is None:
            self.refresh_audio()
        if self.reference_audio is None:
            self._error("Load a reference audio file first.")
            return
        try:
            self.play_buffer(self.reference_audio, "reference", 0.0, kind="reference")
        except Exception as exc:
            self._error(str(exc))

    def export_work_mix(self) -> None:
        self._ui_to_project()
        work = self.current_work_audio()
        if work is None:
            self._error("Load a main audio file first.")
            return
        region = self.project.active_repair
        path = self.standard_repair_output_dir() / f"{region.repair_id}_work_mix.wav"
        if not self.confirm_overwrite([path], "Work mix WAV"):
            return
        save_audio(path, self.build_visible_work_mix(work))
        self.status.setText(f"Exported work mix: {path}")

    def standard_repair_output_dir(self) -> Path:
        """Return the deterministic review folder for the current repair."""

        return Path("work/review/repair_tool") / (self.project.episode or "episode") / self.project.active_repair.repair_id

    def confirm_overwrite(self, paths: list[Path], label: str) -> bool:
        """Ask before replacing existing export artifacts."""

        existing = [path for path in paths if path.exists()]
        if not existing:
            return True
        details = "\n".join(str(path) for path in existing)
        answer = QMessageBox.question(
            self,
            "Replace Existing Export?",
            f"{label} already exists and will be replaced:\n\n{details}\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def export_bed_work_mix(self) -> None:
        self._ui_to_project()
        bed = self.current_bed_work_audio()
        work = self.current_work_audio()
        if bed is None or work is None:
            self._error("Load both main work audio and de-voiced bed audio first.")
            return
        default = f"work/review/repair_tool/{self.project.active_repair.repair_id}_bed_plus_work_mix.wav"
        path, _ = QFileDialog.getSaveFileName(self, "Export bed plus work WAV", default, "WAV (*.wav)")
        if not path:
            return
        output_path = Path(path)
        if not self.confirm_overwrite([output_path], "Bed plus work WAV"):
            return
        save_audio(output_path, self.build_bed_plus_work_mix(bed, work))
        self.status.setText(f"Exported bed plus work mix: {path}")

    def export_full_main_patch(self) -> None:
        self._ui_to_project()
        if self.main_audio is None:
            self.refresh_audio()
        work = self.current_work_audio()
        if self.main_audio is None or work is None:
            self._error("Load a main audio file first.")
            return
        default = f"work/review/repair_tool/{self.project.active_repair.repair_id}_full_main_patched.wav"
        path, _ = QFileDialog.getSaveFileName(self, "Export full patched main WAV", default, "WAV (*.wav)")
        if not path:
            return
        output_path = Path(path)
        if not self.confirm_overwrite([output_path], "Full patched main WAV"):
            return
        work_mix = self.build_visible_work_mix(work)
        region = self.project.active_repair
        start_frame = time_to_frame(region.work_start_seconds, self.main_audio.sample_rate)
        end_frame = min(start_frame + time_to_frame(region.work_window_seconds, self.main_audio.sample_rate), self.main_audio.frames)
        channels = max(self.main_audio.channels, work_mix.channels)
        main_samples = np.pad(self.main_audio.samples, ((0, channels - self.main_audio.channels), (0, 0)))
        work_samples = np.pad(work_mix.samples, ((0, channels - work_mix.channels), (0, 0)))
        patched = np.concatenate([main_samples[:, :start_frame], work_samples, main_samples[:, end_frame:]], axis=1)
        save_audio(output_path, AudioBuffer(samples=np.clip(patched, -0.98, 0.98), sample_rate=self.main_audio.sample_rate))
        self.status.setText(f"Exported full patched main: {path}")

    def export_ready_patch(self) -> None:
        """Export a hot-swappable replacement clip plus pipeline manifest."""

        self._ui_to_project()
        work = self.current_work_audio()
        if work is None:
            self._error("Load a main audio file first.")
            return
        cut_start = min(max(self.cut_start.value(), 0.0), work.duration_seconds)
        cut_end = min(max(self.cut_end.value(), cut_start), work.duration_seconds)
        if cut_end <= cut_start:
            self._error("Select a non-empty work area first.")
            return
        region = self.project.active_repair
        patch_dir = Path("work/ready_audio_patches") / (self.project.episode or "episode") / region.repair_id
        replacement_path = patch_dir / "replacement.wav"
        manifest_path = patch_dir / "patch.json"
        if not self.confirm_overwrite([replacement_path, manifest_path], "Ready patch"):
            return
        patch_dir.mkdir(parents=True, exist_ok=True)
        work_mix = self.build_visible_work_mix(work)
        export_cut_only = self.ready_patch_cut_only.isChecked()
        if export_cut_only:
            replacement = crop(work_mix, cut_start, cut_end - cut_start)
            start_episode = region.work_start_seconds + cut_start
            end_episode = region.work_start_seconds + cut_end
            local_start = cut_start
            local_end = cut_end
            export_scope = "cut_only"
        else:
            replacement = work_mix
            start_episode = region.work_start_seconds
            end_episode = region.work_start_seconds + replacement.duration_seconds
            local_start = 0.0
            local_end = replacement.duration_seconds
            export_scope = "full_work_window"
        save_audio(replacement_path, replacement)
        manifest = {
            "kind": "robotech_ready_audio_patch",
            "patch_id": region.repair_id,
            "episode": self.project.episode,
            "title": self.project.title,
            "description": (
                "Direct ready replacement clip exported from the repair tool. "
                "The pipeline replaces the target section with replacement.wav as-is."
            ),
            "method": "ready_replacement_clip",
            "replacement_is_final": True,
            "export_scope": export_scope,
            "targets": ["dialogue"],
            "start_seconds": start_episode,
            "end_seconds": end_episode,
            "replacement_path": "replacement.wav",
            "replacement_source_seconds": replacement.duration_seconds,
            "source_project_main_track": self.project.main_track,
            "work_start_seconds": region.work_start_seconds,
            "work_window_seconds": region.work_window_seconds,
            "local_replacement_start_seconds": local_start,
            "local_replacement_end_seconds": local_end,
            "local_cut_start_seconds": cut_start,
            "local_cut_end_seconds": cut_end,
            "clip_lanes": [asdict(lane) for lane in region.lanes],
            "applied_actions": [asdict(action) for action in region.actions],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.save_open_project_quietly()
        self.status.setText(f"Exported ready patch: {patch_dir}")

    def export_preview(self) -> None:
        self.export_work_mix()

    def export_recipe(self) -> None:
        self._ui_to_project()
        default = f"work/repair_projects/{self.project.active_repair.repair_id}.recipe.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export recipe JSON", default, "Recipe JSON (*.json)")
        if not path:
            return
        output_path = Path(path)
        if not self.confirm_overwrite([output_path], "Recipe JSON"):
            return
        self._ui_to_project()
        export_recipe(output_path, self.project)
        self.save_open_project_quietly()
        self.status.setText(f"Exported recipe: {path}")

    def _error(self, message: str) -> None:
        QMessageBox.warning(self, "Repair Tool", message)

    def pause_playback(self) -> None:
        if self.playhead_timer.isActive():
            self._paused_scope = self._active_play_scope
            self._paused_seconds = self._active_play_start + self._active_play_elapsed_ms / 1000.0
            self.playback.pause()
            self.video_player.pause()
            self.playhead_timer.stop()
            self._playback_paused = True
            if self._last_play_kind == "selected":
                self.update_play_selected_button_icon(False)
            if self._last_play_kind == "lane" and self._active_lane_widget is not None:
                self._active_lane_widget.set_clip_playing(False)
            return
        if self._last_play_buffer is not None:
            playback_resume = self._last_playback_start_seconds + self._active_play_elapsed_ms / 1000.0
            self.play_buffer(
                self._last_play_buffer,
                self._paused_scope or self._active_play_scope,
                display_start_seconds=self._paused_seconds,
                playback_start_seconds=playback_resume,
                loop_display_start_seconds=self._last_loop_display_start_seconds,
                kind=self._last_play_kind,
            )

    def play_or_pause_focused(self) -> None:
        if self.playhead_timer.isActive():
            self.pause_playback()
            return
        if self.active_focus == "main":
            self.play_full_main()
        elif self.active_focus == "bed":
            self.play_bed_clip()
        elif self.active_focus == "reference":
            self.play_reference()
        elif self.active_focus == "work":
            self.play_work_clip()
        else:
            for lane_widget in self.lane_widgets:
                if self.active_focus == lane_widget.lane.name:
                    self.play_lane(lane_widget)
                    return
            self.play_work_clip()

    def stop_playback(self) -> None:
        self.playback.stop()
        self.video_player.pause()
        self.playhead_timer.stop()
        self._playback_paused = False
        if self._active_play_scope == "work":
            self._work_start_marker_seconds = 0.0
            self.work_waveform.set_start_marker(0.0)
        if self._active_lane_widget is not None:
            self._active_lane_widget.set_clip_playing(False)
            self._active_lane_widget = None
        self.update_play_selected_button_icon(False)
        self.timeline.set_playhead(0.0)
        self.work_waveform.set_playhead(0.0)
        self.bed_waveform.set_playhead(0.0)
        self.reference_waveform.set_playhead(0.0)

    def play_buffer(
        self,
        buffer: AudioBuffer,
        scope: str,
        display_start_seconds: float,
        playback_start_seconds: float | None = None,
        loop_display_start_seconds: float | None = None,
        kind: str = "",
    ) -> None:
        self._last_play_buffer = buffer
        self._last_play_kind = kind
        self._playback_paused = False
        self.update_play_selected_button_icon(kind == "selected")
        if kind == "lane" and self._active_lane_widget is not None:
            self._active_lane_widget.set_clip_playing(True)
        playback_start = display_start_seconds if playback_start_seconds is None else playback_start_seconds
        self._last_playback_start_seconds = playback_start
        self._last_display_start_seconds = display_start_seconds
        self._last_loop_display_start_seconds = (
            display_start_seconds if loop_display_start_seconds is None else loop_display_start_seconds
        )
        self.playback.play(buffer, start_seconds=playback_start)
        self.sync_video_to_scope(scope, display_start_seconds, play=True)
        self.start_playhead(scope, display_start_seconds)

    def sync_video_to_scope(self, scope: str, display_start_seconds: float, play: bool = False) -> None:
        """Seek the muted video reference to the episode time for the current audio scope."""

        if not hasattr(self, "video_player") or not self.video_follow.isChecked() or self._video_loaded_path is None:
            return
        episode_seconds = self.video_episode_seconds(scope, display_start_seconds)
        self.video_player.setPosition(max(int(round(episode_seconds * 1000)), 0))
        if play:
            self.video_player.play()

    def video_episode_seconds(self, scope: str, display_seconds: float) -> float:
        if scope == "main":
            return display_seconds
        if scope in {"work", "bed", "lane", "reference"}:
            return self.project.active_repair.work_start_seconds + display_seconds
        return self.project.active_repair.work_start_seconds + display_seconds

    def start_playhead(self, scope: str, start_seconds: float) -> None:
        self._active_play_scope = scope
        self._active_play_start = start_seconds
        self._active_play_elapsed_ms = 0
        self.playhead_timer.start()
        self.update_playhead()

    def update_playhead(self) -> None:
        self._active_play_elapsed_ms += self.playhead_timer.interval()
        seconds = self._active_play_start + self._active_play_elapsed_ms / 1000.0
        playback_seconds = self._last_playback_start_seconds + self._active_play_elapsed_ms / 1000.0
        if self._active_play_scope == "main":
            if self.main_audio is not None and seconds >= self.main_audio.duration_seconds:
                if self.main_loop.isChecked():
                    self.play_full_main()
                else:
                    self.playhead_timer.stop()
                    self._playback_paused = False
                    self.update_play_selected_button_icon(False)
                return
            self.timeline.set_playhead(seconds)
        elif self._active_play_scope == "work":
            active_duration = self._last_play_buffer.duration_seconds if self._last_play_buffer is not None else 0.0
            if active_duration and playback_seconds >= active_duration:
                if self.work_loop.isChecked():
                    assert self._last_play_buffer is not None
                    self.play_buffer(
                        self._last_play_buffer,
                        "work",
                        display_start_seconds=self._last_loop_display_start_seconds,
                        playback_start_seconds=0.0,
                        loop_display_start_seconds=self._last_loop_display_start_seconds,
                        kind=self._last_play_kind,
                    )
                else:
                    self.playhead_timer.stop()
                    self._playback_paused = False
                    self.update_play_selected_button_icon(False)
                return
            self.work_waveform.set_playhead(seconds)
        elif self._active_play_scope == "bed":
            active_duration = self._last_play_buffer.duration_seconds if self._last_play_buffer is not None else 0.0
            if active_duration and playback_seconds >= active_duration:
                if self.bed_loop.isChecked():
                    assert self._last_play_buffer is not None
                    self.play_buffer(
                        self._last_play_buffer,
                        "bed",
                        display_start_seconds=self._last_loop_display_start_seconds,
                        playback_start_seconds=0.0,
                        loop_display_start_seconds=self._last_loop_display_start_seconds,
                        kind=self._last_play_kind,
                    )
                else:
                    self.playhead_timer.stop()
                    self._playback_paused = False
                    self.update_play_selected_button_icon(False)
                return
            self.bed_waveform.set_playhead(seconds)
        elif self._active_play_scope == "reference":
            if self.reference_audio is not None and seconds >= self.reference_audio.duration_seconds:
                if self.reference_loop.isChecked():
                    self.play_reference()
                else:
                    self.playhead_timer.stop()
                    self._playback_paused = False
                    self.update_play_selected_button_icon(False)
                return
            self.reference_waveform.set_playhead(seconds)
        elif self._active_play_scope == "lane":
            active_duration = self._last_play_buffer.duration_seconds if self._last_play_buffer is not None else 0.0
            if active_duration and playback_seconds >= active_duration:
                self.playhead_timer.stop()
                self._playback_paused = False
                if self._active_lane_widget is not None:
                    self._active_lane_widget.set_clip_playing(False)
                return


def run_app(project: RepairProject | None = None, project_path: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(project, project_path=project_path)
    window.show()
    return app.exec()
