"""Data models for the Robotech audio repair tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


SCHEMA_VERSION = 1

LaneRole = Literal["replacement_voice", "texture", "music", "reference", "other"]


@dataclass
class ClipItem:
    """One selectable audio item inside a lane's clip bank."""

    label: str
    path: str = ""
    role: LaneRole = "other"
    offset_seconds: float = 0.0
    gain_db: float = 0.0
    speed_percent: float = 100.0
    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.0
    fit_to_cut: bool = False
    muted: bool = True
    locked: bool = False

    @property
    def resolved_path(self) -> Path | None:
        return Path(self.path) if self.path else None


@dataclass
class ClipLane:
    """A short clip that can be aligned and mixed over the work window."""

    name: str
    path: str = ""
    role: LaneRole = "other"
    offset_seconds: float = 0.0
    gain_db: float = 0.0
    speed_percent: float = 100.0
    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.0
    fit_to_cut: bool = False
    muted: bool = True
    solo: bool = False
    locked: bool = False
    clip_items: list[ClipItem] = field(default_factory=list)
    selected_clip_index: int = 0

    @property
    def resolved_path(self) -> Path | None:
        return Path(self.path) if self.path else None

    def ensure_clip_items(self) -> None:
        """Backfill old single-path lanes into a one-item clip bank."""

        if not self.clip_items and self.path:
            self.clip_items.append(
                ClipItem(
                    label=Path(self.path).name,
                    path=self.path,
                    role=self.role,
                    offset_seconds=self.offset_seconds,
                    gain_db=self.gain_db,
                    speed_percent=self.speed_percent,
                    fade_in_seconds=self.fade_in_seconds,
                    fade_out_seconds=self.fade_out_seconds,
                    fit_to_cut=self.fit_to_cut,
                    muted=self.muted,
                    locked=self.locked,
                )
            )
        if self.clip_items:
            self.selected_clip_index = min(max(self.selected_clip_index, 0), len(self.clip_items) - 1)
            self.apply_selected_clip_item()

    def selected_clip_item(self) -> ClipItem | None:
        if not self.clip_items:
            return None
        self.selected_clip_index = min(max(self.selected_clip_index, 0), len(self.clip_items) - 1)
        return self.clip_items[self.selected_clip_index]

    def apply_selected_clip_item(self) -> None:
        item = self.selected_clip_item()
        if item is None:
            return
        self.path = item.path
        self.role = item.role
        self.offset_seconds = item.offset_seconds
        self.gain_db = item.gain_db
        self.speed_percent = item.speed_percent
        self.fade_in_seconds = item.fade_in_seconds
        self.fade_out_seconds = item.fade_out_seconds
        self.fit_to_cut = item.fit_to_cut
        self.muted = True if not item.path else item.muted
        self.locked = item.locked

    def store_selected_clip_item(self) -> None:
        item = self.selected_clip_item()
        if item is None:
            return
        item.path = self.path
        item.role = self.role
        item.offset_seconds = self.offset_seconds
        item.gain_db = self.gain_db
        item.speed_percent = self.speed_percent
        item.fade_in_seconds = self.fade_in_seconds
        item.fade_out_seconds = self.fade_out_seconds
        item.fit_to_cut = self.fit_to_cut
        item.muted = True if not self.path else self.muted
        item.locked = self.locked

    def add_clip_item(self, path: str, label: str | None = None) -> None:
        self.store_selected_clip_item()
        self.clip_items.append(
            ClipItem(
                label=label or Path(path).name,
                path=path,
                role=self.role,
                offset_seconds=self.offset_seconds,
                gain_db=self.gain_db,
                speed_percent=self.speed_percent,
                fade_in_seconds=self.fade_in_seconds,
                fade_out_seconds=self.fade_out_seconds,
                muted=False,
                locked=self.locked,
            )
        )
        self.selected_clip_index = len(self.clip_items) - 1
        self.apply_selected_clip_item()


@dataclass
class EdgeSettings:
    """Simple transition settings for a repair insert."""

    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.0
    curve: str = "tri"


@dataclass
class RepairAction:
    """A reproducible action applied to a local work/bed preview region."""

    target: str
    action: str
    local_start_seconds: float
    local_end_seconds: float
    episode_start_seconds: float
    episode_end_seconds: float
    selected_gain_db: float = 0.0
    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.0
    edge_source_seconds: float = 0.050
    voice_keep_original_percent: float = 0.0
    notes: str = ""


@dataclass
class RepairRegion:
    """Current repair window and cut markers."""

    repair_id: str = "untitled_repair"
    marker_seconds: float = 0.0
    work_window_seconds: float = 7.0
    cut_start_seconds: float = 0.0
    cut_end_seconds: float = 0.0
    snap_ms: int = 10
    selected_gain_db: float = 0.0
    voice_keep_original_percent: float = 0.0
    edge_source_seconds: float = 0.050
    edge: EdgeSettings = field(default_factory=EdgeSettings)
    lanes: list[ClipLane] = field(default_factory=list)
    actions: list[RepairAction] = field(default_factory=list)

    @property
    def cut_duration(self) -> float:
        return max(self.cut_end_seconds - self.cut_start_seconds, 0.0)

    @property
    def work_start_seconds(self) -> float:
        return max(self.marker_seconds - self.work_window_seconds / 2, 0.0)

    @property
    def work_end_seconds(self) -> float:
        return self.work_start_seconds + self.work_window_seconds


@dataclass
class RepairProject:
    """Serializable project state for a repair session."""

    episode: str = ""
    title: str = ""
    main_track: str = ""
    bed_track: str = ""
    bed_preview_mode: str = "as_loaded"
    video_track: str = ""
    reference_tracks: list[str] = field(default_factory=list)
    sample_rate: int = 48000
    channels: int = 2
    active_repair: RepairRegion = field(default_factory=RepairRegion)

    def ensure_default_lanes(self) -> None:
        """Keep three edit lanes available for the first UI iteration."""

        names = ["Clip Bank 01", "Clip Bank 02", "Clip Bank 03"]
        while len(self.active_repair.lanes) < 3:
            self.active_repair.lanes.append(ClipLane(name=names[len(self.active_repair.lanes)], muted=True))
        for lane in self.active_repair.lanes:
            lane.ensure_clip_items()
            if not lane.path:
                lane.muted = True
