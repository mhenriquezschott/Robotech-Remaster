"""Project serialization helpers for the audio repair tool."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import ClipItem, ClipLane, EdgeSettings, RepairAction, RepairProject, RepairRegion, SCHEMA_VERSION


def load_project(path: Path) -> RepairProject:
    """Load a repair project JSON file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported repair project schema: {data.get('schema_version')!r}")
    project_data = data["project"]
    region_data = project_data.get("active_repair", {})
    edge_data = region_data.get("edge", {})
    lanes = []
    for lane in region_data.get("lanes", []):
        lane["clip_items"] = [ClipItem(**item) for item in lane.get("clip_items", [])]
        lane.setdefault("speed_percent", 100.0)
        lane.setdefault("fade_in_seconds", 0.0)
        lane.setdefault("fade_out_seconds", 0.0)
        lane.setdefault("selected_clip_index", 0)
        lane.setdefault("muted", not bool(lane.get("path", "")))
        if not lane.get("path", ""):
            lane["muted"] = True
        clip_lane = ClipLane(**lane)
        clip_lane.ensure_clip_items()
        lanes.append(clip_lane)
    actions = []
    for action in region_data.get("actions", []):
        action.setdefault("voice_keep_original_percent", 0.0)
        actions.append(RepairAction(**action))
    region = RepairRegion(
        repair_id=region_data.get("repair_id", "untitled_repair"),
        marker_seconds=float(region_data.get("marker_seconds", 0.0)),
        work_window_seconds=float(region_data.get("work_window_seconds", 7.0)),
        cut_start_seconds=float(region_data.get("cut_start_seconds", 0.0)),
        cut_end_seconds=float(region_data.get("cut_end_seconds", 0.0)),
        snap_ms=int(region_data.get("snap_ms", 10)),
        selected_gain_db=float(region_data.get("selected_gain_db", 0.0)),
        voice_keep_original_percent=float(region_data.get("voice_keep_original_percent", 0.0)),
        edge_source_seconds=float(region_data.get("edge_source_seconds", 0.050)),
        edge=EdgeSettings(**edge_data),
        lanes=lanes,
        actions=actions,
    )
    project = RepairProject(
        episode=project_data.get("episode", ""),
        title=project_data.get("title", ""),
        main_track=project_data.get("main_track", ""),
        bed_track=project_data.get("bed_track", ""),
        bed_preview_mode=project_data.get("bed_preview_mode", "as_loaded"),
        video_track=project_data.get("video_track", ""),
        reference_tracks=list(project_data.get("reference_tracks", [])),
        sample_rate=int(project_data.get("sample_rate", 48000)),
        channels=int(project_data.get("channels", 2)),
        active_repair=region,
    )
    project.ensure_default_lanes()
    return project


def save_project(path: Path, project: RepairProject) -> None:
    """Save a repair project JSON file."""

    for lane in project.active_repair.lanes:
        lane.store_selected_clip_item()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": asdict(project),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
