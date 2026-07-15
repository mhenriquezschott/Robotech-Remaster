"""Apply repair-tool recipe JSON files to full-length audio tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio_engine import (
    AudioBuffer,
    active_lanes,
    apply_fade,
    apply_selection_action,
    crop,
    db_to_linear,
    load_audio,
    mix_into,
    prepare_lane_audio,
    save_audio,
    time_to_frame,
)
from .models import ClipItem, ClipLane, EdgeSettings, RepairProject, RepairRegion


SUPPORTED_RECIPE_ACTIONS = {
    "keep",
    "silence",
    "delete",
    "fade_silence",
    "rubberband_bridge",
    "stretch_edge",
    "repeat_edge",
    "blend_edges",
    "interp_ambience",
}


def load_recipe_project(recipe_path: Path, main_track_override: Path | None = None) -> RepairProject:
    """Load a recipe export into the in-memory project shape used by renderers."""

    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "robotech_audio_repair_recipe":
        raise ValueError(f"Not a repair recipe: {recipe_path}")
    sources = payload.get("sources", {})
    edge = EdgeSettings(**payload.get("edge", {}))
    lanes = []
    for lane_data in payload.get("lanes", []):
        lane_data["clip_items"] = [ClipItem(**item) for item in lane_data.get("clip_items", [])]
        lane_data.setdefault("speed_percent", 100.0)
        lane_data.setdefault("fade_in_seconds", 0.0)
        lane_data.setdefault("fade_out_seconds", 0.0)
        lane = ClipLane(**lane_data)
        lane.ensure_clip_items()
        lanes.append(lane)
    region = RepairRegion(
        repair_id=payload.get("repair_id", recipe_path.stem),
        marker_seconds=float(payload.get("marker_seconds", 0.0)),
        work_window_seconds=float(payload.get("work_window_seconds", 0.0)),
        cut_start_seconds=float(payload.get("cut_start_seconds", 0.0)),
        cut_end_seconds=float(payload.get("cut_end_seconds", 0.0)),
        snap_ms=int(payload.get("snap_ms", 10)),
        selected_gain_db=float(payload.get("selected_gain_db", 0.0)),
        voice_keep_original_percent=float(payload.get("voice_keep_original_percent", 0.0)),
        edge_source_seconds=float(payload.get("edge_source_seconds", 0.050)),
        edge=edge,
        lanes=lanes,
    )
    project = RepairProject(
        episode=payload.get("episode", ""),
        title=payload.get("title", ""),
        main_track=str(main_track_override or sources.get("main_track", "")),
        bed_track=sources.get("bed_track", ""),
        video_track=sources.get("video_track", ""),
        reference_tracks=list(sources.get("reference_tracks", [])),
        sample_rate=int(payload.get("sample_rate", 48000)),
        channels=int(payload.get("channels", 2)),
        active_repair=region,
    )
    return project


def recipe_actions(recipe_path: Path) -> list[dict[str, Any]]:
    """Return recorded recipe actions as dictionaries."""

    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    return list(payload.get("actions", []))


def apply_recipe_to_audio(
    recipe_path: Path,
    input_audio: Path,
    output_audio: Path,
    *,
    action_fallback: str = "keep",
) -> dict[str, Any]:
    """Apply one GUI recipe to a full audio file and save the patched result.

    The recipe replaces the full work-window span with the rendered work mix.
    This is deliberately the same mental model as the GUI's `Export Full Main
    WAV`: the exported window length is `work_window_seconds`, not only the
    cut selection.
    """

    project = load_recipe_project(recipe_path, main_track_override=input_audio)
    main = load_audio(input_audio)
    work = crop(main, project.active_repair.work_start_seconds, project.active_repair.work_window_seconds)
    rendered = render_recipe_work_window(project, work, recipe_actions(recipe_path), action_fallback=action_fallback)
    start_frame = time_to_frame(project.active_repair.work_start_seconds, main.sample_rate)
    end_frame = min(start_frame + work.frames, main.frames)
    channels = max(main.channels, rendered.channels)
    main_samples = np.pad(main.samples, ((0, channels - main.channels), (0, 0)))
    rendered_samples = np.pad(rendered.samples, ((0, channels - rendered.channels), (0, 0)))
    patched = np.concatenate([main_samples[:, :start_frame], rendered_samples, main_samples[:, end_frame:]], axis=1)
    save_audio(output_audio, AudioBuffer(samples=np.clip(patched, -0.98, 0.98), sample_rate=main.sample_rate))
    return {
        "recipe": str(recipe_path),
        "input": str(input_audio),
        "output": str(output_audio),
        "repair_id": project.active_repair.repair_id,
        "work_start_seconds": project.active_repair.work_start_seconds,
        "work_window_seconds": project.active_repair.work_window_seconds,
        "cut_start_seconds": project.active_repair.cut_start_seconds,
        "cut_end_seconds": project.active_repair.cut_end_seconds,
        "actions": recipe_actions(recipe_path),
        "lanes": [lane.path for lane in active_lanes(project.active_repair)],
    }


def render_recipe_work_window(
    project: RepairProject,
    work: AudioBuffer,
    actions: list[dict[str, Any]],
    *,
    action_fallback: str = "keep",
) -> AudioBuffer:
    """Render deterministic recipe actions and lane overlays for one work window."""

    rendered = AudioBuffer(samples=work.samples.copy(), sample_rate=work.sample_rate)
    relevant_actions = [action for action in actions if action.get("target", "work") == "work"]
    if not relevant_actions and action_fallback:
        relevant_actions = [
            {
                "action": action_fallback,
                "local_start_seconds": project.active_repair.cut_start_seconds - project.active_repair.work_start_seconds,
                "local_end_seconds": project.active_repair.cut_end_seconds - project.active_repair.work_start_seconds,
                "selected_gain_db": project.active_repair.selected_gain_db,
                "fade_in_seconds": project.active_repair.edge.fade_in_seconds,
                "fade_out_seconds": project.active_repair.edge.fade_out_seconds,
                "edge_source_seconds": project.active_repair.edge_source_seconds,
            }
        ]
    for action in relevant_actions:
        action_name = str(action.get("action", "keep"))
        if action_name == "remove_voice":
            raise ValueError(
                "Recipe contains remove_voice, which is not replayed by the final-build importer yet. "
                "Apply/export that preview to a clip or full patched WAV first, or use deterministic actions."
            )
        if action_name not in SUPPORTED_RECIPE_ACTIONS:
            raise ValueError(f"Unsupported recipe action: {action_name}")
        start = time_to_frame(float(action.get("local_start_seconds", 0.0)), rendered.sample_rate)
        end = time_to_frame(float(action.get("local_end_seconds", 0.0)), rendered.sample_rate)
        rendered = apply_selection_action(
            rendered,
            start,
            end,
            action_name,
            selected_gain_db=float(action.get("selected_gain_db", 0.0)),
            fade_in_seconds=float(action.get("fade_in_seconds", 0.0)),
            fade_out_seconds=float(action.get("fade_out_seconds", 0.0)),
            edge_source_seconds=float(action.get("edge_source_seconds", 0.050)),
        )
    mixed = rendered.samples.copy()
    region = project.active_repair
    for lane in active_lanes(region):
        lane_audio = prepare_lane_audio(lane, sample_rate=rendered.sample_rate, channels=rendered.channels)
        start_seconds = (region.cut_start_seconds + lane.offset_seconds) - region.work_start_seconds
        start = int(round(start_seconds * rendered.sample_rate))
        lane_samples = lane_audio.samples * db_to_linear(lane.gain_db)
        lane_samples = apply_fade(lane_samples, rendered.sample_rate, lane.fade_in_seconds, lane.fade_out_seconds)
        mix_into(mixed, lane_samples, start)
    return AudioBuffer(samples=np.clip(mixed, -0.98, 0.98), sample_rate=rendered.sample_rate)
