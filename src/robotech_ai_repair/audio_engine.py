"""Audio loading, preview mixing, and export helpers."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from .models import ClipLane, RepairProject, RepairRegion


@dataclass
class AudioBuffer:
    """In-memory floating-point audio buffer in channels-first shape."""

    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def db_to_linear(db: float) -> float:
    return float(10 ** (db / 20.0))


def load_audio(path: Path, target_sample_rate: int | None = None, channels: int = 2) -> AudioBuffer:
    """Load audio as float32 channels-first data."""

    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    samples = data.T
    if samples.shape[0] == 1 and channels == 2:
        samples = np.repeat(samples, 2, axis=0)
    elif samples.shape[0] > channels:
        samples = samples[:channels]
    elif samples.shape[0] < channels:
        samples = np.pad(samples, ((0, channels - samples.shape[0]), (0, 0)))
    if target_sample_rate and target_sample_rate != sample_rate:
        samples = resample_linear(samples, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return AudioBuffer(samples=samples.astype(np.float32, copy=False), sample_rate=sample_rate)


def load_audio_downmix(path: Path, target_sample_rate: int | None = None) -> AudioBuffer:
    """Load audio and create a stereo preview downmix from all available channels."""

    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    samples = data.T
    if samples.shape[0] == 1:
        stereo = np.repeat(samples, 2, axis=0)
    elif samples.shape[0] >= 6:
        fl, fr, fc, lfe, sl, sr = samples[:6]
        left = fl + 0.707 * fc + 0.5 * lfe + 0.707 * sl
        right = fr + 0.707 * fc + 0.5 * lfe + 0.707 * sr
        stereo = np.vstack([left, right]) * 0.5
    elif samples.shape[0] == 2:
        stereo = samples
    else:
        mono = samples.mean(axis=0)
        stereo = np.vstack([mono, mono])
    if target_sample_rate and target_sample_rate != sample_rate:
        stereo = resample_linear(stereo, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return AudioBuffer(samples=stereo.astype(np.float32, copy=False), sample_rate=sample_rate)


def save_audio(path: Path, buffer: AudioBuffer) -> None:
    """Save audio as 24-bit PCM WAV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(buffer.samples, -0.98, 0.98).T
    sf.write(path, data, buffer.sample_rate, subtype="PCM_24")


def resample_linear(samples: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Small dependency-light resampler for GUI preview use."""

    if source_sr == target_sr:
        return samples
    old_frames = samples.shape[1]
    new_frames = max(int(round(old_frames * target_sr / source_sr)), 1)
    old_x = np.linspace(0.0, 1.0, old_frames, endpoint=False)
    new_x = np.linspace(0.0, 1.0, new_frames, endpoint=False)
    out = np.vstack([np.interp(new_x, old_x, channel) for channel in samples])
    return out.astype(np.float32)


def time_to_frame(seconds: float, sample_rate: int) -> int:
    return max(int(round(seconds * sample_rate)), 0)


def crop(buffer: AudioBuffer, start_seconds: float, duration_seconds: float) -> AudioBuffer:
    start = min(time_to_frame(start_seconds, buffer.sample_rate), buffer.frames)
    length = time_to_frame(duration_seconds, buffer.sample_rate)
    end = min(start + length, buffer.frames)
    clipped = buffer.samples[:, start:end]
    if clipped.shape[1] < length:
        clipped = np.pad(clipped, ((0, 0), (0, length - clipped.shape[1])))
    return AudioBuffer(samples=clipped.copy(), sample_rate=buffer.sample_rate)


def fit_to_duration(buffer: AudioBuffer, duration_seconds: float) -> AudioBuffer:
    target_frames = time_to_frame(duration_seconds, buffer.sample_rate)
    if target_frames <= 0:
        return AudioBuffer(samples=np.zeros((buffer.channels, 0), dtype=np.float32), sample_rate=buffer.sample_rate)
    old_frames = max(buffer.frames, 1)
    old_x = np.linspace(0.0, 1.0, old_frames, endpoint=False)
    new_x = np.linspace(0.0, 1.0, target_frames, endpoint=False)
    out = np.vstack([np.interp(new_x, old_x, channel) for channel in buffer.samples])
    return AudioBuffer(samples=out.astype(np.float32), sample_rate=buffer.sample_rate)


def apply_speed(buffer: AudioBuffer, speed_percent: float) -> AudioBuffer:
    """Return a preview-speed adjusted buffer.

    Values above 100 play faster/shorter; values below 100 play slower/longer.
    """

    speed = max(speed_percent, 1.0) / 100.0
    return fit_to_duration(buffer, buffer.duration_seconds / speed)


def prepare_lane_audio(
    lane: ClipLane,
    sample_rate: int,
    channels: int,
    fit_duration_seconds: float | None = None,
) -> AudioBuffer:
    """Load and apply lane stretch settings for preview/mix use."""

    lane_audio = load_audio(Path(lane.path), target_sample_rate=sample_rate, channels=channels)
    if fit_duration_seconds is not None and fit_duration_seconds > 0:
        return fit_to_duration(lane_audio, fit_duration_seconds)
    if abs(lane.speed_percent - 100.0) > 0.001:
        return apply_speed(lane_audio, lane.speed_percent)
    return lane_audio


def apply_fade(samples: np.ndarray, sample_rate: int, fade_in: float, fade_out: float) -> np.ndarray:
    out = samples.copy()
    frames = out.shape[1]
    fade_in_frames = min(time_to_frame(fade_in, sample_rate), frames // 2)
    fade_out_frames = min(time_to_frame(fade_out, sample_rate), frames // 2)
    if fade_in_frames:
        out[:, :fade_in_frames] *= np.linspace(0.0, 1.0, fade_in_frames, dtype=np.float32)
    if fade_out_frames:
        out[:, -fade_out_frames:] *= np.linspace(1.0, 0.0, fade_out_frames, dtype=np.float32)
    return out


def stretch_edge_over_gap(work: AudioBuffer, source_start: int, source_end: int, target_frames: int) -> np.ndarray:
    source = work.samples[:, source_start:source_end]
    if source.shape[1] <= 0 or target_frames <= 0:
        return np.zeros((work.channels, max(target_frames, 0)), dtype=np.float32)
    return fit_to_duration(AudioBuffer(samples=source.copy(), sample_rate=work.sample_rate), target_frames / work.sample_rate).samples


def repeat_edge_over_gap(work: AudioBuffer, source_start: int, source_end: int, target_frames: int) -> np.ndarray:
    source = work.samples[:, source_start:source_end]
    if source.shape[1] <= 0 or target_frames <= 0:
        return np.zeros((work.channels, max(target_frames, 0)), dtype=np.float32)
    repeats = int(np.ceil(target_frames / source.shape[1]))
    return np.tile(source, (1, repeats))[:, :target_frames].astype(np.float32, copy=False)


def blend_edges_over_gap(work: AudioBuffer, cut_start: int, cut_end: int, edge_frames: int) -> np.ndarray:
    """Fill a tiny gap by stretching both neighboring edges and crossblending them."""

    target_frames = cut_end - cut_start
    if target_frames <= 0:
        return np.zeros((work.channels, 0), dtype=np.float32)
    pre_start = max(cut_start - edge_frames, 0)
    post_end = min(cut_end + edge_frames, work.frames)
    pre = work.samples[:, pre_start:cut_start]
    post = work.samples[:, cut_end:post_end]
    if pre.shape[1] <= 0 and post.shape[1] <= 0:
        return np.zeros((work.channels, target_frames), dtype=np.float32)
    if pre.shape[1] <= 0:
        return fit_to_duration(AudioBuffer(samples=post.copy(), sample_rate=work.sample_rate), target_frames / work.sample_rate).samples
    if post.shape[1] <= 0:
        return fit_to_duration(AudioBuffer(samples=pre.copy(), sample_rate=work.sample_rate), target_frames / work.sample_rate).samples
    pre_fill = fit_to_duration(AudioBuffer(samples=pre.copy(), sample_rate=work.sample_rate), target_frames / work.sample_rate).samples
    post_fill = fit_to_duration(AudioBuffer(samples=post.copy(), sample_rate=work.sample_rate), target_frames / work.sample_rate).samples
    ramp = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    return pre_fill * (1.0 - ramp) + post_fill * ramp


def interpolate_gap_with_ambience(work: AudioBuffer, cut_start: int, cut_end: int, edge_frames: int) -> np.ndarray:
    """Fill a very small gap with a click-free interpolation plus low-level local ambience."""

    target_frames = cut_end - cut_start
    if target_frames <= 0:
        return np.zeros((work.channels, 0), dtype=np.float32)
    before = work.samples[:, max(cut_start - 1, 0) : max(cut_start, 1)]
    after_index = min(cut_end, work.frames - 1)
    after = work.samples[:, after_index : after_index + 1]
    if before.shape[1] == 0:
        before = after
    if after.shape[1] == 0:
        after = before
    ramp = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    bridge = before * (1.0 - ramp) + after * ramp
    edge = max(edge_frames, 1)
    local_left = work.samples[:, max(cut_start - edge, 0) : cut_start]
    local_right = work.samples[:, cut_end : min(cut_end + edge, work.frames)]
    local = np.concatenate([local_left, local_right], axis=1)
    if local.shape[1] > 4:
        ambience = fit_to_duration(AudioBuffer(samples=local.copy(), sample_rate=work.sample_rate), target_frames / work.sample_rate).samples
        bridge += ambience * 0.08
    return bridge.astype(np.float32, copy=False)


def rubberband_bridge_action(
    work: AudioBuffer,
    cut_start: int,
    cut_end: int,
    bridge_pre_seconds: float,
    crossfade_seconds: float,
) -> AudioBuffer:
    """Render the old approved bridge style with FFmpeg's rubberband filter.

    The source is the audio immediately before the cut. It is time-stretched to
    span from the bridge start through the cut end, with short crossfades at the
    joins. This is intentionally close to the built-in S01E01 patch behavior.
    """

    cut_start = min(max(cut_start, 0), work.frames)
    cut_end = min(max(cut_end, cut_start), work.frames)
    bridge_pre = max(bridge_pre_seconds, 0.0)
    crossfade = max(crossfade_seconds, 0.0)
    start_seconds = cut_start / work.sample_rate
    end_seconds = cut_end / work.sample_rate
    bridge_start_seconds = max(start_seconds - bridge_pre, 0.0)
    source_bridge_duration = start_seconds - bridge_start_seconds
    replacement_duration = end_seconds - bridge_start_seconds
    rendered_bridge_duration = replacement_duration + (2 * crossfade)
    if source_bridge_duration <= 0 or rendered_bridge_duration <= 0 or cut_end <= cut_start:
        return AudioBuffer(samples=work.samples.copy(), sample_rate=work.sample_rate)
    tempo = source_bridge_duration / rendered_bridge_duration
    filter_complex = (
        f"[0:a]atrim=0:{bridge_start_seconds:.6f},asetpts=N/SR/TB[a];"
        f"[0:a]atrim={bridge_start_seconds:.6f}:{start_seconds:.6f},"
        f"asetpts=N/SR/TB,rubberband=tempo={tempo:.8f}[b];"
        f"[0:a]atrim=start={end_seconds:.6f},asetpts=N/SR/TB[c];"
        f"[a][b]acrossfade=d={crossfade:.6f}:c1=tri:c2=tri[ab];"
        f"[ab][c]acrossfade=d={crossfade:.6f}:c1=tri:c2=tri[out]"
    )
    with tempfile.TemporaryDirectory(prefix="robotech_rubberband_") as temp:
        temp_dir = Path(temp)
        input_path = temp_dir / "input.wav"
        output_path = temp_dir / "output.wav"
        save_audio(input_path, work)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(input_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s24le",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        rendered = load_audio(output_path, target_sample_rate=work.sample_rate, channels=work.channels)
    if rendered.frames != work.frames:
        rendered = fit_to_duration(rendered, work.duration_seconds)
    return rendered


def apply_selection_action(
    work: AudioBuffer,
    cut_start: int,
    cut_end: int,
    action: str,
    lower_gain_db: float = -12.0,
    selected_gain_db: float = 0.0,
    fade_in_seconds: float = 0.0,
    fade_out_seconds: float = 0.0,
    edge_source_seconds: float = 0.050,
) -> AudioBuffer:
    """Apply the selected-area operation to a work clip."""

    cut_start = min(max(cut_start, 0), work.frames)
    cut_end = min(max(cut_end, cut_start), work.frames)
    if cut_end <= cut_start:
        return AudioBuffer(samples=work.samples.copy(), sample_rate=work.sample_rate)
    if action == "delete":
        return AudioBuffer(
            samples=np.concatenate([work.samples[:, :cut_start], work.samples[:, cut_end:]], axis=1),
            sample_rate=work.sample_rate,
        )
    action_base = work.samples.copy()
    if selected_gain_db:
        action_base[:, cut_start:cut_end] *= db_to_linear(selected_gain_db)
    out = action_base.copy()
    if action == "keep":
        return AudioBuffer(samples=out, sample_rate=work.sample_rate)
    if action == "lower":
        out[:, cut_start:cut_end] *= db_to_linear(lower_gain_db)
        return AudioBuffer(samples=out, sample_rate=work.sample_rate)
    if action == "rubberband_bridge":
        crossfade = max(fade_in_seconds, fade_out_seconds)
        rendered = rubberband_bridge_action(work, cut_start, cut_end, edge_source_seconds, crossfade)
        if selected_gain_db:
            rendered.samples[:, cut_start:cut_end] *= db_to_linear(selected_gain_db)
        return rendered
    if action in {"stretch_edge", "repeat_edge", "blend_edges", "interp_ambience"}:
        target_frames = cut_end - cut_start
        edge_frames = min(time_to_frame(edge_source_seconds, work.sample_rate), cut_start)
        source_start = max(cut_start - edge_frames, 0)
        if action == "stretch_edge":
            replacement = stretch_edge_over_gap(work, source_start, cut_start, target_frames)
        elif action == "repeat_edge":
            replacement = repeat_edge_over_gap(work, source_start, cut_start, target_frames)
        elif action == "blend_edges":
            replacement = blend_edges_over_gap(work, cut_start, cut_end, edge_frames)
        else:
            replacement = interpolate_gap_with_ambience(work, cut_start, cut_end, edge_frames)
        if selected_gain_db:
            replacement *= db_to_linear(selected_gain_db)
        out[:, cut_start:cut_end] = replacement[:, :target_frames]
        return AudioBuffer(samples=out, sample_rate=work.sample_rate)
    if action == "fade_silence":
        out[:, cut_start:cut_end] = 0.0
        fade_out_frames = min(time_to_frame(fade_out_seconds, work.sample_rate), cut_end - cut_start)
        fade_in_frames = min(time_to_frame(fade_in_seconds, work.sample_rate), cut_end - cut_start)
        if fade_out_frames:
            ramp = np.linspace(1.0, 0.0, fade_out_frames, dtype=np.float32)
            out[:, cut_start : cut_start + fade_out_frames] = action_base[:, cut_start : cut_start + fade_out_frames] * ramp
        if fade_in_frames:
            ramp = np.linspace(0.0, 1.0, fade_in_frames, dtype=np.float32)
            out[:, cut_end - fade_in_frames : cut_end] = action_base[:, cut_end - fade_in_frames : cut_end] * ramp
        return AudioBuffer(samples=out, sample_rate=work.sample_rate)
    out[:, cut_start:cut_end] = 0.0
    return AudioBuffer(samples=out, sample_rate=work.sample_rate)


def build_work_action_preview(
    project: RepairProject,
    main_audio: AudioBuffer,
    action: str,
    lower_gain_db: float = -12.0,
    selected_gain_db: float | None = None,
) -> AudioBuffer:
    """Render only the selected-area action on the base work clip, without lanes."""

    region = project.active_repair
    work = crop(main_audio, region.work_start_seconds, region.work_window_seconds)
    cut_start = time_to_frame(region.cut_start_seconds - region.work_start_seconds, work.sample_rate)
    cut_end = time_to_frame(region.cut_end_seconds - region.work_start_seconds, work.sample_rate)
    return apply_selection_action(
        work,
        cut_start,
        cut_end,
        action,
        lower_gain_db=lower_gain_db,
        selected_gain_db=region.selected_gain_db if selected_gain_db is None else selected_gain_db,
        fade_in_seconds=region.edge.fade_in_seconds,
        fade_out_seconds=region.edge.fade_out_seconds,
        edge_source_seconds=region.edge_source_seconds,
    )


def build_work_mix(
    project: RepairProject,
    main_audio: AudioBuffer,
    selection_action: str = "silence",
    lower_gain_db: float = -12.0,
    selected_gain_db: float | None = None,
) -> AudioBuffer:
    """Render the current work-window mix from the project state."""

    region = project.active_repair
    work = crop(main_audio, region.work_start_seconds, region.work_window_seconds)
    cut_start = time_to_frame(region.cut_start_seconds - region.work_start_seconds, work.sample_rate)
    cut_end = time_to_frame(region.cut_end_seconds - region.work_start_seconds, work.sample_rate)
    cut_start = min(max(cut_start, 0), work.frames)
    cut_end = min(max(cut_end, cut_start), work.frames)
    action_preview = apply_selection_action(
        work,
        cut_start,
        cut_end,
        selection_action,
        lower_gain_db=lower_gain_db,
        selected_gain_db=region.selected_gain_db if selected_gain_db is None else selected_gain_db,
        fade_in_seconds=region.edge.fade_in_seconds,
        fade_out_seconds=region.edge.fade_out_seconds,
        edge_source_seconds=region.edge_source_seconds,
    )
    mixed = action_preview.samples.copy()
    if selection_action == "delete":
        return AudioBuffer(samples=np.clip(mixed, -0.98, 0.98), sample_rate=work.sample_rate)
    for lane in active_lanes(region):
        lane_audio = prepare_lane_audio(lane, sample_rate=work.sample_rate, channels=work.channels)
        start_seconds = (region.cut_start_seconds + lane.offset_seconds) - region.work_start_seconds
        start = int(round(start_seconds * work.sample_rate))
        lane_samples = lane_audio.samples * db_to_linear(lane.gain_db)
        lane_samples = apply_fade(lane_samples, work.sample_rate, lane.fade_in_seconds, lane.fade_out_seconds)
        mix_into(mixed, lane_samples, start)
    return AudioBuffer(samples=np.clip(mixed, -0.98, 0.98), sample_rate=work.sample_rate)


def active_lanes(region: RepairRegion) -> list[ClipLane]:
    lanes: list[ClipLane] = []
    for lane in region.lanes:
        if lane.clip_items:
            for item in lane.clip_items:
                if not item.path or item.muted:
                    continue
                lanes.append(
                    replace(
                        lane,
                        path=item.path,
                        role=item.role,
                        offset_seconds=item.offset_seconds,
                        gain_db=item.gain_db,
                        speed_percent=item.speed_percent,
                        fade_in_seconds=item.fade_in_seconds,
                        fade_out_seconds=item.fade_out_seconds,
                        fit_to_cut=item.fit_to_cut,
                        muted=item.muted,
                        locked=item.locked,
                        clip_items=[],
                        selected_clip_index=0,
                    )
                )
        elif lane.path and not lane.muted:
            lanes.append(lane)
    return lanes


def mix_into(base: np.ndarray, insert: np.ndarray, start: int) -> None:
    if start >= base.shape[1]:
        return
    insert_start = 0
    if start < 0:
        insert_start = -start
        start = 0
    available = min(base.shape[1] - start, insert.shape[1] - insert_start)
    if available <= 0:
        return
    base[:, start : start + available] += insert[:, insert_start : insert_start + available]


def waveform_overview(buffer: AudioBuffer, points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return min/max envelope for display."""

    mono = buffer.samples.mean(axis=0)
    if mono.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    bucket = max(int(np.ceil(mono.size / points)), 1)
    padded = np.pad(mono, (0, (-mono.size) % bucket), mode="constant")
    shaped = padded.reshape(-1, bucket)
    return shaped.min(axis=1), shaped.max(axis=1)
