"""Playback helpers for the repair GUI."""

from __future__ import annotations

import numpy as np

from .audio_engine import AudioBuffer


class PlaybackEngine:
    """Small wrapper around sounddevice with a graceful import failure."""

    def __init__(self) -> None:
        self._sd = None
        self.error: str | None = None
        self._buffer: AudioBuffer | None = None
        self._start_seconds = 0.0
        try:
            import sounddevice as sd

            self._sd = sd
        except Exception as exc:  # pragma: no cover - depends on host audio setup
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self._sd is not None

    def play(self, buffer: AudioBuffer, start_seconds: float = 0.0) -> None:
        if self._sd is None:
            raise RuntimeError(f"sounddevice is not available: {self.error}")
        self._buffer = buffer
        self._start_seconds = max(start_seconds, 0.0)
        start_frame = min(int(round(self._start_seconds * buffer.sample_rate)), buffer.frames)
        data = np.clip(buffer.samples[:, start_frame:].T, -0.98, 0.98)
        self._sd.stop()
        self._sd.play(data, buffer.sample_rate, blocking=False)

    def pause(self) -> None:
        """Pause current playback.

        The first MVP does not track the exact live cursor yet, so resume starts
        from the original marker used by the last play command.
        """

        if self._sd is not None:
            self._sd.stop()

    def resume(self) -> None:
        if self._buffer is not None:
            self.play(self._buffer, self._start_seconds)

    def stop(self) -> None:
        if self._sd is not None:
            self._sd.stop()
        self._buffer = None
        self._start_seconds = 0.0
