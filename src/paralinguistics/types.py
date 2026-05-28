from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmotionSignal:
    emotion: str
    events: tuple[str, ...] = ()
    language: str | None = None
    confidence: float | None = None
    raw_tags: tuple[str, ...] = ()
    source: str = "sensevoice"
    latency_ms: int | None = None
    audio_ms: int | None = None
    model: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class VoiceStyle:
    emotion: str
    speed: float | None = None

    def to_cartesia_kwargs(self) -> dict[str, str | float]:
        kwargs: dict[str, str | float] = {"emotion": self.emotion}
        if self.speed is not None:
            kwargs["speed"] = self.speed
        return kwargs


@dataclass(frozen=True)
class TaggedText:
    text: str
    style: VoiceStyle | None = None


@dataclass(frozen=True)
class BufferedAudio:
    pcm16: bytes
    sample_rate: int
    audio_ms: int
