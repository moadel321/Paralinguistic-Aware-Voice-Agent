from __future__ import annotations

from livekit import rtc

from ._future import require_attr


def make_frame(
    *, fill: int, duration_ms: int = 100, sample_rate: int = 16000
) -> rtc.AudioFrame:
    samples = sample_rate * duration_ms // 1000
    sample = int(fill).to_bytes(2, byteorder="little", signed=True)
    return rtc.AudioFrame(
        data=sample * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def test_turn_audio_buffer_snapshots_pcm16_and_metadata() -> None:
    buffer_cls = require_attr("paralinguistics.audio_buffer", "TurnAudioBuffer")
    buffer = buffer_cls(target_sample_rate=16000, max_duration_ms=30000)

    buffer.push_frame(make_frame(fill=7, duration_ms=100))
    audio = buffer.snapshot()

    assert audio.sample_rate == 16000
    assert audio.audio_ms == 100
    assert audio.pcm16 == (7).to_bytes(2, "little", signed=True) * 1600


def test_turn_audio_buffer_drain_returns_audio_and_clears_state() -> None:
    buffer_cls = require_attr("paralinguistics.audio_buffer", "TurnAudioBuffer")
    buffer = buffer_cls(target_sample_rate=16000, max_duration_ms=30000)

    buffer.push_frame(make_frame(fill=1, duration_ms=100))
    drained = buffer.drain()
    empty = buffer.snapshot()

    assert drained.audio_ms == 100
    assert drained.pcm16
    assert empty.audio_ms == 0
    assert empty.pcm16 == b""


def test_turn_audio_buffer_keeps_recent_audio_within_max_duration() -> None:
    buffer_cls = require_attr("paralinguistics.audio_buffer", "TurnAudioBuffer")
    buffer = buffer_cls(target_sample_rate=16000, max_duration_ms=300)

    for fill in range(5):
        buffer.push_frame(make_frame(fill=fill, duration_ms=100))

    audio = buffer.snapshot()

    assert audio.audio_ms == 300
    assert len(audio.pcm16) == 16000 * 2 * 300 // 1000
    assert audio.pcm16[:2] == (2).to_bytes(2, "little", signed=True)
