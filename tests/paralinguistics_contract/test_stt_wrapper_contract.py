from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from livekit import rtc
from livekit.agents import APIConnectOptions, stt, vad
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice.io import TimedString

from ._future import require_attr


def make_frame(*, fill: int = 1, duration_ms: int = 100) -> rtc.AudioFrame:
    sample_rate = 16000
    samples = sample_rate * duration_ms // 1000
    return rtc.AudioFrame(
        data=fill.to_bytes(2, "little", signed=True) * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class FakeWrappedStream:
    def __init__(self, events: list[stt.SpeechEvent]) -> None:
        self.events = events
        self.frames: list[rtc.AudioFrame] = []
        self.flushed = False
        self.ended = False
        self.start_time_offset = 0.0

    async def __aenter__(self) -> FakeWrappedStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self.frames.append(frame)

    def flush(self) -> None:
        self.flushed = True

    def end_input(self) -> None:
        self.ended = True

    def __aiter__(self) -> AsyncIterator[stt.SpeechEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[stt.SpeechEvent]:
        for event in self.events:
            yield event


class FakeWrappedSTT(stt.STT):
    def __init__(self, events: list[stt.SpeechEvent]) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=False,
            )
        )
        self.stream_instance = FakeWrappedStream(events)

    async def _recognize_impl(self, *args: object, **kwargs: object) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> FakeWrappedStream:
        return self.stream_instance


class FakeVADStream:
    def __init__(self, events: list[vad.VADEvent]) -> None:
        self.events = events
        self.frames: list[rtc.AudioFrame] = []
        self.flushed = False
        self.ended = False
        self.closed = False

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self.frames.append(frame)

    def flush(self) -> None:
        self.flushed = True

    def end_input(self) -> None:
        self.ended = True

    async def aclose(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[vad.VADEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[vad.VADEvent]:
        for event in self.events:
            yield event


class FakeVAD:
    def __init__(self, events: list[vad.VADEvent]) -> None:
        self.stream_instance = FakeVADStream(events)

    def stream(self) -> FakeVADStream:
        return self.stream_instance


class RecordingAnalyzer:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def analyze_buffer(self, buffered_audio: object) -> None:
        self.calls.append(buffered_audio)


@pytest.mark.asyncio
async def test_paralinguistic_stt_preserves_wrapped_capabilities() -> None:
    wrapper_cls = require_attr("paralinguistics.stt_wrapper", "ParalinguisticSTT")
    wrapped = FakeWrappedSTT(events=[])

    wrapper = wrapper_cls(wrapped_stt=wrapped, analyzer=RecordingAnalyzer())

    assert wrapper.capabilities == wrapped.capabilities
    assert wrapper.provider == wrapped.provider
    assert wrapper.model == wrapped.model


@pytest.mark.asyncio
async def test_paralinguistic_stt_forwards_events_and_audio_to_wrapped_stt() -> None:
    wrapper_cls = require_attr("paralinguistics.stt_wrapper", "ParalinguisticSTT")
    final_event = stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="en", text="hello")],
    )
    events = [
        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH),
        final_event,
        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH),
    ]
    wrapped = FakeWrappedSTT(events=events)
    analyzer = RecordingAnalyzer()
    wrapper = wrapper_cls(wrapped_stt=wrapped, analyzer=analyzer)
    frame = make_frame()

    async with wrapper.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS) as stream:
        stream.push_frame(frame)
        stream.flush()
        stream.end_input()
        observed = [event async for event in stream]

    assert observed == events
    assert wrapped.stream_instance.frames == [frame]
    assert wrapped.stream_instance.flushed is True
    assert wrapped.stream_instance.ended is True


@pytest.mark.asyncio
async def test_paralinguistic_stt_schedules_sidecar_analysis_at_end_of_speech() -> None:
    wrapper_cls = require_attr("paralinguistics.stt_wrapper", "ParalinguisticSTT")
    events = [stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)]
    wrapped = FakeWrappedSTT(events=events)
    analyzer = RecordingAnalyzer()
    wrapper = wrapper_cls(wrapped_stt=wrapped, analyzer=analyzer)

    async with wrapper.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS) as stream:
        stream.push_frame(make_frame(duration_ms=600))
        stream.end_input()
        _ = [event async for event in stream]

    assert len(analyzer.calls) == 1
    assert analyzer.calls[0].sample_rate == 16000
    assert analyzer.calls[0].audio_ms == 600


@pytest.mark.asyncio
async def test_paralinguistic_stt_uses_vad_segment_for_sidecar_when_available() -> None:
    wrapper_cls = require_attr("paralinguistics.stt_wrapper", "ParalinguisticSTT")
    broad_frame = make_frame(fill=1, duration_ms=1200)
    speech_frame = make_frame(fill=7, duration_ms=600)
    vad_events = [
        vad.VADEvent(
            type=vad.VADEventType.START_OF_SPEECH,
            samples_index=0,
            timestamp=0.0,
            speech_duration=0.6,
            silence_duration=0.0,
            frames=[speech_frame],
            speaking=True,
        ),
        vad.VADEvent(
            type=vad.VADEventType.END_OF_SPEECH,
            samples_index=9600,
            timestamp=0.6,
            speech_duration=0.6,
            silence_duration=0.6,
            frames=[speech_frame],
            speaking=False,
        ),
    ]
    wrapped = FakeWrappedSTT(
        events=[stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)]
    )
    analyzer = RecordingAnalyzer()
    fake_vad = FakeVAD(vad_events)
    wrapper = wrapper_cls(wrapped_stt=wrapped, analyzer=analyzer, vad=fake_vad)

    async with wrapper.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS) as stream:
        stream.push_frame(broad_frame)
        stream.push_frame(speech_frame)
        stream.end_input()
        _ = [event async for event in stream]

    assert fake_vad.stream_instance.frames == [broad_frame, speech_frame]
    assert len(analyzer.calls) == 1
    assert analyzer.calls[0].sample_rate == 16000
    assert analyzer.calls[0].audio_ms == 600
    assert analyzer.calls[0].pcm16 == speech_frame.data.tobytes()


@pytest.mark.asyncio
async def test_paralinguistic_stt_prefers_transcript_word_window_over_broad_vad() -> (
    None
):
    wrapper_cls = require_attr("paralinguistics.stt_wrapper", "ParalinguisticSTT")
    leading_frame = make_frame(fill=1, duration_ms=1000)
    speech_frame = make_frame(fill=7, duration_ms=800)
    trailing_frame = make_frame(fill=2, duration_ms=1000)
    final_event = stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language="en",
                text="hello",
                words=[TimedString("hello", start_time=3.0, end_time=3.8)],
            )
        ],
    )
    fake_vad = FakeVAD(
        [
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=44800,
                timestamp=2.8,
                speech_duration=2.8,
                silence_duration=0.6,
                frames=[leading_frame, speech_frame, trailing_frame],
                speaking=False,
            )
        ]
    )
    wrapped = FakeWrappedSTT(
        events=[final_event, stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)]
    )
    analyzer = RecordingAnalyzer()
    wrapper = wrapper_cls(
        wrapped_stt=wrapped,
        analyzer=analyzer,
        vad=fake_vad,
        sidecar_audio_padding_ms=0,
    )

    async with wrapper.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS) as stream:
        stream.start_time_offset = 2.0
        stream.push_frame(leading_frame)
        stream.push_frame(speech_frame)
        stream.push_frame(trailing_frame)
        stream.end_input()
        _ = [event async for event in stream]

    assert len(analyzer.calls) == 1
    assert wrapped.stream_instance.start_time_offset == pytest.approx(2.0)
    assert analyzer.calls[0].audio_ms == 800
    assert analyzer.calls[0].pcm16 == speech_frame.data.tobytes()


@pytest.mark.asyncio
async def test_paralinguistic_stt_ignores_broad_alternative_window_when_words_exist() -> (
    None
):
    wrapper_cls = require_attr("paralinguistics.stt_wrapper", "ParalinguisticSTT")
    leading_frame = make_frame(fill=1, duration_ms=2500)
    speech_frame = make_frame(fill=7, duration_ms=700)
    trailing_frame = make_frame(fill=2, duration_ms=2500)
    final_event = stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language="en",
                text="hello",
                start_time=0.0,
                end_time=5.7,
                words=[TimedString("hello", start_time=2.5, end_time=3.2)],
            )
        ],
    )
    fake_vad = FakeVAD(
        [
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=91200,
                timestamp=5.7,
                speech_duration=5.7,
                silence_duration=0.6,
                frames=[leading_frame, speech_frame, trailing_frame],
                speaking=False,
            )
        ]
    )
    wrapped = FakeWrappedSTT(
        events=[final_event, stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)]
    )
    analyzer = RecordingAnalyzer()
    wrapper = wrapper_cls(
        wrapped_stt=wrapped,
        analyzer=analyzer,
        vad=fake_vad,
        sidecar_audio_padding_ms=0,
    )

    async with wrapper.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS) as stream:
        stream.push_frame(leading_frame)
        stream.push_frame(speech_frame)
        stream.push_frame(trailing_frame)
        stream.end_input()
        _ = [event async for event in stream]

    assert len(analyzer.calls) == 1
    assert analyzer.calls[0].audio_ms == 700
    assert analyzer.calls[0].pcm16 == speech_frame.data.tobytes()


@pytest.mark.asyncio
async def test_paralinguistic_stt_falls_back_to_vad_when_transcript_window_is_empty() -> (
    None
):
    wrapper_cls = require_attr("paralinguistics.stt_wrapper", "ParalinguisticSTT")
    speech_frame = make_frame(fill=7, duration_ms=600)
    final_event = stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language="en",
                text="hello",
                words=[TimedString("hello", start_time=10.0, end_time=10.6)],
            )
        ],
    )
    fake_vad = FakeVAD(
        [
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=9600,
                timestamp=0.6,
                speech_duration=0.6,
                silence_duration=0.6,
                frames=[speech_frame],
                speaking=False,
            )
        ]
    )
    wrapped = FakeWrappedSTT(
        events=[final_event, stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)]
    )
    analyzer = RecordingAnalyzer()
    wrapper = wrapper_cls(wrapped_stt=wrapped, analyzer=analyzer, vad=fake_vad)

    async with wrapper.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS) as stream:
        stream.push_frame(speech_frame)
        stream.end_input()
        _ = [event async for event in stream]

    assert len(analyzer.calls) == 1
    assert analyzer.calls[0].audio_ms == 600
    assert analyzer.calls[0].pcm16 == speech_frame.data.tobytes()


@pytest.mark.asyncio
async def test_paralinguistic_stt_skips_too_short_vad_segments() -> None:
    wrapper_cls = require_attr("paralinguistics.stt_wrapper", "ParalinguisticSTT")
    short_frame = make_frame(fill=3, duration_ms=100)
    fake_vad = FakeVAD(
        [
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=1600,
                timestamp=0.1,
                speech_duration=0.1,
                silence_duration=0.6,
                frames=[short_frame],
                speaking=False,
            )
        ]
    )
    wrapped = FakeWrappedSTT(
        events=[stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)]
    )
    analyzer = RecordingAnalyzer()
    wrapper = wrapper_cls(wrapped_stt=wrapped, analyzer=analyzer, vad=fake_vad)

    async with wrapper.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS) as stream:
        stream.push_frame(short_frame)
        stream.end_input()
        _ = [event async for event in stream]

    assert analyzer.calls == []
