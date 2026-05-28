from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from livekit import rtc
from livekit.agents import APIConnectOptions, stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from ._future import require_attr


def make_frame() -> rtc.AudioFrame:
    sample_rate = 16000
    samples = sample_rate // 10
    return rtc.AudioFrame(
        data=b"\x01\x00" * samples,
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
        stream.push_frame(make_frame())
        stream.end_input()
        _ = [event async for event in stream]

    assert len(analyzer.calls) == 1
    assert analyzer.calls[0].sample_rate == 16000
    assert analyzer.calls[0].audio_ms == 100
