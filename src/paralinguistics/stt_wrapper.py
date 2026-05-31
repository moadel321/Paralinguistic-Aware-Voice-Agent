from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from livekit import rtc
from livekit.agents import APIConnectOptions, stt, utils, vad
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

from .audio_buffer import TurnAudioBuffer
from .types import BufferedAudio

logger = logging.getLogger(__name__)


class SpeechAnalyzer(Protocol):
    async def analyze_buffer(self, buffered_audio: BufferedAudio) -> object: ...


class SpeechActivityDetector(Protocol):
    def stream(self) -> vad.VADStream: ...


@dataclass(frozen=True)
class _TimedAudioChunk:
    pcm16: bytes
    start_ms: int
    end_ms: int


class _TimestampedAudioBuffer:
    def __init__(
        self,
        *,
        target_sample_rate: int,
        max_duration_ms: int,
    ) -> None:
        self._target_sample_rate = target_sample_rate
        self._max_duration_ms = max_duration_ms
        self._chunks: deque[_TimedAudioChunk] = deque()
        self._next_start_ms: int | None = None

    def push_frame(self, frame: rtc.AudioFrame, *, stream_start_offset_ms: int) -> None:
        if self._next_start_ms is None:
            self._next_start_ms = stream_start_offset_ms

        frame_buffer = TurnAudioBuffer(
            target_sample_rate=self._target_sample_rate,
            max_duration_ms=self._max_duration_ms,
        )
        frame_buffer.push_frame(frame)
        audio = frame_buffer.snapshot()
        if not audio.pcm16:
            return

        start_ms = self._next_start_ms
        end_ms = start_ms + audio.audio_ms
        self._chunks.append(
            _TimedAudioChunk(pcm16=audio.pcm16, start_ms=start_ms, end_ms=end_ms)
        )
        self._next_start_ms = end_ms
        self._trim()

    def slice(self, *, start_ms: int, end_ms: int) -> BufferedAudio:
        if end_ms <= start_ms:
            return BufferedAudio(
                pcm16=b"", sample_rate=self._target_sample_rate, audio_ms=0
            )

        chunks: list[bytes] = []
        for chunk in self._chunks:
            if chunk.end_ms <= start_ms:
                continue
            if chunk.start_ms >= end_ms:
                break

            overlap_start_ms = max(start_ms, chunk.start_ms)
            overlap_end_ms = min(end_ms, chunk.end_ms)
            start_sample = (
                (overlap_start_ms - chunk.start_ms) * self._target_sample_rate // 1000
            )
            end_sample = (
                (overlap_end_ms - chunk.start_ms) * self._target_sample_rate // 1000
            )
            if end_sample > start_sample:
                chunks.append(chunk.pcm16[start_sample * 2 : end_sample * 2])

        pcm16 = b"".join(chunks)
        audio_ms = len(pcm16) // 2 * 1000 // self._target_sample_rate
        return BufferedAudio(
            pcm16=pcm16,
            sample_rate=self._target_sample_rate,
            audio_ms=audio_ms,
        )

    def _trim(self) -> None:
        if self._next_start_ms is None:
            return

        cutoff_ms = self._next_start_ms - self._max_duration_ms
        while self._chunks and self._chunks[0].end_ms <= cutoff_ms:
            self._chunks.popleft()


class ParalinguisticSTT(stt.STT):
    def __init__(
        self,
        *,
        wrapped_stt: stt.STT,
        analyzer: SpeechAnalyzer,
        vad: SpeechActivityDetector | None = None,
        target_sample_rate: int = 16000,
        max_buffer_duration_ms: int = 30000,
        min_sidecar_audio_ms: int = 500,
        max_sidecar_audio_ms: int = 15000,
        sidecar_audio_padding_ms: int = 200,
        sidecar_analysis_wait_s: float = 3.0,
    ) -> None:
        super().__init__(capabilities=wrapped_stt.capabilities)
        self._wrapped_stt = wrapped_stt
        self._analyzer = analyzer
        self._vad = vad
        self._target_sample_rate = target_sample_rate
        self._max_buffer_duration_ms = max_buffer_duration_ms
        self._min_sidecar_audio_ms = min_sidecar_audio_ms
        self._max_sidecar_audio_ms = max_sidecar_audio_ms
        self._sidecar_audio_padding_ms = sidecar_audio_padding_ms
        self._sidecar_analysis_wait_s = sidecar_analysis_wait_s

    @property
    def model(self) -> str:
        return self._wrapped_stt.model

    @property
    def provider(self) -> str:
        return self._wrapped_stt.provider

    async def _recognize_impl(
        self,
        buffer: object,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        return await self._wrapped_stt.recognize(
            buffer=buffer,
            language=language,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        return ParalinguisticSpeechStream(
            stt=self,
            wrapped_stt=self._wrapped_stt,
            analyzer=self._analyzer,
            vad=self._vad,
            target_sample_rate=self._target_sample_rate,
            max_buffer_duration_ms=self._max_buffer_duration_ms,
            min_sidecar_audio_ms=self._min_sidecar_audio_ms,
            max_sidecar_audio_ms=self._max_sidecar_audio_ms,
            sidecar_audio_padding_ms=self._sidecar_audio_padding_ms,
            sidecar_analysis_wait_s=self._sidecar_analysis_wait_s,
            language=language,
            conn_options=conn_options,
        )


class ParalinguisticSpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: ParalinguisticSTT,
        wrapped_stt: stt.STT,
        analyzer: SpeechAnalyzer,
        vad: SpeechActivityDetector | None,
        target_sample_rate: int,
        max_buffer_duration_ms: int,
        min_sidecar_audio_ms: int,
        max_sidecar_audio_ms: int,
        sidecar_audio_padding_ms: int,
        sidecar_analysis_wait_s: float,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._wrapped_stt = wrapped_stt
        self._analyzer = analyzer
        self._vad = vad
        self._language = language
        self._wrapped_conn_options = conn_options
        self._fallback_buffer = TurnAudioBuffer(
            target_sample_rate=target_sample_rate,
            max_duration_ms=max_buffer_duration_ms,
        )
        self._target_sample_rate = target_sample_rate
        self._max_buffer_duration_ms = max_buffer_duration_ms
        self._min_sidecar_audio_ms = min_sidecar_audio_ms
        self._max_sidecar_audio_ms = max_sidecar_audio_ms
        self._sidecar_audio_padding_ms = sidecar_audio_padding_ms
        self._sidecar_analysis_wait_s = sidecar_analysis_wait_s
        self._timestamped_buffer = _TimestampedAudioBuffer(
            target_sample_rate=target_sample_rate,
            max_duration_ms=max(max_buffer_duration_ms, max_sidecar_audio_ms, 30000),
        )
        self._audio_seen = asyncio.Event()
        self._input_done = asyncio.Event()
        self._vad_segment_started = asyncio.Event()
        self._vad_segment_done = asyncio.Event()
        self._vad_segment_done.set()
        self._pending_transcript_window_ms: tuple[int, int] | None = None
        self._pending_vad_audio: BufferedAudio | None = None

    async def _run(self) -> None:
        async with self._wrapped_stt.stream(
            language=self._language,
            conn_options=self._wrapped_conn_options,
        ) as wrapped_stream:
            vad_stream = self._vad.stream() if self._vad is not None else None
            forward_input_task = asyncio.create_task(
                self._forward_input(wrapped_stream, vad_stream),
                name="paralinguistic_stt_forward_input",
            )
            forward_events_task = asyncio.create_task(
                self._forward_events(wrapped_stream),
                name="paralinguistic_stt_forward_events",
            )
            tasks = [forward_input_task, forward_events_task]
            if vad_stream is not None:
                tasks.append(
                    asyncio.create_task(
                        self._forward_vad_events(vad_stream),
                        name="paralinguistic_stt_forward_vad_events",
                    )
                )
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.cancel_and_wait(*tasks)
                if vad_stream is not None:
                    await vad_stream.aclose()

    async def _forward_input(
        self,
        wrapped_stream: stt.RecognizeStream,
        vad_stream: vad.VADStream | None,
    ) -> None:
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                self._sync_wrapped_timing(wrapped_stream)
                self._timestamped_buffer.push_frame(
                    item,
                    stream_start_offset_ms=int(self.start_time_offset * 1000),
                )
                if vad_stream is None:
                    self._fallback_buffer.push_frame(item)
                else:
                    vad_stream.push_frame(item)
                self._audio_seen.set()
                wrapped_stream.push_frame(item)
            elif isinstance(item, self._FlushSentinel):
                wrapped_stream.flush()
                if vad_stream is not None:
                    vad_stream.flush()

        wrapped_stream.end_input()
        if vad_stream is not None:
            vad_stream.end_input()
        self._input_done.set()

    async def _forward_events(self, wrapped_stream: stt.RecognizeStream) -> None:
        async for event in wrapped_stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                self._capture_transcript_window(event)
            if event.type == stt.SpeechEventType.END_OF_SPEECH:
                await self._analyze_sidecar_audio_for_turn()
            self._event_ch.send_nowait(event)

    async def _forward_vad_events(self, vad_stream: vad.VADStream) -> None:
        try:
            async for event in vad_stream:
                if event.type == vad.VADEventType.START_OF_SPEECH:
                    self._vad_segment_started.set()
                    self._vad_segment_done.clear()
                    self._pending_vad_audio = None
                elif event.type == vad.VADEventType.END_OF_SPEECH:
                    try:
                        self._pending_vad_audio = self._buffered_audio_from_frames(
                            event.frames
                        )
                    finally:
                        self._vad_segment_started.clear()
                        self._vad_segment_done.set()
        finally:
            self._vad_segment_done.set()

    async def _wait_for_vad_sidecar_analysis(self) -> None:
        try:
            if (
                self._pending_vad_audio is None
                and self._vad_segment_done.is_set()
                and not self._vad_segment_started.is_set()
            ):
                await asyncio.wait_for(
                    self._wait_for_pending_vad_audio(),
                    timeout=min(0.15, self._sidecar_analysis_wait_s),
                )

            if (
                self._vad_segment_done.is_set()
                and not self._vad_segment_started.is_set()
            ):
                return

            await asyncio.wait_for(
                self._vad_segment_done.wait(),
                timeout=self._sidecar_analysis_wait_s,
            )
        except TimeoutError:
            return

    async def _wait_for_pending_vad_audio(self) -> None:
        while (
            self._pending_vad_audio is None
            and self._vad_segment_done.is_set()
            and not self._vad_segment_started.is_set()
        ):
            await asyncio.sleep(0.01)

    async def _analyze_sidecar_audio_for_turn(self) -> None:
        try:
            if (
                self._pending_transcript_window_ms is not None
                and await self._analyze_transcript_window()
            ):
                return

            if self._vad is not None:
                await self._wait_for_vad_sidecar_analysis()
                if self._pending_vad_audio is not None:
                    await self._analyze_buffered_audio(
                        self._pending_vad_audio,
                        source="vad_fallback",
                    )
                return

            await self._analyze_current_turn()
        finally:
            self._pending_transcript_window_ms = None
            self._pending_vad_audio = None

    async def _analyze_transcript_window(self) -> bool:
        if self._pending_transcript_window_ms is None:
            return False

        start_ms, end_ms = self._pending_transcript_window_ms
        start_ms = max(0, start_ms - self._sidecar_audio_padding_ms)
        end_ms += self._sidecar_audio_padding_ms
        max_start_ms = max(start_ms, end_ms - self._max_sidecar_audio_ms)
        buffered_audio = self._timestamped_buffer.slice(
            start_ms=max_start_ms,
            end_ms=end_ms,
        )
        return await self._analyze_buffered_audio(
            buffered_audio, source="stt_word_window"
        )

    async def _analyze_current_turn(self) -> None:
        if not self._fallback_buffer.snapshot().pcm16 and not self._input_done.is_set():
            audio_seen_task = asyncio.create_task(self._audio_seen.wait())
            input_done_task = asyncio.create_task(self._input_done.wait())
            try:
                await asyncio.wait(
                    (audio_seen_task, input_done_task),
                    timeout=0.05,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                await utils.aio.cancel_and_wait(audio_seen_task, input_done_task)

        buffered_audio = self._fallback_buffer.drain()
        self._audio_seen.clear()
        await self._analyze_buffered_audio(buffered_audio, source="fallback_turn")

    async def _analyze_buffered_audio(
        self, buffered_audio: BufferedAudio, *, source: str
    ) -> bool:
        if not buffered_audio.pcm16:
            return False
        if buffered_audio.audio_ms < self._min_sidecar_audio_ms:
            logger.debug(
                "sidecar audio skipped source=%s audio_ms=%s min_audio_ms=%s",
                source,
                buffered_audio.audio_ms,
                self._min_sidecar_audio_ms,
            )
            return False
        logger.info(
            "sidecar audio selected source=%s audio_ms=%s",
            source,
            buffered_audio.audio_ms,
        )
        await self._analyzer.analyze_buffer(buffered_audio)
        return True

    def _buffered_audio_from_frames(
        self, frames: list[rtc.AudioFrame]
    ) -> BufferedAudio:
        max_duration_ms = max(
            self._max_buffer_duration_ms,
            self._max_sidecar_audio_ms,
            30000,
        )
        buffer = TurnAudioBuffer(
            target_sample_rate=self._target_sample_rate,
            max_duration_ms=max_duration_ms,
        )
        for frame in frames:
            buffer.push_frame(frame)

        audio = buffer.snapshot()
        max_samples = self._target_sample_rate * self._max_sidecar_audio_ms // 1000
        max_bytes = max_samples * 2
        if len(audio.pcm16) <= max_bytes:
            return audio

        return BufferedAudio(
            pcm16=audio.pcm16[-max_bytes:],
            sample_rate=audio.sample_rate,
            audio_ms=self._max_sidecar_audio_ms,
        )

    def _capture_transcript_window(self, event: stt.SpeechEvent) -> None:
        window = _speech_event_audio_window_ms(event)
        if window is None:
            return

        if self._pending_transcript_window_ms is None:
            self._pending_transcript_window_ms = window
            return

        current_start_ms, current_end_ms = self._pending_transcript_window_ms
        start_ms, end_ms = window
        self._pending_transcript_window_ms = (
            min(current_start_ms, start_ms),
            max(current_end_ms, end_ms),
        )

    def _sync_wrapped_timing(self, wrapped_stream: stt.RecognizeStream) -> None:
        if not hasattr(wrapped_stream, "start_time_offset"):
            return
        if wrapped_stream.start_time_offset != self.start_time_offset:
            wrapped_stream.start_time_offset = self.start_time_offset


def _speech_event_audio_window_ms(event: stt.SpeechEvent) -> tuple[int, int] | None:
    word_start_s: float | None = None
    word_end_s: float | None = None
    fallback_start_s: float | None = None
    fallback_end_s: float | None = None

    for alternative in event.alternatives:
        if alternative.words:
            for word in alternative.words:
                word_start = _maybe_float(word.start_time)
                word_end = _maybe_float(word.end_time)
                if word_start is None or word_end is None or word_end <= word_start:
                    continue
                word_start_s = (
                    word_start
                    if word_start_s is None
                    else min(word_start_s, word_start)
                )
                word_end_s = (
                    word_end if word_end_s is None else max(word_end_s, word_end)
                )

        alt_start = _maybe_float(alternative.start_time)
        alt_end = _maybe_float(alternative.end_time)
        if alt_start is not None and alt_end is not None and alt_end > alt_start:
            fallback_start_s = (
                alt_start
                if fallback_start_s is None
                else min(fallback_start_s, alt_start)
            )
            fallback_end_s = (
                alt_end if fallback_end_s is None else max(fallback_end_s, alt_end)
            )

    start_s = word_start_s if word_start_s is not None else fallback_start_s
    end_s = word_end_s if word_end_s is not None else fallback_end_s

    if start_s is None or end_s is None or end_s <= start_s:
        return None

    return int(start_s * 1000), int(end_s * 1000)


def _maybe_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
