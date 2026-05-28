from __future__ import annotations

import asyncio
from typing import Protocol

from livekit import rtc
from livekit.agents import APIConnectOptions, stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

from .audio_buffer import TurnAudioBuffer
from .types import BufferedAudio


class SpeechAnalyzer(Protocol):
    async def analyze_buffer(self, buffered_audio: BufferedAudio) -> object: ...


class ParalinguisticSTT(stt.STT):
    def __init__(
        self,
        *,
        wrapped_stt: stt.STT,
        analyzer: SpeechAnalyzer,
        target_sample_rate: int = 16000,
        max_buffer_duration_ms: int = 30000,
    ) -> None:
        super().__init__(capabilities=wrapped_stt.capabilities)
        self._wrapped_stt = wrapped_stt
        self._analyzer = analyzer
        self._target_sample_rate = target_sample_rate
        self._max_buffer_duration_ms = max_buffer_duration_ms

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
            target_sample_rate=self._target_sample_rate,
            max_buffer_duration_ms=self._max_buffer_duration_ms,
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
        target_sample_rate: int,
        max_buffer_duration_ms: int,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._wrapped_stt = wrapped_stt
        self._analyzer = analyzer
        self._language = language
        self._wrapped_conn_options = conn_options
        self._buffer = TurnAudioBuffer(
            target_sample_rate=target_sample_rate,
            max_duration_ms=max_buffer_duration_ms,
        )
        self._audio_seen = asyncio.Event()
        self._input_done = asyncio.Event()

    async def _run(self) -> None:
        async with self._wrapped_stt.stream(
            language=self._language,
            conn_options=self._wrapped_conn_options,
        ) as wrapped_stream:
            forward_input_task = asyncio.create_task(
                self._forward_input(wrapped_stream),
                name="paralinguistic_stt_forward_input",
            )
            forward_events_task = asyncio.create_task(
                self._forward_events(wrapped_stream),
                name="paralinguistic_stt_forward_events",
            )
            try:
                await asyncio.gather(forward_input_task, forward_events_task)
            finally:
                await utils.aio.cancel_and_wait(forward_input_task, forward_events_task)

    async def _forward_input(self, wrapped_stream: stt.RecognizeStream) -> None:
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                self._buffer.push_frame(item)
                self._audio_seen.set()
                wrapped_stream.push_frame(item)
            elif isinstance(item, self._FlushSentinel):
                wrapped_stream.flush()

        wrapped_stream.end_input()
        self._input_done.set()

    async def _forward_events(self, wrapped_stream: stt.RecognizeStream) -> None:
        async for event in wrapped_stream:
            if event.type == stt.SpeechEventType.END_OF_SPEECH:
                await self._analyze_current_turn()
            self._event_ch.send_nowait(event)

    async def _analyze_current_turn(self) -> None:
        if not self._buffer.snapshot().pcm16 and not self._input_done.is_set():
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

        buffered_audio = self._buffer.drain()
        self._audio_seen.clear()
        if not buffered_audio.pcm16:
            return
        await self._analyzer.analyze_buffer(buffered_audio)
