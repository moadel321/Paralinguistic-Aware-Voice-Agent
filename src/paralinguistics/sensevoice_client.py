from __future__ import annotations

import asyncio
import io
import wave
from typing import Protocol

import aiohttp

from .sensevoice import parse_sensevoice_output
from .types import BufferedAudio, EmotionSignal


class SenseVoiceTransport(Protocol):
    async def post_audio(
        self, *, base_url: str, pcm16: bytes, sample_rate: int, timeout_s: float
    ) -> dict: ...


class AiohttpSenseVoiceTransport:
    async def post_audio(
        self, *, base_url: str, pcm16: bytes, sample_rate: int, timeout_s: float
    ) -> dict:
        form = aiohttp.FormData()
        form.add_field(
            "files",
            _wav_bytes(pcm16, sample_rate),
            filename="turn.wav",
            content_type="audio/wav",
        )
        form.add_field("lang", "auto")
        form.add_field("use_itn", "false")

        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(f"{base_url.rstrip('/')}/api/v1/ser", data=form) as response,
        ):
            response.raise_for_status()
            return await response.json()


class SenseVoiceSidecarClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 0.2,
        transport: SenseVoiceTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._transport = transport or AiohttpSenseVoiceTransport()
        self.latest_signal: EmotionSignal | None = None

    async def analyze_pcm(
        self, pcm16: bytes, *, sample_rate: int
    ) -> EmotionSignal | None:
        if not pcm16:
            self.latest_signal = None
            return None

        try:
            payload = await asyncio.wait_for(
                self._transport.post_audio(
                    base_url=self._base_url,
                    pcm16=pcm16,
                    sample_rate=sample_rate,
                    timeout_s=self._timeout_s,
                ),
                timeout=self._timeout_s,
            )
        except TimeoutError:
            self.latest_signal = None
            return None
        except (aiohttp.ClientError, OSError):
            self.latest_signal = None
            return None

        signal = parse_sensevoice_output(payload)
        self.latest_signal = signal
        return signal

    async def analyze_buffer(
        self, buffered_audio: BufferedAudio
    ) -> EmotionSignal | None:
        return await self.analyze_pcm(
            buffered_audio.pcm16,
            sample_rate=buffered_audio.sample_rate,
        )


def _wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return buffer.getvalue()
