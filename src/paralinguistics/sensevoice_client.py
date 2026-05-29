from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import Protocol

import aiohttp

from .sensevoice import parse_sensevoice_output
from .types import BufferedAudio, EmotionSignal

logger = logging.getLogger(__name__)
DEFAULT_SENSEVOICE_TIMEOUT_S = 1.5


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
        timeout_s: float = DEFAULT_SENSEVOICE_TIMEOUT_S,
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
            logger.debug("sensevoice analysis skipped: empty audio buffer")
            return None

        audio_ms = _audio_duration_ms(pcm16, sample_rate)
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
            logger.warning(
                "sensevoice analysis timed out after %.2fs for %sms of audio",
                self._timeout_s,
                audio_ms,
            )
            return None
        except (aiohttp.ClientError, OSError) as exc:
            self.latest_signal = None
            logger.warning(
                "sensevoice analysis failed for %sms of audio: %s",
                audio_ms,
                exc,
            )
            return None

        signal = parse_sensevoice_output(payload)
        self.latest_signal = signal
        logger.info(
            "sensevoice emotion detected: emotion=%s events=%s language=%s audio_ms=%s latency_ms=%s",
            signal.emotion,
            ",".join(signal.events) if signal.events else "none",
            signal.language or "unknown",
            signal.audio_ms if signal.audio_ms is not None else audio_ms,
            signal.latency_ms,
        )
        return signal

    async def analyze_buffer(
        self, buffered_audio: BufferedAudio
    ) -> EmotionSignal | None:
        return await self.analyze_pcm(
            buffered_audio.pcm16,
            sample_rate=buffered_audio.sample_rate,
        )


def resolve_sensevoice_timeout_s(configured: str | None) -> float:
    if configured is None or not configured.strip():
        return DEFAULT_SENSEVOICE_TIMEOUT_S

    try:
        timeout_s = float(configured)
    except ValueError:
        logger.warning(
            "invalid SENSEVOICE_TIMEOUT_S=%r; using %.2fs",
            configured,
            DEFAULT_SENSEVOICE_TIMEOUT_S,
        )
        return DEFAULT_SENSEVOICE_TIMEOUT_S

    if timeout_s < DEFAULT_SENSEVOICE_TIMEOUT_S:
        logger.warning(
            "SENSEVOICE_TIMEOUT_S=%.2fs is below the supported default %.2fs; using %.2fs",
            timeout_s,
            DEFAULT_SENSEVOICE_TIMEOUT_S,
            DEFAULT_SENSEVOICE_TIMEOUT_S,
        )
        return DEFAULT_SENSEVOICE_TIMEOUT_S

    return timeout_s


def _wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return buffer.getvalue()


def _audio_duration_ms(pcm16: bytes, sample_rate: int) -> int:
    if sample_rate <= 0:
        return 0
    return int((len(pcm16) // 2) * 1000 / sample_rate)
