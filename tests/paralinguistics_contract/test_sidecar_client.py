from __future__ import annotations

import asyncio
import logging

import pytest

from ._future import assert_no_transcript, require_attr


class FakeSenseVoiceTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def post_audio(
        self, *, base_url: str, pcm16: bytes, sample_rate: int, timeout_s: float
    ) -> dict:
        self.calls.append(
            {
                "base_url": base_url,
                "pcm16": pcm16,
                "sample_rate": sample_rate,
                "timeout_s": timeout_s,
            }
        )
        return self.response


class SlowSenseVoiceTransport:
    async def post_audio(
        self, *, base_url: str, pcm16: bytes, sample_rate: int, timeout_s: float
    ) -> dict:
        await asyncio.sleep(timeout_s * 5)
        return {"result": [{"raw_text": "<|en|><|ANGRY|><|Speech|>ignored"}]}


@pytest.mark.asyncio
async def test_sidecar_client_posts_pcm_and_returns_emotion_signal() -> None:
    client_cls = require_attr(
        "paralinguistics.sensevoice_client", "SenseVoiceSidecarClient"
    )
    transport = FakeSenseVoiceTransport(
        {
            "result": [
                {
                    "raw_text": "<|en|><|Speech|>ignore transcript<|ANGRY|>",
                    "clean_text": "ignore transcript",
                }
            ],
            "latency_ms": 31,
            "audio_ms": 100,
            "model": "iic/SenseVoiceSmall",
        }
    )
    client = client_cls(
        base_url="http://sensevoice.test",
        timeout_s=0.2,
        transport=transport,
    )

    signal = await client.analyze_pcm(b"\x00\x00" * 1600, sample_rate=16000)

    assert signal.emotion == "angry"
    assert signal.events == ("speech",)
    assert signal.latency_ms == 31
    assert_no_transcript(signal)
    assert transport.calls == [
        {
            "base_url": "http://sensevoice.test",
            "pcm16": b"\x00\x00" * 1600,
            "sample_rate": 16000,
            "timeout_s": 0.2,
        }
    ]


@pytest.mark.asyncio
async def test_sidecar_client_default_timeout_tracks_local_sidecar_latency() -> None:
    client_cls = require_attr(
        "paralinguistics.sensevoice_client", "SenseVoiceSidecarClient"
    )
    transport = FakeSenseVoiceTransport({"emotion": "neutral"})
    client = client_cls(base_url="http://sensevoice.test", transport=transport)

    await client.analyze_pcm(b"\x00\x00" * 1600, sample_rate=16000)

    assert transport.calls[0]["timeout_s"] == 1.5


def test_resolve_sensevoice_timeout_clamps_stale_env_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolve_timeout = require_attr(
        "paralinguistics.sensevoice_client", "resolve_sensevoice_timeout_s"
    )

    with caplog.at_level(logging.WARNING, logger="paralinguistics.sensevoice_client"):
        timeout_s = resolve_timeout("0.2")

    assert timeout_s == 1.5
    assert "SENSEVOICE_TIMEOUT_S=0.20s is below the supported default" in caplog.text


def test_resolve_sensevoice_timeout_uses_valid_higher_value() -> None:
    resolve_timeout = require_attr(
        "paralinguistics.sensevoice_client", "resolve_sensevoice_timeout_s"
    )

    assert resolve_timeout("2.0") == 2.0


@pytest.mark.asyncio
async def test_sidecar_client_logs_detected_emotion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client_cls = require_attr(
        "paralinguistics.sensevoice_client", "SenseVoiceSidecarClient"
    )
    transport = FakeSenseVoiceTransport(
        {
            "emotion": "happy",
            "events": ["speech"],
            "language": "en",
            "audio_ms": 1200,
            "latency_ms": 540,
        }
    )
    client = client_cls(
        base_url="http://sensevoice.test",
        timeout_s=1.5,
        transport=transport,
    )

    with caplog.at_level(logging.INFO, logger="paralinguistics.sensevoice_client"):
        await client.analyze_pcm(b"\x00\x00" * 1600, sample_rate=16000)

    assert "sensevoice emotion detected: emotion=happy" in caplog.text
    assert "latency_ms=540" in caplog.text


@pytest.mark.asyncio
async def test_sidecar_client_timeout_returns_none_instead_of_blocking_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client_cls = require_attr(
        "paralinguistics.sensevoice_client", "SenseVoiceSidecarClient"
    )
    client = client_cls(
        base_url="http://sensevoice.test",
        timeout_s=0.01,
        transport=SlowSenseVoiceTransport(),
    )

    with caplog.at_level(logging.WARNING, logger="paralinguistics.sensevoice_client"):
        signal = await client.analyze_pcm(b"\x00\x00" * 1600, sample_rate=16000)

    assert signal is None
    assert "sensevoice analysis timed out after 0.01s" in caplog.text
