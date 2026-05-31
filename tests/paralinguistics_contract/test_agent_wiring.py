from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeTTS:
    def update_options(self, **kwargs: object) -> None:
        return None


class FakeAgentSession:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    async def start(self, **kwargs: object) -> None:
        return None


class FakeContext:
    def __init__(self, vad: object) -> None:
        self.proc = SimpleNamespace(userdata={"vad": vad})
        self.room = SimpleNamespace(name="test-room")
        self.connected = False
        self.log_context_fields: dict[str, object] = {}

    async def connect(self) -> None:
        self.connected = True


@pytest.mark.asyncio
async def test_agent_wires_prewarmed_vad_into_paralinguistic_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent

    captured: dict[str, object] = {}
    fake_vad = object()

    def fake_paralinguistic_stt(**kwargs: object) -> object:
        captured["stt_kwargs"] = kwargs
        return object()

    def fake_agent_session(**kwargs: object) -> FakeAgentSession:
        captured["session_kwargs"] = kwargs
        return FakeAgentSession(**kwargs)

    monkeypatch.setattr(agent.cartesia, "TTS", lambda **kwargs: FakeTTS())
    monkeypatch.setattr(agent.deepgram, "STTv2", lambda **kwargs: object())
    monkeypatch.setattr(agent.groq, "LLM", lambda **kwargs: object())
    monkeypatch.setattr(agent, "ParalinguisticSTT", fake_paralinguistic_stt)
    monkeypatch.setattr(agent, "AgentSession", fake_agent_session)
    monkeypatch.setattr(
        agent.ai_coustics, "audio_enhancement", lambda **kwargs: object()
    )

    ctx = FakeContext(fake_vad)
    await agent.my_agent(ctx)

    assert captured["stt_kwargs"]["vad"] is fake_vad
    assert captured["session_kwargs"]["vad"] is fake_vad
    assert captured["session_kwargs"]["turn_handling"]["turn_detection"] == "stt"
    assert captured["session_kwargs"]["turn_handling"]["interruption"] == {
        "mode": "vad"
    }
    assert captured["session_kwargs"]["turn_handling"]["preemptive_generation"] == {
        "enabled": True
    }
    assert "preemptive_generation" not in captured["session_kwargs"]
    assert ctx.connected is True
