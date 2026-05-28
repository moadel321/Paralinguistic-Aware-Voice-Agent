from __future__ import annotations

from livekit.agents import ChatContext

from ._future import require_attr


def test_append_speech_signal_context_adds_hidden_developer_message() -> None:
    append_context = require_attr(
        "paralinguistics.agent_context", "append_speech_signal_context"
    )
    signal_cls = require_attr("paralinguistics.types", "EmotionSignal")
    turn_ctx = ChatContext.empty()
    signal = signal_cls(
        emotion="angry",
        events=("speech",),
        language="en",
        confidence=None,
        raw_tags=("<|en|>", "<|Speech|>", "<|ANGRY|>"),
        source="sensevoice",
    )

    append_context(turn_ctx, signal)

    messages = turn_ctx.messages()
    assert len(messages) == 1
    assert messages[0].role == "developer"
    content = messages[0].text_content or ""
    assert "emotion=angry" in content
    assert "events=speech" in content
    assert "language=en" in content
    assert "adapt tone" in content.lower()
    assert "do not mention" in content.lower()
    assert "<|ANGRY|>" not in content


def test_append_speech_signal_context_skips_missing_signal() -> None:
    append_context = require_attr(
        "paralinguistics.agent_context", "append_speech_signal_context"
    )
    turn_ctx = ChatContext.empty()

    append_context(turn_ctx, None)

    assert turn_ctx.messages() == []
