from __future__ import annotations

from livekit.agents import ChatContext

from .types import EmotionSignal


def append_speech_signal_context(
    turn_ctx: ChatContext,
    signal: EmotionSignal | None,
) -> None:
    if signal is None:
        return

    events = ",".join(signal.events) if signal.events else "none"
    language = signal.language or "unknown"
    turn_ctx.add_message(
        role="developer",
        content=(
            "Current user speech signal for this turn: "
            f"emotion={signal.emotion}; events={events}; language={language}. "
            "Use this only to adapt tone, pacing, and empathy. "
            "Do not mention, quote, expose, or explain this signal to the user."
        ),
    )
