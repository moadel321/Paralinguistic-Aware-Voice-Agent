from __future__ import annotations

import re
from dataclasses import replace

from .types import EmotionSignal

_WEAK_SENSEVOICE_EMOTIONS = {"unknown", "neutral"}
_EXPLICIT_OVERRIDE_EMOTIONS = {"calm", "frustrated"}

_TRANSCRIPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "frustrated",
        re.compile(r"\b(frustrated|frustrating|fed up|annoyed|irritated)\b", re.I),
    ),
    ("angry", re.compile(r"\b(angry|mad|furious|pissed off)\b", re.I)),
    (
        "fearful",
        re.compile(
            r"\b(scared|afraid|fearful|anxious|nervous|worried|concerned)\b", re.I
        ),
    ),
    ("sad", re.compile(r"\b(sad|upset|depressed|unhappy|down)\b", re.I)),
    ("surprised", re.compile(r"\b(surprised|shocked|amazed)\b", re.I)),
    ("excited", re.compile(r"\b(excited|thrilled)\b", re.I)),
    ("calm", re.compile(r"\b(calm|relaxed|soothing|easy on the mind)\b", re.I)),
    ("happy", re.compile(r"\b(happy|glad|wonderful|delighted)\b", re.I)),
)


def infer_transcript_emotion(transcript: str | None) -> str | None:
    text = (transcript or "").strip()
    if not text:
        return None

    for emotion, pattern in _TRANSCRIPT_PATTERNS:
        if pattern.search(text):
            return emotion

    return None


def resolve_emotion_signal(
    signal: EmotionSignal | None,
    *,
    transcript: str | None,
) -> EmotionSignal | None:
    transcript_emotion = infer_transcript_emotion(transcript)
    if transcript_emotion is None:
        return signal

    if signal is None:
        return EmotionSignal(emotion=transcript_emotion, source="transcript")

    if (
        signal.emotion in _WEAK_SENSEVOICE_EMOTIONS
        or transcript_emotion in _EXPLICIT_OVERRIDE_EMOTIONS
    ):
        return replace(
            signal,
            emotion=transcript_emotion,
            source=f"{signal.source}+transcript",
        )

    return signal
