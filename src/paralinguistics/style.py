from __future__ import annotations

from .types import VoiceStyle

SUPPORTED_CARTESIA_EMOTIONS = (
    "happy",
    "excited",
    "curious",
    "calm",
    "sympathetic",
    "neutral",
)

_EMOTION_STYLE_MAP = {
    "angry": VoiceStyle(emotion="calm", speed=0.90),
    "frustrated": VoiceStyle(emotion="calm", speed=0.90),
    "disgusted": VoiceStyle(emotion="calm", speed=0.95),
    "sad": VoiceStyle(emotion="sympathetic", speed=0.95),
    "fearful": VoiceStyle(emotion="sympathetic", speed=0.90),
    "happy": VoiceStyle(emotion="excited", speed=1.05),
    "excited": VoiceStyle(emotion="excited", speed=1.05),
    "surprised": VoiceStyle(emotion="curious", speed=1.03),
    "neutral": VoiceStyle(emotion="calm", speed=1.00),
    "calm": VoiceStyle(emotion="calm", speed=1.00),
    "unknown": VoiceStyle(emotion="calm", speed=1.00),
}


def map_user_emotion_to_cartesia_style(emotion: str | None) -> VoiceStyle:
    normalized = (emotion or "unknown").strip().lower()
    return _EMOTION_STYLE_MAP.get(normalized, _EMOTION_STYLE_MAP["unknown"])
