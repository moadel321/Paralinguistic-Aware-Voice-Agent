from __future__ import annotations

import re

from .types import TaggedText, VoiceStyle

VOICE_TAG_RE = re.compile(r"^\s*\[voice(?P<body>[^\]]*)\]\s*", re.IGNORECASE)
KEY_VALUE_RE = re.compile(
    r"(?P<key>emotion|speed)\s*=\s*(?P<value>[^\s\]]+)", re.IGNORECASE
)


def parse_voice_tagged_text(text: str) -> TaggedText:
    match = VOICE_TAG_RE.match(text)
    if not match:
        return TaggedText(text=text, style=None)

    style = _parse_style(match.group("body"))
    return TaggedText(text=text[match.end() :], style=style)


class VoiceTagStreamFilter:
    def __init__(self) -> None:
        self._buffer = ""
        self._decided = False
        self.style: VoiceStyle | None = None

    def push(self, chunk: str) -> str:
        if self._decided:
            return chunk

        self._buffer += chunk
        stripped = self._buffer.lstrip()
        if not stripped:
            return ""

        if not stripped.startswith("["):
            self._decided = True
            text = self._buffer
            self._buffer = ""
            return text

        lower = stripped.lower()
        if not lower.startswith("[voice"):
            self._decided = True
            text = self._buffer
            self._buffer = ""
            return text

        if "]" not in stripped:
            return ""

        parsed = parse_voice_tagged_text(self._buffer)
        self.style = parsed.style
        self._decided = True
        self._buffer = ""
        return parsed.text

    def flush(self) -> str:
        if not self._decided and self._buffer:
            parsed = parse_voice_tagged_text(self._buffer)
            self.style = parsed.style
            self._decided = True
            self._buffer = ""
            return parsed.text
        return ""


def _parse_style(body: str) -> VoiceStyle:
    values = {
        match.group("key").lower(): match.group("value")
        for match in KEY_VALUE_RE.finditer(body)
    }
    emotion = values.get("emotion")
    speed_value = values.get("speed")
    speed = None
    if speed_value is not None:
        try:
            speed = float(speed_value)
        except ValueError:
            speed = None

    return VoiceStyle(emotion=(emotion or "calm").lower(), speed=speed)
