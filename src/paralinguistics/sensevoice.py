from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .types import EmotionSignal

TAG_RE = re.compile(r"<\|[^|]+?\|>")

LANGUAGE_TAGS = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|yue|>": "yue",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|nospeech|>": "nospeech",
}

EMOTION_TAGS = {
    "<|HAPPY|>": "happy",
    "<|SAD|>": "sad",
    "<|ANGRY|>": "angry",
    "<|NEUTRAL|>": "neutral",
    "<|FEARFUL|>": "fearful",
    "<|DISGUSTED|>": "disgusted",
    "<|SURPRISED|>": "surprised",
    "<|EMO_UNKNOWN|>": "unknown",
}

EVENT_TAGS = {
    "<|Speech|>": "speech",
    "<|Speech_Noise|>": "speech_noise",
    "<|BGM|>": "bgm",
    "<|Applause|>": "applause",
    "<|Laughter|>": "laughter",
    "<|Cry|>": "cry",
    "<|Sneeze|>": "sneeze",
    "<|Breath|>": "breath",
    "<|Cough|>": "cough",
    "<|Sing|>": "sing",
    "<|Event_UNK|>": "unknown_event",
}


def parse_sensevoice_output(output: str | dict[str, Any]) -> EmotionSignal:
    if isinstance(output, dict) and "emotion" in output:
        return _parse_structured_signal(output)
    if isinstance(output, dict):
        result = output.get("result")
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and "emotion" in first:
                return _parse_structured_signal({**output, **first})

    raw_text, metadata = _extract_raw_text_and_metadata(output)
    tags = tuple(TAG_RE.findall(raw_text))
    emotions = [EMOTION_TAGS[tag] for tag in tags if tag in EMOTION_TAGS]

    emotion = "unknown"
    if emotions:
        counts = Counter(emotions)
        emotion = max(counts, key=lambda item: (counts[item], emotions.index(item)))

    language = next((LANGUAGE_TAGS[tag] for tag in tags if tag in LANGUAGE_TAGS), None)
    events = _ordered_unique(EVENT_TAGS[tag] for tag in tags if tag in EVENT_TAGS)

    return EmotionSignal(
        emotion=emotion,
        events=events,
        language=language,
        raw_tags=tags,
        source="sensevoice",
        latency_ms=_optional_int(metadata.get("latency_ms")),
        audio_ms=_optional_int(metadata.get("audio_ms")),
        model=metadata.get("model"),
    )


def _parse_structured_signal(output: dict[str, Any]) -> EmotionSignal:
    raw_tags = output.get("raw_tags") or ()
    events = output.get("events") or ()
    return EmotionSignal(
        emotion=str(output.get("emotion") or "unknown").lower(),
        events=tuple(str(event).lower() for event in events),
        language=output.get("language"),
        confidence=output.get("confidence"),
        raw_tags=tuple(str(tag) for tag in raw_tags),
        source=str(output.get("source") or "sensevoice"),
        latency_ms=_optional_int(output.get("latency_ms")),
        audio_ms=_optional_int(output.get("audio_ms")),
        model=output.get("model"),
    )


def _extract_raw_text_and_metadata(
    output: str | dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if isinstance(output, str):
        return output, {}

    metadata = dict(output)
    result = output.get("result")
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("raw_text") or first.get("text") or ""), metadata

    raw_text = output.get("raw_text") or output.get("text") or ""
    return str(raw_text), metadata


def _ordered_unique(values: object) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
