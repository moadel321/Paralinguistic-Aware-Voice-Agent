from __future__ import annotations

from ._future import require_attr


def test_transcript_emotion_overrides_unknown_sensevoice_for_frustration() -> None:
    signal_cls = require_attr("paralinguistics.types", "EmotionSignal")
    resolve = require_attr(
        "paralinguistics.emotion_resolution", "resolve_emotion_signal"
    )
    signal = signal_cls(
        emotion="unknown",
        events=("speech",),
        raw_tags=("<|en|>", "<|EMO_UNKNOWN|>", "<|Speech|>"),
        source="sensevoice",
    )

    resolved = resolve(
        signal,
        transcript="I'm feeling frustrated. It's frustrating overall.",
    )

    assert resolved.emotion == "frustrated"
    assert resolved.source == "sensevoice+transcript"
    assert resolved.raw_tags == signal.raw_tags


def test_transcript_emotion_overrides_concrete_sensevoice_for_explicit_calm() -> None:
    signal_cls = require_attr("paralinguistics.types", "EmotionSignal")
    resolve = require_attr(
        "paralinguistics.emotion_resolution", "resolve_emotion_signal"
    )
    signal = signal_cls(emotion="sad", events=("speech",), source="sensevoice")

    resolved = resolve(
        signal,
        transcript="I'm feeling really calm. It's soothing and easy on the mind.",
    )

    assert resolved.emotion == "calm"
    assert resolved.source == "sensevoice+transcript"


def test_concrete_sensevoice_emotion_wins_without_explicit_override() -> None:
    signal_cls = require_attr("paralinguistics.types", "EmotionSignal")
    resolve = require_attr(
        "paralinguistics.emotion_resolution", "resolve_emotion_signal"
    )
    signal = signal_cls(emotion="happy", events=("speech",), source="sensevoice")

    resolved = resolve(signal, transcript="This is working.")

    assert resolved is signal


def test_transcript_can_supply_signal_when_sidecar_is_missing() -> None:
    resolve = require_attr(
        "paralinguistics.emotion_resolution", "resolve_emotion_signal"
    )

    resolved = resolve(None, transcript="I'm excited to talk about this.")

    assert resolved.emotion == "excited"
    assert resolved.source == "transcript"
