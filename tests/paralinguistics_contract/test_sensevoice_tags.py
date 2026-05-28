from __future__ import annotations

import pytest

from ._future import assert_no_transcript, require_attr


def test_parse_sensevoice_output_extracts_emotion_events_and_language_only() -> None:
    parse = require_attr("paralinguistics.sensevoice", "parse_sensevoice_output")

    signal = parse(
        {
            "result": [
                {
                    "raw_text": "<|en|><|Speech|>Do not keep this text<|ANGRY|><|woitn|>",
                    "clean_text": "Do not keep this text",
                }
            ],
            "latency_ms": 42,
            "audio_ms": 1800,
            "model": "iic/SenseVoiceSmall",
        }
    )

    assert signal.emotion == "angry"
    assert signal.events == ("speech",)
    assert signal.language == "en"
    assert signal.raw_tags == ("<|en|>", "<|Speech|>", "<|ANGRY|>", "<|woitn|>")
    assert signal.latency_ms == 42
    assert signal.audio_ms == 1800
    assert signal.model == "iic/SenseVoiceSmall"
    assert_no_transcript(signal)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("<|HAPPY|>", "happy"),
        ("<|SAD|>", "sad"),
        ("<|ANGRY|>", "angry"),
        ("<|NEUTRAL|>", "neutral"),
        ("<|FEARFUL|>", "fearful"),
        ("<|DISGUSTED|>", "disgusted"),
        ("<|SURPRISED|>", "surprised"),
        ("<|EMO_UNKNOWN|>", "unknown"),
    ],
)
def test_parse_sensevoice_output_normalizes_supported_emotion_tags(
    tag: str, expected: str
) -> None:
    parse = require_attr("paralinguistics.sensevoice", "parse_sensevoice_output")

    signal = parse(f"<|en|><|Speech|>ignored transcript {tag}")

    assert signal.emotion == expected
    assert_no_transcript(signal)


def test_parse_sensevoice_output_uses_most_frequent_emotion_tag() -> None:
    parse = require_attr("paralinguistics.sensevoice", "parse_sensevoice_output")

    signal = parse("<|en|><|HAPPY|><|SAD|><|SAD|><|Speech|>ignored")

    assert signal.emotion == "sad"


def test_parse_sensevoice_output_preserves_ordered_unique_events() -> None:
    parse = require_attr("paralinguistics.sensevoice", "parse_sensevoice_output")

    signal = parse(
        "<|en|><|Laughter|><|Speech|><|Cough|><|Laughter|><|NEUTRAL|>ignored"
    )

    assert signal.events == ("laughter", "speech", "cough")


def test_parse_sensevoice_output_defaults_to_unknown_when_no_emotion_tag_exists() -> (
    None
):
    parse = require_attr("paralinguistics.sensevoice", "parse_sensevoice_output")

    signal = parse("<|en|><|Speech|>ignored")

    assert signal.emotion == "unknown"
    assert signal.events == ("speech",)
    assert signal.language == "en"
