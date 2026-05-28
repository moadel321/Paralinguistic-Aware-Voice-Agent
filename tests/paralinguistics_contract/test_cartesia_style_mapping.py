from __future__ import annotations

import pytest

from ._future import require_attr


@pytest.mark.parametrize(
    ("emotion", "expected_emotion", "expected_speed"),
    [
        ("angry", "calm", 0.90),
        ("frustrated", "calm", 0.90),
        ("disgusted", "calm", 0.95),
        ("sad", "sympathetic", 0.95),
        ("fearful", "sympathetic", 0.90),
        ("happy", "excited", 1.05),
        ("excited", "excited", 1.05),
        ("surprised", "curious", 1.03),
        ("neutral", "calm", 1.00),
        ("calm", "calm", 1.00),
        ("unknown", "calm", 1.00),
    ],
)
def test_map_user_emotion_to_cartesia_style(
    emotion: str, expected_emotion: str, expected_speed: float
) -> None:
    map_style = require_attr(
        "paralinguistics.style", "map_user_emotion_to_cartesia_style"
    )

    style = map_style(emotion)

    assert style.emotion == expected_emotion
    assert style.speed == pytest.approx(expected_speed)
    assert 0.6 <= style.speed <= 1.5


def test_cartesia_style_exports_update_options_shape() -> None:
    map_style = require_attr(
        "paralinguistics.style", "map_user_emotion_to_cartesia_style"
    )

    style = map_style("angry")

    assert style.to_cartesia_kwargs() == {"emotion": "calm", "speed": 0.90}


def test_cartesia_supported_emotions_include_all_mapped_outputs() -> None:
    map_style = require_attr(
        "paralinguistics.style", "map_user_emotion_to_cartesia_style"
    )
    supported = require_attr("paralinguistics.style", "SUPPORTED_CARTESIA_EMOTIONS")

    mapped = {
        map_style(emotion).emotion
        for emotion in (
            "angry",
            "frustrated",
            "disgusted",
            "sad",
            "fearful",
            "happy",
            "excited",
            "surprised",
            "neutral",
            "unknown",
        )
    }

    assert mapped <= set(supported)
