from __future__ import annotations

from ._future import require_attr


def test_parse_voice_control_tag_returns_style_and_clean_text() -> None:
    parse_tagged_text = require_attr(
        "paralinguistics.voice_tags", "parse_voice_tagged_text"
    )

    result = parse_tagged_text(
        "[voice emotion=calm speed=0.90] I understand. Let us slow this down."
    )

    assert result.text == "I understand. Let us slow this down."
    assert result.style.emotion == "calm"
    assert result.style.speed == 0.90


def test_parse_voice_control_tag_accepts_emotion_only() -> None:
    parse_tagged_text = require_attr(
        "paralinguistics.voice_tags", "parse_voice_tagged_text"
    )

    result = parse_tagged_text("[voice emotion=excited] That sounds good.")

    assert result.text == "That sounds good."
    assert result.style.emotion == "excited"
    assert result.style.speed is None


def test_parse_voice_control_tag_leaves_plain_text_untouched() -> None:
    parse_tagged_text = require_attr(
        "paralinguistics.voice_tags", "parse_voice_tagged_text"
    )

    result = parse_tagged_text("Plain response with no control tag.")

    assert result.text == "Plain response with no control tag."
    assert result.style is None


def test_voice_tag_stream_filter_handles_split_opening_tag_without_leaking_it() -> None:
    stream_filter_cls = require_attr(
        "paralinguistics.voice_tags", "VoiceTagStreamFilter"
    )
    stream_filter = stream_filter_cls()

    chunks = [
        "[voice emo",
        "tion=calm speed=0.90]I",
        " understand.",
        "",
    ]
    emitted = "".join(stream_filter.push(chunk) for chunk in chunks)
    emitted += stream_filter.flush()

    assert emitted == "I understand."
    assert stream_filter.style.emotion == "calm"
    assert stream_filter.style.speed == 0.90


def test_voice_tag_stream_filter_passes_through_plain_streaming_text() -> None:
    stream_filter_cls = require_attr(
        "paralinguistics.voice_tags", "VoiceTagStreamFilter"
    )
    stream_filter = stream_filter_cls()

    emitted = "".join(
        stream_filter.push(chunk) for chunk in ("Hello", ", ", "this has no tag.")
    )
    emitted += stream_filter.flush()

    assert emitted == "Hello, this has no tag."
    assert stream_filter.style is None
