from __future__ import annotations

import os
import re
import time
from enum import Enum
from io import BytesIO
from typing import Annotated, Any

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from funasr import AutoModel

TARGET_FS = 16000
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


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


def resolve_device() -> str:
    configured = os.getenv("SENSEVOICE_DEVICE", "auto")
    if configured != "auto":
        return configured
    return "cuda:0" if torch.cuda.is_available() else "cpu"


model_dir = os.getenv("SENSEVOICE_MODEL", "iic/SenseVoiceSmall")
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    device=resolve_device(),
)

app = FastAPI(title="SenseVoice SER sidecar")


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """
    <!DOCTYPE html>
    <html>
      <head><meta charset=utf-8><title>SenseVoice SER sidecar</title></head>
      <body><a href='./docs'>Documents of API</a></body>
    </html>
    """


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": model_dir}


@app.post("/api/v1/ser")
async def analyze_speech_emotion(
    files: Annotated[list[UploadFile], File(description="wav or mp3 audios")],
    keys: Annotated[str | None, Form(description="comma-separated audio keys")] = None,
    lang: Annotated[
        Language, Form(description="language of audio content")
    ] = Language.auto,
    use_itn: Annotated[
        bool, Form(description="apply inverse text normalization")
    ] = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    signals: list[dict[str, Any]] = []
    audio_ms = 0
    file_keys = keys.split(",") if keys else [file.filename for file in files]

    for index, file in enumerate(files):
        audio, file_audio_ms = await _load_audio(file)
        audio_ms += file_audio_ms
        result = model.generate(
            input=audio,
            cache={},
            language=lang.value or "auto",
            use_itn=use_itn,
            batch_size=1,
        )
        raw_text = _raw_text_from_result(result)
        key = file_keys[index] if index < len(file_keys) else file.filename
        signals.append(_signal_from_raw_text(raw_text, key))

    latency_ms = int((time.perf_counter() - started) * 1000)
    if len(signals) == 1:
        return {
            **signals[0],
            "latency_ms": latency_ms,
            "audio_ms": audio_ms,
            "model": model_dir,
        }

    return {
        "result": signals,
        "latency_ms": latency_ms,
        "audio_ms": audio_ms,
        "model": model_dir,
    }


async def _load_audio(file: UploadFile) -> tuple[np.ndarray, int]:
    file_io = BytesIO(await file.read())
    audio, audio_fs = torchaudio.load(file_io)
    if audio_fs != TARGET_FS:
        resampler = torchaudio.transforms.Resample(
            orig_freq=audio_fs,
            new_freq=TARGET_FS,
        )
        audio = resampler(audio)

    audio = audio.mean(0)
    audio_ms = int(audio.shape[-1] * 1000 / TARGET_FS)
    return audio.numpy().astype(np.float32), audio_ms


def _raw_text_from_result(result: object) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("text") or first.get("raw_text") or "")
        return str(first)
    if isinstance(result, dict):
        return str(result.get("text") or result.get("raw_text") or "")
    return str(result or "")


def _signal_from_raw_text(raw_text: str, key: str | None) -> dict[str, Any]:
    tags = TAG_RE.findall(raw_text)
    emotions = [EMOTION_TAGS[tag] for tag in tags if tag in EMOTION_TAGS]
    emotion = _most_frequent(emotions) if emotions else "unknown"
    language = next((LANGUAGE_TAGS[tag] for tag in tags if tag in LANGUAGE_TAGS), None)
    events = _ordered_unique(EVENT_TAGS[tag] for tag in tags if tag in EVENT_TAGS)
    return {
        "key": key,
        "emotion": emotion,
        "events": events,
        "language": language,
        "raw_tags": tags,
        "source": "sensevoice",
    }


def _most_frequent(values: list[str]) -> str:
    return max(
        set(values), key=lambda value: (values.count(value), -values.index(value))
    )


def _ordered_unique(values: object) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
