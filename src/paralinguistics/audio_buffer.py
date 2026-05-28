from __future__ import annotations

from collections import deque

from livekit import rtc

from .types import BufferedAudio


class TurnAudioBuffer:
    def __init__(
        self, *, target_sample_rate: int = 16000, max_duration_ms: int = 30000
    ) -> None:
        self._target_sample_rate = target_sample_rate
        self._max_samples = target_sample_rate * max_duration_ms // 1000
        self._frames: deque[bytes] = deque()
        self._samples = 0

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        for converted in self._convert_frame(frame):
            pcm16 = converted.data.tobytes()
            samples = converted.samples_per_channel
            self._frames.append(pcm16)
            self._samples += samples
            self._trim_to_limit()

    def snapshot(self) -> BufferedAudio:
        return BufferedAudio(
            pcm16=b"".join(self._frames),
            sample_rate=self._target_sample_rate,
            audio_ms=self._samples * 1000 // self._target_sample_rate,
        )

    def drain(self) -> BufferedAudio:
        audio = self.snapshot()
        self.clear()
        return audio

    def clear(self) -> None:
        self._frames.clear()
        self._samples = 0

    def _convert_frame(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        frames = [frame]
        if frame.sample_rate != self._target_sample_rate:
            resampler = rtc.AudioResampler(
                frame.sample_rate,
                self._target_sample_rate,
                num_channels=frame.num_channels,
                quality=rtc.AudioResamplerQuality.HIGH,
            )
            frames = resampler.push(frame)
            frames.extend(resampler.flush())

        if frame.num_channels == 1:
            return frames

        return [self._downmix_mono(converted) for converted in frames]

    def _downmix_mono(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        data = frame.data.tobytes()
        channels = frame.num_channels
        mono = bytearray()
        for offset in range(0, len(data), channels * 2):
            total = 0
            for channel in range(channels):
                start = offset + channel * 2
                total += int.from_bytes(data[start : start + 2], "little", signed=True)
            mono.extend(int(total / channels).to_bytes(2, "little", signed=True))

        return rtc.AudioFrame(
            data=bytes(mono),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=len(mono) // 2,
        )

    def _trim_to_limit(self) -> None:
        while self._samples > self._max_samples and self._frames:
            first = self._frames.popleft()
            self._samples -= len(first) // 2
