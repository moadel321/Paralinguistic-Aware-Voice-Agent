# paralinguistic-aware voice agent

a real-time livekit voice agent that listens beyond the words.

the agent uses normal cascaded voice components, but adds a side channel for
paralinguistic understanding. that means it can use tone, emotion, pauses, and
speech events as context, not just the transcript.

## what is paralinguistic understanding?

paralinguistic understanding means reading the parts of speech that sit around
the words.

it includes things like tone, pace, stress, hesitation, silence, laughter,
sighs, breath, and emotional color. the transcript might say "i'm fine." the
voice can say "i'm annoyed."

this repo treats those signals as useful context for the agent. not as the only
source of truth, but as one more input that helps the conversation feel less
blind.

## why this exists

current cascaded voice agent systems are reliable. they are easy to debug and
each piece has a clear job.

`stt` turns speech into text. the `llm` decides what to say. `tts` speaks the
response.

that works, but it drops a lot of context.

if a user sounds frustrated, tired, uncertain, excited, or calm, the transcript
often loses that. the agent can respond to the literal words and still miss the
actual state of the conversation.

the motive here is simple. keep the reliability of the cascaded stack, but give
the agent a small amount of audio awareness so it can adapt its response and
voice style.

## what this is

this is a livekit voice agent with a local sensevoice sidecar.

the main agent handles the real-time conversation. the sidecar analyzes short
speech windows and returns paralinguistic tags like emotion, language, and
speech events.

right now the stack is:

- `livekit agents` for the realtime agent runtime
- `deepgram` for streaming `stt`
- `silero vad` for speech gating
- `sensevoice` for emotion, event, and language tags
- `groq` for the `llm`
- `cartesia` for `tts`

## how this fits in

this repo does not replace the normal voice agent pipeline. it adds a second
lane next to it.

normal lane:

```text
user audio -> stt -> transcript -> llm -> tts -> agent audio
```

paralinguistic lane:

```text
user audio -> vad/stt crop -> sensevoice sidecar -> emotion signal -> agent context + tts style
```

the key part is the crop. the sidecar should not receive a huge chunk of room
audio. it should receive the smallest useful speech window for the user turn.

the agent now prefers precise `stt` word timings when they exist. if those are
missing, it falls back to `silero vad`. this keeps sensevoice focused on the
actual speech instead of silence, pauses, and unrelated audio.

## how the emotion signal is used

after each user turn, the agent reads the latest sensevoice signal.

if sensevoice returns a strong emotion, that emotion is used directly. if it
returns `unknown` or `neutral`, the agent can use obvious transcript cues as a
fallback. for example, if the user says "i'm feeling frustrated", the agent can
resolve that as `frustrated` even if the acoustic tag is `unknown`.

the resolved emotion is added to the chat context and also maps into cartesia
voice style. so the agent can answer with a tone that fits the user better.

## repo layout

```text
src/
  agent.py                         livekit agent entrypoint
  paralinguistics/
    stt_wrapper.py                 wraps stt and sends cropped audio to sensevoice
    sensevoice_client.py           http client for the sidecar
    emotion_resolution.py          combines sensevoice tags with transcript cues
    style.py                       maps user emotion to cartesia voice style

sidecars/
  sensevoice/
    ser_api.py                     fastapi sidecar for sensevoice
    docker-compose.yaml            local sidecar service
    Dockerfile                     sidecar image

tests/
  paralinguistics_contract/        behavior tests for the paralinguistic path
```

## setup

install python dependencies:

```powershell
uv sync
```

create `.env` from `.env.example` and fill in the keys:

```text
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

DEEPGRAM_API_KEY=
GROQ_API_KEY=
CARTESIA_API_KEY=

SENSEVOICE_SIDECAR_URL=http://127.0.0.1:50000
SENSEVOICE_TIMEOUT_S=1.5
```

## run the sidecar

start the parent-owned sidecar from `sidecars/sensevoice`:

```powershell
cd sidecars\sensevoice
docker compose up --build
```

do not start the ignored root `SenseVoice/` checkout. that folder is only a
local upstream reference. it is not the sidecar wired to the livekit agent.

the sidecar downloads `iic/SenseVoiceSmall` into the `sensevoice-models` docker
volume the first time it starts.

useful sidecar env vars:

- `SENSEVOICE_DEVICE`: `auto`, `cuda:0`, or `cpu`
- `SENSEVOICE_MODEL`: defaults to `iic/SenseVoiceSmall`
- `SENSEVOICE_LOG_LEVEL`: defaults to `INFO`

## run the agent

from the repo root:

```powershell
uv run python src\agent.py dev
```

the agent expects the sidecar at:

```text
http://127.0.0.1:50000
```

override that with `SENSEVOICE_SIDECAR_URL` if needed.

## useful logs

when the paralinguistic path is working, the agent logs should show lines like:

```text
sidecar audio selected source=stt_word_window audio_ms=...
sensevoice emotion detected: emotion=... events=... language=... raw_tags=...
```

`source=stt_word_window` means the sidecar received a crop based on transcript
word timing. `source=vad_fallback` means word timing was missing and the agent
used the vad segment instead.

## tests

run the full test suite:

```powershell
uv run pytest -q
```

run lint and format checks:

```powershell
uv run ruff check
uv run ruff format --check
```

## current limits

sensevoice is still an acoustic and tag model. it can return `unknown` for real
speech, especially if the crop is too broad or the emotional cue is mostly in
the words.

that is why this repo uses both signals:

- sensevoice for acoustic emotion and speech tags
- transcript cues for obvious self-described emotion

the goal is not perfect emotion detection. the goal is a practical signal that
makes the voice agent less tone-deaf while keeping the cascaded stack stable.
