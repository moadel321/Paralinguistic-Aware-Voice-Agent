# Paralinguistic-Aware-Voice-Agent

A real-time voice agent that listens beyond words, using prosody, emotion, pauses, and vocal cues for context-aware dialogue.

## SenseVoice Sidecar

The production SenseVoice sidecar lives in `sidecars/sensevoice`. Start it from
that folder:

```powershell
cd D:\Code\Paralinguistic-Aware-Voice-Agent\sidecars\sensevoice
docker compose up --build
```

Do not start the ignored root `SenseVoice/` checkout; it is only a local
upstream reference and is not the sidecar wired to the LiveKit agent.
