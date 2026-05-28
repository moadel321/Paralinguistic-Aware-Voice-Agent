# SenseVoice Sidecar

This sidecar exposes SenseVoice speech emotion and event understanding at
`/api/v1/ser` for the main LiveKit agent.

It does not commit model weights. The container downloads `iic/SenseVoiceSmall`
into the `sensevoice-models` Docker volume the first time it starts.

Run it locally:

```powershell
cd sidecars/sensevoice
docker compose up --build
```

The main agent defaults to `SENSEVOICE_SIDECAR_URL=http://127.0.0.1:50000`.

Useful environment variables:

- `SENSEVOICE_DEVICE`: `auto`, `cuda:0`, or `cpu`.
- `SENSEVOICE_MODEL`: defaults to `iic/SenseVoiceSmall`.
