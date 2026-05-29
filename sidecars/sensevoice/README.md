# SenseVoice Sidecar

This sidecar exposes SenseVoice speech emotion and event understanding at
`/api/v1/ser` for the main LiveKit agent.

It does not commit model weights. The container downloads `iic/SenseVoiceSmall`
into the `sensevoice-models` Docker volume the first time it starts.

Important: run this parent-owned sidecar folder, not the ignored upstream
`SenseVoice/` checkout at the repository root. The root checkout contains a
different `ser_api.py` that imports its local `model.py`; that path can crash
on startup with `SinusoidalPositionEncoder` missing PyTorch module internals.

Run it locally:

```powershell
cd D:\Code\Paralinguistic-Aware-Voice-Agent\sidecars\sensevoice
docker compose up --build
```

The main agent defaults to `SENSEVOICE_SIDECAR_URL=http://127.0.0.1:50000`.

Useful environment variables:

- `SENSEVOICE_DEVICE`: `auto`, `cuda:0`, or `cpu`.
- `SENSEVOICE_MODEL`: defaults to `iic/SenseVoiceSmall`.
- `SENSEVOICE_TIMEOUT_S`: set on the main agent, defaults to `1.5`. Keep it
  above the sidecar's observed inference wall time or the agent will drop the
  response before it can use the detected emotion. Values below `1.5` are
  clamped to the default by the agent.
- `SENSEVOICE_LOG_LEVEL`: set on the sidecar, defaults to `INFO`.

If the old root checkout was started by mistake, stop it before starting this
sidecar:

```powershell
cd D:\Code\Paralinguistic-Aware-Voice-Agent\SenseVoice
docker compose down --remove-orphans

cd D:\Code\Paralinguistic-Aware-Voice-Agent\sidecars\sensevoice
docker compose up --build
```
