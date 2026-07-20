# Running Voicebox in Docker

Voicebox ships as a headless server + browser UI in a container. The native
desktop (Tauri) shell is **not** containerized — you don't need it. The `web`
frontend is the same React app, served by the same FastAPI backend on the same
port, and you reach it in a browser tab.

Everything functional is there: profiles, generation, the Whisper verify loop,
and the per-engine parameter knobs (Qwen sampling, TADA inference options).

## Variants

| Target | Command |
| --- | --- |
| CPU (default) | `docker compose up --build` |
| NVIDIA (CUDA) | `docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build` |
| AMD (ROCm) | `docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build` |

Once up, open **http://localhost:17600** (host port `17600` → container `17493`,
so it never collides with a desktop/dev Voicebox on `17493`).

## NVIDIA / CUDA (WSL2)

The heavy lifting is auto-detected: `get_torch_device()` returns `"cuda"` the
moment the GPU is visible in the container, and the default PyTorch wheel on
Linux is already CUDA-enabled with its own bundled CUDA libraries. So the only
requirement is **passing the GPU through** — which the CUDA overlay does.

### One-time host setup (inside your WSL distro)

The Windows NVIDIA driver already exposes the GPU to WSL2. You only need the
container toolkit so Docker can hand the device to a container:

```bash
# nvidia-container-toolkit (Debian/Ubuntu example)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker   # or restart Docker Desktop
```

Verify the wire-up:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi
```

If your card shows up there, Voicebox will use it — it's the same passthrough
any existing CUDA container (e.g. a chatterbox container) already relies on.

### Run

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

The first run downloads model weights into the `huggingface-cache` volume
(once). Subsequent starts are warm.

## Data & persistence

The compose files mount three locations so nothing is lost on rebuild:

- `./output` → generated audio (bind mount; change the host path as you like)
- `voicebox-data` → profiles, database, cache (named volume)
- `huggingface-cache` → downloaded model weights (named volume)

## Notes & gotchas

- **VRAM vs the memory limit.** The CUDA overlay raises the container's host-RAM
  limit to 16G for model loading; VRAM is governed by the GPU itself. Qwen 1.7B
  (the daily-driver engine) sits around 3.5 GB VRAM.
- **No cuda-libs sidecar.** The desktop app downloads a separate ~2 GB
  `cuda-libs` tarball at runtime. The container does **not** — the CUDA runtime
  comes bundled in the PyTorch wheel, so that whole mechanism is bypassed.
- **`gpus: all` alternative.** The overlay uses the Compose-native
  `deploy.resources.reservations.devices` form. If you prefer, recent Docker
  also accepts a top-level `gpus: all` on the service; both do the same job.
- **Apple Silicon (MLX/Metal)** cannot be used from Docker — Metal isn't exposed
  to Linux containers. Run the desktop app natively on macOS for MLX.
