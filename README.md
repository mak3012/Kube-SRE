---
tags:
  - openenv
---

# Kube-SRE-Gym

Kube-SRE-Gym is a **production-incident RL environment** for diagnosing and resolving Kubernetes outages using **typed, safe `kubectl` actions** and **structured telemetry observations**. It is **OpenEnv** compatible and served over **WebSockets** at `/ws`.

## What you get

- **Curriculum of incident scenarios**
  - **Easy**: *The Ghost Image* — `ImagePullBackOff` caused by a typo image tag
  - **Medium**: *The Memory Leak* — `OOMKilled` until memory limits/requests are fixed
  - **Hard**: *Cascading Mesh Failure* — service-mesh connectivity restored by patching a VirtualService
- **Type-safe action space** in `models.py` using Pydantic (no raw shell strings)
- **Structured observation space** (pods, events, log buffers) to support verifiable reward shaping
- **Potential-Based Reward Shaping** in `environment.py`
- **AdversarialScenarioGenerator** that injects bounded “chaos” based on historical failure patterns
- **Baseline inference runner** in `inference.py` (configured via environment variables)

## OpenEnv protocol (WebSocket)

This environment is served by **openenv-core** (FastAPI + WebSocket). The canonical “discoverable tool” surface is the set of public methods on `KubeSREGymEnv` (see `environment.py`).

## Running locally (no Docker)

Install and run:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open in browser (use localhost, not 0.0.0.0):

- **Docs**: `http://127.0.0.1:8001/docs`
- **Home**: `http://127.0.0.1:8001/`

Important: don’t use VSCode “Live Server” (port `5500`). That server is unrelated and will break WebSocket URLs (you’ll see errors like `ws://127.0.0.1:5500//ws`).

In another terminal, run inference (requires model access):

```bash
set ENV_WS_URL=ws://127.0.0.1:8001/ws

# Gemini (AI Studio) mode:
set GEMINI_API_KEY=YOUR_GEMINI_KEY
set MODEL_NAME=gemini-1.5-flash

# (Optional) OpenAI-compatible mode:
# set OPENAI_API_KEY=YOUR_KEY
# set API_BASE_URL=https://api.openai.com/v1
# set MODEL_NAME=gpt-4.1-mini
python inference.py
```

## RL details (reward shaping)

The environment uses:

- **Final reward**: success `+1.0`, failure `-1.0`
- **Efficiency penalty**: `-0.05` per step
- **Validity penalty**: `-0.2` for invalid/malformed actions
- **Quality-delta**: `+0.1` for first successful log retrieval on a failing pod
- **Potential-based shaping**: \(r' = r + \gamma \Phi(s') - \Phi(s)\) with \(\gamma = 0.99\)

## Sim-to-Real path (how to graduate to real clusters)

This repo is designed to be a safe **simulator first**, then extended to real Kubernetes:

1. **Simulator mode (this repo)**
   - Deterministic, bounded execution
   - Typed actions prevent command injection
   - Fast iterations for RL + evaluation
2. **Hybrid mode**
   - Replace the in-memory executor with a “driver” that uses a Kubernetes client library (not `kubectl` subprocesses)
   - Keep the same Pydantic action schema and observation schema
3. **Real cluster mode**
   - Run against ephemeral test clusters (kind/k3d) in CI
   - Add policy guards: namespace allowlists, resource allowlists, mutation gates
   - Expand telemetry: metrics, traces, kubelet summaries, service mesh stats

Key principle: **actions remain typed** and **never become raw shell strings**.

## Security notes

- No subprocess execution is used for `kubectl` actions; all commands are **interpreted in-memory**.
- Resource identifiers are constrained by regex in `models.py` to reject whitespace / metacharacters.
- Docker runs as a numeric non-root user (`1000:1000`).
- Secrets must be provided via environment variables (for example `GEMINI_API_KEY`). Run `python security_scan.py` to check for accidental hardcoding.
## Baseline Scores
Using `gemini-2.5-flash` with the provided inference script:
* **The Ghost Image (Easy):** Score 1.0 (Success)
* **The Memory Leak (Medium):** Score 1.0 (Success)
* **Cascading Mesh Failure (Hard):** Score 1.0 (Success)
