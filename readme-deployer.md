# d.py 
Deploy vLLM across hundreds of GPU nodes in parallel. One command to deploy, verify, repair, or teardown.
---
## Quick Start
```bash
# Deploy
python3 d.py --config cluster.conf --name gemma27b \
  --model Infomaniak-AI/vllm-translategemma-27b-it \
  --tp 1 --base-port 35000 --ip-file hosts.csv
# Check health
python3 d.py --config cluster.conf --name gemma27b \
  --tp 1 --base-port 35000 --ip-file hosts.csv --verify
# Fix dead containers only
python3 d.py --config cluster.conf --name gemma27b \
  --model Infomaniak-AI/vllm-translategemma-27b-it \
  --tp 1 --base-port 35000 --ip-file hosts.csv --repair
# Retry just the failures
python3 d.py --config cluster.conf --name gemma27b \
  --model Infomaniak-AI/vllm-translategemma-27b-it \
  --tp 1 --base-port 35000 --failed-from deploy_logs_*/summary.json
# Tear it all down
python3 d.py --config cluster.conf --name gemma27b \
  --ip-file hosts.csv --teardown
# Check what's cached on each node
python3 d.py --config cluster.conf --name gemma27b \
  --model Infomaniak-AI/vllm-translategemma-27b-it \
  --ip-file hosts.csv --inspect-cache
```
Add `--no-docker` to any command to run vLLM directly on the host.
---
## What's Required per Mode
Every mode needs `--config`, `--name`, and a node source (`--ip-file`, `--hosts`, or `--failed-from`).
| Flag | Deploy | Verify | Repair | Teardown | Inspect-cache |
|------|:------:|:------:|:------:|:--------:|:-------------:|
| `--model` | ✓ | | ✓ | | ✓ |
| `--tp` | ✓ | ✓ | ✓ | | |
| `--base-port` | ✓ | ✓ | ✓ | | |
---
## Config
Everything lives in one file. Pass it with `--config`.
```ini
[cache]
threshold = 64
# Local (per-node)
/dev/shm/model-cache
/opt/dlami/nvme/cc
# Shared (team) — skipped when >64 nodes
shared:/fsxnew/opensource-models/hub
shared:/fsx/opensource-models/hub
# Shared (personal) — skipped when >64 nodes
shared:/fsx/shamiul.sheikh/hf-cache
shared:/fsxnew/shamiul.sheikh/hf-cache
shared:/fsx/shamiul.sheikh/cache
shared:/fsxnew/shamiul.sheikh/cache
shared:/home/shamiul.sheikh/cache
shared:/home/shamiul.sheikh/important/hf-cache
# S3 download source
s3://bgen-data-team/trans/trans/
[s3]
streams = 5
multipart_chunksize = 64MB
[cluster]
gpus_per_node = 8
[docker]
image = vllm/vllm-openai:v0.12.0
tar_cache = /opt/dlami/nvme/containers/
load_timeout = 900
[tuning]
gpu_mem_threshold_mb = 3000
retry_count = 2
ram_promote_headroom_gb = 20
```
### Cache hierarchy
The same cache hierarchy is used for **both model weights and Docker TARs**. Paths under `[cache]` are checked top to bottom — first hit wins.
| Prefix | Meaning | When used |
|--------|---------|-----------|
| *(none)* | Local per-node path | Always |
| `shared:` | Shared storage (FSx/NFS) | Skipped when nodes > `threshold` |
| `s3://` | Remote download source | Fallback if nothing cached |
**Tuning knobs** (`[tuning]` section):
| Key | Default | What it does |
|-----|---------|-------------|
| `gpu_mem_threshold_mb` | 3000 | GPU VRAM (MB) above which a GPU slot is "busy" |
| `retry_count` | 2 | Retries for SSH and container launches |
| `ram_promote_headroom_gb` | 20 | Skip RAM promote if free RAM < model + this |
### How CHECK works
Walk paths 1 → 2 → 3 → ... → N in order. First path containing the model (or Docker TAR) wins.
### How DOWNLOAD works (space-aware, safetensors-only)
Downloads only safetensors weights and essential config files — `.bin` and `.pt` files are excluded everywhere to avoid doubling download size.
**Included file types:** `*.safetensors`, `*.json`, `*.txt`, `*.model`, `*.tiktoken`, `*.py`
**Excluded file types:** `*.bin`, `*.pt`
This applies to HuggingFace downloads (CLI and Python), S3 sync downloads (per-node), and S3 uploads (preflight). Models that only ship `.bin` weights (no safetensors) will fail with a diagnostic message suggesting the cause.
When nothing is cached, the script picks a download target by checking **free disk space** (`df -BG`) on each persistent path (everything except RAM):
```
Try path #2 (NVMe)       → has space? Download here
  ↓ full
Try path #3 (shared FSx)  → has space? Download here
  ↓ full
Try path #4, #5, ...      → keep walking
  ↓ all full
Download to path #1 (RAM) → always fits, just not persistent
```
The required space is calculated during preflight (`aws s3 ls --summarize`) with a 20% buffer. If size is unknown, defaults to 50GB for models, 25GB for Docker TARs.
### How SERVE works (RAM guard + auto-prune)
Model weights are promoted to RAM (path #1) **only if the node has enough free memory**. Before copying, the script checks:
```
free RAM  ≥  model size + ram_promote_headroom_gb
```
If the check passes (e.g., 256GB nodes with 100GB free, 52GB model + 20GB headroom = 72GB needed), the model is copied to `/dev/shm` for fastest serving.
If it fails (e.g., 123GB nodes with 68GB free), the model stays on NVMe and vLLM serves from there. Startup is ~30s slower but the node keeps 52GB of RAM free for NCCL, PyTorch shared memory, and KV cache management.
| Node RAM | Free | Model + Headroom | Result |
|----------|------|------------------|--------|
| 256GB | ~100GB | 52 + 20 = 72GB | ✓ Promote to RAM |
| 123GB | ~68GB | 52 + 20 = 72GB | ✗ Serve from NVMe |
**RAM auto-prune:** Before promoting a new model, the script checks how much space old model directories in `/dev/shm` are using (excluding the model about to be loaded). If old models collectively use **≥ 300GB or > 30% of total RAM**, they are pruned to make room. The current model is never deleted. If `free` fails, only the 300GB absolute threshold applies (the 30% check is safely skipped).
Docker TARs are loaded from wherever they are — no promotion needed since `docker load` is a one-time operation.
---
## What Happens
```
Preflight ──▶ Pre-warm ──▶ Deploy ──▶ Crash Check
```
1. **Preflight** — A scout node checks S3 for the Docker image and model. Missing? Downloads from the registry/HuggingFace and uploads automatically. Measures asset sizes for space-aware distribution. S3 uploads show live progress (polled every 30s).
2. **Pre-warm** — All nodes pull assets from S3 in parallel. Each node independently picks its download target based on local disk space.
3. **Deploy** — Each node launches vLLM containers. Skips busy GPUs, handles disk-full/Docker issues gracefully.
4. **Crash check** — Waits 30–90s (with live countdown), then verifies every container is still alive. Downgrades results for anything that died.
### Cache flow per node
```
Check: RAM → NVMe → FSx → S3 download
                                │
Persist: always copy to NVMe    │ (survives reboot, 2 GB/s reads)
                                │
Auto-prune: old models in RAM ≥ 300GB or >30% total RAM?
            │              │
           YES            NO
            │              │
     Delete old      Leave them
     (keep current)
            │
RAM guard: free RAM ≥ model + headroom?
            │              │
           YES            NO
            │              │
     Copy to RAM     Serve from NVMe
     (~instant)      (vLLM loads ~30s slower,
                      saves 52GB host RAM)
```
On subsequent deploys after reboot: NVMe still has the model → skip S3 → promote to RAM in ~25s (vs 200s from FSx).
---
## Progress & Logging
The script never goes silent. Every stage shows live progress.
**Parallel stages** — per-host completion counters:
```
[PRE-WARM] [1/12]  10.0.1.1 — docker=cached, model=cached (1s)              ← already in RAM
[PRE-WARM] [2/12]  10.0.1.2 — docker=cached, model=promoted (30s)           ← NVMe → RAM
[PRE-WARM] [3/12]  10.0.1.3 — docker=cached, model=promoted | skipping RAM  ← low RAM, serve from NVMe
[PRE-WARM] [4/12]  10.0.1.4 — docker=staged, model=staged (47s)             ← downloaded from S3
[DEPLOY]   [5/175] 10.0.1.5 — ✓ 8 started, 0 skipped (31s)
```
**Preflight S3 uploads** — live progress via background monitor:
```
[PRE-FLIGHT]   Model downloaded (47 files, 240.5 GB in 312s, 789 MB/s) — uploading to S3...
[PRE-FLIGHT]   UPLOAD_PROGRESS: 51200MB / 245760MB (20%)
[PRE-FLIGHT]   UPLOAD_PROGRESS: 102400MB / 245760MB (41%)
[PRE-FLIGHT]   UPLOAD_PROGRESS: 204800MB / 245760MB (83%)
[PRE-FLIGHT] ✓ Model uploaded to S3 (240.5 GB in 487s, 505 MB/s)
```
**Long downloads** — live streaming from the remote process:
```
[PRE-FLIGHT]   Downloading model from HuggingFace...
[PRE-FLIGHT]   Fetching 47 files: 12%|████      | 6/47 [02:15<08:30]
[PRE-FLIGHT]   Downloading model-00003-of-00012.safetensors: 45% 2.1GB/4.7GB
[PRE-FLIGHT]   still running... 3m15s
```
**Post-deploy wait** — live countdown:
```
[POST-DEPLOY] Waiting for containers to initialize... 45s remaining
[POST-DEPLOY] Waiting for containers to initialize... done
```
**Per-host logs** — every SSH call, container launch, error, and timing:
```
[2025-01-15 14:23:01] [INFO] [10.0.1.5] SSH connectivity OK
[2025-01-15 14:23:02] [INFO] [10.0.1.5] Docker image already present
[2025-01-15 14:23:03] [INFO] [10.0.1.5] Model found in /opt/dlami/nvme/cc
[2025-01-15 14:23:03] [INFO] [10.0.1.5] RAM prune skipped (52000MB old models, total_ram=2048000MB)
[2025-01-15 14:23:03] [INFO] [10.0.1.5] Skipping RAM promote (68000MB free, need 72480MB) — serving from /opt/dlami/nvme/cc
```
**SSH debug log** — failed SSH commands are logged to `ssh_debug.log` with the full command (first 500 chars) for post-mortem debugging. Never printed to console.
**Log directory** — picks the first writable location:
```
1. ./deploy_logs_YYYYMMDD_HHMMSS    (CWD — preferred)
2. /tmp/deploy_logs_...              (fallback, symlink in CWD)
3. /dev/shm/deploy_logs_...          (last resort, symlink in CWD)
```
Override with `DEPLOY_LOG_DIR` env var. All logs upload to S3 automatically after the run.
---
## Flags
**Always required** — `--config`, `--name`, and one of `--ip-file` / `--hosts` / `--failed-from`
| Flag | What it does |
|------|-------------|
| `--ip-file FILE` | Node IPs, one per line |
| `--hosts IP ...` | Inline IPs instead of file |
| `--failed-from JSON` | Retry failures from a previous `summary.json` |
| `--model ID` | HuggingFace model ID *(required for deploy, repair, inspect-cache)* |
| `--tp N` | Tensor parallelism *(required for deploy, verify, repair)* |
| `--base-port PORT` | Starting port *(required for deploy, verify, repair)* |
| `--verify` | Health check only |
| `--teardown` | Remove everything |
| `--repair` | Restart only dead containers |
| `--no-docker` | Run vLLM directly on host |
| `--docker-image IMG` | Override image from config |
| `--vllm-args ARGS` | Extra vLLM serve flags |
| `--skip-prewarm` | Skip S3 pre-staging |
| `--workers N` | Parallel threads *(default: 32)* |
| `--dry-run` | Print plan without executing |
| `--keep-alive SEC` | vLLM HTTP keepalive *(default: 600)* |
| `--ssh-timeout SEC` | SSH timeout *(default: 15)* |
| `--batch-delay SEC` | Stagger host launches |
| `--temp-folder PATH` | Temp dir on scout node for preflight |
| `--inspect-cache` | Check what's cached where across all nodes |
---
## Outputs
Each run creates `deploy_logs_YYYYMMDD_HHMMSS/`:
| File | Purpose |
|------|---------|
| `summary.json` | Machine-readable results — feed to `--failed-from` |
| `summary.txt` | Human-readable summary |
| `{IP}.log` | Per-host detailed log with timestamps |
| `ssh_debug.log` | Failed SSH commands with full command text (debug) |
| `successful_*.csv` | Working IPs — feed to `--ip-file` for next deploy |
| `containers.csv` | Every container: `ip,container_name` |
| `failed_ips.csv` | All failed IPs — feed to `--ip-file` to retry |
| `no_space_left.csv` | Disk full failures — need cleanup or prune |
| `docker_issue.csv` | Docker daemon failures — restart dockerd or use `--no-docker` |
| `nvidia_broken.csv` | NVIDIA runtime failures — need driver fix |
| `post_crash.csv` | Containers that died after startup — OOM / init failure |
| `docker_retry_nodkr.csv` | All Docker-related failures — retry with `--no-docker --ip-file` |
Logs also upload to `{s3_source}/logs/` automatically.
---
## Good to Know
- `--tp` auto-injects `--tensor-parallel-size` — don't put it in `--vllm-args`
- Containers per node = `gpus_per_node ÷ tp`, ports assigned sequentially from `--base-port`
- Verify/repair must match the original deploy's `--name`, `--tp`, `--base-port`
- Host file supports one IP per line, comma-separated, `#` comments (inline and full-line), blank lines
- Model serves from RAM or NVMe depending on available memory (see RAM guard above)
- Docker TARs also use the cache hierarchy — found on shared FSx? No S3 download needed
- `/dev/shm` is a tmpfs (RAM-backed) shared by model cache, NCCL, and PyTorch — the RAM guard prevents oversubscription
- NVMe persist survives reboots — only RAM cache is lost, NVMe → RAM promotion takes ~25-30s
- Measured speeds: NVMe → RAM ~2 GB/s, FSx → RAM ~460 MB/s (single node), ~260 MB/s (16 nodes concurrent)
- `--inspect-cache` shows what's cached where across all nodes without deploying
- Downloads are **safetensors-only** — `.bin` and `.pt` weights are excluded to halve download size for repos that ship both formats
- Models that only have `.bin` weights (no `.safetensors`) will fail at preflight with a clear diagnostic
- Old models in RAM are auto-pruned when they exceed 300GB or 30% of total RAM — the current model is always kept
- The confirmation prompt shows target node count, parallel workers, and config file before you type `yes`
## Prerequisites
- Passwordless SSH to all GPU nodes
- Docker on nodes (or vLLM installed for `--no-docker`)
- AWS CLI with S3 access if using an S3 source
