#!/usr/bin/env python3
"""
Multi-host vLLM deployment script with parallel execution and per-IP logging.
Usage: run directly. Set DRY_RUN=True at top to only print commands without executing.

Logs are written to LOG_DIR/<ip>.log (one file per host) plus a combined summary.
"""

import subprocess
import sys
import os
import time
import math
import shlex
import json
import argparse
import socket
import getpass
import re
import glob
import select
from pathlib import Path
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading
from datetime import datetime

# ================= HOST CONFIG =================

TOTAL_GPUS = 8
BASE_PORT = 30001
DEFAULT_TP_SIZE = 2  # Tensor parallelism size (1, 2, 4, or 8)

# Directory on remote hosts for PID files, logs, and launch scripts (no-docker mode).
# Using shared storage so files survive node reboots and are visible from head node.
# SAFETY: Must be non-empty absolute path. rm -f globs use this path; empty="" → operates at /.
RUN_DIR = "/fsxnew/shamiul.sheikh/tmp"
assert RUN_DIR and RUN_DIR.startswith("/") and RUN_DIR != "/", \
    f"RUN_DIR must be a non-empty absolute path (not '/'). Got: {RUN_DIR!r}"

def build_gpu_port_map(tp_size: int = DEFAULT_TP_SIZE) -> dict:
    """Build GPU-to-port mapping based on tensor parallelism size.
    TP=1: 8 containers → {30001: "0", 30002: "1", ...}
    TP=2: 4 containers → {30001: "0,1", 30002: "2,3", ...}
    TP=4: 2 containers → {30001: "0,1,2,3", 30002: "4,5,6,7"}
    TP=8: 1 container  → {30001: "0,1,2,3,4,5,6,7"}
    """
    assert TOTAL_GPUS % tp_size == 0, f"TOTAL_GPUS ({TOTAL_GPUS}) must be divisible by tp_size ({tp_size})"
    num_containers = TOTAL_GPUS // tp_size
    port_map = {}
    for i in range(num_containers):
        port = BASE_PORT + i
        gpu_start = i * tp_size
        gpu_ids = ",".join(str(g) for g in range(gpu_start, gpu_start + tp_size))
        port_map[port] = gpu_ids
    return port_map

# ================= DOCKER / MODEL CONFIG =================

DOCKER_IMAGE = "vllm/vllm-openai:v0.12.0"
DOCKER_LOAD_TIMEOUT = 900

def _tar_name_from_image(docker_image: str) -> str:
    """Derive TAR filename from Docker image tag. e.g. vllm/vllm-openai:v0.12.0 → vllm-v0.12.0.tar
    Handles registry:port format (e.g. registry:5000/image → latest)."""
    if ":" in docker_image:
        tag = docker_image.split(":")[-1]
        # If tag contains "/" it's a registry port, not an image tag
        if "/" in tag:
            tag = "latest"
    else:
        tag = "latest"
    return f"vllm-{tag}.tar"


# Local NVMe cache for Docker TARs (used once for docker load, not worth RAM).
LOCAL_TAR_CACHE = "/opt/dlami/nvme/containers"

# ── Configuration ──
# Single config file controls all cluster settings.
# Cache hierarchy is a flat ordered list — checked top to bottom, first hit wins.
# Paths prefixed with "shared:" are skipped when nodes > threshold.
# An s3:// line is the remote download source (optional).
#
# Format:
#   [cache]
#   threshold = 64
#   /dev/shm/model-cache
#   shared:/fsx/model-cache
#   ~/.cache/vllm-models
#   s3://bgen-data-team/trans/trans/
#
#   [s3]
#   streams = 5
#   multipart_chunksize = 64MB
#
#   [cluster]
#   gpus_per_node = 8

CACHE_THRESHOLD = 64
CACHE_HIERARCHY: List[tuple] = []   # [(path, is_shared), ...]  — user-defined order
CACHE_S3_SOURCE = ""
NUM_DEPLOY_NODES = 0

S3_MAX_CONCURRENT_REQUESTS = 5
S3_MULTIPART_CHUNKSIZE = "64MB"


def load_config(path: str):
    """Parse unified config file."""
    global CACHE_THRESHOLD, CACHE_HIERARCHY, CACHE_S3_SOURCE
    global S3_MAX_CONCURRENT_REQUESTS, S3_MULTIPART_CHUNKSIZE
    global TOTAL_GPUS, DOCKER_IMAGE, DOCKER_LOAD_TIMEOUT
    global LOCAL_TAR_CACHE, GPU_MEM_THRESHOLD_MB, RETRY_COUNT, RAM_PROMOTE_HEADROOM_GB
    hierarchy = []
    section = None

    def _int(val_str, key):
        try:
            return int(val_str.strip())
        except ValueError:
            print(f"ERROR: Config '{key}' must be an integer, got: {val_str.strip()!r}")
            sys.exit(1)

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].lower()
                continue
            if section == "cache":
                if line.lower().startswith("threshold"):
                    _, _, val = line.partition("=")
                    CACHE_THRESHOLD = _int(val, "threshold")
                elif line.startswith("s3://"):
                    CACHE_S3_SOURCE = line.rstrip("/") + "/"
                elif line.lower().startswith("shared:"):
                    hierarchy.append((os.path.expanduser(line[7:].strip()), True))
                else:
                    hierarchy.append((os.path.expanduser(line), False))
            elif section == "s3":
                if line.lower().startswith("streams"):
                    _, _, val = line.partition("=")
                    S3_MAX_CONCURRENT_REQUESTS = _int(val, "streams")
                elif line.lower().startswith("multipart_chunksize"):
                    _, _, val = line.partition("=")
                    S3_MULTIPART_CHUNKSIZE = val.strip()
            elif section == "cluster":
                if line.lower().startswith("gpus_per_node"):
                    _, _, val = line.partition("=")
                    TOTAL_GPUS = _int(val, "gpus_per_node")
            elif section == "docker":
                if line.lower().startswith("image"):
                    _, _, val = line.partition("=")
                    DOCKER_IMAGE = val.strip()
                elif line.lower().startswith("tar_cache"):
                    _, _, val = line.partition("=")
                    LOCAL_TAR_CACHE = os.path.expanduser(val.strip())
                elif line.lower().startswith("load_timeout"):
                    _, _, val = line.partition("=")
                    DOCKER_LOAD_TIMEOUT = _int(val, "load_timeout")
            elif section == "tuning":
                if line.lower().startswith("gpu_mem_threshold_mb"):
                    _, _, val = line.partition("=")
                    GPU_MEM_THRESHOLD_MB = _int(val, "gpu_mem_threshold_mb")
                elif line.lower().startswith("retry_count"):
                    _, _, val = line.partition("=")
                    RETRY_COUNT = _int(val, "retry_count")
                elif line.lower().startswith("ram_promote_headroom_gb"):
                    _, _, val = line.partition("=")
                    RAM_PROMOTE_HEADROOM_GB = _int(val, "ram_promote_headroom_gb")
    if hierarchy:
        CACHE_HIERARCHY = hierarchy
    if not CACHE_HIERARCHY:
        print("ERROR: Config [cache] must define at least one cache path")
        sys.exit(1)


def effective_cache_paths(num_nodes: int) -> List[str]:
    """Return ordered cache paths, skipping shared if nodes > threshold.
    Falls back to ALL paths if filtering would leave none (safety net)."""
    paths = [p for p, shared in CACHE_HIERARCHY if not (shared and num_nodes > CACHE_THRESHOLD)]
    if not paths:
        # All paths are shared and nodes > threshold — use them anyway rather than crash
        paths = [p for p, _ in CACHE_HIERARCHY]
    return paths


# Tunables
DRY_RUN = False
MAX_WORKERS = 32
RETRY_COUNT = 2
RETRY_SLEEP = 5
GPU_MEM_THRESHOLD_MB = 3000
RAM_PROMOTE_HEADROOM_GB = 20  # skip RAM promote if free RAM < model_size + this
VLLM_KEEP_ALIVE = 600

# Logging — prefer CWD, fall back to /tmp, last resort /dev/shm (RAM)
_LOG_BASENAME = f"deploy_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def _pick_log_dir():
    """Pick best writable location for logs: CWD → /tmp → /dev/shm."""
    env = os.environ.get("DEPLOY_LOG_DIR")
    if env:
        return env
    for base in [".", "/tmp", "/dev/shm"]:
        candidate = os.path.join(base, _LOG_BASENAME)
        try:
            # Test if base dir is writable without creating the log dir yet
            test_file = os.path.join(base, f".deploy_write_test_{os.getpid()}")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            return candidate
        except OSError:
            continue
    return f"/tmp/{_LOG_BASENAME}"

LOG_DIR = _pick_log_dir()

# SSH optimization
SSH_CONTROL_PATH = "/tmp/ssh_mux_%h_%p_%r"
SSH_CONTROL_PERSIST = "10m"

# Thread-safe printing
print_lock = Lock()

# Global abort flag — set on Ctrl+C to stop new work
_abort = False

# Debug log for failed SSH commands (written lazily so LOG_DIR doesn't need to exist yet)
_debug_log_lock = Lock()

def _log_ssh_debug(ip: str, command: str, rc: int, err: str):
    """Append failed SSH command to debug log (file-only, never printed to console)."""
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        with _debug_log_lock:
            with open(log_dir / "ssh_debug.log", "a") as f:
                f.write(f"[{ts}] [{ip}] rc={rc} err={err.strip()[:200]}\n")
                f.write(f"  cmd: {command[:500]}\n")
    except OSError:
        pass  # best-effort — don't crash on log failure

# =========================================================


def _get_deployer_ip() -> str:
    """Get the IP address of the deployer machine."""
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "unknown"


class HostLogger:
    """Per-host logger that writes to both console and a per-IP log file."""

    def __init__(self, ip: str, log_dir: str):
        self.ip = ip
        self.log_path = Path(log_dir) / f"{ip}.log"
        self._lock = Lock()
        self._lines: List[str] = []

    def log(self, msg: str, level: str = "INFO", file_only: bool = False):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] [{self.ip}] {msg}"
        with self._lock:
            self._lines.append(line)
        # Also print to console (thread-safe) unless file_only
        if not file_only:
            with print_lock:
                print(line)

    def flush(self):
        """Write all buffered lines to the log file (overwrites — always writes complete log)."""
        with self._lock:
            lines = list(self._lines)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def get_lines(self) -> List[str]:
        with self._lock:
            return list(self._lines)


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def ssh_command(ip: str, command: str, timeout: int = 60, use_multiplexing: bool = True, retries: int = 2) -> Tuple[int, str, str]:
    """Execute SSH command with optional connection multiplexing and retries."""
    for attempt in range(1, retries + 1):
        ssh_args = ["ssh"]

        if use_multiplexing:
            ssh_args.extend([
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={SSH_CONTROL_PATH}",
                "-o", f"ControlPersist={SSH_CONTROL_PERSIST}",
            ])

        ssh_args.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            ip,
            command
        ])

        if DRY_RUN:
            return 0, "DRY_RUN", ""

        try:
            result = subprocess.run(
                ssh_args,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            # If mux socket went bad, clean it up and retry
            if result.returncode != 0 and attempt < retries:
                if result.returncode == 255 or "mux" in result.stderr.lower():
                    _cleanup_mux_socket(ip)
                    time.sleep(1)
                    continue
            if result.returncode != 0:
                _log_ssh_debug(ip, command, result.returncode, result.stderr)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            # Always clean mux socket on timeout — stale socket may be the cause
            _cleanup_mux_socket(ip)
            if attempt < retries:
                time.sleep(RETRY_SLEEP)
                continue
            _err = f"Command timed out after {timeout}s"
            _log_ssh_debug(ip, command, -1, _err)
            return -1, "", _err
        except Exception as e:
            _cleanup_mux_socket(ip)
            if attempt < retries:
                time.sleep(RETRY_SLEEP)
                continue
            _log_ssh_debug(ip, command, -1, str(e))
            return -1, "", str(e)

    _log_ssh_debug(ip, command, -1, "All retries exhausted")
    return -1, "", "All retries exhausted"


def ssh_command_stream(ip: str, command: str, timeout: int = 3600,
                       prefix: str = "", heartbeat: int = 30) -> Tuple[int, str, str]:
    """Execute SSH command with live output streaming.
    Used for long-running commands (HF download, S3 upload) where the user
    would otherwise see nothing for 10-60 minutes.

    Streams stdout+stderr lines to console in real-time with [prefix].
    Still collects full output for sentinel parsing (DL_OK, UPLOAD_OK, etc).
    Prints a heartbeat every N seconds if no output is received.

    Returns (returncode, full_stdout, full_stderr) — same interface as ssh_command.
    """
    if DRY_RUN:
        return 0, "DRY_RUN", ""

    ssh_args = [
        "ssh",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={SSH_CONTROL_PATH}",
        "-o", f"ControlPersist={SSH_CONTROL_PERSIST}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        ip,
        command
    ]

    try:
        proc = subprocess.Popen(
            ssh_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for unified streaming
            text=True,
            bufsize=1,  # Line-buffered
        )
    except Exception as e:
        return -1, "", str(e)

    collected = []
    start = time.time()
    last_output = start

    try:
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                proc.kill()
                proc.wait()
                return -1, "\n".join(collected), f"Command timed out after {timeout}s"

            # Wait for output with short timeout for heartbeat checking
            ready, _, _ = select.select([proc.stdout], [], [], min(heartbeat, 5))

            if ready:
                line = proc.stdout.readline()
                if not line:
                    # EOF — process finished
                    break
                line = line.rstrip("\n")
                collected.append(line)
                last_output = time.time()
                # Print progress lines (skip empty and pure whitespace)
                if line.strip():
                    # Filter: show download progress, percentage, speed, important status
                    low = line.lower()
                    show = any(k in low for k in [
                        "%", "downloading", "fetching", "eta", "mb/s", "gb/s",
                        "upload", "sync", "pull", "save", "load", "error", "fail",
                        "dl_ok", "dl_fail", "upload_ok", "upload_fail", "save_ok", "save_fail",
                        "blob", "complete", "already", "s3_ok", "s3_fail",
                    ])
                    if show and prefix:
                        safe_print(f"{prefix} {line.strip()}")
            else:
                # No output — print heartbeat if enough time passed
                since_last = time.time() - last_output
                if since_last >= heartbeat:
                    mins = int(elapsed) // 60
                    secs = int(elapsed) % 60
                    if prefix:
                        safe_print(f"{prefix} still running... {mins}m{secs:02d}s")
                    last_output = time.time()

            # Check if process exited
            if proc.poll() is not None:
                # Read any remaining output
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    collected.append(line)
                break

    except Exception as e:
        proc.kill()
        proc.wait()
        return -1, "\n".join(collected), str(e)

    proc.wait()
    return proc.returncode, "\n".join(collected), ""


def _cleanup_mux_socket(ip: str):
    """Remove stale SSH multiplexing socket for a host."""
    pattern = SSH_CONTROL_PATH.replace("%h", ip).replace("%p", "*").replace("%r", "*")
    for sock in glob.glob(pattern):
        try:
            os.remove(sock)
        except OSError:
            pass


def _s3_parallel_preamble() -> str:
    """Shell preamble to configure parallel S3 downloads via a temp AWS config.
    Merges parallel settings with existing config to preserve credentials/region.
    Falls back to /dev/shm if /tmp is not writable."""
    return (
        f"_S3P=/tmp/.s3parallel.cfg; "
        f"touch $_S3P 2>/dev/null || _S3P=/dev/shm/.s3parallel.cfg; "
        f"{{ cat ~/.aws/config 2>/dev/null || true; "
        f"printf '\\n[default]\\ns3 =\\n  max_concurrent_requests = {S3_MAX_CONCURRENT_REQUESTS}\\n"
        f"  multipart_chunksize = {S3_MULTIPART_CHUNKSIZE}\\n'; }} > $_S3P && "
        f"export AWS_CONFIG_FILE=$_S3P && "
    )


def _snap_select_bash(model_dir: str) -> str:
    """Generate bash to select the correct snapshot directory using refs/main.
    Returns bash that sets $snap to the best snapshot path.
    Prefers refs/main (written by HF CLI) → falls back to head -1."""
    d = shlex.quote(model_dir)
    return (
        f"_ref=$(cat {d}/refs/main 2>/dev/null | tr -d '[:space:]'); "
        f"if [ -n \"$_ref\" ] && [ -d {d}/snapshots/$_ref ]; then "
        f"  snap={d}/snapshots/$_ref/; "
        f"else "
        f"  snap=$(ls -d {d}/snapshots/*/ 2>/dev/null | head -1); "
        f"fi"
    )


def load_docker_image_from_tar(ip: str, logger: HostLogger, docker_image: str = DOCKER_IMAGE,
                               s3_bucket: str = "") -> str:
    """Load Docker image using cache hierarchy with space-aware download.
    Phase 1: Check if image already loaded in Docker
    Phase 2: Walk cache hierarchy looking for existing TAR file
    Phase 3: Space-aware download from S3 (persistent paths first, RAM fallback)
    Phase 4: docker load from wherever the TAR is
    Returns: 'cached' if already present, 'loaded' if freshly loaded, '' if failed."""
    tar_name = _tar_name_from_image(docker_image)
    paths = effective_cache_paths(NUM_DEPLOY_NODES)
    ram_path = paths[0] if paths else "/dev/shm/model-cache"
    persistent_paths = paths[1:] if len(paths) > 1 else []
    nvme_path = persistent_paths[0] if persistent_paths else None
    disk_full_pruned = False

    # Phase 1: Check docker image + walk ALL paths for TAR in single SSH call
    check_parts = [
        f"docker images -q {shlex.quote(docker_image)} 2>/dev/null | head -1 | "
        f"grep -q . && echo 'IMG_OK' || echo 'IMG_MISS'"
    ]
    for i, p in enumerate(paths):
        tar_at = f"{p}/containers/{tar_name}"
        check_parts.append(
            f"test -f {shlex.quote(tar_at)} && echo 'TAR_HIT:{i}:{tar_at}'"
        )
    # Also check legacy LOCAL_TAR_CACHE if not already in paths
    legacy_tar = f"{LOCAL_TAR_CACHE}/{tar_name}"
    check_parts.append(f"test -f {shlex.quote(legacy_tar)} && echo 'TAR_HIT:legacy:{legacy_tar}'")
    # Check free space on persistent paths for potential download
    if persistent_paths:
        df_paths = " ".join(shlex.quote(p) for p in persistent_paths)
        check_parts.append(f"echo '===DF==='; df -BG {df_paths} 2>/dev/null | awk 'NR>1 {{print $6, $4}}'")

    rc, out, _ = ssh_command(ip, "\n".join(check_parts), timeout=20)

    if rc == 0 and "IMG_OK" in (out or ""):
        logger.log("Docker image already present", "INFO")
        return "cached"

    # Find existing TAR
    tar_path = None
    if rc == 0 and out:
        for line in out.strip().splitlines():
            if line.startswith("TAR_HIT:"):
                parts = line.split(":", 2)
                tar_path = parts[2] if len(parts) == 3 else None
                logger.log(f"Found Docker TAR at {tar_path}")
                break

    # Phase 2: If no TAR found, space-aware download from S3
    if not tar_path and s3_bucket:
        s3_tar_path = f"{s3_bucket.rstrip('/')}/containers/{tar_name}"

        # Parse df output for space-aware target selection
        dl_target = None
        if rc == 0 and out and "===DF===" in out:
            df_section = out.split("===DF===")[1].strip()
            space_map = {}
            for line in df_section.splitlines():
                parts = line.split()
                if len(parts) == 2:
                    mount = parts[0]
                    free = parts[1].rstrip("G")
                    try:
                        space_map[mount] = int(free)
                    except ValueError:
                        pass
            # Walk persistent paths, pick first with enough space (25GB default)
            need_gb = 25
            for p in persistent_paths:
                free_gb = space_map.get(p, 0)
                if free_gb == 0:
                    for mount, gb in space_map.items():
                        if p.startswith(mount) or mount.startswith(p):
                            free_gb = max(free_gb, gb)
                if free_gb >= need_gb:
                    dl_target = f"{p}/containers/{tar_name}"
                    dl_dir = f"{p}/containers"
                    logger.log(f"Download target: {p} ({free_gb}GB free)")
                    break
                else:
                    logger.log(f"Skipping {p} ({free_gb}GB free, need {need_gb}GB)", file_only=True)

        # Fall back to RAM
        if not dl_target:
            dl_dir = f"{ram_path}/containers"
            dl_target = f"{dl_dir}/{tar_name}"
            if persistent_paths:
                logger.log("All persistent paths full — downloading Docker TAR to RAM")
            else:
                logger.log("No persistent paths configured — downloading Docker TAR to RAM")

        logger.log(f"Downloading Docker TAR from S3: {s3_tar_path}")
        s3_cmd = (
            f"mkdir -p {shlex.quote(dl_dir)} && "
            f"{_s3_parallel_preamble()}"
            f"aws s3 cp {shlex.quote(s3_tar_path)} {shlex.quote(dl_target)} "
            f"--no-progress 2>&1 && echo 'S3_OK' || echo 'S3_FAIL'"
        )
        rc, out, _ = ssh_command(ip, s3_cmd, timeout=DOCKER_LOAD_TIMEOUT)
        if rc == 0 and "S3_OK" in (out or ""):
            logger.log("Docker TAR downloaded from S3")
            tar_path = dl_target
        else:
            logger.log(f"S3 download failed: {(out or '').strip()}", "WARN")

    # Phase 3: Persist TAR to NVMe (background) + docker load
    if tar_path:
        logger.log(f"Loading Docker image from {tar_path}...")
        # Kick off NVMe persist in background alongside docker load (single SSH)
        persist_bg = ""
        if nvme_path and tar_path != f"{nvme_path}/containers/{tar_name}":
            nvme_tar = f"{nvme_path}/containers/{tar_name}"
            persist_bg = (
                f"( test -f {shlex.quote(nvme_tar)} || "
                f"( mkdir -p {shlex.quote(nvme_path + '/containers')} && "
                f"cp -a {shlex.quote(tar_path)} {shlex.quote(nvme_tar)} || "
                f"rm -f {shlex.quote(nvme_tar)} ) ) &>/dev/null & "
            )
        # First attempt includes background persist; retries skip it (may already be running)
        cmd_first = f"{persist_bg}docker load -i {shlex.quote(tar_path)}"
        cmd_retry = f"docker load -i {shlex.quote(tar_path)}"
        for attempt in range(1, RETRY_COUNT + 1):
            cmd = cmd_first if attempt == 1 else cmd_retry
            rc, out, err = ssh_command(ip, cmd, timeout=DOCKER_LOAD_TIMEOUT)
            if rc == 0:
                logger.log("Docker load from TAR succeeded")
                return "loaded"
            if "no space left on device" in err.lower() and not disk_full_pruned:
                logger.log("Disk full during docker load — attempting prune...", "WARN")
                disk_full_pruned = True
                _docker_prune(ip, logger)
                rc2, out2, err2 = ssh_command(ip, cmd_retry, timeout=DOCKER_LOAD_TIMEOUT)
                if rc2 == 0:
                    logger.log("Docker load from TAR succeeded after prune")
                    return "loaded"
                logger.log(f"Docker load still failing after prune: {err2.strip()}", "ERROR")
            else:
                logger.log(f"Docker load failed (attempt {attempt}/{RETRY_COUNT}): {err.strip()}", "ERROR")
            time.sleep(RETRY_SLEEP)
        logger.log("TAR load exhausted retries — docker load failed on all attempts", "ERROR")
        logger.log("Docker image not available — docker load from TAR failed (TAR exists but won't load)", "ERROR")
    else:
        logger.log("Docker image not available — TAR not found in cache and S3 download failed", "ERROR")

    return ""


def _docker_prune(ip: str, logger: HostLogger):
    """Run docker system prune to free disk space. Removes stopped containers,
    dangling images, unused networks, and build cache. Does NOT remove volumes.
    NOTE: Uses -f (not -af) to avoid removing other users' images on shared hosts.
    Only dangling (untagged) images are removed — named images stay."""
    rc, out, _ = ssh_command(ip, "docker system prune -f 2>&1", timeout=120)
    if rc == 0:
        for line in out.strip().splitlines():
            if "reclaimed" in line.lower() or "total" in line.lower():
                logger.log(f"Prune result: {line.strip()}")
                return
        logger.log("Prune completed")
    else:
        logger.log("Docker prune failed", "WARN")


def capture_container_logs(ip: str, name: str, logger: HostLogger, tail: int = 50):
    """Capture last N lines of a container's logs for debugging."""
    cmd = f"docker logs --tail {tail} {name} 2>&1"
    rc, out, err = ssh_command(ip, cmd, timeout=15)
    if rc == 0 and out.strip():
        logger.log(f"--- Container logs for {name} (last {tail} lines) ---")
        for line in out.strip().splitlines():
            logger.log(f"  | {line}")
        logger.log(f"--- End container logs for {name} ---")
    else:
        logger.log(f"Could not retrieve logs for {name}: {err.strip()}", "WARN")


def _maybe_prune_ram_models(ip: str, ram_path: str, model_cache_name: str, logger: HostLogger):
    """Prune old model dirs from RAM (/dev/shm) if they collectively exceed 300GB or 30% of total RAM.
    Keeps the current model (model_cache_name) if already present."""
    prune_cmd = (
        # Get total RAM in MB
        f"_total_ram_mb=$(free -m 2>/dev/null | awk '/Mem/{{print $2}}'); "
        # Get total size of OTHER model dirs in ram_path (exclude current model)
        f"_old_usage_mb=0; "
        f"for d in {shlex.quote(ram_path)}/models--*; do "
        f'  [ -d "$d" ] || continue; '
        f'  _dname=$(basename "$d"); '
        f'  [ "$_dname" = {shlex.quote(model_cache_name)} ] && continue; '
        f'  _sz=$(du -sm "$d" 2>/dev/null | cut -f1); '
        f'  _old_usage_mb=$((_old_usage_mb + ${{_sz:-0}})); '
        f"done; "
        # Check thresholds: >= 300GB (307200 MB) OR > 30% of total RAM
        # Guard: only check 30% if free succeeded (total_ram > 0), otherwise only use 300GB absolute
        f"_do_prune=0; "
        f'if [ "$_old_usage_mb" -ge 307200 ] 2>/dev/null; then _do_prune=1; '
        f'elif [ "${{_total_ram_mb:-0}}" -gt 0 ] 2>/dev/null; then '
        f'  _threshold_30pct=$((_total_ram_mb * 30 / 100)); '
        f'  [ "$_old_usage_mb" -gt "$_threshold_30pct" ] 2>/dev/null && _do_prune=1; '
        f"fi; "
        f'if [ "$_do_prune" -eq 1 ]; then '
        f'  echo "PRUNE:YES:${{_old_usage_mb}}MB old models (total_ram=${{_total_ram_mb:-unknown}}MB)"; '
        f"  for d in {shlex.quote(ram_path)}/models--*; do "
        f'    [ -d "$d" ] || continue; '
        f'    _dname=$(basename "$d"); '
        f'    [ "$_dname" = {shlex.quote(model_cache_name)} ] && continue; '
        f'    rm -rf "$d"; '
        f'    echo "PRUNED:$_dname"; '
        f"  done; "
        f'else '
        f'  echo "PRUNE:NO:${{_old_usage_mb}}MB old models (total_ram=${{_total_ram_mb:-unknown}}MB)"; '
        f"fi"
    )
    rc, out, _ = ssh_command(ip, prune_cmd, timeout=120)
    if rc == 0 and out:
        for line in out.strip().splitlines():
            if line.startswith("PRUNE:YES:"):
                logger.log(f"RAM prune triggered: {line.split(':', 2)[2]}")
            elif line.startswith("PRUNE:NO:"):
                logger.log(f"RAM prune skipped: {line.split(':', 2)[2]}")
            elif line.startswith("PRUNED:"):
                logger.log(f"Pruned from RAM: {line.split(':', 1)[1]}")


def ensure_model_available(ip: str, logger: HostLogger, model_id: str,
                           s3_bucket: str = "") -> tuple:
    """Walk cache hierarchy, first hit wins. Download from S3 if not found.
    Returns (model_container_path, actual_host_cache_dir) or (None, None) on failure."""
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    paths = effective_cache_paths(NUM_DEPLOY_NODES)

    def _copy_model(src_base, dst_base):
        src = f"{src_base}/{model_cache_name}"
        dst = f"{dst_base}/{model_cache_name}"
        logger.log(f"Copying model {src_base} → {dst_base}")
        rc, _, _ = ssh_command(ip,
            f"mkdir -p {shlex.quote(dst_base)} && "
            f"cp -a {shlex.quote(src)} {shlex.quote(dst_base)}/",
            timeout=600)
        if rc != 0:
            # Cleanup partial copy to avoid corrupt cache on next run
            logger.log(f"Copy failed — cleaning partial at {dst_base}", "WARN")
            ssh_command(ip, f"rm -rf {shlex.quote(dst)}", timeout=30)
        return rc == 0

    # Phase 1: Walk ALL cache paths in single SSH call — report ALL hits + memory info
    check_parts = []
    for i, path in enumerate(paths):
        model_dir = f"{path}/{model_cache_name}"
        check_parts.append(
            f"{_snap_select_bash(model_dir)} && "
            f'[ -n "$snap" ] && echo "HIT:{i}:$snap"'
        )
    # Also get model size (from first available) and free RAM for headroom check
    check_parts.append('echo "===MEM==="')
    check_parts.append("free -m 2>/dev/null | awk '/Mem/{print $7}'")
    # Model size from first available path (same model everywhere, just need one du)
    du_targets = " ".join(shlex.quote(f"{p}/{model_cache_name}") for p in paths)
    check_parts.append(f'echo "===SIZE==="; for _d in {du_targets}; do test -d "$_d" && du -sm "$_d" 2>/dev/null | cut -f1 && break; done')
    rc, out, _ = ssh_command(ip, "\n".join(check_parts), timeout=max(15, len(paths) * 5))

    found_path = None
    found_snap = None
    hit_indices = set()  # all paths that have the model
    avail_mb = 0
    model_mb = 0
    if rc == 0 and out:
        section = "hits"
        for line in out.strip().splitlines():
            if line == "===MEM===":
                section = "mem"
                continue
            elif line == "===SIZE===":
                section = "size"
                continue
            if section == "mem":
                try:
                    avail_mb = int(line.strip())
                except ValueError:
                    pass
            elif section == "size":
                try:
                    model_mb = int(line.strip())
                except ValueError:
                    pass
            elif line.startswith("HIT:"):
                parts = line.split(":", 2)
                idx = int(parts[1])
                hit_indices.add(idx)
                if found_path is None:  # first hit = serve candidate
                    found_snap = parts[2].strip().rstrip("/").split("/")[-1]
                    found_path = paths[idx]
        if found_path:
            logger.log(f"Model found in {found_path}")

    if found_snap:
        # Persist to NVMe (first local persistent path) + promote to RAM for serving
        ram_path = paths[0]
        nvme_idx = 1 if len(paths) > 1 else None
        nvme_path = paths[nvme_idx] if nvme_idx is not None else None
        serve_path = found_path
        promote_src = found_path

        # Use hit_indices to check NVMe without extra SSH
        if nvme_path and found_path != nvme_path:
            if nvme_idx in hit_indices:
                promote_src = nvme_path  # already on NVMe
            else:
                if _copy_model(found_path, nvme_path):
                    promote_src = nvme_path

        # Promote to RAM for serving (with headroom check using pre-gathered data)
        if found_path != ram_path:
            # Prune old models from RAM if they exceed thresholds
            _maybe_prune_ram_models(ip, ram_path, model_cache_name, logger)
            # Re-check available memory after potential prune (pre-gathered value may be stale)
            rc_refr, refr_out, _ = ssh_command(ip,
                "free -m 2>/dev/null | awk '/Mem/{print $7}'", timeout=10)
            if rc_refr == 0 and refr_out and refr_out.strip().isdigit():
                avail_mb = int(refr_out.strip())
            need_mb = model_mb + RAM_PROMOTE_HEADROOM_GB * 1024
            if model_mb > 0 and avail_mb < need_mb:
                logger.log(f"Skipping RAM promote ({avail_mb}MB free, need {need_mb}MB) — serving from {promote_src}")
                serve_path = promote_src  # serve from NVMe (or wherever promote_src points)
            else:
                if _copy_model(promote_src, ram_path):
                    serve_path = ram_path
                else:
                    serve_path = promote_src  # copy failed, serve from NVMe

        container_path = f"/root/.cache/huggingface/hub/{model_cache_name}/snapshots/{found_snap}"
        return container_path, serve_path

    # Phase 2: Space-aware download from S3, promote to RAM (path[0]) for serving
    s3_source = CACHE_S3_SOURCE or (s3_bucket.rstrip("/") + "/" if s3_bucket else "")
    if not s3_source:
        logger.log("Model not found and no S3 source configured", "ERROR")
        return None, None

    if not paths:
        logger.log("No cache paths configured", "ERROR")
        return None, None

    ram_path = paths[0]
    persistent_paths = paths[1:] if len(paths) > 1 else []
    nvme_path_p2 = persistent_paths[0] if persistent_paths else None

    # Check free space on all persistent paths in single SSH call
    dl_path = None
    if persistent_paths:
        df_parts = " ".join(shlex.quote(p) for p in persistent_paths)
        rc_df, df_out, _ = ssh_command(ip,
            f"df -BG {df_parts} 2>/dev/null | awk 'NR>1 {{print $6, $4}}'",
            timeout=15)
        space_map = {}
        if rc_df == 0 and df_out:
            for line in df_out.strip().splitlines():
                parts_line = line.split()
                if len(parts_line) == 2:
                    mount = parts_line[0]
                    free = parts_line[1].rstrip("G")
                    try:
                        space_map[mount] = int(free)
                    except ValueError:
                        pass
        # Walk persistent paths, pick first with enough space (50GB fallback if size unknown)
        need_gb = 50
        for p in persistent_paths:
            # df reports by mount point — find the mount for this path
            free_gb = space_map.get(p, 0)
            if free_gb == 0:
                # Path might not be its own mount — check parent mounts
                for mount, gb in space_map.items():
                    if p.startswith(mount) or mount.startswith(p):
                        free_gb = max(free_gb, gb)
            if free_gb >= need_gb:
                dl_path = p
                logger.log(f"Download target: {p} ({free_gb}GB free)")
                break
            else:
                logger.log(f"Skipping {p} ({free_gb}GB free, need {need_gb}GB)", file_only=True)

    # Fall back to RAM if no persistent path has space
    if not dl_path:
        dl_path = ram_path
        if persistent_paths:
            logger.log("All persistent paths full — downloading directly to RAM")
        else:
            logger.log("No persistent paths configured — downloading directly to RAM")

    dl_dir = f"{dl_path}/{model_cache_name}"
    s3_model = f"{s3_source}models/{model_cache_name}/"
    logger.log(f"Downloading from S3: {s3_model} → {dl_path}")
    s3_cmd = (
        f"mkdir -p {shlex.quote(dl_dir)} && "
        f"{_s3_parallel_preamble()}"
        f"aws s3 sync {shlex.quote(s3_model)} {shlex.quote(dl_dir)}/ "
        f"--exclude 'blobs/*' --exclude '*.bin' --exclude '*.pt' --no-progress 2>&1; "
        f"{_snap_select_bash(dl_dir)}; "
        f"[ -n \"$snap\" ] && echo \"S3_OK:$snap\" || echo 'S3_FAIL'"
    )
    rc, out, _ = ssh_command(ip, s3_cmd, timeout=1800)
    if rc == 0 and "S3_OK:" in (out or ""):
        snap = out.strip().split("S3_OK:")[-1].strip().rstrip("/").split("/")[-1]
        logger.log(f"Downloaded to {dl_path}")
        serve_path = dl_path
        promote_src = dl_path

        # Persist to NVMe if downloaded elsewhere
        if nvme_path_p2 and dl_path != nvme_path_p2:
            if _copy_model(dl_path, nvme_path_p2):
                promote_src = nvme_path_p2  # promote from NVMe (faster)

        # Promote to RAM for serving (with headroom check)
        if dl_path != ram_path:
            # Prune old models from RAM if they exceed thresholds
            _maybe_prune_ram_models(ip, ram_path, model_cache_name, logger)
            rc_mem, mem_out, _ = ssh_command(ip,
                f"_sz=$(du -sm {shlex.quote(promote_src)}/{model_cache_name} 2>/dev/null | cut -f1); "
                f"_avail=$(free -m 2>/dev/null | awk '/Mem/{{print $7}}'); "
                f"echo \"$_sz $_avail\"",
                timeout=15)
            skip_ram = False
            if rc_mem == 0 and mem_out.strip():
                parts_mem = mem_out.strip().split()
                if len(parts_mem) == 2:
                    try:
                        model_mb = int(parts_mem[0])
                        avail_mb = int(parts_mem[1])
                        need_mb = model_mb + RAM_PROMOTE_HEADROOM_GB * 1024
                        if avail_mb < need_mb:
                            logger.log(f"Skipping RAM promote ({avail_mb}MB free, need {need_mb}MB) — serving from {promote_src}")
                            skip_ram = True
                    except ValueError:
                        pass
            if not skip_ram:
                if _copy_model(promote_src, ram_path):
                    serve_path = ram_path
                else:
                    serve_path = promote_src  # copy failed, serve from NVMe/dl_path
            else:
                serve_path = promote_src  # serve from NVMe (RAM too tight)
        container_path = f"/root/.cache/huggingface/hub/{model_cache_name}/snapshots/{snap}"
        return container_path, serve_path
    else:
        logger.log(f"S3 download failed: {(out or '').strip()[-200:]}", "ERROR")

    logger.log(f"Model not available", "ERROR")
    return None, None


def build_docker_run_cmd(port: int, gpus: str, name: str, model_id: str,
                         vllm_args: str = "", docker_image: str = DOCKER_IMAGE,
                         host_cache_dir: str = None) -> str:
    """Build Docker run command with proper quoting."""
    _mount_dir = host_cache_dir or effective_cache_paths(1)[0]
    # Use model HF ID — vLLM resolves from the mounted HF cache automatically
    vllm_cmd_str = f"vllm serve {shlex.quote(model_id)} --port {port}"
    if vllm_args:
        # Validate vllm_args: whitelist safe characters only.
        # Allow: alphanumeric, dashes, dots, equals, spaces, slashes, underscores, commas, colons
        if not re.match(r'^[a-zA-Z0-9 _.=/,:-]+\Z', vllm_args):
            raise ValueError(f"vllm_args contains unsafe characters: {vllm_args!r}")
        vllm_cmd_str += f" {vllm_args}"

    docker_args: List[str] = [
        "docker", "run", "-d",
        "--pull", "never",  # NEVER pull from registry — image must be pre-loaded
        "--name", name,
        "--ipc=host",  # gives container full access to host /dev/shm (no --shm-size needed)
        "--network=host",
        # Force offline mode: use only cached model files, never contact HuggingFace.
        # This avoids 401 errors on gated models when weights are already cached.
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", f"VLLM_HTTP_TIMEOUT_KEEP_ALIVE={VLLM_KEEP_ALIVE}",
        # Mount model cache into container's HF cache path
        "-v", f"{_mount_dir}:/root/.cache/huggingface/hub",
        "--entrypoint", "/bin/bash",
        docker_image,
        "-c", vllm_cmd_str
    ]
    # Build command with proper quoting. The --gpus flag needs special handling:
    # Docker requires --gpus "device=X,Y" with double quotes to avoid the
    # "cannot set both Count and DeviceIDs" CDI error. We insert it manually
    # so the quotes survive the SSH → remote bash → docker CLI chain.
    parts = [shlex.quote(a) for a in docker_args]
    # Insert --gpus "device=X,Y" after --network=host
    # Find where to insert (after --network=host)
    net_idx = parts.index("--network=host")
    gpu_flag = f'--gpus \'"device={gpus}"\''
    parts.insert(net_idx + 1, gpu_flag)
    return " ".join(parts)


def deploy_vllm_direct(ip: str, logger: HostLogger, prefix: str,
                       model_id: str,
                       vllm_args: str = "", tp_size: int = DEFAULT_TP_SIZE, repair: bool = False,
                       host_cache_dir: str = None):
    """Deploy vLLM directly on host without Docker (TP-aware)."""
    _cache = host_cache_dir or effective_cache_paths(1)[0]
    containers_started = 0
    containers_skipped = 0
    containers_busy = 0
    repair_kept = 0  # Containers that were already healthy (repair mode only)
    skip_remaining = False
    skip_reason = ""
    started_names = []

    gpu_port_map = build_gpu_port_map(tp_size)
    ip_tag = ip.replace(".", "-")

    # Query GPU memory once before the loop
    gpu_mem_map = {}
    rc_mem, out_mem, _ = ssh_command(ip,
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
        timeout=10, retries=2)
    if rc_mem == 0 and out_mem.strip():
        for mem_line in out_mem.strip().splitlines():
            parts = [p.strip() for p in mem_line.split(",")]
            if len(parts) == 2:
                try:
                    gpu_mem_map[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    else:
        logger.log("Failed to read GPU memory — treating all as busy", "WARN")
        return 0, len(gpu_port_map), 0, "nvidia-smi failed", [], 0

    # ── Phase 1: Launch all containers (pre-check + nohup for each) ──
    launched = []  # (name, pid_file, log_file) for containers that got launched

    # Ensure run directory exists on host
    rc_mkdir, _, err_mkdir = ssh_command(ip, f"mkdir -p {shlex.quote(RUN_DIR)}", timeout=10)
    if rc_mkdir != 0:
        logger.log(f"Failed to create RUN_DIR {RUN_DIR}: {err_mkdir.strip()}", "ERROR")
        return 0, len(gpu_port_map), 0, f"cannot create {RUN_DIR}", [], 0

    for port, gpu_ids_str in gpu_port_map.items():
        gpu_list = gpu_ids_str.split(",")
        gpu_label = "".join(gpu_list)
        name = f"{prefix}_{ip_tag}_gpu{gpu_label}"

        if _abort:
            logger.log("Aborting — Ctrl+C received", "WARN")
            containers_skipped += len(gpu_port_map) - (containers_started + containers_skipped + containers_busy + len(launched))
            break

        if skip_remaining:
            logger.log(f"Skipping {name} — {skip_reason}", "WARN")
            containers_skipped += 1
            continue

        # Check GPU memory (from cached results)
        # In repair mode, check process health FIRST — running processes legitimately use GPU memory
        any_busy = False
        old_pid_file = f"{RUN_DIR}/{name}.pid"

        # In repair mode: check if process is already alive and serving — skip if healthy
        if repair:
            repair_check = (
                f'pid=$(cat "{old_pid_file}" 2>/dev/null); '
                f'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && '
                f'cat /proc/$pid/cmdline 2>/dev/null | tr "\\0" " " | grep -q vllm; then '
                f'  http_code=$(curl -s -o /dev/null -w "%{{http_code}}" --connect-timeout 2 --max-time 3 http://localhost:{port}/health 2>/dev/null); '
                f'  echo "REPAIR:$http_code"; '
                f'else echo "REPAIR:DEAD"; fi'
            )
            rc_rep, rep_out, _ = ssh_command(ip, repair_check, timeout=10)
            if rc_rep == 0 and rep_out:
                if "REPAIR:200" in rep_out:
                    logger.log(f"✓ {name} already healthy on port {port} — skipping")
                    started_names.append(name)
                    containers_started += 1
                    repair_kept += 1
                    continue
                elif "REPAIR:DEAD" not in rep_out:
                    # Process alive but not healthy yet — could be loading, skip anyway
                    logger.log(f"~ {name} process alive (health={rep_out.strip().split(':')[-1]}) — skipping")
                    started_names.append(name)
                    containers_started += 1
                    repair_kept += 1
                    continue
                # DEAD — fall through to normal deploy
                logger.log(f"  {name} dead — restarting")

        for single_gpu in gpu_list:
            used_mem = gpu_mem_map.get(single_gpu, 10**9)
            if used_mem > GPU_MEM_THRESHOLD_MB:
                logger.log(f"GPU {single_gpu} busy ({used_mem} MB) → skipping {name}", "WARN")
                any_busy = True
                break
        if any_busy:
            containers_busy += 1
            continue

        # Kill old process (if ours) and check port availability in single SSH call

        pre_check_script = f"""
# Kill old vLLM process if PID file exists and process is actually vllm
KILLED=""
if [ -f "{old_pid_file}" ]; then
    pid=$(cat "{old_pid_file}" 2>/dev/null)
    if [ -n "$pid" ] && cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | grep -q vllm; then
        kill "$pid" 2>/dev/null; pkill -P "$pid" 2>/dev/null
        KILLED="$pid"
    fi
    rm -f "{old_pid_file}"
fi
# Check if port is in use (with retry if we just killed something)
port_free=0
attempts=0
max_attempts=1
[ -n "$KILLED" ] && max_attempts=5
while [ $attempts -lt $max_attempts ]; do
    if ! ss -tlnp 'sport = :{port}' 2>/dev/null | grep -q ':{port} '; then
        port_free=1
        break
    fi
    attempts=$((attempts+1))
    [ $attempts -lt $max_attempts ] && sleep 2
done
[ -n "$KILLED" ] && echo "KILLED:$KILLED"
[ $port_free -eq 1 ] && echo "PORT_FREE" || echo "PORT_BUSY"
"""
        rc_pre, pre_out, _ = ssh_command(ip, pre_check_script.strip(), timeout=20)
        if rc_pre == 0 and pre_out:
            if "KILLED:" in pre_out:
                killed_pid = pre_out.split("KILLED:")[1].split()[0]
                logger.log(f"Killed old vLLM process (PID={killed_pid})")
            if "PORT_BUSY" in pre_out:
                logger.log(f"✗ Port {port} already in use on host — skipping {name}", "WARN")
                containers_skipped += 1
                continue

        log_file = f"{RUN_DIR}/{name}.log"
        pid_file = f"{RUN_DIR}/{name}.pid"

        # Build the vLLM command with env vars
        env_parts = [f"CUDA_VISIBLE_DEVICES={gpu_ids_str}"]
        env_parts.append("HF_HUB_OFFLINE=1")
        env_parts.append("TRANSFORMERS_OFFLINE=1")
        env_parts.append(f"VLLM_HTTP_TIMEOUT_KEEP_ALIVE={VLLM_KEEP_ALIVE}")
        env_parts.append(f"HF_HUB_CACHE={_cache}")
        vllm_cmd_parts = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id, "--port", str(port),
            "--download-dir", _cache,
        ]
        if vllm_args:
            vllm_cmd_parts.extend(shlex.split(vllm_args))

        env_str = " ".join(shlex.quote(e) for e in env_parts)
        args_str = " ".join(shlex.quote(a) for a in vllm_cmd_parts)
        launch_script = f"{RUN_DIR}/{name}.sh"
        write_script_cmd = f"""cat > "{launch_script}" << 'VLLM_EOF'
#!/bin/bash
echo $$ > "{pid_file}"
exec env {env_str} {args_str}
VLLM_EOF
chmod +x "{launch_script}"
nohup "{launch_script}" > "{log_file}" 2>&1 &"""

        logger.log(f"Starting {name} on GPU(s) {gpu_ids_str} (port {port}, TP={tp_size})...")
        rc, out, err = ssh_command(ip, write_script_cmd, timeout=30)

        if rc == 0:
            launched.append((name, pid_file, log_file))
        else:
            logger.log(f"✗ Failed to start {name}: {err.strip()}", "ERROR")
            containers_skipped += 1

    # ── Phase 2: Single sleep, then batched PID + liveness check ──
    if launched:
        _crash_check_delay = {1: 2, 2: 3, 4: 5, 8: 8}
        time.sleep(_crash_check_delay.get(tp_size, 5))
        # Build single script to check all PIDs at once
        check_parts = []
        for name, pid_file, log_file in launched:
            check_parts.append(
                f'pid=$(cat "{pid_file}" 2>/dev/null); '
                f'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && '
                f'cat /proc/$pid/cmdline 2>/dev/null | tr "\\0" " " | grep -q vllm; then '
                f'echo "ALIVE:{name}:$pid"; '
                f'else echo "DEAD:{name}"; '
                f'tail -20 "{log_file}" 2>/dev/null | sed "s/^/LOG:{name}:/"; '
                f'fi'
            )
        rc_check, check_out, _ = ssh_command(ip, "\n".join(check_parts), timeout=max(15, len(launched) * 3))

        if rc_check == 0 and check_out:
            alive_set = set()
            dead_set = set()
            dead_logs = {}  # name -> log lines
            for line in check_out.strip().splitlines():
                if line.startswith("ALIVE:"):
                    parts = line[6:].split(":", 1)
                    cname = parts[0]
                    pid = parts[1] if len(parts) > 1 else "?"
                    alive_set.add(cname)
                    logger.log(f"✓ {cname} started (PID={pid})")
                elif line.startswith("DEAD:"):
                    cname = line[5:]
                    dead_set.add(cname)
                elif line.startswith("LOG:"):
                    # Format: LOG:container_name:log_line
                    rest = line[4:]
                    colon = rest.find(":")
                    if colon > 0:
                        cname = rest[:colon]
                        log_line = rest[colon+1:]
                        dead_logs.setdefault(cname, []).append(log_line)

            for name, pid_file, log_file in launched:
                if name in alive_set:
                    started_names.append(name)
                    containers_started += 1
                elif name in dead_set:
                    logger.log(f"✗ {name} started but crashed immediately", "ERROR")
                    log_lines = dead_logs.get(name, [])
                    if log_lines:
                        logger.log(f"--- Process log for {name} ---")
                        for ll in log_lines:
                            logger.log(f"  | {ll}")
                        logger.log(f"--- End log ---")
                        log_lower = "\n".join(log_lines).lower()
                        if "cuda" in log_lower or "nvidia" in log_lower:
                            skip_remaining = True
                            skip_reason = "CUDA/NVIDIA error"
                    containers_skipped += 1
                else:
                    # No output for this container — PID file missing
                    logger.log(f"✗ {name} — could not read PID file", "ERROR")
                    containers_skipped += 1
        else:
            # Batch check failed — count all as unknown/started (optimistic)
            logger.log("⚠ Could not verify launched processes", "WARN")
            for name, _, _ in launched:
                started_names.append(name)
                containers_started += 1

    return containers_started, containers_skipped, containers_busy, skip_reason, started_names, repair_kept


def deploy_vllm_containers(ip: str, logger: HostLogger, prefix: str,
                           model_id: str,
                           vllm_args: str = "", docker_image: str = DOCKER_IMAGE,
                           tp_size: int = DEFAULT_TP_SIZE, image_freshly_loaded: bool = False,
                           repair: bool = False, host_cache_dir: str = None):
    """Deploy vLLM containers on all available GPUs (TP-aware)."""
    containers_started = 0
    containers_skipped = 0
    containers_busy = 0
    repair_kept = 0  # Containers that were already healthy (repair mode only)
    skip_remaining = False
    skip_reason = ""
    started_names = []  # Track container names for CSV output

    gpu_port_map = build_gpu_port_map(tp_size)
    ip_tag = ip.replace(".", "-")

    # ── Pre-loop batch: GPU memory + port checks + stale container cleanup (1 SSH) ──
    # Build a single script that: queries GPU mem, checks all ports, removes stale containers
    container_names_for_ports = {}
    pre_script_parts = [
        'echo "===GPU_MEM==="',
        'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "NVIDIA_FAIL"',
        'echo "===PORTS==="',
    ]
    for port, gpu_ids_str in gpu_port_map.items():
        gpu_list = gpu_ids_str.split(",")
        gpu_label = "".join(gpu_list)
        name = f"{prefix}_{ip_tag}_gpu{gpu_label}"
        container_names_for_ports[port] = (name, gpu_ids_str)
        if repair:
            # Repair mode: check container status + health, don't remove anything
            pre_script_parts.append(
                f'status=$(docker inspect --format "{{{{.State.Status}}}}" {name} 2>/dev/null) || status="missing"; '
                f'if [ "$status" = "running" ]; then '
                f'  http_code=$(curl -s -o /dev/null -w "%{{http_code}}" --connect-timeout 2 --max-time 3 http://localhost:{port}/health 2>/dev/null); '
                f'  echo "REPAIR:{port}:$status:$http_code"; '
                f'else echo "REPAIR:{port}:$status:000"; fi'
            )
        else:
            # Normal mode: remove stale container and check port in one shot
            pre_script_parts.append(f'docker rm -f {name} 2>/dev/null; '
                                    f'ss -tlnp "sport = :{port}" 2>/dev/null | grep -q ":{port} " && echo "BUSY:{port}" || echo "FREE:{port}"')

    rc_pre, pre_out, _ = ssh_command(ip, "\n".join(pre_script_parts), timeout=30, retries=2)

    # Parse GPU memory
    gpu_mem_map = {}
    port_status = {}  # port -> "FREE" or "BUSY"
    repair_status = {}  # port -> (container_status, health_code) for repair mode
    if rc_pre == 0 and pre_out:
        section = None
        for line in pre_out.strip().splitlines():
            line = line.strip()
            if line == "===GPU_MEM===":
                section = "gpu"
                continue
            elif line == "===PORTS===":
                section = "ports"
                continue
            if section == "gpu" and line and line != "NVIDIA_FAIL":
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    try:
                        gpu_mem_map[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
            elif section == "ports":
                if line.startswith("BUSY:"):
                    try:
                        port_status[int(line[5:])] = "BUSY"
                    except ValueError:
                        pass
                elif line.startswith("FREE:"):
                    try:
                        port_status[int(line[5:])] = "FREE"
                    except ValueError:
                        pass
                elif line.startswith("REPAIR:"):
                    # REPAIR:{port}:{container_status}:{health_code}
                    rparts = line[7:].split(":", 2)
                    if len(rparts) == 3:
                        try:
                            rport = int(rparts[0])
                            repair_status[rport] = (rparts[1], rparts[2])
                        except ValueError:
                            pass
    else:
        logger.log("Failed to run pre-deploy checks — treating all as busy", "WARN")
        return 0, len(gpu_port_map), 0, "pre-deploy checks failed", [], 0

    if not gpu_mem_map:
        logger.log("Failed to read GPU memory — treating all as busy", "WARN")
        return 0, len(gpu_port_map), 0, "nvidia-smi failed", [], 0

    # ── Phase 1: Fire all docker run -d, handle launch errors inline ──
    launched = []  # (name, container_id) for successfully created containers
    for port, gpu_ids_str in gpu_port_map.items():
        name, _ = container_names_for_ports[port]
        gpu_list = gpu_ids_str.split(",")
        gpu_label = "".join(gpu_list)

        # Check global abort flag
        if _abort:
            logger.log(f"Aborting — Ctrl+C received", "WARN")
            containers_skipped += len(gpu_port_map) - (containers_started + containers_skipped + containers_busy + len(launched))
            break

        # Skip remaining GPUs if a fatal host-level error was detected
        if skip_remaining:
            logger.log(f"Skipping {name} — {skip_reason}", "WARN")
            containers_skipped += 1
            continue

        # Repair mode: skip containers that are already running and healthy
        # Must check BEFORE GPU busy — healthy containers will show GPU as "busy"
        if repair and port in repair_status:
            cstatus, hcode = repair_status[port]
            if cstatus == "running" and hcode == "200":
                logger.log(f"✓ {name} already healthy on port {port} — skipping")
                started_names.append(name)
                containers_started += 1
                repair_kept += 1
                continue
            elif cstatus == "running":
                # Running but not healthy (loading, or partially up) — leave it alone
                logger.log(f"~ {name} running (health={hcode}) — skipping")
                started_names.append(name)
                containers_started += 1
                repair_kept += 1
                continue
            else:
                # Dead/exited/missing — remove stale and restart
                logger.log(f"  {name} is {cstatus} — removing and restarting")
                rc_rm, rm_out, _ = ssh_command(ip,
                    f"docker rm -f {name} 2>/dev/null; "
                    f"ss -tlnp 'sport = :{port}' 2>/dev/null | grep -q ':{port} ' && echo 'PORT_BUSY' || echo 'PORT_FREE'",
                    timeout=10)
                if rc_rm == 0 and "PORT_BUSY" in (rm_out or ""):
                    logger.log(f"✗ Port {port} occupied by another process after removing {name} — skipping", "WARN")
                    containers_skipped += 1
                    continue

        # Check ALL GPUs in this TP group are free (from cached results)
        any_busy = False
        for single_gpu in gpu_list:
            used_mem = gpu_mem_map.get(single_gpu, 10**9)
            if used_mem > GPU_MEM_THRESHOLD_MB:
                logger.log(f"GPU {single_gpu} busy ({used_mem} MB) → skipping {name}", "WARN")
                any_busy = True
                break
        if any_busy:
            containers_busy += 1
            continue

        # Check if port is already in use (from pre-loop batch results)
        if port_status.get(port) == "BUSY":
            logger.log(f"✗ Port {port} already in use on host — skipping {name}", "WARN")
            containers_skipped += 1
            continue

        cmd = build_docker_run_cmd(port, gpu_ids_str, name, model_id, vllm_args, docker_image, host_cache_dir=host_cache_dir)

        logger.log(f"Starting {name} on GPU(s) {gpu_ids_str} (port {port}, TP={tp_size})...")
        # Scale timeout with TP: more GPUs = longer CUDA/NCCL init
        # TP1=60s, TP2=90s, TP4=135s, TP8=300s
        _tp_timeouts = {1: 60, 2: 90, 4: 135, 8: 300}
        docker_run_timeout = _tp_timeouts.get(tp_size, 202)
        # Freshly loaded images (TAR/pull) need extra time — Docker is still
        # indexing layers and the first container start is significantly slower
        if image_freshly_loaded:
            docker_run_timeout = docker_run_timeout * 2
            if len(launched) == 0 and containers_started == 0:
                logger.log(f"Image freshly loaded — using extended timeout ({docker_run_timeout}s)")
        rc, out, err = ssh_command(ip, cmd, timeout=docker_run_timeout)

        if rc == 0:
            container_id = out.strip()[:12]
            launched.append((name, container_id))
        else:
            err_lower = err.lower()

            # Name conflict — try harder to remove and retry once
            if "already in use" in err_lower or "conflict" in err_lower:
                if repair:
                    # REPAIR SAFETY: name conflict means container exists but wasn't in repair_status
                    # (partial pre-script output). Do NOT kill it — it might be healthy.
                    logger.log(f"✗ Name conflict for {name} in repair mode — skipping (container may be healthy)", "WARN")
                    containers_skipped += 1
                    continue
                logger.log(f"✗ Name conflict for {name} — trying stop + rm...", "WARN")
                ssh_command(ip, f"docker stop -t 5 {name} 2>/dev/null; docker rm -f {name} 2>/dev/null", timeout=20)
                time.sleep(2)  # Give Docker daemon a moment to release the name
                # Retry once
                rc2, out2, err2 = ssh_command(ip, cmd, timeout=docker_run_timeout)
                if rc2 == 0:
                    container_id = out2.strip()[:12]
                    launched.append((name, container_id))
                    continue
                err2_lower = err2.lower()
                if "already in use" in err2_lower or "conflict" in err2_lower:
                    logger.log(f"✗ Name conflict persists for {name} — Docker daemon stuck, skipping remaining GPUs", "ERROR")
                    skip_remaining = True
                    skip_reason = "Docker daemon stuck (persistent name conflict)"
                    containers_skipped += 1
                else:
                    logger.log(f"✗ Retry failed for {name}: {err2.strip()}", "ERROR")
                    containers_skipped += 1
            elif "no space left on device" in err_lower:
                logger.log(f"✗ Disk full on host — attempting docker prune...", "WARN")
                prune_rc, prune_out, _ = ssh_command(ip, "docker system prune -f 2>&1", timeout=60)
                if prune_rc == 0:
                    logger.log(f"Prune output: {prune_out.strip()}", file_only=True)
                    for line in prune_out.strip().splitlines():
                        if "reclaimed" in line.lower():
                            logger.log(f"Prune result: {line.strip()}")
                            break
                    else:
                        logger.log("Prune completed")

                    # Quick check: is Docker still responsive after prune?
                    check_rc, _, check_err = ssh_command(ip, "docker images -q 2>&1 | head -1", timeout=10)
                    if check_rc == -1 and "timeout" in check_err.lower():
                        logger.log(f"✗ Docker daemon hung after prune — skipping remaining GPUs", "ERROR")
                        skip_remaining = True
                        skip_reason = "Docker daemon hung"
                        containers_skipped += 1
                        continue

                    logger.log(f"Retrying {name} after prune...")
                    rc2, out2, err2 = ssh_command(ip, cmd, timeout=docker_run_timeout)
                    if rc2 == 0:
                        container_id = out2.strip()[:12]
                        launched.append((name, container_id))
                        continue
                    else:
                        err2_lower = err2.lower()
                        if "command timed out" in err2_lower:
                            logger.log(f"✗ Docker daemon hung on retry — skipping remaining GPUs", "ERROR")
                            skip_remaining = True
                            skip_reason = "Docker daemon hung"
                            containers_skipped += 1
                            continue
                        else:
                            logger.log(f"✗ Still failing after prune: {err2.strip()}", "ERROR")

                skip_remaining = True
                skip_reason = "disk full on host"
                logger.log(f"Disk full persists — skipping remaining GPUs on this host", "ERROR")
                containers_skipped += 1
            elif "command timed out" in err_lower:
                logger.log(f"✗ Docker command timed out — skipping remaining GPUs", "ERROR")
                skip_remaining = True
                skip_reason = "Docker commands timing out"
                containers_skipped += 1
            elif "address already in use" in err_lower or "bind" in err_lower:
                logger.log(f"✗ Port {port} already in use on host (another container or process) — skipping {name}", "ERROR")
                containers_skipped += 1
                # Don't skip remaining — other ports may be free
            else:
                logger.log(f"✗ Failed to start {name}: {err.strip()}", "ERROR")
                capture_container_logs(ip, name, logger)
                containers_skipped += 1

                # Detect nvidia toolkit errors — skip remaining if runtime is broken
                if "disable-device-node-modification" in err_lower or "oci runtime" in err_lower:
                    logger.log(f"NVIDIA runtime broken on this host — skipping remaining GPUs", "ERROR")
                    skip_remaining = True
                    skip_reason = "NVIDIA runtime broken"
                # Image not found (--pull=never prevents silent pull from registry)
                elif "no such image" in err_lower or "not found" in err_lower:
                    logger.log(f"Docker image missing — skipping remaining GPUs", "ERROR")
                    skip_remaining = True
                    skip_reason = "Docker image not found"

    # ── Phase 2: Single sleep, then batched status check for all launched containers ──
    if launched:
        _crash_check_delay = {1: 2, 2: 3, 4: 5, 8: 8}
        time.sleep(_crash_check_delay.get(tp_size, 5))

        # Build single inspect command for all launched containers
        inspect_parts = []
        for name, _ in launched:
            inspect_parts.append(
                f'status=$(docker inspect --format "{{{{.State.Status}}}}" {name} 2>/dev/null) && '
                f'echo "ST:{name}:$status" || echo "ST:{name}:missing"'
            )
        rc_check, check_out, _ = ssh_command(ip, "\n".join(inspect_parts), timeout=max(15, len(launched) * 3))

        # Parse results
        status_map = {}  # name -> status
        if rc_check == 0 and check_out:
            for line in check_out.strip().splitlines():
                if line.startswith("ST:"):
                    rest = line[3:]
                    colon = rest.rfind(":")
                    if colon > 0:
                        cname = rest[:colon]
                        cstatus = rest[colon+1:]
                        status_map[cname] = cstatus

        for name, container_id in launched:
            status = status_map.get(name, "unknown")
            if status == "running":
                logger.log(f"✓ {name} started and running (container={container_id})")
                started_names.append(name)
                containers_started += 1
            elif status in ("exited", "dead", "created"):
                logger.log(f"✗ {name} started but crashed immediately (status={status})", "ERROR")
                # Fetch logs for crashed container
                rc_logs, log_out, _ = ssh_command(ip, f"docker logs --tail 50 {name} 2>&1", timeout=15)
                if rc_logs == 0 and log_out.strip():
                    logger.log(f"--- Container logs for {name} (last 50 lines) ---")
                    for line in log_out.strip().splitlines():
                        logger.log(f"  | {line}")
                    logger.log(f"--- End container logs for {name} ---")
                    log_lower = log_out.lower()
                    if "cuda" in log_lower or "nvidia" in log_lower or "out of memory" in log_lower:
                        logger.log("CUDA/NVIDIA error detected — skipping remaining containers", "ERROR")
                        skip_remaining = True
                        skip_reason = "CUDA/NVIDIA error on container start"
                else:
                    logger.log(f"Could not retrieve logs for {name}", "WARN")
                containers_skipped += 1
            elif status == "missing":
                logger.log(f"✗ {name} vanished after launch (container={container_id})", "ERROR")
                containers_skipped += 1
            else:
                # Ambiguous state (still starting, restarting) — count as started optimistically
                logger.log(f"⚠ {name} created (container={container_id}, status={status}) — may still be loading")
                started_names.append(name)
                containers_started += 1

    return containers_started, containers_skipped, containers_busy, skip_reason, started_names, repair_kept


def capture_host_info(ip: str, logger: HostLogger):
    """Capture basic host info for the log (GPU status, disk, docker status) in a single SSH call."""
    script = """
echo "===GPU==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null
echo "===CONTAINERS==="
docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep vllm || echo 'No vllm containers running'
echo "===DISK==="
df -h /opt/dlami/nvme 2>/dev/null | tail -1 || echo 'N/A'
"""
    rc, out, _ = ssh_command(ip, script.strip(), timeout=15)
    if rc != 0 or not out:
        return
    # Split output into sections by markers
    sections = {}
    current = None
    for line in out.strip().splitlines():
        s = line.strip()
        if s in ("===GPU===", "===CONTAINERS===", "===DISK==="):
            current = s
            sections[current] = []
        elif current and s:
            sections[current].append(s)
    gpu_lines = sections.get("===GPU===", [])
    if gpu_lines:
        logger.log("GPU status:")
        for g in gpu_lines:
            logger.log(f"  {g}")
    for c in sections.get("===CONTAINERS===", []):
        logger.log(f"Existing containers: {c}")
    for d in sections.get("===DISK===", []):
        if d != "N/A":
            logger.log(f"Disk NVMe: {d}")


def process_host(ip: str, ssh_timeout: int = 15, prefix: str = "", model_id: str = "",
                 s3_bucket: str = "", vllm_args: str = "", docker_image: str = DOCKER_IMAGE,
                 tp_size: int = DEFAULT_TP_SIZE, no_docker: bool = False, repair: bool = False) -> dict:
    """Process a single host - check connectivity, load image, deploy containers."""
    if _abort:
        return {
            "ip": ip, "success": False, "containers_started": 0,
            "containers_skipped": 0, "error": "Aborted (Ctrl+C)",
            "log_file": str(Path(LOG_DIR) / f"{ip}.log"),
        }
    logger = HostLogger(ip, LOG_DIR)
    result = {
        "ip": ip,
        "success": False,
        "containers_started": 0,
        "containers_skipped": 0,
        "error": None,
        "log_file": str(logger.log_path),
    }

    logger.log("=" * 60)
    logger.log("Starting deployment")
    logger.log("=" * 60)

    # Check SSH connectivity with retries (no mux — avoids race with other threads)
    rc, out, err = -1, "", "No connection attempts made"
    for attempt in range(1, RETRY_COUNT + 1):
        rc, out, err = ssh_command(ip, "echo ok", timeout=ssh_timeout, use_multiplexing=False)
        if rc == 0:
            break
        logger.log(f"SSH check failed (attempt {attempt}/{RETRY_COUNT}): {err.strip()}", "WARN")
        if attempt < RETRY_COUNT:
            time.sleep(RETRY_SLEEP)

    if rc != 0:
        result["error"] = f"SSH failed: {err.strip()}"
        result["failure_category"] = "ssh_failed"
        logger.log(result["error"], "ERROR")
        logger.flush()
        return result

    logger.log("SSH connectivity OK")

    # DRY_RUN: skip actual host work — just simulate success
    if DRY_RUN:
        total_containers = len(build_gpu_port_map(tp_size))
        result["success"] = True
        result["containers_started"] = total_containers
        result["container_names"] = [f"{prefix}_{ip.replace('.', '-')}_dryrun_{i}" for i in range(total_containers)]
        logger.log(f"DRY RUN — would deploy {total_containers} containers")
        logger.flush()
        return result

    # Capture host info before deployment
    capture_host_info(ip, logger)

    if _abort:
        result["error"] = "Aborted (Ctrl+C)"
        logger.flush()
        return result

    # Load Docker image (skip in no-docker mode)
    image_freshly_loaded = False
    if not no_docker:
        image_status = load_docker_image_from_tar(ip, logger, docker_image, s3_bucket)
        if not image_status:
            result["error"] = "Docker image load failed"
            result["failure_category"] = "docker_issue"
            logger.log(result["error"], "ERROR")
            logger.flush()
            return result
        image_freshly_loaded = (image_status == "loaded")
    else:
        # Verify vLLM is installed on host
        rc_vllm, out_vllm, _ = ssh_command(ip, "python3 -c 'import vllm; print(vllm.__version__)' 2>&1", timeout=15)
        if rc_vllm != 0:
            result["error"] = f"vLLM not installed on host (no-docker mode requires vLLM)"
            result["failure_category"] = "vllm_missing"
            logger.log(result["error"], "ERROR")
            logger.flush()
            return result
        logger.log(f"Host vLLM version: {out_vllm.strip()}")

    if _abort:
        result["error"] = "Aborted (Ctrl+C)"
        logger.flush()
        return result

    # Ensure model is available in local cache (download from S3 if needed)
    model_path, _cache_dir = ensure_model_available(ip, logger, model_id, s3_bucket)

    if model_path is None:
        result["error"] = "Model download failed on all strategies"
        result["failure_category"] = "model_download"
        logger.log(result["error"], "ERROR")
        logger.flush()
        return result

    if _abort:
        result["error"] = "Aborted (Ctrl+C)"
        logger.flush()
        return result

    # Bulk cleanup: remove ALL old containers/processes matching our prefix on this host
    # In repair mode, skip cleanup — we want to keep healthy containers running
    if repair:
        logger.log("Repair mode — skipping cleanup, will only start missing containers")
    elif no_docker:
        # Kill old processes by PID files (batched into single SSH call with recycling protection)
        ip_tag = ip.replace(".", "-")
        cleanup_script = f"""
killed=0
for pf in "{RUN_DIR}/{prefix}_{ip_tag}_gpu"*.pid; do
    [ -f "$pf" ] || continue
    pid=$(cat "$pf" 2>/dev/null)
    if [ -n "$pid" ] && cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | grep -q vllm; then
        kill "$pid" 2>/dev/null; pkill -P "$pid" 2>/dev/null
        killed=$((killed+1))
    fi
    rm -f "$pf"
done
echo $killed
"""
        rc, out, _ = ssh_command(ip, cleanup_script.strip(), timeout=20)
        if rc == 0 and out.strip() and out.strip() != "0":
            logger.log(f"Cleaned up {out.strip()} old {prefix}_* processes")
    else:
        # Docker cleanup — list, remove, verify in single SSH call (saves 2-3 round trips)
        cleanup_script = f"""echo "===OTHERS==="
docker ps --format '{{{{.Names}}}}' 2>/dev/null | grep -v '^{prefix}_' | head -5
echo "===CLEANUP==="
ids=$(docker ps -aq --filter name=^{prefix}_ 2>/dev/null)
if [ -n "$ids" ]; then
    count=$(echo "$ids" | wc -l)
    docker rm -f $ids >/dev/null 2>&1
    remaining=$(docker ps -aq --filter name=^{prefix}_ 2>/dev/null)
    if [ -n "$remaining" ]; then
        docker rm -f $remaining >/dev/null 2>&1
        remaining2=$(docker ps -aq --filter name=^{prefix}_ 2>/dev/null)
        [ -n "$remaining2" ] && echo "STUCK" || echo "CLEANED:$count"
    else
        echo "CLEANED:$count"
    fi
else
    echo "CLEANED:0"
fi
"""
        rc, out, err = ssh_command(ip, cleanup_script.strip(), timeout=30)
        if rc == 0 and out:
            # Parse others warning
            in_others = False
            others = []
            for line in out.strip().splitlines():
                line = line.strip()
                if line == "===OTHERS===":
                    in_others = True
                    continue
                elif line == "===CLEANUP===":
                    in_others = False
                    continue
                if in_others and line:
                    others.append(line)
            if others:
                logger.log(f"Other running containers on host (may consume GPUs): {', '.join(others)}", "WARN")
            # Parse cleanup result
            if "STUCK" in out:
                result["error"] = "Docker daemon stuck — cannot remove old containers"
                result["failure_category"] = "docker_issue"
                logger.log(result["error"], "ERROR")
                logger.flush()
                return result
            for line in out.strip().splitlines():
                if line.startswith("CLEANED:") and line[8:].strip() != "0":
                    logger.log(f"Cleaned up {line[8:].strip()} old {prefix}_* containers")
        elif "command timed out" in (err or "").lower():
            result["error"] = "Docker daemon hung — cannot remove old containers"
            result["failure_category"] = "docker_issue"
            logger.log(result["error"], "ERROR")
            logger.flush()
            return result

    # Deploy
    if no_docker:
        started, skipped, busy, skip_reason, started_names, repair_kept = deploy_vllm_direct(
            ip, logger, prefix, model_id, vllm_args, tp_size, repair=repair, host_cache_dir=_cache_dir)
    else:
        started, skipped, busy, skip_reason, started_names, repair_kept = deploy_vllm_containers(
            ip, logger, prefix, model_id, vllm_args, docker_image, tp_size, image_freshly_loaded, repair=repair, host_cache_dir=_cache_dir)
    result["containers_started"] = started
    result["containers_skipped"] = skipped
    result["containers_busy"] = busy
    result["skip_reason"] = skip_reason
    result["container_names"] = started_names
    newly_launched = started - repair_kept
    result["newly_launched"] = newly_launched
    if repair_kept > 0:
        result["repair_kept"] = repair_kept
    total_containers = len(build_gpu_port_map(tp_size))
    available = total_containers - busy
    if started > 0 and skipped == 0:
        # All available GPUs deployed successfully
        result["success"] = True
    elif started > 0:
        result["success"] = True
        result["partial"] = True
        result["error"] = f"{started}/{available} available slots started, {skipped} failed ({busy} busy)"
    else:
        result["success"] = False
        if available == 0:
            result["error"] = f"All {total_containers} GPU groups busy"
        else:
            result["error"] = f"All {skipped} containers skipped ({busy} GPU groups busy)"
        # Categorize failure
        if "disk full" in (skip_reason or "").lower() or "no space" in (skip_reason or "").lower():
            result["failure_category"] = "disk_full"
        elif "docker" in (skip_reason or "").lower() or "daemon" in (skip_reason or "").lower():
            result["failure_category"] = "docker_issue"
        elif "nvidia" in (skip_reason or "").lower():
            result["failure_category"] = "nvidia_broken"

    log_msg = f"Deployment complete: {started} started"
    if repair_kept > 0:
        log_msg += f" ({repair_kept} already healthy, {newly_launched} newly launched)"
    if skipped > 0:
        log_msg += f", {skipped} failed"
    if busy > 0:
        log_msg += f", {busy} GPUs busy"
    logger.log(log_msg)
    logger.flush()
    return result


def write_summary(results: List[dict], elapsed: float, deploy_config: dict = None):
    """Write a JSON summary and human-readable summary to the log directory."""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "deployer": f"{getpass.getuser()}@{socket.gethostname()}",
        "deployer_ip": _get_deployer_ip(),
        "deployer_cwd": os.getcwd(),
        "command": sys.argv,
        "elapsed_seconds": round(elapsed, 1),
        "total_hosts": len(results),
        "successful_hosts": sum(1 for r in results if r["success"]),
        "total_containers_started": sum(r["containers_started"] for r in results),
        "total_containers_skipped": sum(r["containers_skipped"] for r in results),
        "hosts": results,
    }
    if deploy_config:
        summary["deploy_config"] = deploy_config

    json_path = log_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable summary
    txt_path = log_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DEPLOYMENT SUMMARY\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Deployer:  {summary['deployer']} ({summary.get('deployer_ip', 'unknown')})\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total time:          {elapsed:.1f}s\n")
        f.write(f"Successful hosts:    {summary['successful_hosts']}/{summary['total_hosts']}\n")
        f.write(f"Containers started:  {summary['total_containers_started']}\n")
        f.write(f"Containers skipped:  {summary['total_containers_skipped']}\n")
        f.write(f"Log directory:       {LOG_DIR}\n")
        f.write("\n")

        failed = [r for r in results if not r["success"]]
        if failed:
            f.write(f"FAILED HOSTS ({len(failed)}):\n")
            for r in failed:
                f.write(f"  {r['ip']:20s} → {r['error']}\n")
            f.write("\n")

        f.write("PER-HOST BREAKDOWN:\n")
        for r in sorted(results, key=lambda x: x["ip"]):
            status = "✓" if r["success"] else "✗"
            f.write(f"  {status} {r['ip']:20s}  started={r['containers_started']}  skipped={r['containers_skipped']}")
            if r["error"]:
                f.write(f"  error={r['error']}")
            f.write(f"  log={r['log_file']}\n")
        f.write("=" * 80 + "\n")

    return json_path, txt_path


def load_ips_from_file(filepath: str) -> List[str]:
    """Load IPs from a file. Supports:
    - One IP per line
    - Comma-separated IPs
    - Mixed (comma-separated across multiple lines)
    - Blank lines and comments (#) are ignored
    """
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: IP file '{filepath}' not found")
        sys.exit(1)

    ips = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()   # strip inline comments
            if not line:
                continue
            for part in line.split(","):
                part = part.strip()
                if part:
                    ips.append(part)

    if not ips:
        print(f"ERROR: No IPs found in '{filepath}'")
        sys.exit(1)

    # Deduplicate while preserving order
    seen = set()
    unique_ips = []
    for ip in ips:
        if ip not in seen:
            seen.add(ip)
            unique_ips.append(ip)

    return unique_ips


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-host vLLM parallel deployment")
    parser.add_argument("--hosts", nargs="+", metavar="IP",
                        help="Deploy only to these specific hosts")
    parser.add_argument("--ip-file", type=str, default=None,
                        help="CSV/text file with host IPs (one per line)")
    parser.add_argument("--failed-from", metavar="SUMMARY_JSON",
                        help="Re-deploy only failed hosts from a previous summary.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Number of parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--ssh-timeout", type=int, default=15,
                        help="SSH connectivity check timeout in seconds (default: 15)")
    parser.add_argument("--batch-delay", type=float, default=0,
                        help="Delay in seconds between launching each host (default: 0)")
    parser.add_argument("--name", type=str, required=True,
                        help="Container name prefix (REQUIRED). e.g. gptoss120b, llama405b")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID (e.g. openai/gpt-oss-120b). "
                             "Required for deploy, repair, inspect-cache.")
    parser.add_argument("--config", type=str, required=True,
                        help="Config file defining cache hierarchy, S3 source, streams, gpus_per_node. "
                             "See sample.conf for format.")
    parser.add_argument("--tp", type=int, default=None,
                        help="Tensor parallelism size (REQUIRED for deploy/verify). "
                             "Must divide gpus_per_node (from config) evenly.")
    parser.add_argument("--vllm-args", type=str, default="--async-scheduling --trust-remote-code --max-model-len 32768 --gpu-memory-utilization 0.90",
                        help="Extra vLLM serve arguments (default: --async-scheduling --trust-remote-code --max-model-len 32768 --gpu-memory-utilization 0.90). "
                             "Note: --tensor-parallel-size is auto-injected from --tp, do NOT include it here.")
    parser.add_argument("--docker-image", type=str, default=None,
                        help="Override Docker image from config")
    parser.add_argument("--keep-alive", type=int, default=600, help="vLLM keepalive timeout seconds (default: 600)")
    parser.add_argument("--base-port", type=int, default=None,
                        help="Starting port number for vLLM endpoints (REQUIRED for deploy/verify, e.g. 35000). "
                             "Ports are assigned sequentially: base, base+1, base+2, ...")
    parser.add_argument("--no-docker", action="store_true",
                        help="Run vLLM directly on host (no Docker). Requires vLLM pre-installed on nodes.")
    parser.add_argument("--verify", action="store_true",
                        help="Verify containers are running and healthy (no deployment)")
    parser.add_argument("--teardown", action="store_true",
                        help="Remove all containers matching prefix on target hosts")
    parser.add_argument("--repair", action="store_true",
                        help="Only start missing/dead containers — leave healthy ones running")
    parser.add_argument("--inspect-cache", action="store_true",
                        help="Check cache status on all nodes — show where model/TAR exist, no changes")
    parser.add_argument("--skip-prewarm", action="store_true",
                        help="Skip pre-warm stage (nodes download from S3 during deploy instead)")
    parser.add_argument("--temp-folder", type=str, default=None,
                        help="Temp folder on scout node for preflight downloads (default: auto-detect /tmp → /dev/shm)")
    return parser.parse_args()


def resolve_ip_list(args) -> List[str]:
    """Determine which IPs to deploy to based on CLI args."""
    if args.failed_from:
        path = Path(args.failed_from)
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)
        try:
            with open(path) as f:
                summary = json.load(f)
            failed = [h["ip"] for h in summary["hosts"] if not h["success"]]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"ERROR: Invalid summary file {path}: {e}")
            sys.exit(1)
        if not failed:
            print("No failed hosts found in summary — nothing to re-deploy!")
            sys.exit(0)
        print(f"Re-deploying {len(failed)} failed hosts from {path}")
        ips = failed
    elif args.hosts:
        ips = args.hosts
    elif args.ip_file:
        ips = load_ips_from_file(args.ip_file)
    else:
        print("ERROR: No target nodes specified.")
        print("Use --ip-file <file> or --hosts <IP ...> or --failed-from <summary.json>")
        sys.exit(1)

    # Validate IP format (defense-in-depth: IPs flow into container names used unquoted in shell)
    ip_re = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    for ip in ips:
        if not ip_re.match(ip):
            print(f"ERROR: Invalid IP format: {ip!r}")
            print("Only IPv4 addresses (e.g. 10.0.1.1) are supported")
            sys.exit(1)

    # Deduplicate while preserving order (duplicates cause race conditions in parallel deploy)
    seen = set()
    unique_ips = []
    for ip in ips:
        if ip not in seen:
            seen.add(ip)
            unique_ips.append(ip)
    if len(unique_ips) < len(ips):
        print(f"NOTE: Removed {len(ips) - len(unique_ips)} duplicate IPs")
    return unique_ips


def verify_host(ip: str, prefix: str, ssh_timeout: int = 15, tp_size: int = DEFAULT_TP_SIZE, no_docker: bool = False) -> dict:
    """Verify all containers/processes on a single host — batched into minimal SSH calls."""
    ip_tag = ip.replace(".", "-")
    result = {
        "ip": ip,
        "reachable": False,
        "endpoints": [],
        "healthy": 0,
        "unhealthy": 0,
        "missing": 0,
        "gpu_info": "",
    }

    # SSH check
    rc, _, err = ssh_command(ip, "echo ok", timeout=ssh_timeout, retries=2)
    if rc != 0:
        return result
    result["reachable"] = True

    verify_port_map = build_gpu_port_map(tp_size)

    # Build a single script that checks everything: GPU mem + all container statuses + all health endpoints + GPU processes
    script_parts = ['echo "===GPU==="', 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "NVIDIA_FAIL"']
    # Also collect what processes are using each GPU (for diagnosing busy GPUs with missing containers)
    script_parts.append('echo "===GPUMAP==="')
    script_parts.append('nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader 2>/dev/null || echo "MAP_FAIL"')
    script_parts.append('echo "===GPUPROCS==="')
    script_parts.append('nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null || echo "PROCS_FAIL"')
    script_parts.append('echo "===PROCCMD==="')
    script_parts.append(
        'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do '
        '  cmd=$(cat /proc/$p/cmdline 2>/dev/null | tr "\\0" " " | head -c 200); '
        '  echo "PID:$p:$cmd"; '
        'done'
    )

    for port, gpu_ids_str in verify_port_map.items():
        gpu_list = gpu_ids_str.split(",")
        gpu_label = "".join(gpu_list)
        name = f"{prefix}_{ip_tag}_gpu{gpu_label}"

        script_parts.append(f'echo "===EP:{name}:{port}:{gpu_ids_str}==="')
        if no_docker:
            # Check PID file and process liveness
            pid_file = f"{RUN_DIR}/{name}.pid"
            script_parts.append(f"""
if [ -f "{pid_file}" ]; then
    pid=$(cat "{pid_file}" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | grep -q vllm; then
        echo "STATUS:running"
    else
        echo "STATUS:exited"
    fi
else
    echo "STATUS:missing"
fi""")
        else:
            script_parts.append(f'status=$(docker inspect --format "{{{{.State.Status}}}}" {name} 2>/dev/null) && echo "STATUS:$status" || echo "STATUS:missing"')

        # Health check (only if container/process might be running)
        script_parts.append(f'http_code=$(curl -s -o /dev/null -w "%{{http_code}}" --connect-timeout 3 --max-time 5 http://localhost:{port}/health 2>/dev/null) && echo "HEALTH:$http_code" || echo "HEALTH:000"')

    full_script = "\n".join(script_parts)
    rc, out, _ = ssh_command(ip, full_script, timeout=max(30, len(verify_port_map) * 8))

    if rc != 0 or not out:
        return result

    # Parse GPU info
    gpu_mem = {}
    lines = out.strip().splitlines()
    in_gpu = False
    for line in lines:
        if line.strip() == "===GPU===":
            in_gpu = True
            continue
        if line.strip().startswith("==="):
            in_gpu = False
        if in_gpu and line.strip() and line.strip() != "NVIDIA_FAIL":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                try:
                    gpu_mem[parts[0]] = {"used": int(parts[1]), "total": int(parts[2])}
                except ValueError:
                    continue
    if gpu_mem:
        total_used = sum(g["used"] for g in gpu_mem.values())
        total_avail = sum(g["total"] for g in gpu_mem.values())
        result["gpu_info"] = f"GPU mem: {total_used:,} / {total_avail:,} MiB"

    # Parse per-endpoint results
    current_ep = None
    for line in lines:
        line = line.strip()
        if line.startswith("===EP:") and line.endswith("==="):
            # Parse "===EP:name:port:gpus==="
            ep_parts = line[6:-3].split(":")
            if len(ep_parts) >= 3:
                ep_name = ep_parts[0]
                try:
                    ep_port = int(ep_parts[1])
                except ValueError:
                    continue
                ep_gpus = ep_parts[2]
                gpu_list = ep_gpus.split(",")
                current_ep = {
                    "name": ep_name,
                    "port": ep_port,
                    "gpu": ep_gpus,
                    "container_status": "missing",
                    "health": "unknown",
                    "gpu_mem_used": sum(gpu_mem.get(g, {}).get("used", 0) for g in gpu_list),
                }
        elif current_ep and line.startswith("STATUS:"):
            current_ep["container_status"] = line[7:]
        elif current_ep and line.startswith("HEALTH:"):
            http_code = line[7:]
            if current_ep["container_status"] == "missing":
                result["missing"] += 1
                current_ep["health"] = "unknown"
                result["endpoints"].append(current_ep)
                current_ep = None
            elif current_ep["container_status"] in ("exited", "dead"):
                result["unhealthy"] += 1
                current_ep["health"] = f"✗ ({current_ep['container_status']})"
                result["endpoints"].append(current_ep)
                current_ep = None
            else:
                if http_code == "200":
                    current_ep["health"] = "✓"
                    result["healthy"] += 1
                else:
                    current_ep["health"] = f"✗ (HTTP {http_code})" if http_code != "000" else "✗"
                    result["unhealthy"] += 1
                result["endpoints"].append(current_ep)
                current_ep = None

    # Handle last endpoint if not flushed (SSH output truncated before HEALTH line)
    if current_ep:
        result["unhealthy"] += 1
        current_ep["health"] = "✗ (incomplete)"
        result["endpoints"].append(current_ep)

    # Parse GPU process info — only relevant when containers are missing/dead but GPUs busy
    gpu_procs = []  # list of {pid, gpu_index, cmdline}
    if result["missing"] > 0 or result["unhealthy"] > 0:
        # Build uuid→index map
        uuid_to_idx = {}
        pid_to_cmd = {}
        pid_to_uuids = {}
        in_section = None
        for line in lines:
            line = line.strip()
            if line == "===GPUPROCS===":
                in_section = "procs"; continue
            elif line == "===GPUMAP===":
                in_section = "map"; continue
            elif line == "===PROCCMD===":
                in_section = "cmd"; continue
            elif line.startswith("==="):
                in_section = None; continue

            if in_section == "map" and line and "FAIL" not in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    uuid_to_idx[parts[1]] = parts[0]
            elif in_section == "procs" and line and "FAIL" not in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    pid, uuid = parts[0], parts[1]
                    gpu_idx = uuid_to_idx.get(uuid, "?")
                    pid_to_uuids.setdefault(pid, []).append(gpu_idx)
            elif in_section == "cmd" and line.startswith("PID:"):
                # PID:12345:python -m vllm...
                cmd_parts = line.split(":", 2)
                if len(cmd_parts) >= 3:
                    pid_to_cmd[cmd_parts[1]] = cmd_parts[2].strip()

        # Build per-process info
        for pid, gpu_indices in pid_to_uuids.items():
            cmd = pid_to_cmd.get(pid, "unknown")
            # Classify the process
            if "vllm" in cmd:
                ptype = "vllm"
            elif "torchrun" in cmd or "train" in cmd:
                ptype = "training"
            elif "python" in cmd:
                ptype = "python"
            else:
                ptype = "other"
            gpu_procs.append({
                "pid": pid,
                "gpu_indices": sorted(gpu_indices),
                "cmdline": cmd[:120],
                "type": ptype,
            })
        result["other_gpu_procs"] = gpu_procs

    return result


def run_inspect_cache(ip_list: List[str], model_id: str, docker_image: str,
                      workers: int, ssh_timeout: int, no_docker: bool = False):
    """Inspect cache status on all nodes — show where model/TAR exist and disk space."""
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    tar_name = _tar_name_from_image(docker_image) if not no_docker else None
    paths = effective_cache_paths(len(ip_list))
    all_paths = [p for p, _ in CACHE_HIERARCHY]  # ALL paths including shared (show everything)

    safe_print(f"\n{'=' * 80}")
    safe_print(f"[INSPECT] Checking cache on {len(ip_list)} nodes")
    safe_print(f"[INSPECT] Model: {model_id} ({model_cache_name})")
    if tar_name:
        safe_print(f"[INSPECT] Docker: {docker_image} ({tar_name})")
    safe_print(f"[INSPECT] Cache paths ({len(CACHE_HIERARCHY)}):")
    for i, (p, is_shared) in enumerate(CACHE_HIERARCHY):
        if p.startswith("s3://"):
            label = "s3"
        elif i == 0:
            label = "RAM"
        elif is_shared:
            label = "shared"
        else:
            label = "local"
        safe_print(f"[INSPECT]   [{i}] {p} ({label})")
    safe_print(f"{'=' * 80}\n")

    def _check_node(ip: str) -> dict:
        """Single SSH call to check everything on one node."""
        parts = []

        # Check docker image
        if tar_name:
            parts.append(
                f"docker images -q {shlex.quote(docker_image)} 2>/dev/null | "
                f"head -1 | grep -q . && echo 'DOCKER_IMG:yes' || echo 'DOCKER_IMG:no'"
            )

        # Check model + TAR in each path + free space
        for p in all_paths:
            if p.startswith("s3://"):
                continue
            sp = shlex.quote(p)
            # Model check
            parts.append(
                f"[ -d {sp}/{model_cache_name}/snapshots ] && "
                f"ls -d {sp}/{model_cache_name}/snapshots/*/ >/dev/null 2>&1 && "
                f"echo 'MODEL:{p}:yes' || echo 'MODEL:{p}:no'"
            )
            # TAR check
            if tar_name:
                tar_at = f"{p}/containers/{tar_name}"
                parts.append(
                    f"[ -f {shlex.quote(tar_at)} ] && "
                    f"echo 'TAR:{p}:yes' || echo 'TAR:{p}:no'"
                )
            # Free space
            parts.append(
                f"_free=$(df -BG {sp} 2>/dev/null | tail -1 | awk '{{print $4}}' | tr -d 'G') && "
                f"echo 'SPACE:{p}:${{_free}}GB' || echo 'SPACE:{p}:?'"
            )

        # Also check legacy tar_cache
        if tar_name:
            legacy = f"{LOCAL_TAR_CACHE}/{tar_name}"
            parts.append(
                f"[ -f {shlex.quote(legacy)} ] && "
                f"echo 'TAR:{LOCAL_TAR_CACHE}:yes' || echo 'TAR:{LOCAL_TAR_CACHE}:no'"
            )

        rc, out, err = ssh_command(ip, "\n".join(parts), timeout=ssh_timeout + 10)
        result = {"ip": ip, "reachable": rc == 0, "docker_img": False,
                  "model": {}, "tar": {}, "space": {}}

        if rc == 0 and out:
            for line in out.strip().splitlines():
                line = line.strip()
                if line.startswith("DOCKER_IMG:"):
                    result["docker_img"] = "yes" in line
                elif line.startswith("MODEL:"):
                    _, path, status = line.split(":", 2)
                    result["model"][path] = status == "yes"
                elif line.startswith("TAR:"):
                    _, path, status = line.split(":", 2)
                    result["tar"][path] = status == "yes"
                elif line.startswith("SPACE:"):
                    _, path, space = line.split(":", 2)
                    result["space"][path] = space

        return result

    # Run in parallel
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(ip_list))) as executor:
        futures = {executor.submit(_check_node, ip): ip for ip in ip_list}
        for future in as_completed(futures):
            completed += 1
            try:
                r = future.result()
                results.append(r)
                status = "✓" if r["reachable"] else "✗ unreachable"
                safe_print(f"[INSPECT] [{completed}/{len(ip_list)}] {r['ip']} — {status}")
            except Exception as e:
                ip = futures[future]
                results.append({"ip": ip, "reachable": False})
                safe_print(f"[INSPECT] [{completed}/{len(ip_list)}] {ip} — ✗ {e}")

    # Summary
    reachable = [r for r in results if r.get("reachable")]
    unreachable = [r for r in results if not r.get("reachable")]

    safe_print(f"\n{'=' * 80}")
    safe_print(f"[INSPECT] SUMMARY — {len(reachable)} reachable, {len(unreachable)} unreachable")
    safe_print(f"{'=' * 80}")

    if not reachable:
        safe_print("[INSPECT] No reachable nodes!")
        return

    # Aggregate: how many nodes have model/TAR at each path
    clean_paths = [p for p in all_paths if not p.startswith("s3://")]

    safe_print(f"\n[INSPECT] MODEL ({model_cache_name}):")
    for p in clean_paths:
        count = sum(1 for r in reachable if r.get("model", {}).get(p, False))
        bar = "█" * count + "░" * (len(reachable) - count)
        safe_print(f"  {p:50s} {count:3d}/{len(reachable)} [{bar}]")

    if tar_name:
        safe_print(f"\n[INSPECT] DOCKER TAR ({tar_name}):")
        # Collect all TAR paths (hierarchy + legacy)
        tar_paths = list(clean_paths)
        if LOCAL_TAR_CACHE not in tar_paths:
            tar_paths.append(LOCAL_TAR_CACHE)
        for p in tar_paths:
            count = sum(1 for r in reachable if r.get("tar", {}).get(p, False))
            bar = "█" * count + "░" * (len(reachable) - count)
            safe_print(f"  {p:50s} {count:3d}/{len(reachable)} [{bar}]")

        # Docker image loaded
        img_count = sum(1 for r in reachable if r.get("docker_img", False))
        bar = "█" * img_count + "░" * (len(reachable) - img_count)
        safe_print(f"\n[INSPECT] DOCKER IMAGE LOADED ({docker_image}):")
        safe_print(f"  {'(in docker)':50s} {img_count:3d}/{len(reachable)} [{bar}]")

    safe_print(f"\n[INSPECT] DISK SPACE:")
    for p in clean_paths:
        spaces = [r.get("space", {}).get(p, "?") for r in reachable]
        # Parse numeric values
        nums = []
        for s in spaces:
            try:
                nums.append(int(s.replace("GB", "")))
            except (ValueError, AttributeError):
                pass
        if nums:
            safe_print(f"  {p:50s} min={min(nums)}GB  avg={sum(nums)//len(nums)}GB  max={max(nums)}GB")

    if unreachable:
        safe_print(f"\n[INSPECT] UNREACHABLE ({len(unreachable)}):")
        for r in unreachable:
            safe_print(f"  {r['ip']}")

    safe_print("")


def run_verify(ip_list: List[str], prefix: str, workers: int, ssh_timeout: int, tp_size: int = DEFAULT_TP_SIZE, no_docker: bool = False):
    """Run verification across all hosts in parallel."""
    mode_str = "processes" if no_docker else "containers"
    safe_print("=" * 80)
    safe_print(f"VERIFYING: {prefix}_* {mode_str}")
    safe_print(f"Hosts: {len(ip_list)}  |  Workers: {workers}")
    safe_print("=" * 80)

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ip = {executor.submit(verify_host, ip, prefix, ssh_timeout, tp_size, no_docker): ip for ip in ip_list}
        # Scale timeout: ~30s per wave, plus headroom
        num_waves = math.ceil(len(ip_list) / max(workers, 1))
        verify_timeout = max(120, num_waves * 45 + 30)
        verify_completed = 0
        verify_total = len(ip_list)
        try:
            for future in as_completed(future_to_ip, timeout=verify_timeout):
                ip = future_to_ip[future]
                verify_completed += 1
                try:
                    results.append(future.result())
                    safe_print(f"[VERIFY] [{verify_completed}/{verify_total}] {ip} — checked ({time.time() - start_time:.0f}s)")
                except Exception as e:
                    safe_print(f"[VERIFY] [{verify_completed}/{verify_total}] {ip} — ✗ Exception: {e}")
                    results.append({"ip": ip, "reachable": False, "endpoints": [], "healthy": 0, "unhealthy": 0, "missing": 0})
        except TimeoutError:
            timed_out_ips = [ip for f, ip in future_to_ip.items() if not f.done()]
            for ip in timed_out_ips:
                safe_print(f"[{ip}] ✗ Verify timed out")
                results.append({"ip": ip, "reachable": False, "endpoints": [], "healthy": 0, "unhealthy": 0, "missing": 0})

    elapsed = time.time() - start_time

    # Print per-host results
    total_healthy = 0
    total_unhealthy = 0
    total_missing = 0
    unreachable_hosts = []
    healthy_ips = []

    for r in sorted(results, key=lambda x: x["ip"]):
        if not r["reachable"]:
            safe_print(f"\n  ✗ {r['ip']}  — SSH unreachable")
            unreachable_hosts.append(r["ip"])
            continue

        total_healthy += r["healthy"]
        total_unhealthy += r["unhealthy"]
        total_missing += r["missing"]

        host_status = "✓" if r["unhealthy"] == 0 and r["missing"] == 0 else "✗"
        gpu_info = f"  | {r.get('gpu_info', '')}" if r.get('gpu_info') else ""
        safe_print(f"\n  {host_status} {r['ip']}  healthy={r['healthy']}  unhealthy={r['unhealthy']}  missing={r['missing']}{gpu_info}")

        if r["unhealthy"] == 0 and r["missing"] == 0 and r["healthy"] > 0:
            healthy_ips.append(r["ip"])

        for ep in r["endpoints"]:
            status_icon = "✓" if ep["health"] == "✓" else "✗" if "✗" in str(ep["health"]) else "?"
            mem_str = f"  gpu_mem={ep['gpu_mem_used']}MB" if ep.get("gpu_mem_used", 0) > 0 else ""
            safe_print(f"      {status_icon} {ep['name']}  port={ep['port']}  container={ep['container_status']}  health={ep['health']}{mem_str}")

        # Show other GPU processes only when our containers are missing/dead but GPUs are busy
        other_procs = r.get("other_gpu_procs", [])
        if other_procs and (r["missing"] > 0 or r["unhealthy"] > 0):
            # Collect GPU indices of missing/unhealthy endpoints
            problem_gpus = set()
            for ep in r["endpoints"]:
                if ep["health"] != "✓":
                    for g in ep["gpu"].split(","):
                        problem_gpus.add(g)
            # Only show processes using those problem GPUs
            relevant_procs = [p for p in other_procs
                              if any(g in problem_gpus for g in p["gpu_indices"])]
            if relevant_procs:
                safe_print(f"      ⚠ Processes on affected GPUs:")
                for proc in relevant_procs:
                    gpus = ",".join(proc["gpu_indices"])
                    cmd_short = proc["cmdline"][:80]
                    safe_print(f"        GPU {gpus} — {proc['type']} (pid {proc['pid']}): {cmd_short}")

    # Summary
    total_endpoints = len(build_gpu_port_map(tp_size)) * len(ip_list)
    safe_print("\n" + "=" * 80)
    safe_print("VERIFICATION SUMMARY")
    safe_print("=" * 80)
    safe_print(f"Time:              {elapsed:.1f}s")
    safe_print(f"Hosts reachable:   {len(results) - len(unreachable_hosts)}/{len(ip_list)}")
    safe_print(f"Endpoints healthy: {total_healthy}/{total_endpoints}")
    safe_print(f"Endpoints down:    {total_unhealthy}")
    safe_print(f"Containers missing:{total_missing}")

    if unreachable_hosts:
        safe_print(f"\nUnreachable hosts: {unreachable_hosts}")

    if healthy_ips:
        safe_print(f"\nFully healthy IPs ({len(healthy_ips)}):")
        ip_strs = ', '.join(f'"{ip}"' for ip in sorted(healthy_ips))
        safe_print(f"[{ip_strs}]")

    safe_print("=" * 80)


def teardown_host(ip: str, prefix: str, ssh_timeout: int = 15) -> dict:
    """Remove all containers matching prefix on a host."""
    result = {"ip": ip, "removed": 0, "error": None}

    # SSH check
    rc, out, err = ssh_command(ip, "echo ok", timeout=ssh_timeout)
    if rc != 0:
        result["error"] = f"SSH failed: {err}"
        return result

    # Count running containers first
    rc, out, _ = ssh_command(ip,
        f"docker ps -aq --filter name=^{prefix}_ 2>/dev/null | wc -l",
        timeout=10)
    count = int(out.strip()) if rc == 0 and out.strip().isdigit() else 0

    if count == 0:
        return result

    # Remove all matching containers (xargs handles empty input gracefully)
    rc, out, err = ssh_command(ip,
        f"docker ps -aq --filter name=^{prefix}_ | xargs -r docker rm -f 2>/dev/null",
        timeout=30)

    if rc == 0 and out.strip():
        result["removed"] = len(out.strip().splitlines())
    elif rc != 0:
        if "timeout" in err.lower():
            result["error"] = "Docker daemon hung"
        else:
            result["error"] = err.strip()[:100]

    return result


def teardown_host_direct(ip: str, prefix: str, ssh_timeout: int = 15) -> dict:
    """Remove all vLLM processes matching prefix on a host (no-docker mode)."""
    result = {"ip": ip, "removed": 0, "error": None}

    rc, _, err = ssh_command(ip, "echo ok", timeout=ssh_timeout)
    if rc != 0:
        result["error"] = f"SSH failed: {err}"
        return result

    ip_tag = ip.replace(".", "-")
    # Kill all matching processes and clean up in a single SSH call
    teardown_script = f"""
killed=0
for pf in "{RUN_DIR}/{prefix}_{ip_tag}_gpu"*.pid; do
    [ -f "$pf" ] || continue
    pid=$(cat "$pf" 2>/dev/null)
    if [ -n "$pid" ] && cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | grep -q vllm; then
        kill "$pid" 2>/dev/null; pkill -P "$pid" 2>/dev/null
        killed=$((killed+1))
    fi
    rm -f "$pf"
done
rm -f "{RUN_DIR}/{prefix}_{ip_tag}_gpu"*.log 2>/dev/null
rm -f "{RUN_DIR}/{prefix}_{ip_tag}_gpu"*.sh 2>/dev/null
echo $killed
"""
    rc, out, _ = ssh_command(ip, teardown_script.strip(), timeout=20)
    if rc == 0 and out.strip():
        try:
            result["removed"] = int(out.strip())
        except ValueError:
            pass

    return result


def run_teardown(ip_list: List[str], prefix: str, workers: int, ssh_timeout: int, no_docker: bool = False):
    """Teardown containers/processes on all hosts in parallel."""
    mode_str = "processes" if no_docker else "containers"
    safe_print("=" * 80)
    safe_print(f"TEARDOWN: Removing {prefix}_* {mode_str}")
    safe_print(f"Hosts: {len(ip_list)}  |  Workers: {workers}")
    safe_print("=" * 80)

    start = time.time()
    results = []

    def _teardown_fn(ip):
        if no_docker:
            return teardown_host_direct(ip, prefix, ssh_timeout)
        return teardown_host(ip, prefix, ssh_timeout)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_teardown_fn, ip): ip for ip in ip_list}
        tear_completed = 0
        tear_total = len(ip_list)
        try:
            for future in as_completed(futures, timeout=120):
                ip = futures[future]
                tear_completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    removed = result.get("removed", 0)
                    safe_print(f"[TEARDOWN] [{tear_completed}/{tear_total}] {ip} — {removed} removed ({time.time() - start:.0f}s)")
                except Exception as e:
                    safe_print(f"[TEARDOWN] [{tear_completed}/{tear_total}] {ip} — ✗ {e} ({time.time() - start:.0f}s)")
                    results.append({"ip": ip, "removed": 0, "error": str(e)})
        except TimeoutError:
            timed_out_ips = [ip for f, ip in futures.items() if not f.done()]
            for ip in timed_out_ips:
                safe_print(f"  [{ip}] ✗ Teardown timed out")
                results.append({"ip": ip, "removed": 0, "error": "Timed out"})

    elapsed = time.time() - start

    total_removed = sum(r["removed"] for r in results)
    errors = [r for r in results if r["error"]]

    safe_print("")
    for r in sorted(results, key=lambda x: x["ip"]):
        if r["error"]:
            safe_print(f"  ✗ {r['ip']:16s}  error: {r['error']}")
        elif r["removed"] > 0:
            safe_print(f"  ✓ {r['ip']:16s}  removed {r['removed']} {mode_str}")
        else:
            safe_print(f"  - {r['ip']:16s}  no {mode_str} found")

    safe_print("")
    safe_print("=" * 80)
    safe_print(f"TEARDOWN COMPLETE  |  Time: {elapsed:.1f}s  |  "
               f"Removed: {total_removed} {mode_str}  |  Errors: {len(errors)}")
    safe_print("=" * 80)


def _find_scout_node(ip_list: List[str], ssh_timeout: int = 15) -> str:
    """Find the first reachable node from the IP list to use as scout."""
    for ip in ip_list[:10]:  # Try first 10 nodes max
        rc, _, _ = ssh_command(ip, "echo ok", timeout=ssh_timeout, use_multiplexing=False)
        if rc == 0:
            return ip
    return ""


def preflight_ensure_shared_assets(scout_ip: str, docker_image: str, model_id: str,
                                    s3_bucket: str = "", skip_docker: bool = False,
                                    temp_folder: str = None) -> bool:
    """Pre-flight: verify Docker TAR and model exist on S3 before parallel deployment.

    Returns (success, asset_info) where asset_info contains:
      - s3_tar_path: S3 path to Docker TAR (or None)
      - s3_model_path: S3 path to model directory (or None)
      - model_cache_name: model directory name (models--org--name format)
    """
    logger = HostLogger("preflight", LOG_DIR)

    # Determine temp folder on scout: user-provided → /tmp (if writable+space) → /dev/shm
    # Always append a unique subdirectory to avoid collisions between concurrent runs
    _preflight_uid = f"_preflight_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    if temp_folder:
        _preflight_tmp = f"{temp_folder.rstrip('/')}/{_preflight_uid}"
        logger.log(f"Using user-specified temp folder: {_preflight_tmp}")
    else:
        rc_tmp, tmp_out, _ = ssh_command(scout_ip,
            "df -BG /tmp 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G'",
            timeout=10)
        tmp_free = int(tmp_out.strip()) if rc_tmp == 0 and tmp_out.strip().isdigit() else 0
        if tmp_free >= 5:
            _preflight_tmp = f"/tmp/{_preflight_uid}"
        else:
            _preflight_tmp = f"/dev/shm/{_preflight_uid}"
            logger.log(f"/tmp has only {tmp_free}GB free — using /dev/shm for preflight downloads", "WARN")
            safe_print(f"[PRE-FLIGHT] ⚠ /tmp has {tmp_free}GB free — using /dev/shm (RAM) for temp downloads")
    # Ensure temp folder exists on scout
    rc_mk, _, mk_err = ssh_command(scout_ip, f"mkdir -p {shlex.quote(_preflight_tmp)}", timeout=10)
    if rc_mk != 0:
        logger.log(f"Cannot create {_preflight_tmp}: {(mk_err or '').strip()} — trying fallbacks", "WARN")
        safe_print(f"[PRE-FLIGHT] ⚠ Cannot create {_preflight_tmp} — trying fallbacks...")
        _preflight_tmp = None
        # Try persistent cache paths from config (NVMe etc.)
        for p in effective_cache_paths(1)[1:]:  # skip RAM (index 0)
            fb = f"{p}/{_preflight_uid}"
            rc_fb, _, _ = ssh_command(scout_ip, f"mkdir -p {shlex.quote(fb)}", timeout=10)
            if rc_fb == 0:
                _preflight_tmp = fb
                logger.log(f"Fallback temp folder: {_preflight_tmp}")
                safe_print(f"[PRE-FLIGHT]   Using fallback: {_preflight_tmp}")
                break
        if not _preflight_tmp:
            for base in ["/tmp", "/dev/shm"]:
                fb = f"{base}/{_preflight_uid}"
                rc_fb, _, _ = ssh_command(scout_ip, f"mkdir -p {shlex.quote(fb)}", timeout=10)
                if rc_fb == 0:
                    _preflight_tmp = fb
                    safe_print(f"[PRE-FLIGHT]   Using fallback: {_preflight_tmp}")
                    break
        if not _preflight_tmp:
            safe_print(f"[PRE-FLIGHT] ✗ No writable temp folder found on scout node")
            return False, {}
    logger.log("=" * 60)
    logger.log("PRE-FLIGHT: Verifying assets")
    logger.log(f"Scout node: {scout_ip}")
    logger.log(f"S3 bucket: {s3_bucket or '(none)'}")
    logger.log("=" * 60)

    success = True
    s3_tar_path = None
    s3_model_path = None
    model_size_bytes = 0
    tar_size_bytes = 0
    model_cache_name = f"models--{model_id.replace('/', '--')}"

    # ── Docker image TAR on S3 ──
    if not skip_docker:
        tar_name = _tar_name_from_image(docker_image)
        s3_tar = f"{s3_bucket.rstrip('/')}/containers/{tar_name}"
        safe_print(f"\n[PRE-FLIGHT] Checking Docker TAR on S3: {s3_tar}")

        rc, out, _ = ssh_command(scout_ip,
            f"aws s3 ls {shlex.quote(s3_tar)} 2>/dev/null | grep -q . && echo 'S3_FOUND' || echo 'S3_MISS'",
            timeout=30)
        if rc == 0 and "S3_FOUND" in (out or ""):
            logger.log(f"Docker TAR found on S3: {s3_tar}")
            safe_print(f"[PRE-FLIGHT] ✓ Docker TAR found on S3")
            s3_tar_path = s3_tar
            # Get TAR size for space-aware download
            rc_sz, sz_out, _ = ssh_command(scout_ip,
                f"aws s3 ls {shlex.quote(s3_tar)} 2>/dev/null | awk '{{print $3}}'", timeout=15)
            if rc_sz == 0 and sz_out.strip().isdigit():
                tar_size_bytes = int(sz_out.strip())
                safe_print(f"[PRE-FLIGHT]   TAR size: {tar_size_bytes / (1024**3):.1f} GB")
        else:
            logger.log(f"Docker TAR not found on S3 — auto-downloading from registry", "WARN")
            safe_print(f"[PRE-FLIGHT] ⚠ Docker TAR not on S3 — pulling from registry and uploading...")
            # Pull image, save as TAR, upload to S3 (specific path, never overwrites bucket)
            tmp_tar = f"{_preflight_tmp}/{tar_name}"
            # Check space before docker save (TAR can be 10-20GB)
            rc_df, df_out, _ = ssh_command(scout_ip,
                f"df -BG {shlex.quote(_preflight_tmp)} 2>/dev/null | tail -1 | awk '{{print $4}}' | tr -d 'G'",
                timeout=10)
            if rc_df == 0 and df_out.strip().isdigit() and int(df_out.strip()) < 25:
                logger.log(f"{_preflight_tmp} only {df_out.strip()}GB free — docker save may fail", "WARN")
                safe_print(f"[PRE-FLIGHT] ⚠ {_preflight_tmp} only {df_out.strip()}GB free — large images may not fit")
            pull_cmd = (
                f"docker pull {shlex.quote(docker_image)} 2>&1 && "
                f"docker save {shlex.quote(docker_image)} -o {shlex.quote(tmp_tar)} 2>&1 && "
                f"echo 'SAVE_OK' || echo 'SAVE_FAIL'"
            )
            rc, out, _ = ssh_command_stream(scout_ip, pull_cmd, timeout=1200,
                                            prefix="[PRE-FLIGHT]  ", heartbeat=30)
            if rc == 0 and "SAVE_OK" in (out or ""):
                logger.log("Docker image pulled and saved to TAR")
                safe_print(f"[PRE-FLIGHT]   Docker image pulled — uploading TAR to S3...")
                upload_cmd = (
                    f"{_s3_parallel_preamble()}"
                    f"aws s3 cp {shlex.quote(tmp_tar)} {shlex.quote(s3_tar)} "
                    f"--no-progress 2>&1 && echo 'UPLOAD_OK' || echo 'UPLOAD_FAIL'; "
                    f"rm -f {shlex.quote(tmp_tar)}"
                )
                rc, out, _ = ssh_command_stream(scout_ip, upload_cmd, timeout=1200,
                                                prefix="[PRE-FLIGHT]  ", heartbeat=30)
                if rc == 0 and "UPLOAD_OK" in (out or ""):
                    logger.log(f"Docker TAR uploaded to S3: {s3_tar}")
                    safe_print(f"[PRE-FLIGHT] ✓ Docker TAR uploaded to S3")
                    s3_tar_path = s3_tar
                    # Get TAR size
                    rc_sz, sz_out, _ = ssh_command(scout_ip,
                        f"aws s3 ls {shlex.quote(s3_tar)} 2>/dev/null | awk '{{print $3}}'", timeout=15)
                    if rc_sz == 0 and sz_out.strip().isdigit():
                        tar_size_bytes = int(sz_out.strip())
                else:
                    logger.log(f"S3 upload failed: {(out or '').strip()[-200:]}", "ERROR")
                    safe_print(f"[PRE-FLIGHT] ✗ Failed to upload Docker TAR to S3")
                    success = False
            else:
                logger.log(f"Docker pull/save failed: {(out or '').strip()[-200:]}", "ERROR")
                safe_print(f"[PRE-FLIGHT] ✗ Failed to pull Docker image from registry")
                success = False
            # Cleanup tmp TAR on failure too
            ssh_command(scout_ip, f"rm -f {shlex.quote(tmp_tar)}", timeout=10)
    else:
        logger.log("Skipping Docker TAR check (no-docker mode)")

    # ── Model weights on S3 ──
    s3_model = f"{s3_bucket.rstrip('/')}/models/{model_cache_name}/"
    safe_print(f"\n[PRE-FLIGHT] Checking model on S3: {s3_model}")

    rc, out, _ = ssh_command(scout_ip,
        f"aws s3 ls {shlex.quote(s3_model)}.upload_complete 2>/dev/null | grep -q . && echo 'S3_FOUND' || echo 'S3_MISS'",
        timeout=30)
    if rc == 0 and "S3_FOUND" in (out or ""):
        logger.log(f"Model found on S3: {s3_model}")
        safe_print(f"[PRE-FLIGHT] ✓ Model found on S3")
        s3_model_path = s3_model
        # Get model size for space-aware download
        rc_sz, sz_out, _ = ssh_command(scout_ip,
            f"aws s3 ls --summarize --recursive {shlex.quote(s3_model)} 2>/dev/null | grep 'Total Size' | awk '{{print $3}}'",
            timeout=60)
        if rc_sz == 0 and sz_out.strip().isdigit():
            model_size_bytes = int(sz_out.strip())
            safe_print(f"[PRE-FLIGHT]   Model size: {model_size_bytes / (1024**3):.1f} GB")
    else:
        # Check if files exist but marker is missing (partial or manual upload)
        rc_partial, partial_out, _ = ssh_command(scout_ip,
            f"aws s3 ls {shlex.quote(s3_model)} 2>/dev/null | head -1 | grep -q . && echo 'HAS_FILES' || echo 'EMPTY'",
            timeout=15)
        if rc_partial == 0 and "HAS_FILES" in (partial_out or ""):
            logger.log("Model files exist on S3 but no completion marker — may be partial upload", "WARN")
            safe_print(f"[PRE-FLIGHT] ⚠ Model files on S3 but no .upload_complete marker — re-syncing")
            safe_print(f"[PRE-FLIGHT]   (If manually uploaded, run: echo done | aws s3 cp - {s3_model}.upload_complete)")

        logger.log(f"Model not found/incomplete on S3 — auto-downloading from HuggingFace", "WARN")
        safe_print(f"[PRE-FLIGHT] ⚠ Model not on S3 — downloading from HuggingFace and uploading...")
        # Check if we can download via CLI or Python
        rc_hf, _, _ = ssh_command(scout_ip, "command -v huggingface-cli >/dev/null 2>&1", timeout=10)
        rc_py, _, _ = ssh_command(scout_ip,
            "python3 -c 'from huggingface_hub import snapshot_download' 2>/dev/null", timeout=10)
        if rc_hf != 0 and rc_py != 0:
            logger.log("Neither huggingface-cli nor huggingface_hub Python package found on scout node", "ERROR")
            safe_print(f"[PRE-FLIGHT] ✗ huggingface-hub not available on scout node ({scout_ip})")
            safe_print(f"[PRE-FLIGHT]   Fix: pip install huggingface-hub --break-system-packages")
            success = False
        else:
            # Download to temp folder on scout node, then sync to S3 (specific path only, no --delete)
            # Use model-specific temp dir to avoid conflicts with concurrent deploys
            tmp_cache = f"{_preflight_tmp}/_preflight_dl_{model_cache_name}"
            tmp_model_dir = f"{tmp_cache}/{model_cache_name}"
            # Read HF token: first try {s3_bucket}/a, then fall back to HF_TOKEN env var
            s3_token_path = f"{s3_bucket.rstrip('/')}/a"
            rc_tok, tok_out, _ = ssh_command(scout_ip,
                f"aws s3 cp {shlex.quote(s3_token_path)} - 2>/dev/null | head -1",
                timeout=15)
            hf_token = ""
            if rc_tok == 0 and tok_out and tok_out.strip():
                hf_token = tok_out.strip()
                logger.log("HF token loaded from S3 bucket")
            else:
                hf_token = os.environ.get("HF_TOKEN", "")
                if hf_token:
                    logger.log("HF token loaded from HF_TOKEN env var")
            hf_env_prefix = f"HF_TOKEN={shlex.quote(hf_token)} " if hf_token else ""
            # Prefer CLI, fall back to Python
            # Only download safetensors + config files — skip .bin/.pt to avoid doubling download size
            if rc_hf == 0:
                dl_inner = (
                    f"{hf_env_prefix}"
                    f"huggingface-cli download {shlex.quote(model_id)} "
                    f"--include '*.safetensors' --include '*.json' --include '*.txt' "
                    f"--include '*.model' --include '*.tiktoken' --include '*.py' "
                    f"--cache-dir {shlex.quote(tmp_cache)} 2>&1"
                )
            else:
                # Build a Python script string — avoids nested quote hell
                py_token_line = f"token='{hf_token}', " if hf_token else ""
                py_script = (
                    f"from huggingface_hub import snapshot_download; "
                    f"snapshot_download('{model_id}', cache_dir='{tmp_cache}', {py_token_line}"
                    f"allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model', '*.tiktoken', '*.py'])"
                )
                # shlex.quote wraps the entire python -c arg safely for bash
                dl_inner = (
                    f"{hf_env_prefix}"
                    f"python3 -c {shlex.quote(py_script)} 2>&1"
                )
                logger.log("Using Python huggingface_hub (huggingface-cli not in PATH)")
            dl_cmd = (
                f"mkdir -p {shlex.quote(tmp_cache)} && "
                f"{dl_inner}; "
                f"if [ -d {shlex.quote(tmp_model_dir)}/snapshots ]; then "
                f"BLOB_SIZE=$(du -sb {shlex.quote(tmp_model_dir)}/blobs/ 2>/dev/null | cut -f1); "
                f"BLOB_COUNT=$(find {shlex.quote(tmp_model_dir)}/blobs/ -type f 2>/dev/null | wc -l); "
                f"echo \"DL_OK BLOBS=$BLOB_COUNT SIZE=$BLOB_SIZE\"; "
                f"else echo 'DL_FAIL'; fi"
            )
            safe_print(f"[PRE-FLIGHT]   Downloading {model_id} from HuggingFace (this may take a while)...")
            _dl_start = time.time()
            rc, out, _ = ssh_command_stream(scout_ip, dl_cmd, timeout=3600,
                                            prefix="[PRE-FLIGHT]  ", heartbeat=30)
            _dl_elapsed = time.time() - _dl_start
            if rc == 0 and "DL_OK" in (out or ""):
                # Parse blob stats
                _blob_match = re.search(r"BLOBS=(\d+)\s+SIZE=(\d+)", out or "")
                _blob_count = int(_blob_match.group(1)) if _blob_match else 0
                _blob_size = int(_blob_match.group(2)) if _blob_match else 0
                _blob_size_gb = _blob_size / (1024**3)
                _dl_speed = _blob_size_gb / _dl_elapsed * 1024 if _dl_elapsed > 0 else 0  # MB/s
                logger.log(f"Model downloaded from HuggingFace ({_blob_count} blobs, {_blob_size_gb:.1f} GB, {_dl_elapsed:.0f}s, {_dl_speed:.0f} MB/s)")
                if _blob_count == 0 or _blob_size < 1_000_000:  # < 1MB means no real weights
                    logger.log(f"Download appears incomplete — no weight files found ({_blob_count} blobs, {_blob_size} bytes)", "ERROR")
                    safe_print(f"[PRE-FLIGHT] ✗ Download incomplete — only metadata downloaded, no weight files")
                    safe_print(f"[PRE-FLIGHT]   Possible causes:")
                    safe_print(f"[PRE-FLIGHT]   1. Gated model — put HF token in {{s3-bucket}}/a or export HF_TOKEN")
                    safe_print(f"[PRE-FLIGHT]   2. Model repo has only .bin weights (no .safetensors) — download filters exclude .bin")
                    success = False
                else:
                    safe_print(f"[PRE-FLIGHT]   Model downloaded ({_blob_count} files, {_blob_size_gb:.1f} GB in {_dl_elapsed:.0f}s, {_dl_speed:.0f} MB/s) — uploading to S3...")
                    # aws s3 sync: only uploads new/changed files, never deletes anything on S3.
                    # --exclude "blobs/*": HF cache has blobs/ (real files) + snapshots/ (symlinks → blobs).
                    # aws s3 sync follows symlinks, so snapshots/ files are uploaded as real data.
                    # Excluding blobs/ avoids uploading the same data twice (would double S3 and RAM usage).
                    # Write .upload_complete marker ONLY after sync finishes — detects partial uploads.
                    marker_path = f"{s3_model}.upload_complete"
                    _upload_total_mb = int(_blob_size / (1024**2)) or 1  # avoid div-by-zero
                    upload_cmd = (
                        f"{_s3_parallel_preamble()}"
                        # Background progress monitor in its own process group (setsid).
                        # This ensures kill -9 on the group kills the monitor AND any child
                        # (aws s3 ls) atomically — no orphans left holding SSH stdout open.
                        f"setsid sh -c '"
                        f"_ul_total_mb={_upload_total_mb}; while true; do sleep 30; "
                        f"  _done_bytes=$(aws s3 ls --summarize --recursive {shlex.quote(s3_model)} 2>/dev/null "
                        f"    | awk '\"'\"'/Total Size/{{print $3}}'\"'\"'); "
                        f"  _done_mb=$(( ${{_done_bytes:-0}} / 1048576 )); "
                        f"  _pct=$(( _done_mb * 100 / _ul_total_mb )); "
                        f'  echo "UPLOAD_PROGRESS: ${{_done_mb}}MB / ${{_ul_total_mb}}MB (${{_pct}}%)"; '
                        f"done' & _monitor_pid=$!; "
                        # Actual sync
                        f"aws s3 sync {shlex.quote(tmp_model_dir)}/ {shlex.quote(s3_model)} "
                        f"--exclude 'blobs/*' --exclude '*.bin' --exclude '*.pt' "
                        f"--no-progress 2>&1; SYNC_RC=$?; "
                        # Kill entire process group — monitor + all children die atomically
                        f"kill -9 -- -$_monitor_pid 2>/dev/null; wait $_monitor_pid 2>/dev/null; "
                        f"if [ $SYNC_RC -eq 0 ]; then "
                        f"echo 'done' | aws s3 cp - {shlex.quote(marker_path)} --no-progress 2>&1 && "
                        f"echo 'UPLOAD_OK' && rm -rf {shlex.quote(tmp_cache)}; "
                        f"else echo \"UPLOAD_FAIL rc=$SYNC_RC\"; fi"
                    )
                    _upload_start = time.time()
                    rc, out, _ = ssh_command_stream(scout_ip, upload_cmd, timeout=1800,
                                                    prefix="[PRE-FLIGHT]  ", heartbeat=30)
                    _upload_elapsed = time.time() - _upload_start
                    _upload_speed = _blob_size_gb / _upload_elapsed * 1024 if _upload_elapsed > 0 else 0  # MB/s
                    if rc == 0 and "UPLOAD_OK" in (out or ""):
                        logger.log(f"Model uploaded to S3: {s3_model} ({_upload_elapsed:.0f}s, {_upload_speed:.0f} MB/s)")
                        safe_print(f"[PRE-FLIGHT] ✓ Model uploaded to S3 ({_blob_size_gb:.1f} GB in {_upload_elapsed:.0f}s, {_upload_speed:.0f} MB/s)")
                        s3_model_path = s3_model
                        model_size_bytes = _blob_size
                    else:
                        logger.log(f"S3 upload failed after {_upload_elapsed:.0f}s: {(out or '').strip()[-500:]}", "ERROR")
                        safe_print(f"[PRE-FLIGHT] ✗ Failed to upload model to S3 (after {_upload_elapsed:.0f}s)")
                        safe_print(f"[PRE-FLIGHT]   Error: {(out or '').strip()[-200:]}")
                        success = False
            else:
                logger.log(f"HuggingFace download failed after {_dl_elapsed:.0f}s: {(out or '').strip()[-200:]}", "ERROR")
                safe_print(f"[PRE-FLIGHT] ✗ Failed to download model from HuggingFace (after {_dl_elapsed:.0f}s)")
                safe_print(f"[PRE-FLIGHT]   Check: is {model_id} a valid/accessible model? For gated models, put token in {{s3-bucket}}/a")
                success = False
            # Cleanup tmpfs on failure too
            ssh_command(scout_ip, f"rm -rf {shlex.quote(tmp_cache)}", timeout=30)

    logger.log("=" * 60)
    logger.log(f"PRE-FLIGHT complete: {'READY' if success else 'MISSING ASSETS'}")
    logger.log("=" * 60)

    # Cleanup unique temp folder on scout (with safety guards)
    if (_preflight_uid and _preflight_uid in _preflight_tmp
            and "_preflight_" in _preflight_tmp
            and len(_preflight_tmp) > 10):
        ssh_command(scout_ip, f"rm -rf {shlex.quote(_preflight_tmp)}", timeout=30)
    else:
        logger.log(f"Skipping temp cleanup — path didn't pass safety check: {_preflight_tmp}", "WARN")

    logger.flush()

    asset_info = {
        "s3_tar_path": s3_tar_path,
        "s3_model_path": s3_model_path,
        "model_cache_name": model_cache_name,
        "model_size_bytes": model_size_bytes,
        "tar_size_bytes": tar_size_bytes,
    }
    return success, asset_info


def preflight_prewarm_nodes(ip_list: List[str], asset_info: dict, docker_image: str,
                            workers: int, ssh_timeout: int, no_docker: bool = False):
    """Pre-warm ALL nodes: download Docker TAR + model weights from S3 in parallel.
    Runs after preflight_ensure_shared_assets() has confirmed assets on S3.

    Per node (single SSH call):
      1. aws s3 cp Docker TAR → NVMe, docker load (if not already present)
      2. aws s3 sync model weights → local cache with parallel downloads
    """
    s3_tar = asset_info.get("s3_tar_path")
    s3_model = asset_info.get("s3_model_path")
    model_cache_name = asset_info.get("model_cache_name")
    model_size_bytes = asset_info.get("model_size_bytes", 0)
    tar_size_bytes = asset_info.get("tar_size_bytes", 0)
    # Convert to GB with 20% buffer for overhead, minimum 5GB
    model_need_gb = max(5, int(model_size_bytes / (1024**3) * 1.2) + 1) if model_size_bytes else 50
    tar_need_gb = max(2, int(tar_size_bytes / (1024**3) * 1.2) + 1) if tar_size_bytes else 25

    # Skip if nothing to pre-warm
    want_tar = s3_tar and not no_docker
    want_model = s3_model and model_cache_name
    if not want_tar and not want_model:
        return

    task_parts = []
    if want_tar:
        task_parts.append("Docker TAR")
    if want_model:
        task_parts.append("model weights")

    # Per-node timeout: 120s for TAR-only, 900s if model needs download
    per_node_timeout = 900 if want_model else 120
    num_waves = math.ceil(len(ip_list) / max(workers, 1))
    hard_ceiling = max(1200, num_waves * per_node_timeout + 120)

    safe_print(f"\n[PRE-WARM] Staging {' + '.join(task_parts)} on {len(ip_list)} nodes (check cache → S3 fallback)...")
    if want_model:
        safe_print(f"[PRE-WARM] {num_waves} waves × {per_node_timeout}s/node — hard ceiling {hard_ceiling}s ({hard_ceiling // 60}m)")

    tar_name = _tar_name_from_image(docker_image) if want_tar else None
    logger = HostLogger("prewarm", LOG_DIR)

    def _prewarm_node(ip: str) -> tuple:
        """Download assets from S3 with space-aware path selection.
        For each asset (Docker TAR, model weights):
          1. Walk cache hierarchy — first hit wins (existing file)
          2. If not found, walk persistent paths checking disk space
          3. Download to first path with enough space
          4. Fall back to RAM (path[0]) if everything is full
          5. Always promote to RAM for serving
        """
        _paths = effective_cache_paths(NUM_DEPLOY_NODES)
        ram_path = _paths[0] if _paths else "/dev/shm/model-cache"
        # Persistent paths = everything except RAM (index 1+)
        persistent_paths = _paths[1:] if len(_paths) > 1 else []
        # First local persistent path (NVMe) — always persist here for next run
        nvme_path = persistent_paths[0] if persistent_paths else None

        script_parts = []

        # ── Check aws CLI availability ──
        script_parts.append("""
# Fail fast if aws CLI is missing
if ! command -v aws >/dev/null 2>&1; then
    echo "MODEL:FAIL"
    echo "MODEL_LOG:aws CLI not installed on this node"
    echo "DOCKER:FAIL"
    echo "DOCKER_LOG:aws CLI not installed on this node"
    exit 1
fi
""")

        # ── S3 parallel download config (merge with existing to preserve region/creds) ──
        script_parts.append(f"""
# Configure parallel S3 downloads (merge with existing config)
_S3P=/tmp/.s3parallel.cfg
touch $_S3P 2>/dev/null || _S3P=/dev/shm/.s3parallel.cfg
export AWS_CONFIG_FILE=$_S3P
{{ cat ~/.aws/config 2>/dev/null || true; printf '\\n[default]\\ns3 =\\n  max_concurrent_requests = {S3_MAX_CONCURRENT_REQUESTS}\\n  multipart_chunksize = {S3_MULTIPART_CHUNKSIZE}\\n'; }} > $AWS_CONFIG_FILE
""")

        # ── Ensure all cache dirs exist ──
        _mkdir_cmds = " ".join(f"mkdir -p {shlex.quote(p)} 2>/dev/null;" for p in _paths)
        script_parts.append(f"""
# Ensure cache directories exist
{_mkdir_cmds}
""")

        # ── Docker TAR: walk hierarchy → space-aware download → docker load ──
        if want_tar:
            safe_s3_tar = shlex.quote(s3_tar)
            safe_docker_img = shlex.quote(docker_image)

            # Build bash: check each path for existing TAR
            tar_check_parts = []
            for i, p in enumerate(_paths):
                sp = shlex.quote(p)
                tar_at = f"{p}/containers/{tar_name}"
                safe_tar_at = shlex.quote(tar_at)
                cond = "if" if i == 0 else "elif"
                tar_check_parts.append(
                    f'{cond} [ -f {safe_tar_at} ]; then\n'
                    f'    _tar_at={safe_tar_at}\n'
                    f'    echo "DOCKER_LOG:found TAR at {p}"'
                )
            # Also check legacy tar_cache path (may differ from hierarchy paths)
            legacy_tar = f"{LOCAL_TAR_CACHE}/{tar_name}"
            safe_legacy_tar = shlex.quote(legacy_tar)
            tar_check_parts.append(
                f'elif [ -f {safe_legacy_tar} ]; then\n'
                f'    _tar_at={safe_legacy_tar}\n'
                f'    echo "DOCKER_LOG:found TAR at {LOCAL_TAR_CACHE} (legacy)"'
            )
            tar_check_script = "\n".join(tar_check_parts)

            # Build bash: walk persistent paths checking space for download
            tar_space_parts = []
            for p in persistent_paths:
                sp = shlex.quote(p)
                tar_dir = f"{p}/containers"
                safe_tar_dir = shlex.quote(tar_dir)
                tar_space_parts.append(
                    f'    _free=$(df -BG {sp} 2>/dev/null | tail -1 | awk \'{{print $4}}\' | tr -d \'G\')\n'
                    f'    if [ "$_free" -ge {tar_need_gb} ] 2>/dev/null; then\n'
                    f'        if mkdir -p {safe_tar_dir} 2>/dev/null && touch {safe_tar_dir}/.write_test 2>/dev/null; then\n'
                    f'            rm -f {safe_tar_dir}/.write_test\n'
                    f'            _tar_dl={safe_tar_dir}/{shlex.quote(tar_name)}\n'
                    f'            echo "DOCKER_LOG:downloading to {p} (${{_free}}GB free)"\n'
                    f'            break\n'
                    f'        else\n'
                    f'            echo "DOCKER_LOG:{p} not writable (permission denied)"\n'
                    f'        fi\n'
                    f'    else\n'
                    f'        echo "DOCKER_LOG:{p} only ${{_free}}GB free (need {tar_need_gb}GB)"\n'
                    f'    fi'
                )
            tar_space_script = "\n".join(tar_space_parts)

            # RAM fallback for TAR
            ram_tar_dir = f"{ram_path}/containers"
            safe_ram_tar = shlex.quote(f"{ram_tar_dir}/{tar_name}")
            safe_ram_tar_dir = shlex.quote(ram_tar_dir)

            # NVMe persist target for TAR
            nvme_tar_persist = ""
            safe_nvme_tar = ""
            safe_nvme_tar_dir = ""
            if nvme_path:
                nvme_tar_dir = f"{nvme_path}/containers"
                nvme_tar_persist = f"{nvme_tar_dir}/{tar_name}"
                safe_nvme_tar = shlex.quote(nvme_tar_persist)
                safe_nvme_tar_dir = shlex.quote(nvme_tar_dir)

            script_parts.append(f"""
# Docker TAR: check hierarchy → space-aware download → persist to NVMe → docker load
_docker_start=$(date +%s)
if docker image inspect {safe_docker_img} >/dev/null 2>&1; then
    echo "DOCKER:CACHED"
else
    # Check existing TARs in cache hierarchy
    _tar_at=""
{tar_check_script}
    else
        _tar_at=""
    fi

    if [ -z "$_tar_at" ]; then
        # No existing TAR — find download target with enough space
        _tar_dl=""
        for _x in 1; do
{tar_space_script}
        done
        # Fall back to RAM if no persistent path has space
        if [ -z "$_tar_dl" ]; then
            _tar_dl={safe_ram_tar}
            mkdir -p {safe_ram_tar_dir}
            echo "DOCKER_LOG:all persistent paths full — downloading to RAM"
        fi
        # Download from S3
        aws s3 cp {safe_s3_tar} "$_tar_dl" --no-progress 2>&1
        if [ $? -eq 0 ]; then
            _tar_at="$_tar_dl"
            _tar_size=$(du -sm "$_tar_at" 2>/dev/null | cut -f1)
            _docker_dl_done=$(date +%s)
            echo "DOCKER_LOG:downloaded ${{_tar_size}}MB in $((_docker_dl_done - _docker_start))s"
        else
            echo "DOCKER:FAIL"
            _tar_at=""
        fi
    fi

    # Persist TAR to NVMe for next run (if not already there)
    {f'''if [ -n "$_tar_at" ] && [ "$_tar_at" != {safe_nvme_tar} ] && [ ! -f {safe_nvme_tar} ]; then
        mkdir -p {safe_nvme_tar_dir}
        cp -a "$_tar_at" {safe_nvme_tar} 2>/dev/null && echo "DOCKER_LOG:persisted TAR to {nvme_path}" || {{ rm -f {safe_nvme_tar} 2>/dev/null; true; }}
    fi''' if nvme_tar_persist else '# No NVMe path configured for TAR persist'}

    # Load TAR into Docker
    if [ -n "$_tar_at" ]; then
        echo "DOCKER_LOG:loading image..."
        docker load -i "$_tar_at" >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            _docker_end=$(date +%s)
            echo "DOCKER_LOG:loaded in $((_docker_end - _docker_start))s"
            echo "DOCKER:STAGED"
        else
            echo "DOCKER:LOAD_FAIL"
        fi
    fi
fi""")

        # ── Model weights: walk hierarchy → space-aware download → persist NVMe → promote RAM ──
        if want_model:
            safe_s3_model = shlex.quote(s3_model)
            safe_ram = shlex.quote(ram_path)
            safe_nvme = shlex.quote(nvme_path) if nvme_path else None

            # Build bash: check each path for existing model
            model_check_parts = []
            for i, p in enumerate(_paths):
                sp = shlex.quote(p)
                cond = "if" if i == 0 else "elif"
                model_check_parts.append(
                    f'{cond} [ -d {sp}/{model_cache_name}/snapshots ] && '
                    f'ls -d {sp}/{model_cache_name}/snapshots/*/ >/dev/null 2>&1; then\n'
                    f'    _model_size=$(du -sm {sp}/{model_cache_name} 2>/dev/null | cut -f1)\n'
                    f'    echo "MODEL_LOG:found at {p} (${{_model_size}}MB)"\n'
                    f'    _found_at={sp}'
                )
            model_check_script = "\n".join(model_check_parts)

            # Build bash: walk persistent paths checking space AND write permission for download
            model_space_parts = []
            for p in persistent_paths:
                sp = shlex.quote(p)
                model_space_parts.append(
                    f'    _free=$(df -BG {sp} 2>/dev/null | tail -1 | awk \'{{print $4}}\' | tr -d \'G\')\n'
                    f'    if [ "$_free" -ge {model_need_gb} ] 2>/dev/null; then\n'
                    f'        if mkdir -p {sp}/{model_cache_name} 2>/dev/null && touch {sp}/{model_cache_name}/.write_test 2>/dev/null; then\n'
                    f'            rm -f {sp}/{model_cache_name}/.write_test\n'
                    f'            _model_dl={sp}\n'
                    f'            echo "MODEL_LOG:downloading to {p} (${{_free}}GB free, need {model_need_gb}GB)"\n'
                    f'            break\n'
                    f'        else\n'
                    f'            echo "MODEL_LOG:{p} not writable (permission denied)"\n'
                    f'        fi\n'
                    f'    else\n'
                    f'        echo "MODEL_LOG:{p} only ${{_free}}GB free (need {model_need_gb}GB)"\n'
                    f'    fi'
                )
            model_space_script = "\n".join(model_space_parts)

            # NVMe persist + RAM promote bash block (reused for found and downloaded)
            persist_and_promote = ""
            if safe_nvme:
                persist_and_promote += f"""
    # Persist to NVMe for next run
    if [ "$_src" != {safe_nvme} ] && [ ! -d {safe_nvme}/{model_cache_name}/snapshots ]; then
        _nvme_free=$(df -BG {safe_nvme} 2>/dev/null | tail -1 | awk '{{print $4}}' | tr -d 'G')
        _need_gb=$(( (${{_model_size:-0}} / 1024) + 5 ))
        if [ "${{_nvme_free:-0}}" -ge "$_need_gb" ] 2>/dev/null; then
            echo "MODEL_LOG:persisting to {nvme_path}..."
            _p_start=$(date +%s)
            mkdir -p {safe_nvme}
            cp -a "$_src"/{model_cache_name} {safe_nvme}/ 2>&1
            if [ $? -eq 0 ]; then
                echo "MODEL_LOG:persisted in $(( $(date +%s) - _p_start ))s"
                _src={safe_nvme}
            else
                echo "MODEL_LOG:persist to NVMe failed — cleaning partial"
                rm -rf {safe_nvme}/{model_cache_name} 2>/dev/null
            fi
        else
            echo "MODEL_LOG:skipping NVMe persist (${{_nvme_free:-?}}GB free, need ${{_need_gb}}GB)"
        fi
    fi
    # Use NVMe as promote source if available (faster than FSx)
    if [ "$_src" != {safe_nvme} ] && [ -d {safe_nvme}/{model_cache_name}/snapshots ]; then
        _src={safe_nvme}
    fi"""
            persist_and_promote += f"""
    # Prune old models from RAM if they exceed 300GB or 30% of total RAM
    if [ "$_src" != {safe_ram} ] && [ ! -d {safe_ram}/{model_cache_name}/snapshots ]; then
        _total_ram_mb=$(free -m 2>/dev/null | awk '/Mem:/{{print $2}}')
        _old_usage_mb=0
        for _od in {safe_ram}/models--*; do
            [ -d "$_od" ] || continue
            _odname=$(basename "$_od")
            [ "$_odname" = "{model_cache_name}" ] && continue
            _osz=$(du -sm "$_od" 2>/dev/null | cut -f1)
            _old_usage_mb=$((_old_usage_mb + ${{_osz:-0}}))
        done
        # Guard: only check 30% if free succeeded (total_ram > 0), otherwise only use 300GB absolute
        _do_prune=0
        if [ "$_old_usage_mb" -ge 307200 ] 2>/dev/null; then
            _do_prune=1
        elif [ "${{_total_ram_mb:-0}}" -gt 0 ] 2>/dev/null; then
            _threshold_30pct=$((_total_ram_mb * 30 / 100))
            if [ "$_old_usage_mb" -gt "$_threshold_30pct" ] 2>/dev/null; then
                _do_prune=1
            fi
        fi
        if [ "$_do_prune" -eq 1 ]; then
            echo "MODEL_LOG:pruning old models from RAM (${{_old_usage_mb}}MB, total_ram=${{_total_ram_mb:-unknown}}MB)"
            for _od in {safe_ram}/models--*; do
                [ -d "$_od" ] || continue
                _odname=$(basename "$_od")
                [ "$_odname" = "{model_cache_name}" ] && continue
                echo "MODEL_LOG:pruned $_odname from RAM"
                rm -rf "$_od"
            done
        else
            echo "MODEL_LOG:RAM prune skipped (${{_old_usage_mb}}MB old models, total_ram=${{_total_ram_mb:-unknown}}MB)"
        fi
    fi"""
            persist_and_promote += f"""
    # Promote to RAM for serving (with headroom check)
    if [ "$_src" != {safe_ram} ] && [ ! -d {safe_ram}/{model_cache_name}/snapshots ]; then
        _avail_mb=$(free -m 2>/dev/null | awk '/Mem:/{{print $7}}')
        _need_mb=$(( ${{_model_size:-0}} + {RAM_PROMOTE_HEADROOM_GB * 1024} ))
        if [ "${{_avail_mb:-0}}" -ge "$_need_mb" ] 2>/dev/null; then
            echo "MODEL_LOG:promoting to {ram_path} (${{_avail_mb}}MB free, need ${{_need_mb}}MB)..."
            _cp_start=$(date +%s)
            mkdir -p {safe_ram}
            cp -a "$_src"/{model_cache_name} {safe_ram}/ 2>&1
            if [ $? -eq 0 ]; then
                echo "MODEL_LOG:promoted in $(( $(date +%s) - _cp_start ))s"
            else
                echo "MODEL_LOG:promote to RAM failed — cleaning partial, serving from $_src"
                rm -rf {safe_ram}/{model_cache_name} 2>/dev/null
            fi
        else
            echo "MODEL_LOG:skipping RAM promote (${{_avail_mb}}MB free, need ${{_need_mb}}MB) — serving from $_src"
        fi
    fi"""

            script_parts.append(f"""
# Model weights: check hierarchy → space-aware download → persist NVMe → promote RAM
_found_at=""
{model_check_script}
else
    _found_at=""
fi

if [ -n "$_found_at" ]; then
    _src="$_found_at"
{persist_and_promote}
    # Detect if we did any copy work (persist or promote)
    if [ "$_src" != "$_found_at" ] || {{ [ -d {safe_ram}/{model_cache_name}/snapshots ] && [ "$_found_at" != {safe_ram} ]; }}; then
        echo "MODEL:PROMOTED"
    else
        echo "MODEL:CACHED"
    fi
else
    # Find download target: walk persistent paths checking space
    _model_dl=""
    for _x in 1; do
{model_space_script}
    done
    # Fall back to RAM if no persistent path has space
    if [ -z "$_model_dl" ]; then
        _model_dl={safe_ram}
        echo "MODEL_LOG:all persistent paths full — downloading directly to RAM"
    fi

    # Download from S3
    _model_start=$(date +%s)
    mkdir -p "$_model_dl"/{model_cache_name}
    aws s3 sync {safe_s3_model} "$_model_dl"/{model_cache_name}/ --exclude 'blobs/*' --exclude '*.bin' --exclude '*.pt' --no-progress 2>&1
    if [ $? -eq 0 ]; then
        _model_end=$(date +%s)
        _model_size=$(du -sm "$_model_dl"/{model_cache_name} 2>/dev/null | cut -f1)
        echo "MODEL_LOG:downloaded ${{_model_size}}MB in $((_model_end - _model_start))s"
        _src="$_model_dl"
{persist_and_promote}
        echo "MODEL:STAGED"
    else
        echo "MODEL:FAIL"
    fi
fi""")

        script = "\n".join(script_parts)
        timeout = 120
        if want_model:
            timeout = 900
        rc, out, err = ssh_command(ip, script, timeout=timeout)
        return ip, "ok" if rc == 0 else "fail", (out or "").strip()

    succeeded = 0
    skipped = 0
    failed = 0
    completed = 0
    total = len(ip_list)
    start = time.time()

    # Heartbeat thread — prints progress every 30s so the console never goes silent
    _heartbeat_stop = threading.Event()

    def _heartbeat():
        while not _heartbeat_stop.wait(30):
            elapsed = time.time() - start
            mins, secs = divmod(int(elapsed), 60)
            safe_print(f"[PRE-WARM] still waiting... {mins}m{secs:02d}s ({completed}/{total} complete)")

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_prewarm_node, ip): ip for ip in ip_list}
        try:
            for future in as_completed(futures, timeout=hard_ceiling):
                ip = futures[future]
                try:
                    result_ip, status, out = future.result()
                    node_elapsed = time.time() - start
                    if status == "skip":
                        completed += 1
                        skipped += 1
                        safe_print(f"[PRE-WARM] [{completed}/{total}] {result_ip} — skipped ({node_elapsed:.0f}s)")
                        continue
                    # SSH failure itself counts as failed (timeout, unreachable, etc.)
                    if status == "fail":
                        completed += 1
                        logger.log(f"{result_ip}: prewarm failed (rc≠0)")
                        failed += 1
                        safe_print(f"[PRE-WARM] [{completed}/{total}] {result_ip} — failed ({node_elapsed:.0f}s)")
                        continue
                    results = {}
                    log_lines = []
                    for line in out.splitlines():
                        if ":" in line:
                            key, val = line.split(":", 1)
                            key = key.strip()
                            val = val.strip()
                            if key in ("DOCKER_LOG", "MODEL_LOG"):
                                log_lines.append(val)
                            else:
                                results[key] = val
                    docker_status = results.get("DOCKER", "")
                    model_status = results.get("MODEL", "")
                    parts = []
                    if docker_status:
                        parts.append(f"docker={docker_status.lower()}")
                    if model_status:
                        parts.append(f"model={model_status.lower()}")
                    shm_warn = results.get("CACHE_WARN", "")
                    if shm_warn:
                        parts.append(f"⚠ cache dir: {shm_warn}")
                    logger.log(f"{result_ip}: {', '.join(parts)}")
                    for ll in log_lines:
                        logger.log(f"{result_ip}:   {ll}")
                    # Check for failure using parsed sentinels (not raw output)
                    docker_failed = "FAIL" in docker_status.upper() if docker_status else False
                    model_failed = "FAIL" in model_status.upper() if model_status else False
                    # Missing expected sentinels = script crashed
                    docker_missing = want_tar and not docker_status
                    model_missing = want_model and not model_status
                    completed += 1
                    detail = f" | {'; '.join(log_lines)}" if log_lines else ""
                    if docker_failed or model_failed or docker_missing or model_missing:
                        failed += 1
                        safe_print(f"[PRE-WARM] [{completed}/{total}] {result_ip} — FAILED: {', '.join(parts)}{detail} ({node_elapsed:.0f}s)")
                    else:
                        succeeded += 1
                        safe_print(f"[PRE-WARM] [{completed}/{total}] {result_ip} — {', '.join(parts)}{detail} ({node_elapsed:.0f}s)")
                except Exception as e:
                    completed += 1
                    logger.log(f"{ip}: prewarm error: {e}", "WARN")
                    failed += 1
                    safe_print(f"[PRE-WARM] [{completed}/{total}] {ip} — error: {e} ({time.time() - start:.0f}s)")
        except TimeoutError:
            # Count any nodes that didn't complete as failed
            timed_out = len(ip_list) - (succeeded + skipped + failed)
            failed += timed_out
            safe_print(f"[PRE-WARM] ⚠ Timed out waiting for {timed_out} nodes")

    _heartbeat_stop.set()
    hb.join(timeout=2)

    elapsed = time.time() - start
    safe_print(f"[PRE-WARM] Done in {elapsed:.0f}s — {succeeded} staged, {skipped} skipped, {failed} failed")
    if skipped > 0:
        safe_print("[PRE-WARM] Some nodes skipped — will download from S3 during deploy (slower)")
    if failed > 0:
        safe_print("[PRE-WARM] Failed nodes will download from S3 during deploy")
    logger.flush()


def main():
    """Main execution with parallel processing."""
    global DRY_RUN

    args = parse_args()
    if args.dry_run:
        DRY_RUN = True
    workers = args.workers
    ssh_timeout = args.ssh_timeout
    batch_delay = args.batch_delay
    prefix = args.name
    global VLLM_KEEP_ALIVE
    VLLM_KEEP_ALIVE = args.keep_alive
    # Load config (required) — sets cache hierarchy, S3 source, streams, gpus_per_node
    load_config(args.config)
    global BASE_PORT
    if args.base_port is not None:
        BASE_PORT = args.base_port
    # Validate --name: only allow alphanumeric, dashes, underscores (used raw in shell commands)
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$', prefix):
        safe_print(f"ERROR: --name must be alphanumeric (with - or _), got: {prefix!r}")
        sys.exit(1)
    model_id = args.model

    # --model is required for deploy, repair, inspect-cache (not for verify, teardown)
    _needs_model = not (args.teardown or args.verify)
    if _needs_model and not model_id:
        safe_print("ERROR: --model is required for this mode (e.g. --model meta-llama/Llama-3.1-70B)")
        sys.exit(1)

    # Validate model ID: HuggingFace format (org/name, alphanumeric + dashes/dots/underscores)
    if model_id and not re.match(r'^[a-zA-Z0-9_.-]+(/[a-zA-Z0-9_.-]+)*$', model_id):
        safe_print(f"ERROR: --model contains unsafe characters: {model_id!r}")
        safe_print("Expected HuggingFace format: org/model-name (e.g. meta-llama/Llama-3.1-70B)")
        sys.exit(1)
    s3_bucket = CACHE_S3_SOURCE
    vllm_args = args.vllm_args
    docker_image = args.docker_image or DOCKER_IMAGE
    no_docker = args.no_docker
    repair = args.repair
    skip_prewarm = args.skip_prewarm
    tp_size = args.tp
    ip_list = resolve_ip_list(args)
    global NUM_DEPLOY_NODES
    NUM_DEPLOY_NODES = len(ip_list)

    # Validate --vllm-args: whitelist safe characters (blocks ; | & $ ` ( ) < > quotes)
    if vllm_args and not re.match(r'^[a-zA-Z0-9 _.=/,:-]+\Z', vllm_args):
        safe_print(f"ERROR: --vllm-args contains unsafe characters: {vllm_args!r}")
        safe_print("Allowed: alphanumeric, spaces, dots, equals, slashes, commas, colons, dashes")
        sys.exit(1)

    # Teardown only needs --name, not --tp
    if args.teardown:
        run_teardown(ip_list, prefix, workers, ssh_timeout, no_docker)
        return

    # Inspect cache — no --tp or --base-port needed
    if args.inspect_cache:
        run_inspect_cache(ip_list, model_id, docker_image, workers, ssh_timeout, no_docker)
        return

    # Everything else needs --tp and --base-port
    if tp_size is None:
        safe_print("ERROR: --tp is required (e.g. --tp 2)")
        sys.exit(1)

    if tp_size < 1 or tp_size > TOTAL_GPUS or (tp_size & (tp_size - 1)) != 0:
        safe_print(f"ERROR: --tp {tp_size} must be a power of 2 between 1 and {TOTAL_GPUS}")
        sys.exit(1)

    if TOTAL_GPUS % tp_size != 0:
        safe_print(f"ERROR: --gpus-per-node {TOTAL_GPUS} is not divisible by --tp {tp_size}")
        sys.exit(1)

    if args.base_port is None:
        safe_print("ERROR: --base-port is required (e.g. --base-port 35000)")
        sys.exit(1)

    # Validate port range: all generated ports must be valid (1-65535)
    num_containers_per_node = TOTAL_GPUS // tp_size
    max_port = BASE_PORT + num_containers_per_node - 1
    if BASE_PORT < 1 or max_port > 65535:
        safe_print(f"ERROR: --base-port {BASE_PORT} with TP={tp_size} generates ports {BASE_PORT}–{max_port}")
        safe_print(f"All ports must be in range 1–65535. Use a lower --base-port or higher --tp.")
        sys.exit(1)

    # Auto-inject --tensor-parallel-size if TP > 1 and not already present
    if "--tensor-parallel-size" in vllm_args:
        # User explicitly set TP in vllm-args — check for conflict
        tp_match = re.search(r'--tensor-parallel-size[\s=]+(\d+)', vllm_args)
        if tp_match:
            user_tp = int(tp_match.group(1))
            if user_tp != tp_size:
                safe_print(f"ERROR: --tp {tp_size} conflicts with --tensor-parallel-size {user_tp} in --vllm-args")
                safe_print(f"Remove --tensor-parallel-size from --vllm-args and use --tp only")
                sys.exit(1)
            safe_print(f"NOTE: --tensor-parallel-size {user_tp} already in --vllm-args, skipping auto-inject")
    elif tp_size > 1:
        vllm_args = f"--tensor-parallel-size {tp_size} {vllm_args}"
    else:
        # TP=1: inject explicitly for clarity in logs/debugging
        vllm_args = f"--tensor-parallel-size 1 {vllm_args}"

    # Verify mode — check and exit
    if args.verify:
        run_verify(ip_list, prefix, workers, ssh_timeout, tp_size, no_docker)
        return

    # S3 source validation (from config [source] section — optional)
    if s3_bucket and not s3_bucket.startswith("s3://"):
        safe_print(f"ERROR: S3 source in config must start with s3:// — got: {s3_bucket!r}")
        sys.exit(1)
    if not s3_bucket:
        safe_print("NOTE: No S3 source in config — models must already exist in cache paths")

    safe_print("=" * 80)
    safe_print("vLLM Multi-Host PARALLEL Deployment")
    safe_print(f"Started:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"Deployer:          {getpass.getuser()}@{socket.gethostname()} ({_get_deployer_ip()})")
    safe_print(f"Command:           {' '.join(shlex.quote(a) for a in sys.argv)}")
    safe_print(f"Total hosts:       {len(ip_list)}")
    safe_print(f"Parallel workers:  {workers}")
    safe_print(f"Container prefix:  {prefix}")
    safe_print(f"Model:             {model_id}")
    if no_docker:
        safe_print(f"Mode:              no-docker (direct host execution)")
    else:
        safe_print(f"Docker image:      {docker_image}")
    safe_print(f"GPUs per node:     {TOTAL_GPUS}")
    safe_print(f"Tensor parallel:   TP={tp_size} → {num_containers_per_node} containers/node")
    safe_print(f"Port range:        {BASE_PORT}–{BASE_PORT + num_containers_per_node - 1}")
    safe_print(f"Total containers:  {len(ip_list) * num_containers_per_node} (max)")
    safe_print(f"vLLM args:         {vllm_args or '(none)'}")
    safe_print(f"Config:            {args.config}")
    _eff_paths = effective_cache_paths(NUM_DEPLOY_NODES)
    _skipped = [p for p, s in CACHE_HIERARCHY if s and NUM_DEPLOY_NODES > CACHE_THRESHOLD]
    safe_print(f"Cache hierarchy:   {' → '.join(_eff_paths)}")
    if _skipped:
        safe_print(f"Cache skipped:     {', '.join(_skipped)} (>{CACHE_THRESHOLD} nodes)")
    safe_print(f"RAM headroom:      {RAM_PROMOTE_HEADROOM_GB}GB (skip RAM promote if free < model + {RAM_PROMOTE_HEADROOM_GB}GB)")
    # Warn if all paths are shared — fallback means shared paths are used anyway
    _local_paths = [p for p, s in CACHE_HIERARCHY if not s]
    if not _local_paths:
        safe_print(f"⚠ WARNING: All cache paths are shared: — add at least one local path to avoid I/O throttle")
    if CACHE_S3_SOURCE:
        safe_print(f"S3 source:         {CACHE_S3_SOURCE} ({S3_MAX_CONCURRENT_REQUESTS} streams/node)")
    safe_print(f"SSH timeout:       {ssh_timeout}s")
    safe_print(f"Batch delay:       {batch_delay}s")
    safe_print(f"Dry run:           {DRY_RUN}")
    if repair:
        safe_print(f"Mode:              REPAIR (only start missing/dead containers)")
    if skip_prewarm:
        safe_print(f"Skip prewarm:      True (nodes download from S3 during deploy)")
    safe_print(f"Log directory:     {LOG_DIR}")

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Create symlink in CWD if logs ended up elsewhere (/tmp, /dev/shm)
    _cwd = os.path.abspath(".") + os.sep
    _log_in_cwd = os.path.abspath(LOG_DIR).startswith(_cwd)
    if not _log_in_cwd:
        local_link = Path(_LOG_BASENAME)
        if not local_link.exists():
            try:
                local_link.symlink_to(LOG_DIR)
                safe_print(f"Symlink:           ./{_LOG_BASENAME} → {LOG_DIR}")
            except OSError:
                pass
    safe_print("=" * 80)

    # ── Port conflict check: ensure our port range isn't used by someone else ──
    port_conflict_ips = set()
    original_ip_count = len(ip_list)
    if not DRY_RUN and not repair:
        port_map = build_gpu_port_map(tp_size)
        ports = sorted(port_map.keys())
        min_port, max_port = ports[0], ports[-1]
        safe_print(f"\n[PORT-CHECK] Scanning ports {min_port}-{max_port} on {len(ip_list)} node(s)...")
        port_conflicts = []

        def _check_ports(ip):
            ip_tag = ip.replace(".", "-")
            # Single SSH: check all ports
            port_checks = " ".join(
                f'echo "PORT:{p}:$(ss -tlnp \'sport = :{p}\' 2>/dev/null | tail -n+2 | head -1)";'
                for p in ports
            )
            if not no_docker:
                port_checks += f' echo "DOCKER:$(docker ps --format \'{{{{.Names}}}}:{{{{.Ports}}}}\' 2>/dev/null)";'
            else:
                # No-docker: list our PID files to identify owned processes
                for p, gpu_ids in port_map.items():
                    gpu_label = "".join(gpu_ids.split(","))
                    name = f"{prefix}_{ip_tag}_gpu{gpu_label}"
                    pid_file = f"{RUN_DIR}/{name}.pid"
                    port_checks += f' if [ -f "{pid_file}" ] && kill -0 $(cat "{pid_file}" 2>/dev/null) 2>/dev/null; then echo "OURS:{name}"; fi;'
            rc, out, _ = ssh_command(ip, port_checks, timeout=15)
            conflicts = []
            if rc == 0 and out:
                for line in out.strip().splitlines():
                    if not line.startswith("PORT:"):
                        continue
                    parts = line.split(":", 2)
                    if len(parts) < 3:
                        continue
                    try:
                        p = int(parts[1])
                    except ValueError:
                        continue
                    detail = parts[2].strip()
                    if not detail:  # port is free
                        continue
                    # Check if it's our container/process
                    gpu_ids = port_map.get(p, "")
                    gpu_label = "".join(gpu_ids.split(","))
                    expected_name = f"{prefix}_{ip_tag}_gpu{gpu_label}"
                    if expected_name in out:
                        continue  # ours — not a conflict
                    conflicts.append(p)
            return ip, conflicts

        with ThreadPoolExecutor(max_workers=min(workers, len(ip_list))) as pool:
            futures = {pool.submit(_check_ports, ip): ip for ip in ip_list}
            for future in as_completed(futures):
                try:
                    ip, conflicts = future.result()
                    if conflicts:
                        port_conflicts.append((ip, conflicts))
                except Exception:
                    pass  # SSH failure — node will fail at deploy anyway

        if port_conflicts:
            safe_print(f"\n[PORT-CHECK] ✗ PORT CONFLICTS on {len(port_conflicts)} node(s):")
            for ip, conflicts in port_conflicts:
                safe_print(f"  {ip} — ports {', '.join(str(p) for p in conflicts)} in use by another process")
            # Remove conflicting nodes from deploy list
            port_conflict_ips = {ip for ip, _ in port_conflicts}
            ip_list = [ip for ip in ip_list if ip not in port_conflict_ips]
            if not ip_list:
                safe_print(f"\n  All nodes have port conflicts. Pick a different --base-port.")
                safe_print(f"  Aborting.\n")
                sys.exit(1)
            safe_print(f"\n  Skipping {len(port_conflict_ips)} node(s) — deploying on remaining {len(ip_list)}")
            NUM_DEPLOY_NODES = len(ip_list)
        else:
            safe_print(f"[PORT-CHECK] ✓ Ports {min_port}-{max_port} clear on all nodes\n")

    # ── Pre-flight: ensure Docker image + model on shared storage ──
    if not DRY_RUN and not repair:
        safe_print("")
        safe_print("[PRE-FLIGHT] Finding a reachable scout node...")
        scout_ip = _find_scout_node(ip_list, ssh_timeout)
        if not scout_ip:
            safe_print("ERROR: No reachable nodes in first 10 IPs — aborting")
            sys.exit(1)
        safe_print(f"[PRE-FLIGHT] Scout node: {scout_ip}")
        if no_docker:
            safe_print("[PRE-FLIGHT] No-docker mode — skipping Docker image checks")
            # Only check model availability — skip Docker TAR entirely
            preflight_ok, asset_info = preflight_ensure_shared_assets(scout_ip, docker_image, model_id, s3_bucket,
                                                           skip_docker=True, temp_folder=args.temp_folder)
        else:
            preflight_ok, asset_info = preflight_ensure_shared_assets(scout_ip, docker_image, model_id, s3_bucket,
                                                           temp_folder=args.temp_folder)
        if not preflight_ok:
            safe_print("\n[PRE-FLIGHT] ✗ FATAL: Assets not available — auto-download failed, aborting")
            safe_print("[PRE-FLIGHT]   Check logs for details. Common causes:")
            safe_print("[PRE-FLIGHT]   - Gated model: put HF token in {s3-bucket}/a")
            safe_print("[PRE-FLIGHT]   - Docker registry unreachable")
            safe_print("[PRE-FLIGHT]   - S3 bucket permissions (write access needed)")
            sys.exit(1)
        safe_print("")

        # ── Pre-warm: download assets from S3 to all nodes ──
        if skip_prewarm:
            safe_print("[PRE-WARM] Skipped (--skip-prewarm) — nodes will download from S3 during deploy")
        elif asset_info.get("s3_tar_path") or asset_info.get("s3_model_path"):
            preflight_prewarm_nodes(ip_list, asset_info, docker_image, workers, ssh_timeout, no_docker)
            safe_print("")

    start_time = time.time()
    results = []
    # Max time per host: docker_load(900s) + model_download(1800s) + docker_run(202s) + overhead
    per_host_timeout = DOCKER_LOAD_TIMEOUT + 1800 + 300  # ~3000s = 50 min
    # as_completed(timeout) is a global clock from call time, not per-thread.
    # With N waves, last-wave threads start late and need the full per_host_timeout.
    num_deploy_waves = math.ceil(len(ip_list) / max(workers, 1))
    total_deploy_timeout = per_host_timeout + (num_deploy_waves - 1) * per_host_timeout

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ip = {}
        for i, ip in enumerate(ip_list):
            if batch_delay > 0 and i > 0:
                time.sleep(batch_delay)
            future_to_ip[executor.submit(process_host, ip, ssh_timeout, prefix, model_id, s3_bucket, vllm_args, docker_image, tp_size, no_docker, repair)] = ip

        deploy_completed = 0
        deploy_total = len(ip_list)
        try:
            for future in as_completed(future_to_ip, timeout=total_deploy_timeout):
                ip = future_to_ip[future]
                deploy_completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    node_elapsed = time.time() - start_time
                    started = result.get("containers_started", 0)
                    skipped = result.get("containers_skipped", 0)
                    if result.get("success"):
                        safe_print(f"[DEPLOY] [{deploy_completed}/{deploy_total}] {ip} — ✓ {started} started, {skipped} skipped ({node_elapsed:.0f}s)")
                    else:
                        safe_print(f"[DEPLOY] [{deploy_completed}/{deploy_total}] {ip} — ✗ {result.get('error', 'unknown')} ({node_elapsed:.0f}s)")
                except Exception as e:
                    safe_print(f"[DEPLOY] [{deploy_completed}/{deploy_total}] {ip} — ✗ Exception: {str(e)} ({time.time() - start_time:.0f}s)")
                    results.append({
                        "ip": ip,
                        "success": False,
                        "containers_started": 0,
                        "containers_skipped": 0,
                        "error": str(e),
                        "log_file": str(Path(LOG_DIR) / f"{ip}.log"),
                    })
        except TimeoutError:
            # Some threads didn't finish in time
            timed_out_ips = [ip for future, ip in future_to_ip.items() if not future.done()]
            for ip in timed_out_ips:
                safe_print(f"[{ip}] ✗ Timed out after {total_deploy_timeout}s")
                results.append({
                    "ip": ip,
                    "success": False,
                    "containers_started": 0,
                    "containers_skipped": 0,
                    "error": f"Timed out after {total_deploy_timeout}s",
                    "failure_category": "timeout",
                    "log_file": str(Path(LOG_DIR) / f"{ip}.log"),
                })

    elapsed = time.time() - start_time

    # Inject port-conflict nodes as failed results so they appear in summary
    for pc_ip in sorted(port_conflict_ips):
        results.append({
            "ip": pc_ip,
            "success": False,
            "containers_started": 0,
            "containers_skipped": 0,
            "error": "port conflict (skipped)",
            "failure_category": "port_conflict",
            "log_file": "",
        })

    # ── Post-deploy crash detection ──
    # Wait for containers to initialize, then check which ones actually survived.
    # vLLM takes several minutes to load model weights — crashes happen during load.
    successful_results = [r for r in results if r["success"] and r.get("container_names")]
    # In repair mode, only check hosts that actually launched new containers
    total_newly_launched = sum(r.get("newly_launched", r["containers_started"]) for r in successful_results)
    if successful_results and not DRY_RUN and not _abort and total_newly_launched > 0:
        # Scale wait time with TP: larger TP = longer model load
        _crash_wait = {1: 30, 2: 45, 4: 60, 8: 90}
        wait_secs = _crash_wait.get(tp_size, 60)
        # Countdown timer so user knows it's not stuck
        for remaining in range(wait_secs, 0, -5):
            print(f"\r[POST-DEPLOY] Waiting for containers to initialize... {remaining}s remaining  ", end="", flush=True)
            time.sleep(min(5, remaining))
        print(f"\r[POST-DEPLOY] Waiting for containers to initialize... done{' ' * 20}")

        safe_print(f"[POST-DEPLOY] Checking {sum(len(r.get('container_names', [])) for r in successful_results)} containers across {len(successful_results)} hosts...")
        crashed_total = 0

        def _check_containers(r):
            """Check if containers/processes on a host are still running after init delay (single SSH).
            Returns (ip, crashed_count, total_count, survivors_list)."""
            ip = r["ip"]
            container_names = r.get("container_names", [])
            if not container_names:
                return ip, 0, 0, []

            # Build a single script to check all containers at once
            if no_docker:
                checks = []
                for cname in container_names:
                    pid_file = f"{RUN_DIR}/{cname}.pid"
                    checks.append(f"""
pid=$(cat "{pid_file}" 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | grep -q vllm; then
    echo "ALIVE:{cname}"
else
    echo "CRASHED:{cname}"
fi""")
                script = "\n".join(checks)
            else:
                checks = []
                for cname in container_names:
                    checks.append(f'status=$(docker inspect --format "{{{{.State.Status}}}}" {cname} 2>/dev/null) || status="missing"; echo "$status:{cname}"')
                script = "\n".join(checks)

            rc, out, _ = ssh_command(ip, script, timeout=max(15, len(container_names) * 3))
            if rc != 0 or not out:
                return ip, 0, len(container_names), list(container_names)  # Can't tell, don't downgrade

            crashed = 0
            survivors = []
            for line in out.strip().splitlines():
                line = line.strip()
                if no_docker:
                    if line.startswith("CRASHED:"):
                        crashed += 1
                    elif line.startswith("ALIVE:"):
                        survivors.append(line[6:])
                else:
                    # Format: "status:container_name"
                    if ":" in line:
                        status, cname = line.split(":", 1)
                        if status in ("exited", "dead", "created", "missing"):
                            crashed += 1
                        else:
                            survivors.append(cname)
            return ip, crashed, len(container_names), survivors

        with ThreadPoolExecutor(max_workers=workers) as executor:
            crash_futures = {executor.submit(_check_containers, r): r for r in successful_results}
            # Scale timeout: each container needs ~2 SSH calls × 10s timeout each, plus headroom
            crash_check_timeout = max(60, num_containers_per_node * 25)
            try:
                for future in as_completed(crash_futures, timeout=crash_check_timeout):
                    r = crash_futures[future]
                    try:
                        ip, crashed, total, survivors = future.result()
                        if crashed > 0:
                            crashed_total += crashed
                            safe_print(f"[POST-DEPLOY] ⚠ {ip}: {crashed}/{total} containers crashed after startup")
                            # Update container_names to only include survivors
                            r["container_names"] = survivors
                            # Downgrade result
                            if crashed == total:
                                r["success"] = False
                                r["error"] = f"All {total} containers crashed after startup (OOM/init failure)"
                                r["failure_category"] = "post_deploy_crash"
                            else:
                                r["partial"] = True
                                r["containers_started"] -= crashed
                                r["containers_skipped"] += crashed
                                r["error"] = f"{crashed}/{total} containers crashed after startup"
                    except Exception as e:
                        safe_print(f"[POST-DEPLOY] ⚠ Crash check failed for {r.get('ip', '?')}: {e}")
            except TimeoutError:
                safe_print("[POST-DEPLOY] ⚠ Crash check timed out on some hosts")

        if crashed_total == 0:
            safe_print(f"[POST-DEPLOY] ✓ All containers still running after {wait_secs}s")
        else:
            safe_print(f"[POST-DEPLOY] ✗ {crashed_total} containers crashed — check logs with --verify")
    elif successful_results and not DRY_RUN and not _abort and total_newly_launched == 0:
        safe_print(f"\n[POST-DEPLOY] All {sum(r['containers_started'] for r in successful_results)} containers already healthy — skipping crash check")

    # Write summary files
    deploy_config = {
        "name": prefix,
        "tp": tp_size,
        "model": model_id,
        "docker_image": docker_image if not no_docker else "(no-docker)",
        "no_docker": no_docker,
        "repair": repair,
        "vllm_args": vllm_args,
        "s3_bucket": s3_bucket,
    }
    try:
        json_path, txt_path = write_summary(results, elapsed, deploy_config)
    except Exception as e:
        safe_print(f"\n⚠ Failed to write summary files: {e}")
        json_path = "(write failed)"
        txt_path = "(write failed)"

    # Print summary to console
    full_success = sum(1 for r in results if r["success"] and not r.get("partial"))
    partial_success = sum(1 for r in results if r.get("partial"))
    total_containers = sum(r["containers_started"] for r in results)
    total_skipped = sum(r["containers_skipped"] for r in results)
    total_busy = sum(r.get("containers_busy", 0) for r in results)

    safe_print("\n" + "=" * 80)
    safe_print("DEPLOYMENT SUMMARY")
    safe_print("=" * 80)
    safe_print(f"Total time:          {elapsed:.1f}s")
    safe_print(f"Fully successful:    {full_success}/{original_ip_count}")
    if partial_success:
        safe_print(f"Partial success:     {partial_success}/{original_ip_count}")
    safe_print(f"Containers started:  {total_containers}")
    if repair:
        total_kept = sum(r.get("repair_kept", 0) for r in results)
        total_new = sum(r.get("newly_launched", 0) for r in results)
        if total_kept > 0:
            safe_print(f"  Already healthy:   {total_kept}")
            safe_print(f"  Newly launched:    {total_new}")
    if total_skipped:
        safe_print(f"Containers failed:   {total_skipped}")
    if total_busy:
        safe_print(f"GPUs busy (other):   {total_busy}")
    safe_print(f"Log directory:       {LOG_DIR}")
    safe_print(f"Summary JSON:        {json_path}")
    safe_print(f"Summary text:        {txt_path}")

    partial = [r for r in results if r.get("partial")]
    if partial:
        safe_print(f"\nPartial hosts ({len(partial)}):")
        for r in partial:
            safe_print(f"  ⚠ {r['ip']}: {r['error']}  (log: {r['log_file']})")

    failed = [r for r in results if not r["success"]]
    if failed:
        safe_print(f"\nFailed hosts ({len(failed)}):")
        for r in failed:
            safe_print(f"  ✗ {r['ip']}: {r['error']}  (log: {r['log_file']})")

    # Successful IPs — CSV (groups of 8 per line) and Python list
    successful_ips = sorted(r["ip"] for r in results if r["success"])

    if successful_ips:
        # Write CSV — one IP per line, blank line every 8
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(LOG_DIR) / f"successful_{ts}.csv"
        with open(csv_path, "w") as f:
            for i, ip in enumerate(successful_ips):
                f.write(ip + "\n")
                if (i + 1) % 8 == 0 and (i + 1) < len(successful_ips):
                    f.write("\n")
        safe_print(f"\nSuccessful IPs CSV:  {csv_path}")

        # Print as Python list
        safe_print(f"\nSuccessful IPs ({len(successful_ips)}):")
        ip_strs = ', '.join(f'"{ip}"' for ip in successful_ips)
        safe_print(f"[{ip_strs}]")

    # Container names CSV — one entry per line: ip,container_name
    all_container_names = []
    for r in results:
        for cname in r.get("container_names", []):
            all_container_names.append(f"{r['ip']},{cname}")
    if all_container_names:
        containers_csv = Path(LOG_DIR) / "containers.csv"
        with open(containers_csv, "w") as f:
            f.write("ip,container_name\n")
            for entry in sorted(all_container_names):
                f.write(entry + "\n")
        safe_print(f"Containers CSV:      {containers_csv}  ({len(all_container_names)} containers)")

    # Categorized failure CSVs (--ip-file compatible: one IP per line)
    failed_all = [r for r in results if not r["success"]]
    disk_full = [r for r in results if r.get("failure_category") == "disk_full"]
    docker_issue = [r for r in results if r.get("failure_category") == "docker_issue"]
    nvidia_broken = [r for r in results if r.get("failure_category") == "nvidia_broken"]
    post_crash = [r for r in results if r.get("failure_category") == "post_deploy_crash"]
    # "docker_retry" = all docker-related failures that could work with --no-docker
    docker_retry = [r for r in results if r.get("failure_category") in
                    ("docker_issue", "post_deploy_crash", "nvidia_broken")]

    def _write_ip_csv(ips, filename):
        if not ips:
            return None
        path = Path(LOG_DIR) / filename
        with open(path, "w") as f:
            for ip in sorted(ips):
                f.write(ip + "\n")
        return path

    if failed_all:
        p = _write_ip_csv([r["ip"] for r in failed_all], "failed_ips.csv")
        safe_print(f"\nFailed IPs CSV:      {p}  ({len(failed_all)} hosts)")
    if disk_full:
        p = _write_ip_csv([r["ip"] for r in disk_full], "no_space_left.csv")
        safe_print(f"Disk full CSV:       {p}  ({len(disk_full)} hosts)")
    if docker_issue:
        p = _write_ip_csv([r["ip"] for r in docker_issue], "docker_issue.csv")
        safe_print(f"Docker issue CSV:    {p}  ({len(docker_issue)} hosts)")
    if nvidia_broken:
        p = _write_ip_csv([r["ip"] for r in nvidia_broken], "nvidia_broken.csv")
        safe_print(f"NVIDIA broken CSV:   {p}  ({len(nvidia_broken)} hosts)")
    if post_crash:
        p = _write_ip_csv([r["ip"] for r in post_crash], "post_crash.csv")
        safe_print(f"Post-crash CSV:      {p}  ({len(post_crash)} hosts)")
    if docker_retry and not no_docker:
        p = _write_ip_csv([r["ip"] for r in docker_retry], "docker_retry_nodkr.csv")
        safe_print(f"Docker retry CSV:    {p}  ({len(docker_retry)} hosts) ← use with --no-docker --ip-file {p}")

    # Upload logs to S3: {s3_bucket}/logs/{timestamp}/
    if s3_bucket and not DRY_RUN:
        s3_log_dest = f"{s3_bucket.rstrip('/')}/logs/{_LOG_BASENAME}/"
        safe_print(f"\nUploading logs to S3: {s3_log_dest}")
        try:
            sync_result = subprocess.run(
                ["aws", "s3", "sync", str(LOG_DIR), s3_log_dest, "--no-progress"],
                capture_output=True, text=True, timeout=120)
            if sync_result.returncode == 0:
                safe_print(f"✓ Logs uploaded to {s3_log_dest}")
            else:
                safe_print(f"⚠ Log upload failed (rc={sync_result.returncode}): {sync_result.stderr.strip()[:200]}")
        except subprocess.TimeoutExpired:
            safe_print("⚠ Log upload timed out (120s)")
        except FileNotFoundError:
            safe_print("⚠ Log upload skipped — aws CLI not found on deployer")

    safe_print("=" * 80)


if __name__ == "__main__":
    try:
        # Parse args early to check if this is a deploy (needs confirmation)
        _pre_parser = argparse.ArgumentParser(add_help=False)
        _pre_parser.add_argument("--ip-file", default=None)
        _pre_parser.add_argument("--hosts", nargs="*")
        _pre_parser.add_argument("--failed-from", default=None)
        _pre_parser.add_argument("--verify", action="store_true")
        _pre_parser.add_argument("--teardown", action="store_true")
        _pre_parser.add_argument("--repair", action="store_true")
        _pre_parser.add_argument("--inspect-cache", action="store_true")
        _pre_parser.add_argument("--dry-run", action="store_true")
        _pre_parser.add_argument("--config", type=str, default=None)
        _pre_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
        _pre_args, _ = _pre_parser.parse_known_args()

        _is_readonly = (_pre_args.verify or _pre_args.inspect_cache or _pre_args.dry_run)

        # Validate --config exists before asking for destructive-action confirmation
        if _pre_args.config and not os.path.exists(_pre_args.config):
            print(f"\n  ERROR: Config file not found: {_pre_args.config!r}")
            sys.exit(1)

        if not _is_readonly:
            _nodes = []
            if _pre_args.hosts:
                _nodes = _pre_args.hosts
            elif _pre_args.failed_from and os.path.exists(_pre_args.failed_from):
                try:
                    with open(_pre_args.failed_from) as f:
                        _s = json.load(f)
                    _nodes = [h["ip"] for h in _s["hosts"] if not h["success"]]
                except Exception:
                    _nodes = ["(could not parse summary)"]
            elif _pre_args.ip_file and os.path.exists(_pre_args.ip_file):
                with open(_pre_args.ip_file) as f:
                    _nodes = [l.strip().split(",")[0].split("#")[0].strip()
                              for l in f if l.strip() and not l.strip().startswith("#")]

            _action = "TEAR DOWN containers on" if _pre_args.teardown else \
                      "REPAIR containers on" if _pre_args.repair else \
                      "DEPLOY on"

            print(f"\n  {'=' * 62}")
            print(f"  ╔══════════════════════════════════════════════════════════╗")
            print(f"  ║  ⚠⚠⚠  WARNING: DESTRUCTIVE ACTION  ⚠⚠⚠                ║")
            print(f"  ╚══════════════════════════════════════════════════════════╝")
            print(f"\n  You are about to {_action} {len(_nodes)} GPU node(s).")
            print(f"  Parallel workers: {_pre_args.workers}")
            if _pre_args.config:
                print(f"  Config:           {_pre_args.config}")
            print(f"\n  TARGET NODES:")
            for n in _nodes:
                print(f"    ▸ {n}")
            print(f"\n  ┌──────────────────────────────────────────────────────┐")
            print(f"  │  This action WILL consume/kill GPU resources.        │")
            print(f"  │  Double-check the IPs above.                         │")
            print(f"  │  Confirm you have permission from stakeholders.      │")
            print(f"  │  Wrong nodes = someone else's work gets destroyed.   │")
            print(f"  └──────────────────────────────────────────────────────┘")
            print(f"  {'=' * 62}\n")

            inp = input("  Type 'yes' to proceed (anything else aborts): ")
            if inp.strip().lower() != "yes":
                print("\n  ✗ Aborted. No changes made.\n")
                sys.exit(0)
            print()

        main()
    except KeyboardInterrupt:
        _abort = True
        safe_print("\nInterrupted by user — waiting for running threads to finish...")
        safe_print("(Running threads will stop at next checkpoint)")
        sys.exit(1)
