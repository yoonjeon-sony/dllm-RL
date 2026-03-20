#!/usr/bin/env python3
"""SLURM Monitor Web Dashboard - aiohttp server."""

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

from aiohttp import web

LOG_DIR = Path("/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs")
TEMPLATE_DIR = Path("/home/yoonjeon.kim/dLLM-RL/train_sft/slurm_monitor/templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def run_cmd(cmd: list[str], timeout: float = 10.0) -> str:
    """Run a shell command asynchronously and return stdout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode(errors="replace")
    except (asyncio.TimeoutError, FileNotFoundError, OSError):
        return ""


def scan_log_job_ids() -> dict[str, dict[str, Path]]:
    """Scan LOG_DIR for job IDs and their log files.

    Returns {job_id: {"error": Path, "output": Path}}.
    """
    jobs: dict[str, dict[str, Path]] = {}
    if not LOG_DIR.is_dir():
        return jobs
    for p in LOG_DIR.iterdir():
        m = re.match(r"(error|output)\.(\d+)\.log", p.name)
        if m:
            log_type, job_id = m.group(1), m.group(2)
            jobs.setdefault(job_id, {})[log_type] = p
    return jobs


async def get_squeue_jobs() -> dict[str, dict]:
    """Query squeue for current user's jobs.

    Returns {job_id: {name, state, time, nodelist}}.
    """
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    raw = await run_cmd([
        "squeue", "-u", user,
        "-o", "%i|%j|%T|%M|%N",
        "--noheader",
    ])
    jobs: dict[str, dict] = {}
    for line in raw.strip().splitlines():
        parts = line.strip().split("|", 4)
        if len(parts) < 5:
            continue
        jid, name, state, time_used, nodelist = parts
        jid = jid.strip()
        jobs[jid] = {
            "name": name.strip(),
            "state": state.strip(),
            "time": time_used.strip(),
            "nodelist": nodelist.strip(),
        }
    return jobs


def infer_crash_status(log_files: dict[str, Path]) -> str:
    """Check log files to guess if job crashed or completed normally."""
    err_path = log_files.get("error")
    if err_path and err_path.exists():
        try:
            tail = err_path.read_bytes()[-4096:]
            text = tail.decode(errors="replace").lower()
            # Common crash indicators
            for pattern in [
                "error", "traceback", "killed", "oom",
                "segfault", "core dumped", "exception",
                "srun: error", "slurmstepd: error",
            ]:
                if pattern in text:
                    return "CRASHED"
        except OSError:
            pass
    # Check output log for successful completion signals
    out_path = log_files.get("output")
    if out_path and out_path.exists():
        try:
            tail = out_path.read_bytes()[-2048:]
            text = tail.decode(errors="replace").lower()
            for pattern in ["training complete", "finished", "done"]:
                if pattern in text:
                    return "COMPLETED"
        except OSError:
            pass
    return "COMPLETED"


# ---------------------------------------------------------------------------
# API Handlers
# ---------------------------------------------------------------------------

async def handle_index(request: web.Request) -> web.Response:
    html_path = TEMPLATE_DIR / "index.html"
    return web.FileResponse(html_path)


async def handle_jobs(request: web.Request) -> web.Response:
    squeue_jobs = await get_squeue_jobs()
    log_jobs = scan_log_job_ids()

    result = []

    # Active jobs from squeue
    for jid, info in squeue_jobs.items():
        has_logs = jid in log_jobs
        result.append({
            "id": jid,
            "name": info["name"],
            "state": info["state"],
            "time": info["time"],
            "nodelist": info["nodelist"],
            "has_error_log": has_logs and "error" in log_jobs.get(jid, {}),
            "has_output_log": has_logs and "output" in log_jobs.get(jid, {}),
        })

    # Jobs with log files but not in squeue -> crashed/completed
    for jid, log_files in log_jobs.items():
        if jid not in squeue_jobs:
            status = infer_crash_status(log_files)
            result.append({
                "id": jid,
                "name": "",
                "state": status,
                "time": "",
                "nodelist": "",
                "has_error_log": "error" in log_files,
                "has_output_log": "output" in log_files,
            })

    # Sort: RUNNING first, then PENDING, then others, by job id descending
    state_order = {"RUNNING": 0, "PENDING": 1}
    result.sort(key=lambda j: (state_order.get(j["state"], 2), -int(j["id"])))

    return web.json_response(result)


async def handle_logs(request: web.Request) -> web.StreamResponse:
    """SSE endpoint that tails a log file."""
    job_id = request.match_info["job_id"]
    log_type = request.match_info["log_type"]

    if log_type not in ("error", "output"):
        raise web.HTTPBadRequest(text="log_type must be 'error' or 'output'")
    if not re.match(r"^\d+$", job_id):
        raise web.HTTPBadRequest(text="Invalid job_id")

    log_path = LOG_DIR / f"{log_type}.{job_id}.log"
    if not log_path.exists():
        raise web.HTTPNotFound(text=f"Log file not found: {log_path.name}")

    # Check if job is still running
    squeue_jobs = await get_squeue_jobs()
    is_running = job_id in squeue_jobs and squeue_jobs[job_id]["state"] == "RUNNING"

    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    await resp.prepare(request)

    # Send last N lines as initial batch
    tail_lines = 500
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
            initial = lines[-tail_lines:] if len(lines) > tail_lines else lines
            data = "".join(initial)
            await resp.write(f"event: initial\ndata: {json.dumps(data)}\n\n".encode())

            if not is_running:
                # Job not running - send all content and close
                await resp.write(b"event: done\ndata: job_finished\n\n")
                return resp

            # Stream new lines as they appear
            f.seek(0, 2)  # seek to end
            while True:
                line = f.readline()
                if line:
                    await resp.write(
                        f"event: line\ndata: {json.dumps(line)}\n\n".encode()
                    )
                else:
                    await asyncio.sleep(0.5)
                    # Check if connection is still alive
                    try:
                        await resp.write(b": heartbeat\n\n")
                    except (ConnectionResetError, ConnectionAbortedError):
                        break
    except asyncio.CancelledError:
        pass

    return resp


def _parse_nvidia_smi(raw: str) -> list[dict]:
    """Parse nvidia-smi CSV output into a list of GPU dicts."""
    gpus = []
    if not raw.strip():
        return gpus
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            try:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "gpu_util": float(parts[2]) if parts[2] not in ("[N/A]", "N/A", "") else 0,
                    "mem_used": float(parts[3]) if parts[3] not in ("[N/A]", "N/A", "") else 0,
                    "mem_total": float(parts[4]) if parts[4] not in ("[N/A]", "N/A", "") else 0,
                    "temperature": float(parts[5]) if parts[5] not in ("[N/A]", "N/A", "") else 0,
                    "power": parts[6] if parts[6] not in ("[N/A]", "N/A", "") else "N/A",
                })
            except (ValueError, IndexError):
                continue
    return gpus


async def _query_gpu_for_job(job_id: str, job_info: dict) -> dict:
    """Run nvidia-smi on a job's compute node via srun --jobid."""
    nv_query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    raw = await run_cmd(
        [
            "srun", "--jobid", job_id, "--overlap",
            "nvidia-smi",
            f"--query-gpu={nv_query}",
            "--format=csv,noheader,nounits",
        ],
        timeout=15.0,
    )
    gpus = _parse_nvidia_smi(raw)
    return {
        "job_id": job_id,
        "job_name": job_info.get("name", ""),
        "nodelist": job_info.get("nodelist", ""),
        "gpus": gpus,
        "error": "" if gpus else "Could not query GPUs",
    }


async def handle_gpu(request: web.Request) -> web.Response:
    """Return GPU stats per running job via srun --jobid nvidia-smi."""
    squeue_jobs = await get_squeue_jobs()
    running = {jid: info for jid, info in squeue_jobs.items()
               if info["state"] == "RUNNING"}

    if not running:
        return web.json_response({"jobs": [], "error": "No running jobs"})

    # Query all running jobs in parallel
    tasks = [
        _query_gpu_for_job(jid, info) for jid, info in running.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    job_gpus = []
    for r in results:
        if isinstance(r, Exception):
            continue
        job_gpus.append(r)

    return web.json_response({"jobs": job_gpus})


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/jobs", handle_jobs)
    app.router.add_get("/api/logs/{job_id}/{log_type}", handle_logs)
    app.router.add_get("/api/gpu", handle_gpu)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLURM Monitor Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    print(f"SLURM Monitor starting on http://{args.host}:{args.port}")
    print(f"Log directory: {LOG_DIR}")
    web.run_app(create_app(), host=args.host, port=args.port)
