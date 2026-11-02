"""Dynamic one-process-per-GPU scheduling for independent project tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
import threading
import time
from typing import Iterable, Sequence


@dataclass(frozen=True)
class GpuTask:
    """One independent subprocess that occupies exactly one visible GPU."""

    name: str
    command: tuple[str, ...] | list[str]
    environment: dict[str, str] = field(default_factory=dict)


class GpuQueueError(RuntimeError):
    """Raised after one or more dynamically scheduled tasks fail."""

    def __init__(self, records):
        self.records = records
        failed = [record for record in records if record["status"] == "failed"]
        rendered = ", ".join(
            f"{record['task']} (exit={record['returncode']})" for record in failed
        )
        super().__init__(f"GPU task queue failed: {rendered}")


def parse_gpu_ids(values: Sequence[str | int] | str | None) -> list[int]:
    """Parse ``0 1 2`` and ``0,1,2`` forms while preserving order."""
    if values is None:
        return []
    items = [values] if isinstance(values, (str, int)) else list(values)
    result = []
    for item in items:
        for token in str(item).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                gpu = int(token)
            except ValueError as error:
                raise ValueError(f"GPU identifiers must be integers, received {token!r}") from error
            if gpu < 0:
                raise ValueError("GPU identifiers must be non-negative")
            if gpu in result:
                raise ValueError(f"Duplicate GPU identifier: {gpu}")
            result.append(gpu)
    if not result:
        raise ValueError("At least one GPU identifier is required")
    return result


def validate_gpu_ids(gpu_ids: Sequence[int]):
    """Fail before dispatch when a requested visible CUDA device is unavailable."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; multi-GPU queue execution cannot start")
    count = torch.cuda.device_count()
    invalid = [gpu for gpu in gpu_ids if gpu >= count]
    if invalid:
        raise ValueError(
            f"Requested GPU IDs {invalid}, but only visible IDs 0..{count - 1} exist"
        )


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text[:120] or "task"


def _write_summary(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _busy_compute_gpu_ids() -> set[int] | None:
    """Return GPUs with active compute processes, or ``None`` if unavailable."""
    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        process_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if gpu_query.returncode != 0 or process_query.returncode != 0:
        return None
    uuid_to_index = {}
    for line in gpu_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2:
            continue
        try:
            uuid_to_index[fields[1]] = int(fields[0])
        except ValueError:
            continue
    active_uuids = {
        line.strip()
        for line in process_query.stdout.splitlines()
        if line.strip() and "no running" not in line.lower()
    }
    return {
        uuid_to_index[uuid]
        for uuid in active_uuids
        if uuid in uuid_to_index
    }


def run_gpu_queue(
    tasks: Iterable[GpuTask],
    gpu_ids: Sequence[int],
    *,
    cwd: Path,
    log_root: Path,
    continue_on_error: bool = False,
    dry_run: bool = False,
    validate_devices: bool = True,
    wait_for_free_gpus: bool = True,
    gpu_poll_seconds: float = 10.0,
):
    """Run a dynamic FIFO queue with one long-lived worker per selected GPU.

    Each subprocess sees only its assigned physical GPU through
    ``CUDA_VISIBLE_DEVICES`` and therefore always uses ``cuda:0`` internally.
    When a subprocess exits, that worker immediately takes the next queued task.
    """
    tasks = list(tasks)
    gpu_ids = list(gpu_ids)
    if not gpu_ids:
        raise ValueError("gpu_ids cannot be empty")
    if not tasks:
        return []
    if validate_devices and not dry_run:
        validate_gpu_ids(gpu_ids)
    cwd = Path(cwd).resolve()
    log_root = Path(log_root).resolve()
    summary_path = log_root / "queue_summary.json"

    if dry_run:
        records = []
        for index, task in enumerate(tasks):
            gpu = gpu_ids[index % len(gpu_ids)]
            record = {
                "index": index,
                "task": task.name,
                "gpu": gpu,
                "status": "dry_run",
                "command": list(task.command),
            }
            records.append(record)
            print(f"[GPU queue dry-run] GPU {gpu} <- {task.name}", flush=True)
        return records

    log_root.mkdir(parents=True, exist_ok=True)

    pending = Queue()
    for index, task in enumerate(tasks):
        pending.put((index, task))
    records = []
    records_lock = threading.Lock()
    print_lock = threading.Lock()
    stop_dispatch = threading.Event()
    probe_available = [bool(wait_for_free_gpus and validate_devices)]
    probe_lock = threading.Lock()

    def wait_until_free(gpu_id: int):
        announced = False
        while probe_available[0]:
            busy = _busy_compute_gpu_ids()
            if busy is None:
                with probe_lock:
                    if probe_available[0]:
                        probe_available[0] = False
                        with print_lock:
                            print(
                                "[GPU queue] nvidia-smi process probing is unavailable; "
                                "continuing with one queue worker per selected GPU.",
                                flush=True,
                            )
                return
            if gpu_id not in busy:
                if announced:
                    with print_lock:
                        print(f"[GPU queue] GPU {gpu_id} is now free", flush=True)
                return
            if not announced:
                with print_lock:
                    print(
                        f"[GPU queue] GPU {gpu_id} is occupied by another compute "
                        "process; waiting",
                        flush=True,
                    )
                announced = True
            if stop_dispatch.wait(max(float(gpu_poll_seconds), 0.1)):
                return

    def worker(gpu_id: int):
        while True:
            if stop_dispatch.is_set() and not continue_on_error:
                return
            wait_until_free(gpu_id)
            if stop_dispatch.is_set() and not continue_on_error:
                return
            try:
                index, task = pending.get_nowait()
            except Empty:
                return
            started = time.time()
            stamp = datetime.now().isoformat(timespec="seconds")
            log_path = log_root / f"{index:04d}_{_safe_name(task.name)}.log"
            environment = os.environ.copy()
            environment.update({str(k): str(v) for k, v in task.environment.items()})
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            environment.setdefault("PYTHONUNBUFFERED", "1")
            with print_lock:
                print(
                    f"[GPU queue] start GPU {gpu_id}: {task.name} | log={log_path}",
                    flush=True,
                )
            returncode = -1
            error = None
            try:
                with log_path.open("w", encoding="utf-8", errors="replace") as stream:
                    process = subprocess.run(
                        list(task.command),
                        cwd=cwd,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                returncode = int(process.returncode)
            except Exception as exception:  # pragma: no cover - operating-system failure
                error = repr(exception)
            status = "completed" if returncode == 0 and error is None else "failed"
            record = {
                "index": index,
                "task": task.name,
                "gpu": gpu_id,
                "status": status,
                "returncode": returncode,
                "started_at": stamp,
                "elapsed_seconds": time.time() - started,
                "command": list(task.command),
                "log": str(log_path),
            }
            if error is not None:
                record["error"] = error
            with records_lock:
                records.append(record)
                _write_summary(summary_path, sorted(records, key=lambda item: item["index"]))
            with print_lock:
                print(
                    f"[GPU queue] {status} GPU {gpu_id}: {task.name} "
                    f"({record['elapsed_seconds']:.1f}s)",
                    flush=True,
                )
            pending.task_done()
            if status == "failed" and not continue_on_error:
                stop_dispatch.set()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}", daemon=False)
        for gpu in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = sorted(records, key=lambda item: item["index"])
    if any(record["status"] == "failed" for record in records) and not continue_on_error:
        raise GpuQueueError(records)
    if len(records) != len(tasks):
        skipped = len(tasks) - len(records)
        raise GpuQueueError(
            records
            + [
                {
                    "index": len(records),
                    "task": f"{skipped} undispatched task(s)",
                    "status": "failed",
                    "returncode": -1,
                }
            ]
        )
    return records
