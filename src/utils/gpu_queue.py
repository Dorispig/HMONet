"""Dynamic one-process-per-GPU scheduling for independent project tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Iterable, Sequence

from src.utils.progress import wrapped_print


@dataclass(frozen=True)
class GpuTask:
    """One independent subprocess that occupies exactly one visible GPU."""

    name: str
    command: tuple[str, ...] | list[str]
    environment: dict[str, str] = field(default_factory=dict)
    log_relative_path: str | Path | None = None
    clear_log_directory: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class GpuQueueError(RuntimeError):
    """Raised after one or more dynamically scheduled tasks fail."""

    def __init__(self, records):
        self.records = records
        failed = [record for record in records if record["status"] == "failed"]
        rendered = ", ".join(
            f"{record['task']} (exit={record['returncode']})" for record in failed
        )
        super().__init__(f"GPU task queue failed: {rendered}")


class GpuQueueInterrupted(SystemExit):
    """Raised after SIGINT/SIGTERM cancels the whole queue and its children."""

    def __init__(self, records, signal_number):
        self.records = records
        self.signal_number = signal_number
        super().__init__(128 + int(signal_number))


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


def _task_log_path(log_root: Path, index: int, task: GpuTask) -> Path:
    """Resolve one task log while preventing escape from the queue log root."""
    if task.log_relative_path is None:
        relative = Path(f"{index:04d}_{_safe_name(task.name)}.log")
    else:
        relative = Path(task.log_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"Task log path must be relative to the queue root: {relative}"
            )
        if relative.suffix.lower() != ".log":
            raise ValueError(f"Task log path must end in .log: {relative}")
    resolved = (log_root / relative).resolve()
    try:
        resolved.relative_to(log_root)
    except ValueError as error:
        raise ValueError(f"Task log path escapes the queue root: {relative}") from error
    return resolved


def run_gpu_queue(
    tasks: Iterable[GpuTask],
    gpu_ids: Sequence[int],
    *,
    cwd: Path,
    log_root: Path,
    continue_on_error: bool = False,
    dry_run: bool = False,
    validate_devices: bool = True,
    terminal_interval: int = 500,
):
    """Run a dynamic FIFO queue with one long-lived worker per selected GPU.

    Each subprocess sees only its assigned physical GPU through
    ``CUDA_VISIBLE_DEVICES`` and therefore always uses ``cuda:0`` internally.
    A GPU is considered free when this queue has no running task assigned to it;
    processes outside this queue are deliberately ignored. When a subprocess
    exits, that worker immediately takes the next queued task.
    """
    tasks = list(tasks)
    gpu_ids = list(gpu_ids)
    if not gpu_ids:
        raise ValueError("gpu_ids cannot be empty")
    if int(terminal_interval) < 1:
        raise ValueError("terminal_interval must be positive")
    if not tasks:
        return []
    if validate_devices and not dry_run:
        validate_gpu_ids(gpu_ids)
    cwd = Path(cwd).resolve()
    log_root = Path(log_root).resolve()
    summary_path = log_root / "queue_summary.json"

    task_log_paths = [
        _task_log_path(log_root, index, task)
        for index, task in enumerate(tasks)
    ]
    duplicate_logs = {
        path for path in task_log_paths if task_log_paths.count(path) > 1
    }
    if duplicate_logs:
        raise ValueError(
            "GPU queue tasks must have distinct log files: "
            + ", ".join(str(path) for path in sorted(duplicate_logs))
        )

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
                "log": str(task_log_paths[index]),
            }
            if task.metadata:
                record["metadata"] = dict(task.metadata)
            records.append(record)
            wrapped_print(
                f"[GPU queue dry-run] GPU {gpu} <- {task.name} "
                f"| log={task_log_paths[index]}",
            )
        return records

    log_root.mkdir(parents=True, exist_ok=True)

    pending = Queue()
    for index, task in enumerate(tasks):
        pending.put((index, task))
    records = []
    records_lock = threading.Lock()
    print_lock = threading.Lock()
    stop_dispatch = threading.Event()
    cancel_queue = threading.Event()
    active_processes = {}
    active_lock = threading.Lock()
    received_signal = [None]

    def signal_process_tree(process, signal_number, *, force=False):
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                if force:
                    process.kill()
                else:
                    process.terminate()
            else:
                os.killpg(
                    process.pid,
                    signal.SIGKILL if force else signal_number,
                )
        except (OSError, ProcessLookupError):
            pass

    def cancel_all(signal_number, _frame=None):
        if cancel_queue.is_set():
            return
        received_signal[0] = int(signal_number)
        cancel_queue.set()
        stop_dispatch.set()
        wrapped_print(
            f"\n[GPU queue] received signal {signal_number}; cancelling the "
            "entire queue and all running tasks...",
        )
        with active_lock:
            running = list(active_processes.values())
        for process in running:
            signal_process_tree(process, signal_number)

    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, cancel_all)

    def worker(gpu_id: int):
        while True:
            if cancel_queue.is_set():
                return
            if stop_dispatch.is_set() and not continue_on_error:
                return
            try:
                index, task = pending.get_nowait()
            except Empty:
                return
            started = time.time()
            stamp = datetime.now().isoformat(timespec="seconds")
            log_path = task_log_paths[index]
            environment = os.environ.copy()
            environment.update({str(k): str(v) for k, v in task.environment.items()})
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            environment.setdefault("PYTHONUNBUFFERED", "1")
            environment["HMONET_GPU_QUEUE"] = "1"
            with print_lock:
                wrapped_print(
                    f"[GPU queue] start GPU {gpu_id}: {task.name} | log={log_path}"
                )
            returncode = -1
            error = None
            recent_lines = deque(maxlen=30)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if task.clear_log_directory:
                    # One fingerprint directory represents one atomic run.  A
                    # repeated train/evaluate request replaces its old logs
                    # instead of leaving stale phase output beside the new run.
                    for stale_log in log_path.parent.glob("*.log"):
                        stale_log.unlink(missing_ok=True)
                with log_path.open("w", encoding="utf-8", errors="replace") as stream:
                    process = subprocess.Popen(
                        list(task.command),
                        cwd=cwd,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        start_new_session=(os.name != "nt"),
                        creationflags=(
                            subprocess.CREATE_NEW_PROCESS_GROUP
                            if os.name == "nt"
                            else 0
                        ),
                    )
                    with active_lock:
                        active_processes[gpu_id] = process
                    if cancel_queue.is_set():
                        signal_process_tree(
                            process,
                            received_signal[0] or signal.SIGTERM,
                        )
                    assert process.stdout is not None
                    echo_epoch_continuation = False
                    with process.stdout:
                        for line in process.stdout:
                            stream.write(line)
                            stream.flush()
                            clean = line.rstrip("\r\n")
                            recent_lines.append(clean)
                            match = re.match(r"^Epoch\s+(\d+)/(\d+):", clean)
                            if match:
                                epoch, total = map(int, match.groups())
                                echo_epoch_continuation = bool(
                                    epoch == 1
                                    or epoch == total
                                    or epoch % int(terminal_interval) == 0
                                )
                                if echo_epoch_continuation:
                                    with print_lock:
                                        wrapped_print(
                                            f"[GPU queue][GPU {gpu_id}]"
                                            f"[{task.name}] {clean}"
                                        )
                            elif echo_epoch_continuation and clean.startswith("  "):
                                with print_lock:
                                    wrapped_print(
                                        f"[GPU queue][GPU {gpu_id}]"
                                        f"[{task.name}] {clean.strip()}"
                                    )
                            else:
                                echo_epoch_continuation = False
                    returncode = int(process.wait())
            except Exception as exception:  # pragma: no cover - operating-system failure
                error = repr(exception)
            finally:
                with active_lock:
                    active_processes.pop(gpu_id, None)
            if cancel_queue.is_set():
                status = "interrupted"
            else:
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
            if task.metadata:
                record["metadata"] = dict(task.metadata)
            if error is not None:
                record["error"] = error
            with records_lock:
                records.append(record)
                _write_summary(summary_path, sorted(records, key=lambda item: item["index"]))
            with print_lock:
                wrapped_print(
                    f"[GPU queue] {status} GPU {gpu_id}: {task.name} "
                    f"({record['elapsed_seconds']:.1f}s)"
                )
                if status in {"failed", "interrupted"} and recent_lines:
                    wrapped_print(f"[GPU queue] last log lines for {task.name}:")
                    for line in recent_lines:
                        wrapped_print(f"  {line}", indent="    ")
            pending.task_done()
            if status == "failed" and not continue_on_error:
                stop_dispatch.set()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}", daemon=False)
        for gpu in gpu_ids
    ]
    for thread in threads:
        thread.start()
    cancellation_started = None
    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.1)
            if cancel_queue.is_set():
                cancellation_started = cancellation_started or time.time()
                if time.time() - cancellation_started >= 5.0:
                    with active_lock:
                        running = list(active_processes.values())
                    for process in running:
                        signal_process_tree(
                            process,
                            received_signal[0] or signal.SIGTERM,
                            force=True,
                        )
    finally:
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)
    records = sorted(records, key=lambda item: item["index"])
    if cancel_queue.is_set():
        while True:
            try:
                index, task = pending.get_nowait()
            except Empty:
                break
            records.append(
                {
                    "index": index,
                    "task": task.name,
                    "gpu": None,
                    "status": "cancelled",
                    "returncode": None,
                    "command": list(task.command),
                }
            )
            pending.task_done()
        records = sorted(records, key=lambda item: item["index"])
        _write_summary(summary_path, records)
        raise GpuQueueInterrupted(records, received_signal[0])
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
