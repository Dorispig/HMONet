"""Dynamic GPU task queue behavior without requiring CUDA hardware."""

import os
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
from contextlib import redirect_stdout
from unittest.mock import patch

from src.utils.gpu_queue import (
    GpuQueueInterrupted,
    GpuTask,
    parse_gpu_ids,
    run_gpu_queue,
)
import pytest


class TestGpuQueue:
    def test_gpu_id_forms_and_validation(self):
        assert (parse_gpu_ids(["0", "1", "2,3"])) == ([0, 1, 2, 3])
        with pytest.raises(ValueError):
            parse_gpu_ids(["0", "0"])
        with pytest.raises(ValueError):
            parse_gpu_ids(["-1"])

    def test_dynamic_workers_isolate_visible_gpu_and_write_logs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = [
                GpuTask(
                    name=f"task_{index}",
                    command=(
                        sys.executable,
                        "-c",
                        "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])",
                    ),
                )
                for index in range(4)
            ]
            records = run_gpu_queue(
                tasks,
                [2, 5],
                cwd=root,
                log_root=root / "logs",
                validate_devices=False,
            )
            assert (len(records)) == (4)
            assert all(record["status"] == "completed" for record in records)
            assert ({record["gpu"] for record in records}) == ({2, 5})
            for record in records:
                assert (Path(record["log"]).read_text(encoding="utf-8").strip()) == (
                    str(record["gpu"])
                )

    def test_dry_run_does_not_create_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "not-created"
            records = run_gpu_queue(
                [GpuTask("dry", (sys.executable, "-c", "pass"))],
                [0],
                cwd=root,
                log_root=output,
                dry_run=True,
            )
            assert (records[0]["status"]) == ("dry_run")
            assert not output.exists()

    def test_epoch_logs_are_complete_but_terminal_is_periodic(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = (
                sys.executable,
                "-c",
                "print('Epoch 1/1000: training_loss=1, mse=1'); "
                "print('Epoch 499/1000: training_loss=2, mse=2'); "
                "print('Epoch 500/1000: training_loss=3, mse=3'); "
                "print('Epoch 1000/1000: training_loss=4, mse=4')",
            )
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                records = run_gpu_queue(
                    [GpuTask("epochs", command)],
                    [0],
                    cwd=root,
                    log_root=root / "logs",
                    validate_devices=False,
                    terminal_interval=500,
                )
            log = Path(records[0]["log"]).read_text(encoding="utf-8")
            assert ("Epoch 499/1000") in (log)
            rendered = terminal.getvalue()
            assert ("Epoch 1/1000") in (rendered)
            assert ("Epoch 499/1000") not in (rendered)
            assert ("Epoch 500/1000") in (rendered)
            assert ("Epoch 1000/1000") in (rendered)

    def test_readable_run_log_directory_is_reused_and_old_logs_are_cleared(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("dataset_a/experiment/seed_7/paper_v1/train_evaluate.log")
            log_directory = root / "logs" / relative.parent
            log_directory.mkdir(parents=True)
            (log_directory / "training.log").write_text(
                "stale training", encoding="utf-8"
            )
            (log_directory / "evaluation.log").write_text(
                "stale evaluation", encoding="utf-8"
            )
            task = GpuTask(
                "grouped",
                (sys.executable, "-c", "print('first run')"),
                log_relative_path=relative,
                clear_log_directory=True,
                metadata={"training_fingerprint": "abc123"},
            )
            first = run_gpu_queue(
                [task], [0], cwd=root, log_root=root / "logs", validate_devices=False
            )[0]
            assert not (log_directory / "training.log").exists()
            assert not (log_directory / "evaluation.log").exists()
            assert (Path(first["log"]).read_text(encoding="utf-8").strip()) == (
                "first run"
            )
            assert (first["metadata"]["training_fingerprint"]) == ("abc123")

            repeated = GpuTask(
                "grouped",
                (sys.executable, "-c", "print('replacement run')"),
                log_relative_path=relative,
                clear_log_directory=True,
                metadata={"training_fingerprint": "abc123"},
            )
            second = run_gpu_queue(
                [repeated],
                [0],
                cwd=root,
                log_root=root / "logs",
                validate_devices=False,
            )[0]
            assert (first["log"]) == (second["log"])
            content = Path(second["log"]).read_text(encoding="utf-8")
            assert ("replacement run") in (content)
            assert ("first run") not in (content)

    def test_grouped_log_path_cannot_escape_queue_root(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with pytest.raises(ValueError, match="relative to the queue root"):
                run_gpu_queue(
                    [
                        GpuTask(
                            "unsafe",
                            (sys.executable, "-c", "pass"),
                            log_relative_path="../outside.log",
                        )
                    ],
                    [0],
                    cwd=root,
                    log_root=root / "logs",
                    validate_devices=False,
                )

    def test_interrupt_cancels_running_and_pending_tasks_even_when_continuing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "second-task-started"
            tasks = [
                GpuTask(
                    "running",
                    (sys.executable, "-c", "import time; time.sleep(30)"),
                ),
                GpuTask(
                    "pending",
                    (
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).touch()",
                    ),
                ),
            ]
            installed = {}

            def capture_handler(number, handler):
                installed[number] = handler

            def interrupt_later():
                while signal.SIGINT not in installed:
                    time.sleep(0.01)
                time.sleep(0.1)
                installed[signal.SIGINT](signal.SIGINT, None)

            import signal

            interrupter = threading.Thread(target=interrupt_later)
            with patch(
                "src.utils.gpu_queue.signal.signal", side_effect=capture_handler
            ):
                interrupter.start()
                with pytest.raises(GpuQueueInterrupted) as raised:
                    run_gpu_queue(
                        tasks,
                        [0],
                        cwd=root,
                        log_root=root / "logs",
                        validate_devices=False,
                        continue_on_error=True,
                    )
                interrupter.join()
            statuses = {item["task"]: item["status"] for item in raised.value.records}
            assert (statuses["running"]) == ("interrupted")
            assert (statuses["pending"]) == ("cancelled")
            assert not marker.exists()
