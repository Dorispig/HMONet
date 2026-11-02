"""Dynamic GPU task queue behavior without requiring CUDA hardware."""

import os
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.utils.gpu_queue import (
    GpuQueueInterrupted,
    GpuTask,
    parse_gpu_ids,
    run_gpu_queue,
)


class GpuQueueTests(unittest.TestCase):
    def test_gpu_id_forms_and_validation(self):
        self.assertEqual(parse_gpu_ids(["0", "1", "2,3"]), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            parse_gpu_ids(["0", "0"])
        with self.assertRaises(ValueError):
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
            self.assertEqual(len(records), 4)
            self.assertTrue(all(record["status"] == "completed" for record in records))
            self.assertEqual({record["gpu"] for record in records}, {2, 5})
            for record in records:
                self.assertEqual(
                    Path(record["log"]).read_text(encoding="utf-8").strip(),
                    str(record["gpu"]),
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
            self.assertEqual(records[0]["status"], "dry_run")
            self.assertFalse(output.exists())

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
            self.assertIn("Epoch 499/1000", log)
            rendered = terminal.getvalue()
            self.assertIn("Epoch 1/1000", rendered)
            self.assertNotIn("Epoch 499/1000", rendered)
            self.assertIn("Epoch 500/1000", rendered)
            self.assertIn("Epoch 1000/1000", rendered)

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
            with patch("src.utils.gpu_queue.signal.signal", side_effect=capture_handler):
                interrupter.start()
                with self.assertRaises(GpuQueueInterrupted) as raised:
                    run_gpu_queue(
                        tasks,
                        [0],
                        cwd=root,
                        log_root=root / "logs",
                        validate_devices=False,
                        continue_on_error=True,
                    )
                interrupter.join()
            statuses = {item["task"]: item["status"] for item in raised.exception.records}
            self.assertEqual(statuses["running"], "interrupted")
            self.assertEqual(statuses["pending"], "cancelled")
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
