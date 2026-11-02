"""Dynamic GPU task queue behavior without requiring CUDA hardware."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from src.utils.gpu_queue import GpuTask, parse_gpu_ids, run_gpu_queue


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


if __name__ == "__main__":
    unittest.main()
