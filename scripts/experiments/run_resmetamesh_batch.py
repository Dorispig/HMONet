"""Run the documented 2-D ResMetaMesh experiments and summarize failures."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = ROOT / "configs/2D/formal_external_commands.md"
LOG_ROOT = ROOT / "outputs/reports/2D/resmetamesh_batch"
ENTRY = "scripts/train/train_evaluate.py"
PHASES = ("train_and_evaluate", "compare")
STATE = re.compile(r"\btrain=([\w-]+)\s*\|\s*evaluate=([\w-]+)")
ERROR = re.compile(r"^(?:ERROR: |(?:[\w.]+(?:Error|Exception)|Exception): )")


def dataset_commands():
    commands = {}
    for line in COMMANDS.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith(f"python {ENTRY} ") or " --models resmetamesh " not in line:
            continue
        command = shlex.split(line)[1:]
        dataset = command[command.index("--dataset") + 1]
        if dataset in commands:
            raise ValueError(f"Duplicate ResMetaMesh command for {dataset}")
        commands[dataset] = command
    if len(commands) != 9:
        raise ValueError(f"Expected nine ResMetaMesh commands in {COMMANDS}, found {len(commands)}")
    return commands


def stage_command(command, phase, device):
    command = list(command)
    if phase == "train_and_evaluate":
        command[command.index("--evaluate-methods") + 1] = "resmetamesh"
    else:
        command += ["--evaluate-only", "--analysis-only"]
    if device:
        command += ["--device", device]
    return command


def run_stage(command, log):
    training = evaluation = None
    reason = ""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "TQDM_DISABLE": "1"}
    with log.open("w", encoding="utf-8") as stream:
        stream.write(shlex.join([sys.executable, *command]) + "\n")
        try:
            child = subprocess.Popen(
                [sys.executable, *command], cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except OSError as error:
            reason = str(error)
            print(reason, file=stream, flush=True)
            return {"status": "failed", "error": reason, "log": str(log)}
        try:
            with child.stdout:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                    if match := STATE.search(line):
                        training, evaluation = match.groups()
                    if ERROR.match(line.strip()):
                        reason = line.strip()
            code = child.wait()
        except KeyboardInterrupt:
            if child.poll() is None:
                child.terminate()
                child.wait()
            raise
    return {
        "status": "failed" if code else "completed",
        "exit_code": code, "training": training, "evaluation": evaluation,
        "error": (reason or f"Child process exited with status {code}") if code else "",
        "log": str(log),
    }


def main(argv=None):
    commands = dataset_commands()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(commands), help="Default: all nine datasets.")
    parser.add_argument("--device", help="Override device selection, for example cuda:0.")
    parser.add_argument("--dry-run", action="store_true", help="Show the commands without starting tasks.")
    args = parser.parse_args(argv)
    selected = list(dict.fromkeys(args.datasets)) if args.datasets else list(commands)
    if args.dry_run:
        for dataset in selected:
            for phase in PHASES:
                print(shlex.join([sys.executable, *stage_command(commands[dataset], phase, args.device)]))
        return 0

    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = LOG_ROOT / now
    directory.mkdir(parents=True, exist_ok=False)
    history_path = directory / "history.jsonl"
    print(f"Batch logs: {directory}", flush=True)
    records = []
    with history_path.open("w", encoding="utf-8") as history:
        for dataset in selected:
            previous_failed = False
            for phase in PHASES:
                if previous_failed:
                    result = {"status": "skipped", "error": "Training or single-model evaluation failed"}
                else:
                    print(f"\n[{dataset}] {phase}", flush=True)
                    result = run_stage(
                        stage_command(commands[dataset], phase, args.device),
                        directory / f"{dataset}__{phase}.log",
                    )
                    if result["status"] == "failed":
                        print(f"[failed; continuing] {dataset}: {result['error']}", flush=True)
                        previous_failed = True
                record = {"dataset": dataset, "phase": phase, **result}
                records.append(record)
                history.write(json.dumps(record, ensure_ascii=False) + "\n")
                history.flush()

    print("\n=== ResMetaMesh batch completion summary ===")
    for record in records:
        detail = ""
        if record.get("training") is not None:
            detail = f" | train={record['training']} | evaluate={record['evaluation']}"
        if record["error"]:
            detail += f" | {record['error']}"
        print(f"{record['dataset']} | {record['phase']} | {record['status']}{detail}")
    failures = [record for record in records if record["status"] == "failed"]
    print(f"Failed stages: {len(failures)}. History: {history_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
