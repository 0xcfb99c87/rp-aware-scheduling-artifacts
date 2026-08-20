#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 4  # four hours

SCRIPT_ROOT = Path(__file__).resolve().parent
CRYPTOPT_ROOT = SCRIPT_ROOT / "CryptOpt"

print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def fatal(msg: str) -> None:
    print(f"bench.py: {msg}", file=sys.stderr)
    sys.exit(1)


def format_elapsed(seconds: float) -> str:
    total = round(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if h or m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return "".join(parts)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_run_id(state: dict) -> str:
    """Derive a result-dir name from a readState JSON's embedded parsedArgs,
    so it stays unique per (curve, method, scheduling algorithm, seed)."""
    parsed = state.get("parsedArgs") or {}
    curve = parsed.get("curve", "unknown-curve")
    method = parsed.get("method", "unknown-method")
    seed = parsed.get("seed", "unknown-seed")
    algo = parsed.get("schedulingAlgorithm", "default")
    if algo == "pressure-minimized":
        algo = f"pressure-minimized-la{parsed.get('pmLookahead', '?')}"
    return f"{curve}--{method}--{algo}--seed{seed}"


def discover_runs(root: str, base_dir: str) -> list[dict]:
    runs = []
    for path in sorted(Path(root).rglob("*.json")):
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"bench.py: skipping {path}: {e}", file=sys.stderr)
            continue
        if "parsedArgs" not in state:
            print(
                f"bench.py: skipping {path}: no parsedArgs (not a CryptOpt state file)",
                file=sys.stderr,
            )
            continue
        runs.append(
            {
                "readState": str(path.absolute()),
                "resultDir": os.path.join(base_dir, make_run_id(state)),
            }
        )
    return runs


def cli_args(run: dict, proof: bool, evals: str | None) -> list[str]:
    args = ["--single", "--readState", run["readState"]]
    if evals:
        args += ["--evals", evals]
    if not proof:
        args.append("--no-proof")
    args += ["--resultDir", run["resultDir"]]
    return args


class AtomicCounter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


def process_run(
    cpu_id: int,
    run: dict,
    total: int,
    completed: AtomicCounter,
    opts: argparse.Namespace,
) -> None:
    run_id = os.path.basename(run["resultDir"])

    if os.path.exists(run["resultDir"]):
        count = completed.increment()
        log(f"[CPU {cpu_id}] [{count}/{total}] Skipping (already exists): {run_id}")
        return

    # Sibling temp dir so the final rename stays on the same filesystem
    # (avoids cross-device link errors from os.rename).
    parent = os.path.dirname(run["resultDir"]) or "."
    try:
        tmp_dir = tempfile.mkdtemp(prefix=".tmp-cryptopt-bench-", dir=parent)
    except OSError as e:
        log(f"[CPU {cpu_id}] Failed to create temp dir for {run_id}: {e}")
        return

    tmp_run = dict(run)
    tmp_run["resultDir"] = tmp_dir

    cmd = [
        "taskset",
        "-c",
        str(2 * cpu_id),
        (CRYPTOPT_ROOT / "CryptOpt").as_posix(),
        *cli_args(tmp_run, opts.proof, opts.evals),
    ]

    env = {}
    if os.environ.get("PATH"):
        env["PATH"] = os.environ["PATH"]
    if os.environ.get("CC"):
        env["CC"] = os.environ["CC"] or "clang"
    env = env or None

    log(f"[CPU {cpu_id}] {now_str()}: Running: {run_id}")
    if opts.verbose:
        log(f"[CPU {cpu_id}] {now_str()}: Args: {' '.join(cmd)}")

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,  # own process group, so a timeout can kill the whole tree
    )
    try:
        _, stderr = proc.communicate(timeout=DEFAULT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        count = completed.increment()
        log(f"[CPU {cpu_id}] [{count}/{total}] timed out running {run_id}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    if proc.returncode != 0:
        count = completed.increment()
        log(
            f"[CPU {cpu_id}] [{count}/{total}] Error running {run_id}: exit status {proc.returncode}: {stderr.decode(errors='replace')}"
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # Process exited cleanly: atomically promote the temp dir to the final
    # location. If it's taken (e.g. a duplicate trial), append _1, _2, ...
    final_dir = run["resultDir"]
    suffix = 1
    while True:
        try:
            os.rename(tmp_dir, final_dir)
            break
        except OSError as e:
            if not os.path.exists(final_dir):
                log(
                    f"[CPU {cpu_id}] Failed to move result dir to {run['resultDir']}: {e}"
                )
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return
            final_dir = f"{run['resultDir']}_{suffix}"
            suffix += 1

    count = completed.increment()
    elapsed = format_elapsed(time.monotonic() - start)
    log(
        f"[CPU {cpu_id}] [{count}/{total}] Completed {os.path.basename(final_dir)} (took {elapsed})"
    )


def worker(
    cpu_id: int,
    jobs: "queue.Queue[dict | None]",
    total: int,
    completed: AtomicCounter,
    opts: argparse.Namespace,
) -> None:
    while True:
        run = jobs.get()
        try:
            if run is None:
                return
            process_run(cpu_id, run, total, completed, opts)
        finally:
            jobs.task_done()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continue CryptOpt optimization from each starting-state JSON found under a directory, in parallel.",
        usage="%(prog)s [flags] [root]",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="./artifacts",
        help="directory to recursively search for readState JSON files",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        dest="num_workers",
        type=int,
        default=(os.cpu_count() or 2) // 2,
        help="number of parallel jobs (CPUs to use)",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        dest="base_dir",
        default=".",
        help="relative path to where results should be stored",
    )
    parser.add_argument(
        "-e",
        "--evals",
        dest="evals",
        default=None,
        help="override the number of evals to run from each starting state",
    )
    parser.add_argument(
        "--proof",
        dest="proof",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to enable/disable proofs of final assembly",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="enable debug output",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    base_dir = os.path.join(args.base_dir, "artifacts_optimization_comparison")
    os.makedirs(base_dir, exist_ok=True)

    runs = discover_runs(args.root, base_dir)
    if not runs:
        fatal(f"No readState JSON files found under {args.root}")
    random.shuffle(runs)

    print(f"Root: {args.root}")
    print(f"Total runs: {len(runs)}")
    print(f"Workers: {args.num_workers} (CPUs 0-{args.num_workers - 1})")
    print(f"Base dir: {base_dir}")
    print(f"Verbose: {str(args.verbose).lower()}")
    print(f"Evals: {args.evals or '(CryptOpt default)'}")
    print(f"Proof-checking: {str(args.proof).lower()}")
    print()

    jobs: "queue.Queue[dict | None]" = queue.Queue(maxsize=1)
    completed = AtomicCounter()

    threads = [
        threading.Thread(
            target=worker, args=(i, jobs, len(runs), completed, args), daemon=True
        )
        for i in range(args.num_workers)
    ]
    for t in threads:
        t.start()

    for run in runs:
        jobs.put(run)
        time.sleep(0.1)

    for _ in threads:
        jobs.put(
            None
        )  # sentinel: tells each worker to stop, playing the role of close(jobs)

    for t in threads:
        t.join()

    print("All runs completed.")


if __name__ == "__main__":
    main()
