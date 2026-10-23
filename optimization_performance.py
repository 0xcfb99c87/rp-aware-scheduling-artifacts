#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
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
from statistics import median
from pathlib import Path

from gen_starting_states import (
    BUNDLED_NODE,
    CURVES,
    METHODS,
    StepFailed,
    count_cycles,
)

SCRIPT_ROOT = Path(__file__).resolve().parent
CRYPTOPT_ROOT = SCRIPT_ROOT / "CryptOpt"
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 4  # four hours
# Random Bet-and-Run initialization (CryptOpt default) vs. a single run from
# the informed RP-aware ordering.
STRATEGIES = [
    {"scheduler": "default", "single": False},
    {"scheduler": "pressure-minimized", "single": True},
]
# One row per (pressure-minimized, default) run pair; *_pm/*_br suffixes refer
# to the pressure-minimized single run and the default Bet-and-Run run.
SUMMARY_CSV_HEADER = [
    "curve",
    "method",
    "rerun",
    "seed_pm",
    "seed_br",
    "cycles_cc_baseline_pm",
    "cycles_cc_baseline_br",
    "cycles_pm",
    "cycles_br",
    "final_ratio_traj",
    "ttt_pm",  # Number of mutations needed to reach 98% of B&R run's final performance, for the PM strategy.
    "ttt_br",  # Number of mutations needed to reach 98% of B&R run's final performance, for the B&R strategy.
    "start_equiv",
    "upfront_pm",
    "upfront_br",
]

# Trajectory metrics, computed from each run's .dat file. Layout: one row per
# run, one sample every 10 mutations, "?" pads shorter rows. The
# pressure-minimized file holds a single run. The default file holds the 10
# bets, followed by the continued winning bet *including* its bet prefix, i.e.,
# on the winner's lineage axis rather than the total budget axis. The budget
# constants mirror the paper's configuration (-e 100k, CryptOpt's default of
# 10 bets with 2% of the budget each).
MUTATIONS_PER_SAMPLE = 10
N_BETS = 10
TOTAL_BUDGET = 100_000
BET_BUDGET = TOTAL_BUDGET * 2 // 100
BET_STAGE = N_BETS * BET_BUDGET
# Final speedup = median over the last 100 samples (1k mutations).
FINAL_WINDOW = 100
# Running-median window (in samples) to suppress measurement noise before
# computing threshold crossings. Must be odd.
SMOOTH_K = 11
CONV = 0.98
BUDGET = range(MUTATIONS_PER_SAMPLE, TOTAL_BUDGET + 1, MUTATIONS_PER_SAMPLE)

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


def make_run_id(curve: str, method: str, scheduler: str, seed: int) -> str:
    return f"{curve}--{method}--{scheduler}--seed{seed}"


def build_runs(
    curves: list[str], methods: list[str], seed: int, base_dir: str
) -> list[dict]:
    runs = []
    for curve in curves:
        for method in methods:
            for strategy in STRATEGIES:
                run_id = make_run_id(curve, method, strategy["scheduler"], seed)
                runs.append(
                    {
                        "curve": curve,
                        "method": method,
                        "seed": seed,
                        "scheduler": strategy["scheduler"],
                        "single": strategy["single"],
                        "resultDir": os.path.abspath(os.path.join(base_dir, run_id)),
                    }
                )
    return runs


def cli_args(run: dict, proof: bool, evals: str | None) -> list[str]:
    args = [
        "--curve",
        run["curve"],
        "--method",
        run["method"],
        "--seed",
        str(run["seed"]),
        "--schedulingAlgorithm",
        run["scheduler"],
    ]
    if run["single"]:
        args += ["--single"]
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
        str(cpu_id),
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


def read_runs(dat_path: Path) -> list[list[float]]:
    runs = []
    for line in dat_path.read_text().splitlines():
        xs = [float(x) for x in line.split() if x != "?"]
        if xs:
            runs.append(xs)
    return runs


def med3(a: float, b: float, c: float) -> float:
    return sorted((a, b, c))[1]


# TODO: we can use normal med probably?
def runmed(x: list[float], k: int) -> list[float]:
    h = k // 2
    n = len(x)
    y = list(x)
    for i in range(h, n - h):
        y[i] = sorted(x[i - h : i + h + 1])[h]
    # R's smoothEnds() on the running median.
    sm = list(y)
    if h >= 2:
        sm[1] = med3(y[0], y[1], y[2])
        sm[n - 2] = med3(y[n - 1], y[n - 2], y[n - 3])
        for i in range(3, h + 1):
            if 2 * i > n:
                break
            sm[i - 1] = sorted(y[: 2 * i - 1])[i - 1]
            sm[n - i] = sorted(y[n + 1 - 2 * i :])[i - 1]
    sm[0] = med3(y[0], sm[1], 3 * sm[1] - 2 * sm[2])
    sm[n - 1] = med3(y[n - 1], sm[n - 2], 3 * sm[n - 2] - 2 * sm[n - 3])
    return sm


def smooth_run(x: list[float]) -> list[float]:
    # RLS never accepts a worse candidate, so dips in the raw series are
    # measurement noise.
    out, best = [], -math.inf
    for v in runmed(x, SMOOTH_K):
        best = max(best, v)
        out.append(best)
    return out


def final_of(x: list[float]) -> float:
    return median(x[-FINAL_WINDOW:])


def first_reach(y: list[float], target: float) -> float:
    return next((t for t, v in zip(BUDGET, y) if v >= target), math.inf)


def br_total_axis(bets: list[list[float]], final: list[float]) -> list[float]:
    bets = [smooth_run(b) for b in bets]
    final = smooth_run(final)
    min_len = min(len(b) for b in bets)
    out = []
    for t in BUDGET:
        if t <= BET_STAGE:
            k = max(1, min(t // N_BETS // MUTATIONS_PER_SAMPLE, min_len))
            out.append(max(b[k - 1] for b in bets))
        else:
            k = (BET_BUDGET + t - BET_STAGE) // MUTATIONS_PER_SAMPLE
            out.append(final[min(len(final), max(1, k)) - 1])
    return out


def trajectory_metrics(pm_dat: Path, br_dat: Path) -> dict[str, float]:
    pm_raw = read_runs(pm_dat)[0]
    rnd = read_runs(br_dat)
    bets, br_raw = rnd[:N_BETS], rnd[-1]
    pm_smooth = smooth_run(pm_raw)
    pm_t = [pm_smooth[min(i, len(pm_smooth) - 1)] for i in range(len(BUDGET))]
    br_t = br_total_axis(bets, br_raw)
    pm_final, br_final = final_of(pm_raw), final_of(br_raw)
    pm_start = median(pm_raw[:5])
    target = CONV * br_final
    return {
        "final_ratio_traj": pm_final / br_final,
        "ttt_pm": first_reach(pm_t, target),
        "ttt_br": first_reach(br_t, target),
        # Mutations B&R needs to match PM's unoptimized starting state.
        "start_equiv": first_reach(br_t, pm_start),
        # Share of the final speedup already present at the start (B&R: best bet).
        "upfront_pm": pm_start / pm_final,
        "upfront_br": max(median(b[:5]) for b in bets) / br_final,
    }


def csv_value(v) -> str:
    """Format like R's write.csv, which the paper's data files use."""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, float):
        return "Inf" if v == math.inf else f"{v:.15g}"
    return str(v)


def measure_run(run: dict, label: str, force: bool) -> tuple[float, float, Path] | None:
    run_id = os.path.basename(run["resultDir"])
    asms = sorted(Path(run["resultDir"]).rglob("seed*_ratio*.asm"))
    dats = sorted(Path(run["resultDir"]).rglob("seed*.dat"))
    if len(asms) != 1 or len(dats) != 1:
        log(
            f"{label} Skipping {run_id}: expected one final asm and .dat, found {len(asms)} and {len(dats)}"
        )
        return None
    asm_path = asms[0]
    log(f"{label} Measuring cycles: {run_id}")
    try:
        asm_cycles, cc_baseline_cycles = count_cycles(
            asm_path, str(BUNDLED_NODE), asm_path.with_suffix(".cycles"), force
        )
    except StepFailed as e:
        log(f"{label} Failed to measure {run_id}: {e}")
        return None
    return asm_cycles, cc_baseline_cycles, dats[0]


def write_summary(runs: list[dict], csv_out: Path, force: bool, rerun: bool) -> None:
    pairs: dict[tuple[str, str], dict[str, dict]] = {}
    for run in runs:
        pairs.setdefault((run["curve"], run["method"]), {})[run["scheduler"]] = run
    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(SUMMARY_CSV_HEADER)
        for i, key in enumerate(sorted(pairs), start=1):
            label = f"[{i}/{len(pairs)}]"
            pm_run = pairs[key]["pressure-minimized"]
            br_run = pairs[key]["default"]
            pm = measure_run(pm_run, label, force)
            br = measure_run(br_run, label, force)
            if pm is None or br is None:
                log(f"{label} Skipping {' '.join(key)}: incomplete run pair")
                continue
            metrics = trajectory_metrics(pm[2], br[2])
            row = [
                *key,
                rerun,
                pm_run["seed"],
                br_run["seed"],
                pm[1],
                br[1],
                pm[0],
                br[0],
                *(metrics[c] for c in SUMMARY_CSV_HEADER[9:]),
            ]
            writer.writerow([csv_value(v) for v in row])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Fixed seed to use for every curve/method/strategy combination. If unset, chooses randomly.",
    )
    parser.add_argument(
        "--curves",
        default=",".join(CURVES),
        help="Comma-separated list of curves to run (default: all)",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated list of methods to run (default: mul,square)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        dest="num_workers",
        type=int,
        default=3,
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
        help="override the number of evals to run per optimization",
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
    parser.add_argument(
        "--force-measure",
        dest="force_measure",
        action="store_true",
        default=False,
        help="re-measure final cycle counts even if cached .cycles files exist",
    )
    args = parser.parse_args()
    args.curves = [c.strip() for c in args.curves.split(",") if c.strip()]
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for c in args.curves:
        if c not in CURVES:
            parser.error(f"unknown curve {c!r}; known curves: {', '.join(CURVES)}")
    for m in args.methods:
        if m not in METHODS:
            parser.error(f"unknown method {m!r}; known methods: {', '.join(METHODS)}")
    return args


def main() -> None:
    args = parse_args()
    base_dir = os.path.join(args.base_dir, "artifacts_optimization_comparison")
    os.makedirs(base_dir, exist_ok=True)
    seed = args.seed if args.seed is not None else random.randint(1, 9999)
    runs = build_runs(args.curves, args.methods, seed, base_dir)
    if not runs:
        fatal("No curve/method combinations selected")
    random.shuffle(runs)

    print(f"Seed: {seed}")
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

    # Close the "channel", tells the workers to stop awaiting more requests.
    for _ in threads:
        jobs.put(None)

    for t in threads:
        t.join()

    # Filter out state files for bets, all info we need is contained in final
    # run instance.
    for bet in filter(
        lambda p: os.path.isfile(p)
        and "0000000000"  # Our pseudo-random seeds are <= 10000, so leading zeroes tell us that the state file is the chosen bet.
        not in p.as_posix(),
        Path(base_dir).rglob("seed*"),
    ):
        os.remove(bet)

    print("All runs completed.")
    print()

    # Measure only once all optimizations are done, so that measurements don't
    # interfere with running optimizations.
    summary_csv = Path(base_dir) / f"summary_seed{seed}.csv"
    write_summary(runs, summary_csv, args.force_measure, False)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
