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

print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def fatal(msg: str) -> None:
    print(f"bench.py: {msg}", file=sys.stderr)
    sys.exit(1)


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
    "seed",
    "cycles_cc_baseline_pm",
    "cycles_cc_baseline_br",
    "cycles_pm",
    "cycles_br",
    "n_pm",  # Number of mutations needed to reach 98% of B&R run's final performance, for the PM strategy.
    "n_br",  # Number of mutations needed to reach 98% of B&R run's final performance, for the B&R strategy.
    "start_equiv",
    "upfront_pm",
    "upfront_br",
]

MUTATIONS_PER_SAMPLE = 10
N_BETS = 10
# Share of the total budget spread across the bets (CryptOpt default betRatio).
BET_RATIO = 0.2
# CryptOpt samples at mutations 10, 20, ..., so each bet needs more than 10
# mutations to record a sample.
MIN_EVALS = round(N_BETS * MUTATIONS_PER_SAMPLE / BET_RATIO)
# Final speedup = median over the last 100 samples (1k mutations) to smooth out per-sample noise.
FINAL_WINDOW = 100
# Running-median window (in samples), used to compute when each run convergences to 98% of final speedup. Same smoothing deal.
SMOOTH_K = 11
# Convergence threshold (98%).
CONV = 0.98


class Budget:
    def __init__(self, n_eval: int) -> None:
        self.n_eval = n_eval
        self.bet_budget = round(n_eval * BET_RATIO / N_BETS)
        self.bet_stage = N_BETS * self.bet_budget
        self.axis = range(MUTATIONS_PER_SAMPLE, n_eval + 1, MUTATIONS_PER_SAMPLE)

    @classmethod
    def from_single_run(cls, samples: int) -> Budget:
        # CryptOpt samples at mutations 10, 20, ..., n_eval - 10.
        return cls((samples + 1) * MUTATIONS_PER_SAMPLE)


# Used to orchestrate run allocation across threads
class AtomicCounter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


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
    curves: list[str], methods: list[str], seed: int, out_dir: str
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
                        "resultDir": os.path.abspath(os.path.join(out_dir, run_id)),
                    }
                )
    return runs


def cli_args(run: dict, proof: bool, evals: int) -> list[str]:
    args = [
        "--curve",
        run["curve"],
        "--method",
        run["method"],
        "--seed",
        str(run["seed"]),
        "--schedulingAlgorithm",
        run["scheduler"],
        "--evals",
        str(evals),
    ]
    if run["single"]:
        args += ["--single"]
    if not proof:
        args.append("--no-proof")
    args += ["--resultDir", run["resultDir"]]
    return args


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

    # Write optimization progress to temp dir so that tasks broken off in the
    # middle of script execution can be identified and cleaned up.
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
    return [
        [float(x) for x in line.split() if x != "?"]
        for line in dat_path.read_text().splitlines()
    ]


# Compute running median over optimization ratios, to help smooth out per-sample noise.
# Follows R's runmed logic:
#   - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/runmed.html
def runmed(samples: list[float], window: int) -> list[float]:
    n_samples = len(samples)
    # Shrink window to largest odd value <= n_samples. If n is odd, then (n % 2) - 1 = 0, else 1.
    window = min(window, n_samples + (n_samples % 2) - 1)
    half_window = window // 2
    assert half_window > 0
    medians = [
        (
            median(samples[pos - half_window : pos + half_window + 1])
            # Check for pos being far away enough from both ends of the list to allow the full window size.
            if half_window <= pos < n_samples - half_window
            else sample
        )
        for pos, sample in enumerate(samples)
    ]
    # Replicate R's `smoothEnds` (https://stat.ethz.ch/R-manual/R-devel/library/stats/html/smoothEnds.html)
    smoothed = list(medians)
    for dist in range(1, half_window):
        smoothed[dist] = median(medians[: 2 * dist + 1])
        smoothed[-1 - dist] = median(medians[-2 * dist - 1 :])
    smoothed[0] = median([medians[0], smoothed[1], 3 * smoothed[1] - 2 * smoothed[2]])
    smoothed[-1] = median(
        [medians[-1], smoothed[-2], 3 * smoothed[-2] - 2 * smoothed[-3]]
    )
    return smoothed


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


def first_reach(budget: Budget, y: list[float], target: float) -> float:
    return next((t for t, v in zip(budget.axis, y) if v >= target), math.inf)


def br_total_axis(
    budget: Budget, bets: list[list[float]], final: list[float]
) -> list[float]:
    bets = [smooth_run(b) for b in bets]
    final = smooth_run(final)
    min_len = min(len(b) for b in bets)
    out = []
    for t in budget.axis:
        if t <= budget.bet_stage:
            k = max(1, min(t // N_BETS // MUTATIONS_PER_SAMPLE, min_len))
            out.append(max(b[k - 1] for b in bets))
        else:
            # The final run's samples continue the chosen bet's samples.
            k = (budget.bet_budget + t - budget.bet_stage) // MUTATIONS_PER_SAMPLE
            out.append(final[min(len(final), max(1, k)) - 1])
    return out


def trajectory_metrics(pm_dat: Path, br_dat: Path) -> dict[str, float]:
    pm_raw = read_runs(pm_dat)[0]
    rnd = read_runs(br_dat)
    bets, br_raw = rnd[:N_BETS], rnd[-1]
    budget = Budget.from_single_run(len(pm_raw))
    pm_smooth = smooth_run(pm_raw)
    pm_t = [pm_smooth[min(i, len(pm_smooth) - 1)] for i in range(len(budget.axis))]
    br_t = br_total_axis(budget, bets, br_raw)
    pm_final, br_final = final_of(pm_raw), final_of(br_raw)
    pm_start = median(pm_raw[:5])
    target = CONV * br_final
    return {
        "n_pm": first_reach(budget, pm_t, target),
        "n_br": first_reach(budget, br_t, target),
        # Mutations B&R needs to match PM's unoptimized starting state.
        "start_equiv": first_reach(budget, br_t, pm_start),
        # Share of the final speedup already present at the start (B&R: best bet).
        "upfront_pm": pm_start / pm_final,
        "upfront_br": max(median(b[:5]) for b in bets) / br_final,
    }


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


def write_summary(runs: list[dict], csv_out: Path, force: bool) -> None:
    pairs: dict[tuple[str, str], dict[str, dict]] = {}
    for run in runs:
        pairs.setdefault((run["curve"], run["method"]), {})[run["scheduler"]] = run
    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, SUMMARY_CSV_HEADER, lineterminator="\n")
        writer.writeheader()
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
            seed = pm_run["seed"]
            assert seed == br_run["seed"]
            writer.writerow(
                {
                    "curve": key[0],
                    "method": key[1],
                    "seed": seed,
                    "cycles_cc_baseline_pm": pm[1],
                    "cycles_cc_baseline_br": br[1],
                    "cycles_pm": pm[0],
                    "cycles_br": br[0],
                    **metrics,
                }
            )


def parse_evals(s: str) -> int:
    # Same syntax as CryptOpt's --evals: a plain number or a k/M/T suffix.
    multipliers = {"k": 1e3, "M": 1e6, "T": 1e9}
    try:
        if s[-1:] in multipliers:
            evals = float(s[:-1]) * multipliers[s[-1]]
        else:
            evals = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid eval count {s!r}")
    if not evals.is_integer():
        raise argparse.ArgumentTypeError(f"eval count {s!r} is not a whole number")
    return int(evals)


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
        "-e",
        "--evals",
        dest="evals",
        type=parse_evals,
        default="10k",
        help="number of evals to run per optimization, e.g. 1000, 1k, 0.1M (default: 10k, as CryptOpt)",
    )
    parser.add_argument(
        "--proof",
        dest="proof",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to enable/disable proofs of final assembly",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=str(SCRIPT_ROOT / "artifacts_optimization_comparison"),
        help="Directory to store generated JSON/asm/cycle artifacts",
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
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="re-measure final cycle counts even if cached .cycles files exist",
    )
    args = parser.parse_args()
    if args.evals <= MIN_EVALS:
        parser.error(
            f"--evals must be greater than {MIN_EVALS}, so that each of the {N_BETS} bets records at least one sample"
        )
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
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    seed = args.seed if args.seed is not None else random.randint(1, 9999)
    runs = build_runs(args.curves, args.methods, seed, out_dir)
    if not runs:
        fatal("No curve/method combinations selected")
    random.shuffle(runs)

    print(f"Seed: {seed}")
    print(f"Total runs: {len(runs)}")
    print(f"Workers: {args.num_workers} (CPUs 0-{args.num_workers - 1})")
    print(f"Out dir: {out_dir}")
    print(f"Verbose: {str(args.verbose).lower()}")
    print(f"Evals: {args.evals}")
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
        Path(out_dir).rglob("seed*"),
    ):
        os.remove(bet)

    print("All runs completed.")
    print()

    # Measure only once all optimizations are done, so that measurements don't
    # interfere with running optimizations.
    summary_csv = Path(out_dir) / f"summary_seed{seed}.csv"
    write_summary(runs, summary_csv, args.force)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
