#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from random import randint
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT / "CryptOpt"
DIST = REPO_ROOT / "dist"
BUNDLED_NODE = REPO_ROOT / "bins" / "node" / "bin" / "node"
CURVES = [
    "bls12_381_p",
    "bls12_381_q",
    "curve25519",
    "curve25519_solinas",
    "p224",
    "p256",
    "p384",
    "p434",
    "p448_solinas",
    "p521",
    "poly1305",
    "secp256k1_montgomery",
    "secp256k1_dettman",
]
METHODS = ["mul", "square"]

SCHEDULERS = ["default", "pressure-minimized"]
CSV_HEADER = [
    "curve",
    "method",
    "seed",
    "scheduling_strategy",
    "pm_lookahead",
    "cycles_cc_baseline",
    "cycles_cryptopt",
    "scheduling_time_elapsed_ms",
    "stack_size_bytes",
    "num_spills",
    "num_instructions",
]


class StepFailed(Exception):
    """Raised when a subprocess step fails; carries a human-readable reason."""


def run_node(
    script: Path, args: list[str], node: str, env: dict[str, str] | None = None
) -> str:
    if not script.exists():
        raise StepFailed(f"{script} does not exist. Did you run `make build`?")
    try:
        proc = subprocess.run(
            [node, str(script), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError as e:
        raise StepFailed(f"could not execute node ({node!r}): {e}")
    if proc.returncode != 0:
        raise StepFailed(
            f"{script.name} {' '.join(args)} exited {proc.returncode}\n--- stderr ---\n{proc.stderr.strip()}"
        )
    return proc.stdout


def generate_state(
    curve: str,
    method: str,
    scheduler: str,
    lookahead: int,
    seed: int,
    node: str,
    path: Path,
) -> None:
    cli_args = [
        "--seed",
        str(seed),
        "--schedulingAlgorithm",
        scheduler,
        "--pmLookahead",
        str(lookahead),
        "-c",
        curve,
        "-m",
        method,
    ]
    stdout = run_node(DIST / "GenerateStartState.js", cli_args, node)
    try:
        json.loads(stdout)
    except json.JSONDecodeError as e:
        raise StepFailed(
            f"GenerateStartState.js did not emit valid JSON for {curve}/{method}/{scheduler}: {e}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stdout)


def assemble(state_path: Path, node: str, path: Path) -> None:
    stdout = run_node(DIST / "Assemble.js", ["--readState", str(state_path)], node)
    if "GLOBAL" not in stdout:
        raise StepFailed(
            f"Assemble.js output for {state_path} has no GLOBAL symbol line (asm looks empty/broken)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stdout)


def count_cycles(
    asm_path: Path, node: str, cache_path: Path, force: bool
) -> tuple[float, float]:
    if not force and cache_path.exists():
        raw = cache_path.read_text().strip()
    else:
        env = {**os.environ, "CC": "clang"}
        raw = run_node(DIST / "CountCycle.js", [str(asm_path)], node, env=env).strip()
        if not raw:
            raise StepFailed(
                f"CountCycle.js produced no output for {asm_path} (likely couldn't get a stable measurement)"
            )
        cache_path.write_text(raw + "\n")
    try:
        asm_cycles, cc_baseline_cycles = (float(x) for x in raw.split())
    except ValueError:
        raise StepFailed(f"Could not parse CountCycle.js output {raw!r} for {asm_path}")
    return asm_cycles, cc_baseline_cycles


LABEL_RE = re.compile(r"^[A-Za-z_.$][\w.$]*:$")
SUB_RSP_RE = re.compile(r"^sub rsp, (\d+)")


def analyze_asm(asm_path: Path) -> dict[str, int]:
    num_instructions = 0
    num_spills = 0
    stack_size_bytes = 0

    for line in asm_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "SECTION .text" or stripped.startswith("GLOBAL "):
            continue
        if LABEL_RE.match(stripped):
            continue

        num_instructions += 1
        # CryptOpt's assembler inserts small debug statements whenever forced to spill a variable.
        # We could equally just count the number of `mov [mem], [reg]` statements.
        if "; spilling" in stripped:
            num_spills += 1
        if num_instructions == 1:
            m = SUB_RSP_RE.match(stripped)
            if m:
                stack_size_bytes = int(m.group(1))

    return {
        "num_instructions": num_instructions,
        "num_spills": num_spills,
        "stack_size_bytes": stack_size_bytes,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        "--lookahead",
        default="1",
        help=(
            "Comma-separated list of lookahead factors to generate pressure-minimized "
            "states for (default: 1). The default scheduler is unaffected by lookahead, "
            "so it is only ever run once regardless of how many values are given here."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_ROOT / "artifacts"),
        help="Directory to store generated JSON/asm/cycle artifacts",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Path to write the CSV to (default: <comparison-dir>/seed<seed>.csv)",
    )
    parser.add_argument(
        "--node",
        default=str(BUNDLED_NODE),
        help=f"node executable to use (default: bundled node at {BUNDLED_NODE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate/re-measure even if cached artifacts already exist",
    )
    args = parser.parse_args()
    for c in [c.strip() for c in args.curves.split(",") if c.strip()]:
        if c not in CURVES:
            parser.error(f"unknown curve {c!r}; known curves: {', '.join(CURVES)}")
    for m in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if m not in METHODS:
            parser.error(f"unknown method {m!r}; known methods: {', '.join(METHODS)}")
    for la in [la.strip() for la in args.lookahead.split(",") if la.strip()]:
        if not la.isdigit():
            parser.error(f"invalid lookahead {la!r}; must be a positive integer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    curves = [c.strip() for c in args.curves.split(",") if c.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    lookaheads = [int(la.strip()) for la in args.lookahead.split(",") if la.strip()]
    seed = int(args.seed) if args.seed is not None else randint(1, 9999)
    out_dir = Path(args.out_dir)
    csv_out = Path(args.csv_out) if args.csv_out else out_dir / f"seed{seed}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    # Lookahead only affects the pressure-minimized scheduler, so "default" is run
    # once and never multiplied out across the requested lookahead values.
    runs: list[tuple[str, int | None]] = [("default", None)]
    runs += [("pressure-minimized", la) for la in lookaheads]

    total = len(curves) * len(methods) * len(runs)
    i = 0

    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for curve in curves:
            for method in methods:
                for scheduler, lookahead in runs:
                    i += 1
                    print(
                        f"[{i}/{total}] curve={curve} method={method} scheduler={scheduler} "
                        f"seed={seed} lookahead={lookahead if lookahead is not None else 'n/a'}",
                        file=sys.stderr,
                    )

                    base = out_dir / f"seed{seed}" / curve / method
                    name = f"{curve}_{method}_{scheduler}"
                    if lookahead is not None:
                        name += f"_la{lookahead}"
                    state_path = base / f"{name}.json"
                    # CountCycle.js requires the asm filename to match /seed[0-9]+_ratio[0-9]+\.asm/, just add a dummy digit at the end.
                    asm_path = base / f"{name}_seed{seed}_ratio0.asm"
                    cycles_cache_path = base / f"{name}.cycles"

                    elapsed_ms = None
                    if args.force or not state_path.exists():
                        t0 = time.monotonic()
                        generate_state(
                            curve,
                            method,
                            scheduler,
                            lookahead if lookahead is not None else 1,
                            seed,
                            args.node,
                            state_path,
                        )
                        elapsed_ms = (time.monotonic() - t0) * 1000

                    if args.force or not asm_path.exists():
                        assemble(state_path, args.node, asm_path)

                    asm_cycles, cc_baseline_cycles = count_cycles(
                        asm_path, args.node, cycles_cache_path, args.force
                    )

                    stats = analyze_asm(asm_path)

                    writer.writerow(
                        [
                            curve,
                            method,
                            seed,
                            scheduler,
                            lookahead if lookahead is not None else 1,
                            cc_baseline_cycles,
                            asm_cycles,
                            elapsed_ms,
                            stats["stack_size_bytes"],
                            stats["num_spills"],
                            stats["num_instructions"],
                        ]
                    )

    print(f"\nWrote {csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
