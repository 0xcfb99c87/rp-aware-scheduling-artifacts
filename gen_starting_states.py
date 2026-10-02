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
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT / "CryptOpt"
DIST = REPO_ROOT / "dist"
BUNDLED_NODE = REPO_ROOT / "bins" / "node" / "bin" / "node"
CURVES = [
    "bls12_381_p",
    "curve25519",
    "p224",
    "p256",
    "p384",
    "p434",
    "p448_solinas",
    "p521",
    "poly1305",
    "secp256k1_montgomery",
]
METHODS = ["mul", "square"]

SCHEDULERS = ["default", "pressure-minimized"]
CSV_HEADER = [
    "curve",
    "method",
    "seed",
    "scheduling_strategy",
    "cycles_cc_baseline",
    "cycles_cryptopt",
    "stack_size_bytes",
    "num_spills",
    "num_instructions",
    # Per-instruction classes
    "n_core",
    "n_spill_traffic",
    "n_flag_mgmt",
    "n_fixed",
]
INSTRUCTION_CLASSES = ["core", "spill_traffic", "flag_mgmt", "fixed"]


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
    seed: int,
    node: str,
    path: Path,
) -> None:
    cli_args = [
        "--seed",
        str(seed),
        "--schedulingAlgorithm",
        scheduler,
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
    stdout = run_node(
        DIST / "Assemble.js", ["--readState", str(state_path.absolute())], node
    )
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
        raw = run_node(
            DIST / "CountCycle.js", [str(asm_path.absolute())], node, env=env
        ).strip()
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
# Arithmetic mnemonics emitted by CryptOpt for the routine itself.
CORE_OPS = {
    "mulx", "imul", "mul", "adcx", "adox", "adc", "add", "sub", "sbb", "lea",
    "shrd", "shr", "shl", "sar", "shlx", "and", "or", "xchg", "xor", "bzhi",
    "cmovc", "cmovo",
}
IMM_RE = re.compile(r",\s*(-?0x[0-9a-f]+|-?\d+)\s*$")
SAME_REG_RE = re.compile(r"(\w+), \1$")


# Assign one emitted instruction to exactly one class (core, fixed, spill traffic, or flag management)
def classify_instruction(op: str, args: str, comment: str) -> str:
    rsp = "rsp" in args
    if (
        op == "ret"
        or (op in ("sub", "add") and args.startswith("rsp,"))
        or "calSv" in comment  # callee-saved register save
        or comment.strip() == "pop"  # callee-saved register restore
    ):
        return "fixed"
    if op in ("setc", "seto"):
        return "flag_mgmt"
    if op == "clc" or (op in ("xor", "test") and SAME_REG_RE.match(args)):
        return "flag_mgmt"
    if op in ("inc", "dec") and "OF<-" in comment:
        return "flag_mgmt"
    if op == "movzx" or "loading flag" in comment:
        return "flag_mgmt"
    if op == "mov" and args.startswith("byte"):
        return "spill_traffic"
    if op == "mov":
        if args.startswith("["):
            return "spill_traffic" if rsp else "fixed"  # spill store vs. output store
        if rsp:
            return "spill_traffic"  # reload
        if "[" in args:
            return "fixed"  # argument load
        return "core"  # reg-reg copy or immediate (core arith)
    if op in CORE_OPS:
        return "spill_traffic" if rsp else "core"
    raise StepFailed(f"cannot classify instruction {op} {args!r}")


def analyze_asm(asm_path: Path) -> dict[str, int]:
    num_instructions = 0
    num_spills = 0
    stack_size_bytes = 0
    classes = {c: 0 for c in INSTRUCTION_CLASSES}

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

        code, _, comment = stripped.partition(";")
        op, _, args = code.strip().partition(" ")
        classes[classify_instruction(op, args.strip(), comment)] += 1

    assert sum(classes.values()) == num_instructions
    return {
        "num_instructions": num_instructions,
        "num_spills": num_spills,
        "stack_size_bytes": stack_size_bytes,
        **{f"n_{c}": n for c, n in classes.items()},
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
        "--out-dir",
        default=str(SCRIPT_ROOT / "artifacts_start_states"),
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    curves = [c.strip() for c in args.curves.split(",") if c.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seed = int(args.seed) if args.seed is not None else randint(1, 9999)
    out_dir = Path(args.out_dir)
    csv_out = Path(args.csv_out) if args.csv_out else out_dir / f"seed{seed}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    runs: list[str] = ["default", "pressure-minimized"]

    total = len(curves) * len(methods) * len(runs)
    i = 0

    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for curve in curves:
            for method in methods:
                for scheduler in runs:
                    i += 1
                    print(
                        f"[{i}/{total}] curve={curve} method={method} scheduler={scheduler} seed={seed}",
                        file=sys.stderr,
                    )

                    base = out_dir / f"seed{seed}" / curve / method
                    name = f"{curve}_{method}_{scheduler}"
                    state_path = base / f"{name}.json"
                    # CountCycle.js requires the asm filename to match /seed[0-9]+_ratio[0-9]+\.asm/, just add a dummy digit at the end.
                    asm_path = base / f"{name}_seed{seed}_ratio0.asm"
                    cycles_cache_path = base / f"{name}.cycles"

                    if args.force or not state_path.exists():
                        generate_state(
                            curve,
                            method,
                            scheduler,
                            seed,
                            args.node,
                            state_path,
                        )

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
                            cc_baseline_cycles,
                            asm_cycles,
                            stats["stack_size_bytes"],
                            stats["num_spills"],
                            stats["num_instructions"],
                            *(stats[f"n_{c}"] for c in INSTRUCTION_CLASSES),
                        ]
                    )

    print(f"\nWrote {csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
