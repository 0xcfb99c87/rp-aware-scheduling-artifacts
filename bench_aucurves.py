#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
AUCURVES_ROOT = SCRIPT_ROOT / "AUCurves"

# The `_bls12_mul` / `_bls12_square` symbols are what AUCurves' build.sh links
# in place of the Rust CIOS leaves in bls12-381-safe-rust/src/stubs.rs.  Its
# jasmin_aliases.s exists only to bridge the CryptOpt symbol names to those two;
# we rename the globals in the assembly directly instead, so the alias file is
# assembled empty.
JASMIN_ALIASES = """\t.text
"""

# AUCurves' build.sh also assembles libjade's Jasmin SHAKE-128, which is not
# shipped in the repository.  The pure-Rust path that the crate falls back on
# when `cryptopt` is off is itself a stub returning zeros (src/shake128.rs), and
# the caller pre-zeroes the output buffer, so this stub is equivalent.  No
# benchmark here touches SHAKE-128.
SHAKE128_STUB = """\t.text
\t.globl jade_xof_shake128_amd64_ref
jade_xof_shake128_amd64_ref:
\txorl %eax, %eax
\tret
"""

TARGETS = {
    "bls12_381_p": {
        "crate": "bls12-381-safe-rust",
        "package": "bls12-381-safe-rust",
        "operation": "pairing",
        "archive": "libbls12_leaves.a",
        # cryptopt method -> (destination under generated/, exported symbol)
        "leaves": {
            "mul": ("bls12_mul_cryptopt.asm", "_bls12_mul"),
            "square": ("bls12_square_cryptopt.asm", "_bls12_square"),
        },
        # The CryptOpt assembly AUCurves vendors itself.  This crate's
        # generated/ ships none; these are the files the C pairing pipeline in
        # src/Implementations/C/ links (see that directory's README.md).
        "vendored": {
            "mul": "src/Implementations/C/cryptopt/fiat_bls12_381_p_mul.asm",
            "square": "src/Implementations/C/cryptopt/fiat_bls12_381_p_square.asm",
        },
        "aux": {"jasmin_aliases.s": JASMIN_ALIASES, "shake128.s": SHAKE128_STUB},
        "ref_env": {},
        "metric": ("Pairing:", "us", 1000.0),
    },
    "p384": {
        "crate": "p384-safe-rust",
        "package": "p384-safe-rust",
        "operation": "scalarmult",
        "archive": "libp384_cryptopt.a",
        "leaves": {
            "mul": ("p384_mul_cryptopt.asm", "p384_cryptopt_mul"),
            "square": ("p384_square_cryptopt.asm", "p384_cryptopt_square"),
        },
        # Here the vendored files are the crate's own generated/ inputs, already
        # carrying the renamed symbol; they are read before the run overwrites
        # them and put back verbatim for the aucurves-upstream measurement.
        "vendored": {
            "mul": "p384-safe-rust/generated/p384_mul_cryptopt.asm",
            "square": "p384-safe-rust/generated/p384_square_cryptopt.asm",
        },
        "ref_env": {"P384_NO_CRYPTOPT": "1"},
        "aux": {},
        "metric": ("g1_scalar_mul (384-bit)", "ns/op", 1.0),
    },
}

# (csv label, kind, scheduling strategy)
BACKENDS = [
    ("aucurves-ref", "ref", None),
    ("aucurves-upstream", "upstream", None),
    ("aucurves+noopt+default", "noopt", "default"),
    ("aucurves+noopt+pm", "noopt", "pressure-minimized"),
    ("aucurves+opt+default", "opt", "default"),
    ("aucurves+opt+pm", "opt", "pressure-minimized"),
]


def fatal(msg: str) -> None:
    print(f"bench_aucurves.py: {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg: str) -> None:
    print(msg, flush=True)


def detect_seed(start_states: Path) -> str:
    seeds = sorted(
        p.name[len("seed"):]
        for p in start_states.iterdir()
        if p.is_dir() and p.name.startswith("seed")
    )
    if not seeds:
        fatal(f"no seed<N> directory under {start_states}")
    if len(seeds) > 1:
        fatal(f"{start_states} holds several seeds ({', '.join(seeds)}); pass --seed")
    return seeds[0]


def rename_leaf(body: str, curve: str, method: str, symbol: str) -> str:
    """Point a CryptOpt .asm at the symbol the AUCurves crate links against.

    Only the GLOBAL directive and the label are touched, so provenance comments
    keep naming the fiat function.  A file already exporting `symbol` (AUCurves'
    vendored p384 leaves) passes through unchanged.
    """
    fiat = re.escape(f"fiat_{curve}_{method}")
    body = re.sub(rf"^(\s*GLOBAL\s+){fiat}\s*$", rf"\g<1>{symbol}", body, flags=re.M)
    body = re.sub(rf"^{fiat}:", f"{symbol}:", body, flags=re.M)
    if not re.search(rf"^\s*GLOBAL\s+{re.escape(symbol)}\s*$", body, flags=re.M):
        fatal(f"no GLOBAL {symbol} after rewriting the {curve} {method} assembly")
    return body


def read_vendored(curves: list[str]) -> dict[tuple[str, str], tuple[str, str]]:
    """(curve, method) -> (source filename, assembly ready to install).

    Read up front, because installing an earlier backend overwrites the
    destination, which for p384 is the vendored file itself.
    """
    out = {}
    for curve in curves:
        target = TARGETS[curve]
        for method, (_, symbol) in target["leaves"].items():
            src = AUCURVES_ROOT / target["vendored"][method]
            if not src.is_file():
                fatal(f"missing vendored assembly: {src}")
            out[(curve, method)] = (src.name, rename_leaf(src.read_text(), curve, method, symbol))
    return out


def source_asm(
    curve: str, kind: str, strategy: str, method: str,
    start_states: Path, opt_dir: Path, seed: str,
) -> Path:
    if kind == "noopt":
        path = (
            start_states / f"seed{seed}" / curve / method
            / f"{curve}_{method}_{strategy}_seed{seed}_ratio0.asm"
        )
    else:
        run_dir = opt_dir / f"{curve}--{method}--{strategy}--seed{seed}"
        found = sorted((run_dir / "fiat" / f"fiat_{curve}_{method}").glob("*.asm"))
        if len(found) != 1:
            fatal(f"expected exactly one .asm under {run_dir}, found {len(found)}")
        path = found[0]
    if not path.is_file():
        fatal(f"missing assembly: {path}")
    return path


def install(curve: str, kind: str, strategy: str | None, vendored: dict, *args) -> dict[str, str]:
    """Write the generated/ files for one backend; return the asm names used."""
    target = TARGETS[curve]
    gen = AUCURVES_ROOT / target["crate"] / "generated"
    used = {}

    for method, (dest, symbol) in target["leaves"].items():
        if kind == "ref":
            # No assembly: p384's build.rs is told to skip it via P384_NO_CRYPTOPT,
            # and bls12-381's build.sh fails on the absent input and falls back to
            # the Rust leaves, which is the state of a clean AUCurves checkout.
            (gen / dest).unlink(missing_ok=True)
            used[method] = "(none)"
            continue
        if kind == "upstream":
            name, body = vendored[(curve, method)]
        else:
            src = source_asm(curve, kind, strategy, method, *args)
            name, body = src.name, rename_leaf(src.read_text(), curve, method, symbol)
        (gen / dest).write_text(body)
        used[method] = name

    for name, body in target["aux"].items():
        if kind == "ref":
            (gen / name).unlink(missing_ok=True)
        else:
            (gen / name).write_text(body)

    return used


def parse_metric(out: str, label: str, unit: str) -> float | None:
    for line in out.splitlines():
        if not line.startswith(label):
            continue
        fields = line.split()
        if unit not in fields:
            continue
        return float(fields[fields.index(unit) - 1])
    return None


def run_bench(curve: str, kind: str, cpu: int | None, repeat: int) -> float:
    target = TARGETS[curve]
    manifest = AUCURVES_ROOT / target["crate"] / "Cargo.toml"
    env = dict(os.environ)
    env.pop("P384_NO_CRYPTOPT", None)
    if kind == "ref":
        env.update(target["ref_env"])

    # Rebuild the crate from scratch so no stale build-script output or leaf
    # archive from the previous backend can survive into this measurement.
    subprocess.run(
        ["cargo", "clean", "--release", "-p", target["package"]],
        cwd=AUCURVES_ROOT, env=env, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    cmd = ["cargo", "run", "--release", "--quiet",
           "--manifest-path", str(manifest), "--example", "bench"]
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu)] + cmd

    label, unit, scale = target["metric"]
    best = None
    for _ in range(repeat):
        proc = subprocess.run(
            cmd, cwd=AUCURVES_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if proc.returncode != 0:
            fatal(f"{curve}/{kind}: bench failed:\n{proc.stderr}")
        value = parse_metric(proc.stdout, label, unit)
        if value is None:
            fatal(f"{curve}/{kind}: no '{label}' line in bench output:\n{proc.stdout}")
        best = value if best is None else min(best, value)

    # Guard against silently reporting fiat-rust numbers as a CryptOpt backend:
    # the leaf archive exists if and only if the assembly was linked.
    archives = list(
        (AUCURVES_ROOT / "target" / "release" / "build").glob(
            f"{target['package']}-*/out/{target['archive']}"
        )
    )
    if kind == "ref" and archives:
        fatal(f"{curve}/ref: {target['archive']} was built; assembly leaked in")
    if kind != "ref" and not archives:
        fatal(f"{curve}/{kind}: {target['archive']} missing; assembly was not linked")

    return best * scale


def snapshot(paths: list[Path]) -> dict[Path, bytes | None]:
    return {p: p.read_bytes() if p.is_file() else None for p in paths}


def restore(snap: dict[Path, bytes | None]) -> None:
    for path, body in snap.items():
        if body is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AUCurves high-level primitives against each "
                    "CryptOpt field-arithmetic backend."
    )
    parser.add_argument("start_states", help="./artifacts_start_states from gen_starting_states.py")
    parser.add_argument("opt_comparison", help="./artifacts_optimization_comparison from bench.py")
    parser.add_argument("-o", "--out-dir", default=str(SCRIPT_ROOT / "artifacts_aucurves"),
                        help="where to write aucurves_bench.csv")
    parser.add_argument("--seed", default=None, help="seed to select (default: autodetect)")
    parser.add_argument("--curve", action="append", choices=sorted(TARGETS),
                        help="restrict to one curve (repeatable; default: all)")
    parser.add_argument("-r", "--repeat", type=int, default=3,
                        help="benchmark runs per configuration, best is kept (default: 3)")
    parser.add_argument("--cpu", type=int, default=None,
                        help="pin the benchmark to this CPU with taskset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for tool in ("cargo", "nasm", "as", "ar"):
        if shutil.which(tool) is None:
            fatal(f"{tool} not found on PATH")
    if args.cpu is not None and shutil.which("taskset") is None:
        fatal("taskset not found on PATH")

    start_states = Path(args.start_states).resolve()
    opt_dir = Path(args.opt_comparison).resolve()
    for path in (start_states, opt_dir, AUCURVES_ROOT):
        if not path.is_dir():
            fatal(f"not a directory: {path}")

    seed = args.seed or detect_seed(start_states)
    curves = args.curve or sorted(TARGETS)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "aucurves_bench.csv"

    log(f"AUCurves: {AUCURVES_ROOT}")
    log(f"Seed: {seed}")
    log(f"Curves: {', '.join(curves)}")
    log(f"Repeats: {args.repeat}")
    log(f"Pinned to CPU: {args.cpu if args.cpu is not None else '(unpinned)'}")
    log(f"Output: {out_csv}")
    log("")

    vendored = read_vendored(curves)
    touched = [
        AUCURVES_ROOT / TARGETS[c]["crate"] / "generated" / name
        for c in curves
        for name in (
            [dest for dest, _ in TARGETS[c]["leaves"].values()] + list(TARGETS[c]["aux"])
        )
    ]
    saved = snapshot(touched)

    rows = []
    start = time.monotonic()
    try:
        for curve in curves:
            operation = TARGETS[curve]["operation"]
            baseline = None
            for label, kind, strategy in BACKENDS:
                used = install(curve, kind, strategy, vendored, start_states, opt_dir, seed)
                ns = run_bench(curve, kind, args.cpu, args.repeat)
                if kind == "ref":
                    baseline = ns
                log(f"{curve:<12} {operation:<11} {label:<24} {ns:12.1f} ns/op"
                    f"   ({baseline / ns:.2f}x)")
                rows.append({
                    "curve": curve,
                    "operation": operation,
                    "backend": label,
                    "ns_per_op": f"{ns:.1f}",
                    "speedup_vs_ref": f"{baseline / ns:.3f}",
                    "mul_asm": used["mul"],
                    "square_asm": used["square"],
                })
            log("")
    finally:
        restore(saved)

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    log(f"Wrote {len(rows)} rows to {out_csv} in {time.monotonic() - start:.0f}s")


if __name__ == "__main__":
    main()
