#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
AUCURVES_ROOT = SCRIPT_ROOT / "AUCurves"


def fatal(msg: str) -> None:
    print(f"bench_aucurves.py: {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg: str) -> None:
    print(msg, flush=True)


def detect_seed(start_states: Path) -> str:
    seeds = sorted(
        p.name[len("seed") :]
        for p in start_states.iterdir()
        if p.is_dir() and p.name.startswith("seed")
    )
    if not seeds:
        fatal(f"no seed<N> directory under {start_states}")
    if len(seeds) > 1:
        fatal(f"{start_states} holds several seeds ({', '.join(seeds)}); pass --seed")
    return seeds[0]


def rename_leaf(body: str, curve: str, method: str, symbol: str) -> str:
    # AUCurves expects different symbol names than those used in CryptOpt's output.
    fiat = re.escape(f"fiat_{curve}_{method}")
    body = re.sub(rf"^(\s*GLOBAL\s+){fiat}\s*$", rf"\g<1>{symbol}", body, flags=re.M)
    body = re.sub(rf"^{fiat}:", f"{symbol}:", body, flags=re.M)
    if not re.search(rf"^\s*GLOBAL\s+{re.escape(symbol)}\s*$", body, flags=re.M):
        fatal(f"no GLOBAL {symbol} after rewriting the {curve} {method} assembly")
    return body


# Sources assembly to link into AUCurves, depending on the kind requested.
def source_asm(
    curve: str,
    vendored: str,
    kind: str,
    strategy: str | None,
    method: str,
    start_states: Path,
    opt_dir: Path,
    seed: str,
) -> Path:
    if kind == "vendored":
        path = AUCURVES_ROOT / vendored.format(curve=curve, method=method)
    elif kind == "noopt":
        path = (
            start_states
            / f"seed{seed}"
            / curve
            / method
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


def relink(link: Path, target: Path) -> None:
    link.unlink(missing_ok=True)
    link.symlink_to(target)


# bench_compare row label -> csv key. Arkworks exposes no miller loop entry
# point in the shape the other arms use, so those two cells are "-".
COMPARE_ROWS = {
    "Fp mul": "fp_mul",
    "Miller loop": "miller",
    "Pairing (full)": "pairing",
}
COMPARE_COLS = ("cyc", "ns", "blst_cyc", "blst_ns", "ark_cyc", "ark_ns")


def parse_compare(out: str) -> dict[str, float]:
    # Only scan the figures table; the ratios table below repeats the row labels.
    table = out.partition("=== ratios")[0].partition("=== per-arm figures ===")[2]
    metrics = {}
    for label, key in COMPARE_ROWS.items():
        row = next((l for l in table.splitlines() if l.startswith(label)), None)
        if row is None:
            fatal(f"no '{label}' row in bench_compare output:\n{out}")
        cells = row[len(label) :].split()
        if len(cells) != len(COMPARE_COLS):
            fatal(f"expected {len(COMPARE_COLS)} cells in '{label}' row, got: {row!r}")
        metrics.update(
            {f"{key}_{c}": float(v) for c, v in zip(COMPARE_COLS, cells) if v != "-"}
        )
    return metrics


def cargo_bench(package: str, example: str, parse, cpu: int | None) -> dict[str, float]:
    # Ensure cargo rebuilds from scratch
    subprocess.run(
        ["cargo", "clean", "--release", "-p", package],
        cwd=AUCURVES_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    cmd = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--manifest-path",
        str(AUCURVES_ROOT / package / "Cargo.toml"),
        "--example",
        example,
    ]
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu)] + cmd

    # Keep the per-metric minimum, the same rule the example applies internally.
    proc = subprocess.run(
        cmd,
        cwd=AUCURVES_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        fatal(f"{package}: {example} failed:\n{proc.stderr}")
    metrics = parse(proc.stdout)
    return metrics


def linked_archive(package: str, archive: str) -> bool:
    build = AUCURVES_ROOT / "target" / "release" / "build"
    return any(build.glob(f"{package}-*/out/{archive}"))


def bench_bls_pairing(
    out_dir: Path,
    start_states: Path,
    opt_dir: Path,
    seed: str,
    cpu: int | None,
) -> list[dict]:
    curve = "bls12_381_p"
    operation = "pairing"
    package = "bls12-381-safe-rust"
    archive = "libbls12_leaves.a"
    generated = AUCURVES_ROOT / package / "generated"
    # cryptopt method -> symbol the crate links in place of its Rust default impl.
    leaves = {"mul": "_bls12_mul", "square": "_bls12_square"}
    # AUCurves vendors some pre-optimized cryptopt itself. They consist of a
    # 10k-mutation run.
    vendored = "src/Implementations/C/cryptopt/fiat_{curve}_{method}.asm"

    # Init artifact dir with assembly impls we will be linking.
    staged_dir = out_dir / f"{curve}-{operation}"
    staged_dir.mkdir(parents=True, exist_ok=True)
    # Build script expects these files to exist, but the benchmark never
    # actually calls them. So we can just add stubs.
    stubs = ["jasmin_aliases.s", "shake128.s"]
    for name in stubs:
        (staged_dir / name).write_text("")

    # (staged-name suffix, source kind, scheduling strategy).  Kind "none" stages
    # no assembly at all: with the leaves missing, build.sh fails and the crate
    # silently falls back to its own Rust field arithmetic.
    variants = [
        ("rust", "none", None),
        ("ref", "vendored", None),
        ("defnoopt", "noopt", "default"),
        ("pmnoopt", "noopt", "pressure-minimized"),
        ("defopt", "opt", "default"),
        ("pmopt", "opt", "pressure-minimized"),
    ]
    sources = {}
    for tag, kind, strategy in variants:
        if kind == "none":
            sources.update({(tag, method): "(none)" for method in leaves})
            continue
        for method, symbol in leaves.items():
            src = source_asm(
                curve, vendored, kind, strategy, method, start_states, opt_dir, seed
            )
            staged = staged_dir / f"bls12_{method}_cryptopt_{tag}.asm"
            staged.write_text(rename_leaf(src.read_text(), curve, method, symbol))
            sources[tag, method] = src.name

    links = [generated / n for n in stubs]
    links += [generated / f"bls12_{m}_cryptopt.asm" for m in leaves]

    rows = []
    log(
        f"{'variant':<10} {'fp mul':>9} {'vs blst':>9} {'vs ark':>9} "
        f"{'miller':>10} {'vs blst':>9} "
        f"{'pairing':>11} {'vs blst':>9} {'vs ark':>9}"
    )
    try:
        for name in stubs:
            relink(generated / name, staged_dir / name)
        for tag, kind, _ in variants:
            for method in leaves:
                link = generated / f"bls12_{method}_cryptopt.asm"
                if kind == "none":
                    link.unlink(missing_ok=True)
                else:
                    relink(link, staged_dir / f"bls12_{method}_cryptopt_{tag}.asm")
            m = cargo_bench(package, "bench_compare", parse_compare, cpu)
            if (kind != "none") != linked_archive(package, archive):
                fatal(f"{tag}: unexpected {archive}; the wrong leaves were linked")
            # Same convention as the example: ours / theirs, below 1.00 is faster.
            r = {
                f"{op}_vs_{arm}": m[f"{op}_cyc"] / m[f"{op}_{arm}_cyc"]
                for op in COMPARE_ROWS.values()
                for arm in ("blst", "ark")
                if f"{op}_{arm}_cyc" in m
            }
            log(
                f"{tag:<10} {m['fp_mul_cyc']:>9.1f} {r['fp_mul_vs_blst']:>8.2f}x "
                f"{r['fp_mul_vs_ark']:>8.2f}x "
                f"{m['miller_cyc']:>10.0f} {r['miller_vs_blst']:>8.2f}x "
                f"{m['pairing_cyc']:>11.0f} {r['pairing_vs_blst']:>8.2f}x "
                f"{r['pairing_vs_ark']:>8.2f}x"
            )
            rows.append(
                {
                    "curve": curve,
                    "variant": tag,
                    "fp_mul_cyc": f"{m['fp_mul_cyc']:.1f}",
                    "miller_cyc": f"{m['miller_cyc']:.0f}",
                    "pairing_cyc": f"{m['pairing_cyc']:.0f}",
                    "pairing_ns": f"{m['pairing_ns']:.0f}",
                    "blst_fp_mul_cyc": f"{m['fp_mul_blst_cyc']:.1f}",
                    "blst_miller_cyc": f"{m['miller_blst_cyc']:.0f}",
                    "blst_pairing_cyc": f"{m['pairing_blst_cyc']:.0f}",
                    "ark_fp_mul_cyc": f"{m['fp_mul_ark_cyc']:.1f}",
                    "ark_pairing_cyc": f"{m['pairing_ark_cyc']:.0f}",
                    "mul_asm": sources[tag, "mul"],
                    "square_asm": sources[tag, "square"],
                }
            )
    finally:
        for link in links:
            link.unlink(missing_ok=True)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "start_states", help="./artifacts_start_states from gen_starting_states.py"
    )
    parser.add_argument(
        "opt_comparison", help="./artifacts_optimization_comparison from bench.py"
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default=str(SCRIPT_ROOT / "artifacts_aucurves"),
        help="where to write the staged leaves and aucurves_bench.csv",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="seed to use when selecting states (default: autodetect)",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=None,
        help="pin the benchmark to this CPU with taskset",
    )
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
    out_dir = Path(args.out_dir).resolve()
    out_csv = out_dir / "aucurves_bench.csv"

    log(f"AUCurves: {AUCURVES_ROOT}")
    log(f"Pinned to CPU: {args.cpu if args.cpu is not None else '(unpinned)'}")
    log(f"Output: {out_csv}")
    log("")

    start = time.monotonic()
    rows = bench_bls_pairing(out_dir, start_states, opt_dir, seed, args.cpu)

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    log("")
    log(f"Wrote {len(rows)} rows to {out_csv} in {time.monotonic() - start:.0f}s")


if __name__ == "__main__":
    main()
