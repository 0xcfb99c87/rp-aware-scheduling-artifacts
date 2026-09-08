#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUCURVES = ROOT / "AUCurves"
BENCH_BIN = AUCURVES / "target" / "release" / "cryptopt-bench"

# Columns cryptopt-bench prints, then the ones added here.
BENCH_COLS = [
    "curve",
    "operation",
    "asm_linked",
    "iters",
    "ns_per_op_min",
    "ns_per_op_median",
]
CSV_HEADER = (
    ["curve", "operation", "stage", "strategy", "mul_seed", "square_seed"]
    + BENCH_COLS[2:]
    + ["mul_asm", "square_asm", "verified", "host", "timestamp_utc"]
)

GLOBAL_RE = re.compile(r"^\s*GLOBAL\s+(\S+)\s*$", re.MULTILINE)


def warn(msg):
    print(f"bench_aucurves.py: {msg}", file=sys.stderr, flush=True)


@dataclass
class BenchmarkTarget:
    # The curve to evaluate.
    curve: str
    # The crate in AUCurves of the respective curve to evaluate.
    crate: str
    # A prefix used to construct an env var, configuring the build to use our
    # CryptOpt-generated code or the reference rust impl in AUCurves.
    env_prefix: str
    # Dict of method to Path of respective assembly file.
    asm: dict
    # Symbol to link.
    sym: dict
    # The test name to invoke in the AUCurves project: cargo test --test <name>
    test: str


@dataclass
class Candidate:
    curve: str
    method: str
    strategy: str
    seed: str
    stage: str
    path: Path
    rank: float


TARGETS = {
    "p224": BenchmarkTarget(
        "p224",
        "p224-safe-rust",
        "P224",
        {"mul": "p224_mul_cryptopt.asm", "square": "p224_square_cryptopt.asm"},
        {
            "mul": ("fiat_p224_mul", "p224_cryptopt_mul"),
            "square": ("fiat_p224_square", "p224_cryptopt_square"),
        },
        "cryptopt_diff",
    ),
    "p256": BenchmarkTarget(
        "p256",
        "p256-safe-rust",
        "P256",
        {"mul": "p256_mul_cryptopt.asm", "square": "p256_square_cryptopt.asm"},
        {
            "mul": ("fiat_p256_mul", "p256_cryptopt_mul"),
            "square": ("fiat_p256_square", "p256_cryptopt_square"),
        },
        "cryptopt_diff",
    ),
    "p384": BenchmarkTarget(
        "p384",
        "p384-safe-rust",
        "P384",
        {"mul": "p384_mul_cryptopt.asm", "square": "p384_square_cryptopt.asm"},
        {
            "mul": ("fiat_p384_mul", "p384_cryptopt_mul"),
            "square": ("fiat_p384_square", "p384_cryptopt_square"),
        },
        "cryptopt_diff",
    ),
    "bls12_381_p": BenchmarkTarget(
        "bls12_381",
        "bls12-381-safe-rust",
        "BLS12_381",
        {"mul": "bls12_mul_cryptopt.asm", "square": "bls12_square_cryptopt.asm"},
        {
            "mul": ("fiat_bls12_381_p_mul", "fiat_bls12_381_p_mul"),
            "square": ("fiat_bls12_381_p_square", "fiat_bls12_381_p_square"),
        },
        "kat_vectors",
    ),
}


def discover(roots):
    by_sym = {s[0]: (c, m) for c, t in TARGETS.items() for m, s in t.sym.items()}
    out, skipped = [], 0
    for label, root in roots:
        for asm in sorted(root.rglob("*.asm")):
            m = GLOBAL_RE.search(asm.read_text(errors="replace")[:4096])
            if not m or m.group(1) not in by_sym:
                skipped += 1
                continue
            curve, method = by_sym[m.group(1)]
            # Both layouts drop the state file beside the .asm, under the
            # same name minus the _seed/_ratio suffixes CryptOpt appends.
            js = asm.parent / (re.sub(r"(_seed\d+)?_ratio\d+$", "", asm.stem) + ".json")
            if not js.exists():
                warn(f"skipping {asm}: no state file at {js.name}")
                continue
            state = json.loads(js.read_text())
            parsed = state.get("parsedArgs", {})
            # `scheduling-algorithm` (kebab) is authoritative: CryptOpt leaves
            # the camelCase key at its default in the state file an optimized
            # run writes out, which would label every such run "default".
            algo = parsed.get("scheduling-algorithm") or parsed.get(
                "schedulingAlgorithm"
            )
            strategy = (
                f"pressure-minimized-la{parsed.get('pmLookahead', 1)}"
                if algo == "pressure-minimized"
                else algo
            )
            # Rank candidates the way select_best_states.py does: by the
            # CryptOpt-measured cycle count where there is one (starting
            # states write `.cycles`), else by the ratio an optimized run
            # records in its state file.  Lower is better either way.
            cyc = js.with_suffix(".cycles")
            rank = (
                float(cyc.read_text().split()[0])
                if cyc.exists()
                else -float(state.get("ratio", 0))
            )
            # `bench.py` hands CryptOpt a --resultDir, which it lays out as
            # <dir>/fiat/fiat_<curve>_<method>/; gen_starting_states.py writes
            # flat.  That is what separates the two stages.
            stage = label or (
                "optimized" if asm.parent.parent.name == "fiat" else "start-state"
            )
            out.append(
                Candidate(
                    curve,
                    method,
                    strategy,
                    str(parsed.get("seed", "?")),
                    stage,
                    asm,
                    rank,
                )
            )
    if skipped:
        print(f"  ({skipped} leaves skipped: no AUCurves consumer for that target)")
    return out


@dataclass
class Variant:
    curve: str
    stage: str
    strategy: str
    leaves: dict  # method -> Cand, empty for the baselines
    no_cryptopt: bool = False

    @property
    def target(self):
        return TARGETS[self.curve]

    def seed(self, method):
        return self.leaves[method].seed if method in self.leaves else ""

    def asm(self, method):
        return str(self.leaves[method].path) if method in self.leaves else ""

    def __str__(self):
        seeds = "+".join(sorted({c.seed for c in self.leaves.values()}))
        return f"{self.stage}--{self.strategy}" + (f"--seed{seeds}" if seeds else "")


def variants(cands, curves):
    """Baselines, then one variant per (curve, stage, strategy).

    Both leaves are linked together and a pairing or ladder exercises both,
    so a variant needs a mul and a square.  Where they were searched under
    different seeds --- which the shipped paper data does --- the best of
    each is paired and both seeds are recorded.
    """
    out = []
    for c in curves:
        out.append(Variant(c, "baseline", "aucurves-upstream", {}))
        out.append(Variant(c, "baseline", "reference", {}, no_cryptopt=True))

    groups = {}
    for cand in cands:
        groups.setdefault((cand.curve, cand.stage, cand.strategy), []).append(cand)

    for (curve, stage, strategy), cs in sorted(groups.items()):
        best = {}
        for cand in cs:
            if cand.method not in best or cand.rank < best[cand.method].rank:
                best[cand.method] = cand
        if len(best) != 2:
            warn(
                f"{curve}/{stage}/{strategy}: only a {list(best)[0]} leaf; "
                f"both are linked together, so this is not buildable"
            )
            continue
        out.append(Variant(curve, stage, strategy, best))
    return out


def install(variant):
    """Put this variant's leaves in place, renaming the exported symbol.

    AUCurves links the NIST leaves under a `<curve>_cryptopt_*` name so the
    assembly cannot shadow the fiat-rust function it replaces; the BLS12-381
    alias shim jumps to CryptOpt's own name, so that one is copied through
    unchanged.  Nothing else is touched, so each file keeps its CryptOpt
    metadata footer.
    """
    t = variant.target
    for method, cand in variant.leaves.items():
        src, dst = t.sym[method]
        text = cand.path.read_text(errors="replace")
        found = GLOBAL_RE.search(text)
        if not found or found.group(1) != src:
            raise ValueError(
                f"{cand.path}: exports {found and found.group(1)!r}, expected {src!r}"
            )
        if dst != src:
            text, n = re.subn(rf"\b{re.escape(src)}\b", dst, text)
            if n < 2:  # the GLOBAL directive and the label
                raise ValueError(f"{cand.path}: renamed {src!r} only {n} time(s)")
        (AUCURVES / t.crate / "generated" / t.asm[method]).write_text(text)


# Run a process in the AUCurves project directory.
def run(cmd, env, timeout=1800):
    return subprocess.run(
        cmd, cwd=AUCURVES, env=env, capture_output=True, text=True, timeout=timeout
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="*", help="dirs to search for CryptOpt .asm")
    p.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "artifacts_aucurves" / "aucurves_bench.csv"),
    )
    p.add_argument("--curves", default=",".join(TARGETS))
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--round-ms", type=int, default=200)
    p.add_argument("--cpu", default="0", help="CPU to pin to, or 'none'")
    p.add_argument("--no-verify", action="store_true", help="skip the correctness test")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    curves = [c.strip() for c in args.curves.split(",") if c.strip()]
    for c in curves:
        if c not in TARGETS:
            sys.exit(f"unknown curve {c!r}; supported: {', '.join(TARGETS)}")
    if shutil.which("nasm") is None:
        sys.exit("nasm not found on PATH; the leaves cannot be assembled")

    roots = []
    for spec in args.roots:
        label, _, path = spec.rpartition("=")
        d = Path(path).resolve()
        if not d.is_dir():
            sys.exit(f"{d} is not a directory")
        roots.append((label or None, d))

    print("Discovering CryptOpt artifacts...")
    cands = [c for c in discover(roots) if c.curve in curves]
    vs = variants(cands, curves)
    print(f"{len(vs)} variants\n")
    for v in vs:
        print(f"  {v.target.curve:10s} {v}")
    if args.list:
        return 0
    print()

    out_csv = Path(args.output).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Back up every committed leaf we might overwrite, and restore on the way
    # out however we leave -- a failed build or a Ctrl-C must not leave a
    # candidate sitting in the AUCurves tree looking like the shipped one.
    backup = out_csv.parent / "generated_backup"
    for c in curves:
        t = TARGETS[c]
        (backup / t.crate).mkdir(parents=True, exist_ok=True)
        for f in t.asm.values():
            shutil.copy2(AUCURVES / t.crate / "generated" / f, backup / t.crate / f)

    rows, failures = [], 0
    try:
        for i, v in enumerate(vs, 1):
            t = v.target
            tag = f"[{i}/{len(vs)}] {t.curve} {v}"

            # Every variant starts from the committed leaves, so a previous
            # candidate cannot linger in a crate this one does not replace.
            for c in curves:
                for f in TARGETS[c].asm.values():
                    shutil.copy2(
                        backup / TARGETS[c].crate / f,
                        AUCURVES / TARGETS[c].crate / "generated" / f,
                    )
            try:
                install(v)
            except ValueError as e:
                warn(f"{tag}: {e}")
                failures += 1
                continue

            env = dict(os.environ)
            env.pop(f"{t.env_prefix}_NO_CRYPTOPT", None)
            if v.no_cryptopt:
                env[f"{t.env_prefix}_NO_CRYPTOPT"] = "1"

            print(f"{tag}: building")
            r = run(["cargo", "build", "--release", "-p", "cryptopt-bench"], env)
            if r.returncode:
                warn(f"{tag}: build failed:\n{r.stderr[-1500:]}")
                failures += 1
                continue

            verified = "skipped"
            if not args.no_verify:
                print(f"{tag}: verifying ({t.test})")
                r = run(
                    ["cargo", "test", "--release", "-p", t.crate, "--test", t.test], env
                )
                if r.returncode:
                    warn(
                        f"{tag}: CORRECTNESS CHECK FAILED, not benchmarking:\n"
                        f"{(r.stdout + r.stderr)[-1500:]}"
                    )
                    failures += 1
                    continue
                verified = t.test

            print(f"{tag}: measuring")
            # --require-asm unless we asked for the reference leaves: a
            # silent fall back to fiat would otherwise look like a candidate
            # that merely happens to match fiat's speed.
            cmd = ([] if args.cpu == "none" else ["taskset", "-c", args.cpu]) + [
                str(BENCH_BIN),
                "--curve",
                t.curve,
                "--rounds",
                str(args.rounds),
                "--round-ms",
                str(args.round_ms),
            ]
            if not v.no_cryptopt:
                cmd.append("--require-asm")
            r = run(cmd, env)
            if r.returncode:
                warn(f"{tag}: benchmark exited {r.returncode}:\n{r.stderr[-1500:]}")
                failures += 1
                continue

            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for line in r.stdout.splitlines():
                rec = dict(zip(BENCH_COLS, line.split(",")))
                rows.append(
                    {
                        "curve": rec["curve"],
                        "operation": rec["operation"],
                        "stage": v.stage,
                        "strategy": v.strategy,
                        "mul_seed": v.seed("mul"),
                        "square_seed": v.seed("square"),
                        **{k: rec[k] for k in BENCH_COLS[2:]},
                        "mul_asm": v.asm("mul"),
                        "square_asm": v.asm("square"),
                        "verified": verified,
                        "host": platform.node(),
                        "timestamp_utc": now,
                    }
                )
    finally:
        for c in curves:
            t = TARGETS[c]
            for f in t.asm.values():
                shutil.copy2(backup / t.crate / f, AUCURVES / t.crate / "generated" / f)
        print("Restored the committed leaves in AUCurves.")

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")
    if failures:
        warn(f"{failures} variant(s) failed; the CSV covers the rest")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
