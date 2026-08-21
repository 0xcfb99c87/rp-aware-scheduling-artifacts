#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent

CSV_HEADER = [
    "curve",
    "method",
    "scheduling_strategy",
    "pm_lookahead",
    "best_seed",
    "best_cycles",
    "num_trials",
    "state_file",
]


def algo_label(scheduling_algorithm: str, pm_lookahead: int) -> str:
    if scheduling_algorithm == "pressure-minimized":
        return f"pressure-minimized-la{pm_lookahead}"
    return scheduling_algorithm


def read_cycles(json_path: Path) -> float | None:
    cycles_path = json_path.with_suffix(".cycles")
    if not cycles_path.exists():
        return None
    raw = cycles_path.read_text().strip()
    if not raw:
        return None
    try:
        return float(raw.split()[0])
    except ValueError:
        return None


def discover_candidates(root: Path) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}

    for json_path in sorted(root.rglob("*.json")):
        try:
            state = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"skipping {json_path}: {e}", file=sys.stderr)
            continue

        parsed = state.get("parsedArgs")
        if not parsed:
            print(
                f"skipping {json_path}: no parsedArgs (not a CryptOpt state file)",
                file=sys.stderr,
            )
            continue

        cycles = read_cycles(json_path)
        if cycles is None:
            print(
                f"skipping {json_path}: no usable .cycles file next to it",
                file=sys.stderr,
            )
            continue

        curve = parsed.get("curve", "unknown-curve")
        method = parsed.get("method", "unknown-method")
        scheduling_algorithm = parsed.get("schedulingAlgorithm", "default")
        pm_lookahead = parsed.get("pmLookahead", 1)
        seed = parsed.get("seed", "unknown-seed")

        key = (curve, method, scheduling_algorithm, pm_lookahead)
        groups.setdefault(key, []).append(
            {
                "path": json_path,
                "cycles": cycles,
                "seed": seed,
            }
        )

    return groups


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pick the best-performing starting state (lowest achieved cycle count) "
            "for each curve/method/scheduling-strategy combination across all "
            "seed<seed>/ trials under root, and lay the winners out in a single "
            "directory suitable for `bench.py`."
        )
    )
    parser.add_argument(
        "root",
        help="directory to search for seed<seed>/ trials (structure of artifacts_start_state)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_ROOT / "artifacts_best_start_state"),
        help="directory to write the selected starting states to (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="path to write a CSV summary of the selection (default: <out-dir>/selection_summary.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="wipe --out-dir first if it already exists",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        parser.error(f"root {root} is not a directory")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.force:
            parser.error(f"{out_dir} already exists (pass --force to overwrite)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else out_dir / "selection_summary.csv"
    )

    groups = discover_candidates(root)
    if not groups:
        print(
            f"select_best_states.py: no usable starting states found under {root}",
            file=sys.stderr,
        )
        return 1

    rows = []
    for (curve, method, scheduling_algorithm, pm_lookahead), candidates in sorted(
        groups.items()
    ):
        best = min(candidates, key=lambda c: c["cycles"])

        dest_dir = out_dir / curve / method
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_json = dest_dir / best["path"].name
        shutil.copy2(best["path"], dest_json)

        for ext in (".asm", ".cycles"):
            if ext == ".asm":
                sibling = next(
                    best["path"].parent.glob(f"{best['path'].stem}_seed*_ratio*.asm"),
                    None,
                )
            else:
                sibling = best["path"].with_suffix(ext)
            if sibling and sibling.exists():
                shutil.copy2(sibling, dest_dir / sibling.name)

        rows.append(
            [
                curve,
                method,
                scheduling_algorithm,
                pm_lookahead,
                best["seed"],
                best["cycles"],
                len(candidates),
                str(dest_json.relative_to(out_dir)),
            ]
        )

        label = algo_label(scheduling_algorithm, pm_lookahead)
        print(
            f"{curve}/{method}/{label}: best={best['cycles']:.2f} cycles "
            f"(seed{best['seed']}, {len(candidates)} trial(s) considered)"
        )

    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    print(f"\nSelected {len(rows)} starting state(s) -> {out_dir}")
    print(f"Summary written to {summary_csv}")
    if len(rows) != 52:
        print(
            f"select_best_states.py: warning: expected 52 curve/method/scheduler "
            f"combinations, got {len(rows)} (root may only cover a subset)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
