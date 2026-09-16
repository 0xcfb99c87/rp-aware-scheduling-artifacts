#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import sys
import operator
from typing import TypedDict
from pathlib import Path
from itertools import groupby

def fatal(msg: str, *args, **kwargs) -> None:
    log(msg, *args, **kwargs)
    sys.exit(1)


def log(msg: str, prefix: str | None = "extract_best_states.py: ", *args, **kwargs):
    prefix = "" if prefix is None else prefix
    print(f"{prefix}{msg}", file=sys.stderr, *args, **kwargs)


class Run(TypedDict):
    curve: str
    method: str
    seed: int
    scheduler: str
    ratio: float
    cycles: float
    state_file: Path


def discover_starting_states(dir: Path) -> list[Run]:
    runs: list[Run] = []
    for state in filter(
        lambda p: os.path.isfile(p.with_suffix(".cycles")), dir.rglob("*.json")
    ):
        # Yeah, we could technically json parse the state file and get the
        # curve/method/scheduler/seed that way, but since it's encoded into the
        # filepath we might as well do manually parsing since it bypasses the
        # additional IO. Hacky code ftw.
        curve, method, scheduler = state.with_suffix("").name.rsplit("_", 2)
        seed = [
            int(part.lstrip("seed")) for part in state.parts if part.startswith("seed")
        ][0]
        cycles_cryptopt, cycles_cc = [
            float(x) for x in open(state.with_suffix(".cycles")).read().split()
        ]
        run: Run = {
            "curve": curve,
            "seed": seed,
            "method": method,
            "scheduler": scheduler,
            "cycles": cycles_cryptopt,
            "ratio": cycles_cc / cycles_cryptopt,
            "state_file": state,
        }
        runs.append(run)
    return runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursively extract best state files from a collection of CryptOpt starting-state JSONs."
        "Assumes that state files have a corresponding $state.cycles file, as produced by ./gen_starting_states.py.",
        usage="%(prog)s [flags] [root]",
    )
    parser.add_argument(
        "-i",
        "--in",
        dest="in_dir",
        required=True,
        help="directory in which state files should be searched for",
    )
    parser.add_argument(
        "-s",
        "--sort",
        dest="sort_by",
        choices=["ratio", "cycles"],
        default="cycles",
        help="whether to select by ratio (cycles_cc_baseline/cycles_cryptopt) or raw cycle count",
    )
    parser.add_argument(
        "-o",
        "--out",
        dest="out_dir",
        required=True,
        help="directory to extract state files from",
    )
    return parser.parse_args()


def main() -> None:
    pass


if __name__ == "__main__":
    args = parse_args()

    in_dir = Path(args.in_dir)
    if not os.path.isdir(in_dir):
        fatal(f"invalid input directory: {in_dir}")

    out_dir = Path(args.out_dir)
    if os.path.exists(out_dir):
        fatal(f"output directory {out_dir} already exists, refusing to overwrite")

    states = discover_starting_states(in_dir)
    if not states:
        fatal(f"no valid state files found under {in_dir}")

    os.makedirs(out_dir)

    group_key = operator.itemgetter("curve", "method", "scheduler")
    for i, g in groupby(sorted(states, key=group_key), key=group_key):
        log(f"finding candidate for {'/'.join(i)}...", end="")
        winner = sorted(g, key=operator.itemgetter(args.sort_by))[
            0 if args.sort_by == "cycles" else -1
        ]
        log(
            f" using seed {winner['seed']} ({args.sort_by}={winner[args.sort_by]})",
            prefix=None,
        )
        state_file = winner["state_file"]
        seed = winner["seed"]
        out_name = state_file.with_suffix("").name + f"_seed{seed}" + state_file.suffix
        shutil.copy(state_file, out_dir / out_name)
