#!/usr/bin/env bash

set -eoux pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# For added reprodocibility, derive pseudo-random seed from a fixed constant
# via repeated md5-hashes.
seed="1337"
for _ in 1 2 3; do
  seed="$(printf '%s' "$seed" | md5sum | cut -d' ' -f1)"
  seed=$((16#${seed:0:8} % 10000))
done

python3 "$SCRIPT_DIR/gen_starting_states.py" --lookahead 1,2,3 --seed "$seed" "$@"
