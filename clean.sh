#!/usr/bin/env bash

set -eoux pipefail

rm -rf artifacts
rm -rf __pycache__
git submodule foreach --recursive '
  git reset --hard
  git clean -fdx
'
