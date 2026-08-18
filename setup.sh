#!/usr/bin/env bash

set -eoux pipefail

git submodule init
git submodule update

(
  cd CryptOpt
  patch -p1 < ../scheduler.patch
  make
)
