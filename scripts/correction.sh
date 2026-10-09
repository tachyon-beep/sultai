#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec nice -n 10 timeout 120 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 -m sultai.correction "$@"
