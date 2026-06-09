#!/usr/bin/env bash
# Generate publication-ready plots, tables, and results.md from completed
# sweep results.
#
# Run after `run_sweep.sh` (or `run_eval.sh` if training was previously done).
#
# Outputs:
#   results/plots/*.png
#   results/tables/*.csv
#   docs/results.md  (regenerated)

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BENCH_DIR"
export PYTHONPATH="$BENCH_DIR:${PYTHONPATH:-}"

python -m velgate_bench.analyze --results-dir "$BENCH_DIR/results"
