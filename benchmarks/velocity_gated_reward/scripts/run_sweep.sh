#!/usr/bin/env bash
# Run the full velocity-gated reward benchmark sweep.
#
# Usage:
#   bash scripts/run_sweep.sh                          # default config
#   bash scripts/run_sweep.sh configs/sweep_quick.yaml # smoke test
#
# Assumes the env_isaaclab conda environment is active and the benchmarks
# package is importable (run from this benchmark's root, or set PYTHONPATH).

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$BENCH_DIR/configs/sweep_default.yaml}"

cd "$BENCH_DIR"
export PYTHONPATH="$BENCH_DIR:${PYTHONPATH:-}"

echo "[velgate] Bench dir : $BENCH_DIR"
echo "[velgate] Config    : $CONFIG"
echo "[velgate] PYTHONPATH: $PYTHONPATH"
echo ""

python -m velgate_bench.sweep --config "$CONFIG"
