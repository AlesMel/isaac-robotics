#!/usr/bin/env bash
# Run only the eval phase (deterministic rollouts on existing checkpoints).
# Useful if training was completed previously and you just want to (re)collect
# the hold-quality metrics.
#
# Usage:
#   bash scripts/run_eval.sh                          # default config
#   bash scripts/run_eval.sh configs/sweep_quick.yaml

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$BENCH_DIR/configs/sweep_default.yaml}"

cd "$BENCH_DIR"
export PYTHONPATH="$BENCH_DIR:${PYTHONPATH:-}"

python -m velgate_bench.sweep --config "$CONFIG" --eval_only
