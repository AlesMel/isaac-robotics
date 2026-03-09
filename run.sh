#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-}"
ISAACLAB_SCRIPT=""

if [[ -z "${ISAACLAB_DIR}" ]]; then
  if [[ $# -gt 0 && -d "$1" ]]; then
    ISAACLAB_DIR="$1"
    shift
  else
    echo "Usage: $0 /path/to/IsaacLab [train.py args...]" >&2
    echo "   or: ISAACLAB_DIR=/path/to/IsaacLab $0 [train.py args...]" >&2
    exit 1
  fi
fi

if [[ -x "${ISAACLAB_DIR}/isaaclab.sh" ]]; then
  ISAACLAB_SCRIPT="${ISAACLAB_DIR}/isaaclab.sh"
elif [[ -x "${ISAACLAB_DIR}/isaac-sim.sh" ]]; then
  ISAACLAB_SCRIPT="${ISAACLAB_DIR}/isaac-sim.sh"
else
  echo "Could not find an executable isaaclab launcher in ${ISAACLAB_DIR}" >&2
  echo "Expected one of: isaaclab.sh or isaac-sim.sh" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${ISAACLAB_SCRIPT}" -p "${ROOT_DIR}/train.py" "$@"
