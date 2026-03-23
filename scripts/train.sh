#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "${PROJECT_ROOT}/activate.sh" 2>/dev/null || {
    echo "Error: Environment not set up. Run ./run.sh setup first"
    exit 1
}

python3 "${PROJECT_ROOT}/src/train.py" --config "${PROJECT_ROOT}/configs/config.yaml" "$@"
