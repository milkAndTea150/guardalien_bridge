#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MOCK_CODEGEN="${MOCK_CODEGEN:-1}"
export MOCK_REPAIR="${MOCK_REPAIR:-1}"

python -m uvicorn app:app --host 0.0.0.0 --port 8787 --reload
