#!/usr/bin/env bash
# Starts the Warpmate dashboard: the analysis server on :8000 and the page on
# :5173. Open http://localhost:5173. Ctrl+C stops both.
#
# First time only:
#   python3 -m venv .venv && .venv/bin/pip install -r dashboard/server/requirements.txt
#   (cd dashboard/web && npm install)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"

(cd server && "$PY" -m uvicorn app:app --host 127.0.0.1 --port 8000) &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
cd web && npx vite --host 127.0.0.1 --port 5173
