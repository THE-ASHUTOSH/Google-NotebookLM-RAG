#!/usr/bin/env bash
#
# run.sh — start the NotebookLM-style Corrective RAG app.
#
# Brings up ChromaDB (the JS client needs a running server), then the Node app.
# ChromaDB is auto-detected in this order:
#   1. an already-running server on CHROMA_PORT  -> reuse it
#   2. Docker                                    -> run chromadb/chroma
#   3. the `chroma` CLI (pip install chromadb)   -> chroma run
#
# A Chroma container started by this script is stopped again on exit.
#
# Usage:
#   ./run.sh                # start everything
#   PORT=3001 ./run.sh      # override the app port (default 3000, or .env)
#   APP_PORT=3001 ./run.sh  # same thing
#
set -euo pipefail

cd "$(dirname "$0")"

# ---- config (env overrides win) ----
CHROMA_IMAGE="chromadb/chroma:0.6.3"
CHROMA_CONTAINER="notebooklm-rag-chroma"
CHROMA_PORT="${CHROMA_PORT:-8000}"
CHROMA_URL_DEFAULT="http://localhost:${CHROMA_PORT}"
STARTED_CHROMA=""   # set to "docker" or "cli" if we start one (for cleanup)
CHROMA_CLI_PID=""

# Load .env so we can read PORT / CHROMA_URL (without clobbering shell env).
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
export CHROMA_URL="${CHROMA_URL:-$CHROMA_URL_DEFAULT}"
APP_PORT="${APP_PORT:-${PORT:-3000}}"
export PORT="$APP_PORT"

log()  { printf '\033[36m[run]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[run]\033[0m %s\n' "$*"; }
err()  { printf '\033[31m[run]\033[0m %s\n' "$*" >&2; }

chroma_up() {
  curl -fsS -o /dev/null "http://localhost:${CHROMA_PORT}/api/v2/heartbeat" 2>/dev/null \
    || curl -fsS -o /dev/null "http://localhost:${CHROMA_PORT}/api/v1/heartbeat" 2>/dev/null
}

cleanup() {
  echo
  if [ -n "$STARTED_CHROMA" ]; then
    log "Stopping ChromaDB ($STARTED_CHROMA)…"
    if [ "$STARTED_CHROMA" = "docker" ]; then
      docker rm -f "$CHROMA_CONTAINER" >/dev/null 2>&1 || true
    elif [ "$STARTED_CHROMA" = "cli" ] && [ -n "$CHROMA_CLI_PID" ]; then
      kill "$CHROMA_CLI_PID" >/dev/null 2>&1 || true
    fi
  fi
  log "Bye."
}
trap cleanup EXIT INT TERM

# ---- 1. prerequisites ----
command -v node >/dev/null 2>&1 || { err "Node.js not found. Install Node 20+."; exit 1; }

if [ ! -d node_modules ]; then
  log "Installing dependencies (npm install)…"
  npm install
fi

if [ ! -f .env ]; then
  warn "No .env found. Copy .env.example to .env and add your GEMINI_API_KEY."
  warn "  cp .env.example .env"
  exit 1
fi
if grep -q "your_google_ai_studio_api_key_here" .env 2>/dev/null || [ -z "${GEMINI_API_KEY:-}" ]; then
  warn "GEMINI_API_KEY looks unset in .env — get a free key at https://aistudio.google.com/apikey"
fi

# ---- 2. ChromaDB ----
if chroma_up; then
  log "ChromaDB already running on port ${CHROMA_PORT} — reusing it."
elif command -v docker >/dev/null 2>&1; then
  log "Starting ChromaDB via Docker ($CHROMA_IMAGE) on port ${CHROMA_PORT}…"
  docker rm -f "$CHROMA_CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CHROMA_CONTAINER" -p "${CHROMA_PORT}:8000" "$CHROMA_IMAGE" >/dev/null
  STARTED_CHROMA="docker"
elif command -v chroma >/dev/null 2>&1; then
  log "Starting ChromaDB via the chroma CLI on port ${CHROMA_PORT}…"
  chroma run --path ./chroma_db --port "${CHROMA_PORT}" >/dev/null 2>&1 &
  CHROMA_CLI_PID=$!
  STARTED_CHROMA="cli"
else
  err "No running ChromaDB, and neither Docker nor the 'chroma' CLI is available."
  err "Install one of them:"
  err "  • Docker, then re-run this script, or"
  err "  • pip install chromadb   (provides the 'chroma' CLI)"
  exit 1
fi

# wait for Chroma to become reachable
if [ -n "$STARTED_CHROMA" ]; then
  log "Waiting for ChromaDB to be ready…"
  for i in $(seq 1 30); do
    if chroma_up; then log "ChromaDB ready."; break; fi
    if [ "$i" -eq 30 ]; then err "ChromaDB did not start in time."; exit 1; fi
    sleep 1
  done
fi

# ---- 3. the Node app ----
if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${APP_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  err "Port ${APP_PORT} is already in use. Set a different one, e.g.:  PORT=3001 ./run.sh"
  exit 1
fi

log "Starting the app on http://localhost:${APP_PORT}  (Ctrl-C to stop)"
exec node server.js
