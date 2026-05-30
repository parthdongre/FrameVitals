#!/usr/bin/env bash
# ============================================================================
# DataLens AI — one-shot launcher  (macOS / Linux)
#
# Starts:
#   1. Ollama daemon          — if installed and not already on :11434
#   2. Flask backend          — :5055
#   3. Vite React dev server  — :5173
#
# Logs → ./logs/   ·   Ctrl+C tears everything down cleanly.
#
# ── flags ────────────────────────────────────────────────────────────────────
#   SKIP_OLLAMA=1    ./run.sh   don't touch Ollama at all
#   SKIP_FRONTEND=1  ./run.sh   Flask only, no Vite
#   OPEN_BROWSER=1   ./run.sh   auto-open the dashboard
#   LOCAL_MODEL=1    ./run.sh   use qwen3:4b locally instead of Ollama Cloud
#   OLLAMA_MODEL=x   ./run.sh   use any specific model name
# ============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ── terminal colours (teal = 38;5;86  ·  ink = 2  ·  reset = 0) ─────────────
_T='\033[38;5;86m'   # teal accent   ≈ #5eead4
_B='\033[1m'         # bold
_D='\033[2m'         # dim  (ink-3)
_R='\033[0m'         # reset
_G='\033[38;5;84m'   # green ok
_Y='\033[38;5;221m'  # amber warn
_E='\033[38;5;203m'  # red error

log()  { printf "${_T}▸${_R}  %s\n"    "$*"; }
ok()   { printf "${_G}✓${_R}  %s\n"    "$*"; }
warn() { printf "${_Y}!${_R}  %s\n"    "$*"; }
fail() { printf "${_E}✗${_R}  %s\n" "$*" >&2; exit 1; }

have()         { command -v "$1" >/dev/null 2>&1; }
port_in_use()  { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

wait_for_port() {
    local port="$1" label="$2" timeout="${3:-30}" i=0
    while ! port_in_use "${port}"; do
        sleep 0.5
        i=$((i + 1))
        [ $i -ge $((timeout * 2)) ] && { warn "${label} did not come up on :${port}"; return 1; }
    done
}

# ── directories ───────────────────────────────────────────────────────────────
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}" uploads cleaned static/charts

VENV_PY="${ROOT_DIR}/venv/bin/python"
[ -x "${VENV_PY}" ] || fail "venv not found at ${VENV_PY} — run ./install.sh first."

# ── LLM model ─────────────────────────────────────────────────────────────────
if [ "${LOCAL_MODEL:-0}" = "1" ]; then
    export DATALENS_OLLAMA_ALLOW_CLOUD="${DATALENS_OLLAMA_ALLOW_CLOUD:-0}"
    export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b}"
else
    export DATALENS_OLLAMA_ALLOW_CLOUD="${DATALENS_OLLAMA_ALLOW_CLOUD:-1}"
    export OLLAMA_MODEL="${OLLAMA_MODEL:-gpt-oss:120b-cloud}"
fi

# ── opening card — mirrors the site's landing headline ───────────────────────
printf "\n"
printf "${_D}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${_R}\n"
printf "\n"
printf "  ${_T}DATALENS  ·  AI${_R}\n"
printf "\n"
printf "  ${_B}Read the signal in your data.${_R}\n"
printf "\n"
printf "  ${_D}Upload a dataset and get a structured, evidence-backed${_R}\n"
printf "  ${_D}report on quality, ML readiness, anomalies, time-series,${_R}\n"
printf "  ${_D}drift, and a local-LLM narrative — all in one pass.${_R}\n"
printf "\n"
printf "${_D}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${_R}\n"
printf "\n"

# ── 1. Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_PID=""

if [ "${SKIP_OLLAMA:-0}" = "1" ]; then
    warn "SKIP_OLLAMA=1 — skipping Ollama"
elif ! have ollama; then
    warn "ollama not found — LLM agent will use the heuristic fallback"
else
    if port_in_use 11434; then
        ok "Ollama already running on :11434"
    else
        log "Starting Ollama daemon → ${LOG_DIR}/ollama.log"
        ollama serve >"${LOG_DIR}/ollama.log" 2>&1 &
        OLLAMA_PID=$!
        if wait_for_port 11434 "Ollama" 12; then
            ok "Ollama  :11434  (pid ${OLLAMA_PID})"
        else
            warn "Ollama did not come up — agent falls back to heuristic"
            OLLAMA_PID=""
        fi
    fi
fi

# ── 2. Flask backend ──────────────────────────────────────────────────────────
FLASK_PID=""

if port_in_use 5055; then
    warn "Port 5055 already in use — skipping Flask start"
else
    log "Starting Flask backend → ${LOG_DIR}/flask.log"
    "${VENV_PY}" "${ROOT_DIR}/app.py" >"${LOG_DIR}/flask.log" 2>&1 &
    FLASK_PID=$!
    if wait_for_port 5055 "Flask" 30; then
        ok "Flask   :5055  (pid ${FLASK_PID})"
    else
        fail "Flask failed to start. Check ${LOG_DIR}/flask.log"
    fi
fi

# ── 2b. Streamlit visual console ─────────────────────────────────────────────
STREAMLIT_PID=""

if [ "${SKIP_STREAMLIT:-0}" = "1" ]; then
    warn "SKIP_STREAMLIT=1 — skipping Streamlit console"
elif [ ! -f "${ROOT_DIR}/streamlit_app.py" ]; then
    warn "streamlit_app.py not found — skipping Streamlit console"
elif ! "${VENV_PY}" -c "import streamlit" >/dev/null 2>&1; then
    warn "streamlit not installed — skipping console (run ./install.sh)"
elif port_in_use 8501; then
    warn "Port 8501 already in use — skipping Streamlit start"
else
    log "Starting Streamlit console → ${LOG_DIR}/streamlit.log"
    "${VENV_PY}" -m streamlit run "${ROOT_DIR}/streamlit_app.py" \
        --server.port 8501 \
        --server.headless true \
        --browser.gatherUsageStats false \
        >"${LOG_DIR}/streamlit.log" 2>&1 &
    STREAMLIT_PID=$!
    if wait_for_port 8501 "Streamlit" 30; then
        ok "Streamlit :8501  (pid ${STREAMLIT_PID})"
    else
        warn "Streamlit did not come up — see ${LOG_DIR}/streamlit.log"
        STREAMLIT_PID=""
    fi
fi

# ── 3. Vite dev server ────────────────────────────────────────────────────────
VITE_PID=""

if [ "${SKIP_FRONTEND:-0}" = "1" ]; then
    warn "SKIP_FRONTEND=1 — skipping Vite"
elif [ ! -d "${ROOT_DIR}/frontend" ]; then
    warn "frontend/ not found — skipping Vite"
elif ! have npm; then
    warn "npm not found — install Node 18+ to enable the dashboard"
elif [ ! -d "${ROOT_DIR}/frontend/node_modules" ]; then
    warn "frontend/node_modules missing — run ./install.sh first"
elif port_in_use 5173; then
    warn "Port 5173 already in use — skipping Vite start"
else
    log "Starting Vite dev server → ${LOG_DIR}/vite.log"
    (cd "${ROOT_DIR}/frontend" && npm run dev) >"${LOG_DIR}/vite.log" 2>&1 &
    VITE_PID=$!
    if wait_for_port 5173 "Vite" 30; then
        ok "Vite    :5173  (pid ${VITE_PID})"
    else
        warn "Vite did not come up — only Flask is reachable"
        VITE_PID=""
    fi
fi

# ── 4. Open browser ───────────────────────────────────────────────────────────
if [ "${OPEN_BROWSER:-0}" = "1" ]; then
    sleep 0.4
    OPEN_URL="http://127.0.0.1:5055/"
    [ -n "${VITE_PID}" ] && OPEN_URL="http://127.0.0.1:5173/"
    if have open;      then open "${OPEN_URL}"     || true   # macOS
    elif have xdg-open; then xdg-open "${OPEN_URL}" || true  # Linux
    fi
fi

# ── 5. Ready banner ───────────────────────────────────────────────────────────
printf "\n"
printf "${_D}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${_R}\n"
printf "\n"

if   port_in_use 5173; then
    printf "  ${_T}●${_R}  ${_B}Dashboard${_R}  →  ${_B}http://127.0.0.1:5173/${_R}   ${_T}← open this${_R}\n"
fi
if   port_in_use 8501; then
    printf "  ${_T}●${_R}  ${_B}Console${_R}    →  ${_B}http://127.0.0.1:8501/${_R}   ${_T}← step-by-step visual${_R}\n"
fi
printf "  ${_D}●  Flask API  →  http://127.0.0.1:5055/${_R}\n"
printf "  ${_D}●  Health     →  http://127.0.0.1:5055/api/health${_R}\n"
printf "\n"
printf "  ${_D}LLM   ${OLLAMA_MODEL}  (cloud=${DATALENS_OLLAMA_ALLOW_CLOUD})${_R}\n"
printf "  ${_D}Logs  ${LOG_DIR}/${_R}\n"
printf "\n"
printf "  ${_D}Ctrl+C to stop${_R}\n"
printf "\n"
printf "${_D}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${_R}\n"
printf "\n"

# ── 6. Shutdown ───────────────────────────────────────────────────────────────
cleanup() {
    printf "\n"
    log "Shutting down…"
    [ -n "${VITE_PID}"       ] && { kill "${VITE_PID}"       2>/dev/null; ok "Stopped Vite       (${VITE_PID})";       }
    [ -n "${STREAMLIT_PID}"  ] && { kill "${STREAMLIT_PID}"  2>/dev/null; ok "Stopped Streamlit  (${STREAMLIT_PID})";  }
    [ -n "${FLASK_PID}"      ] && { kill "${FLASK_PID}"      2>/dev/null; ok "Stopped Flask      (${FLASK_PID})";      }
    [ -n "${OLLAMA_PID}"     ] && { kill "${OLLAMA_PID}"     2>/dev/null; ok "Stopped Ollama     (${OLLAMA_PID})";     }
    exit 0
}
trap cleanup INT TERM

# ── 7. Wait ───────────────────────────────────────────────────────────────────
if   [ -n "${FLASK_PID}" ]; then wait "${FLASK_PID}"
elif [ -n "${VITE_PID}"  ]; then wait "${VITE_PID}"
else while true; do sleep 60; done   # both were already running
fi
