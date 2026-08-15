#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

log() { printf "\033[1;36m▶\033[0m %s\n" "$*"; }
ok() { printf "\033[1;32m✓\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!\033[0m %s\n" "$*"; }
fail() { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

port_in_use() {
    if have lsof; then
        lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
    else
        return 1
    fi
}

wait_for_port() {
    local port="$1" label="$2" timeout="${3:-30}" i=0
    while ! port_in_use "${port}"; do
        sleep 0.5
        i=$((i + 1))
        if [ "${i}" -ge $((timeout * 2)) ]; then
            warn "${label} did not come up on :${port}"
            return 1
        fi
    done
}

VENV_PY="${ROOT_DIR}/.venv/bin/python"
[ -x "${VENV_PY}" ] || fail ".venv not found; run ./install.sh first."

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}" uploads cleaned static/charts reports

export OPENROUTER_APP_NAME="${OPENROUTER_APP_NAME:-FrameVitals}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b}"

printf "\nFrameVitals development stack\n\n"

OLLAMA_PID=""
FLASK_PID=""
STREAMLIT_PID=""
VITE_PID=""

if [ "${SKIP_OLLAMA:-0}" = "1" ]; then
    warn "Skipping Ollama"
elif have ollama; then
    if port_in_use 11434; then
        ok "Ollama already running on :11434"
    else
        log "Starting Ollama"
        ollama serve >"${LOG_DIR}/ollama.log" 2>&1 &
        OLLAMA_PID=$!
        wait_for_port 11434 "Ollama" 12 || OLLAMA_PID=""
    fi
else
    warn "Ollama is not installed; AI features will use available fallbacks"
fi

if port_in_use 5055; then
    warn "Port 5055 already in use; not starting another Flask process"
else
    log "Starting Flask API"
    "${VENV_PY}" "${ROOT_DIR}/app.py" >"${LOG_DIR}/flask.log" 2>&1 &
    FLASK_PID=$!
    wait_for_port 5055 "Flask" 30 || fail "Flask failed; see logs/flask.log"
fi

if [ "${SKIP_STREAMLIT:-0}" != "1" ] && [ -f streamlit_app.py ]; then
    if "${VENV_PY}" -c "import streamlit" >/dev/null 2>&1; then
        if port_in_use 8501; then
            warn "Port 8501 already in use; not starting another Streamlit process"
        else
            log "Starting Streamlit console"
            "${VENV_PY}" -m streamlit run "${ROOT_DIR}/streamlit_app.py" \
                --server.port 8501 \
                --server.headless true \
                --browser.gatherUsageStats false \
                >"${LOG_DIR}/streamlit.log" 2>&1 &
            STREAMLIT_PID=$!
            wait_for_port 8501 "Streamlit" 30 || STREAMLIT_PID=""
        fi
    else
        warn "Streamlit extra is not installed"
    fi
fi

if [ "${SKIP_FRONTEND:-0}" != "1" ] && [ -d frontend ]; then
    if have npm && [ -d frontend/node_modules ]; then
        if port_in_use 5173; then
            warn "Port 5173 already in use; not starting another Vite process"
        else
            log "Starting React dashboard"
            (cd frontend && npm run dev) >"${LOG_DIR}/vite.log" 2>&1 &
            VITE_PID=$!
            wait_for_port 5173 "Vite" 30 || VITE_PID=""
        fi
    else
        warn "Frontend dependencies are missing; run ./install.sh"
    fi
fi

printf "\n"
port_in_use 5173 && printf "Dashboard: http://127.0.0.1:5173/\n" || true
port_in_use 8501 && printf "Console:   http://127.0.0.1:8501/\n" || true
printf "API:       http://127.0.0.1:5055/\n"
printf "Health:    http://127.0.0.1:5055/api/health\n"
printf "Logs:      %s\n\n" "${LOG_DIR}"
printf "Press Ctrl+C to stop processes started by this script.\n"

if [ "${OPEN_BROWSER:-0}" = "1" ]; then
    URL="http://127.0.0.1:5055/"
    port_in_use 5173 && URL="http://127.0.0.1:5173/"
    if have open; then open "${URL}" || true
    elif have xdg-open; then xdg-open "${URL}" || true
    fi
fi

cleanup() {
    printf "\n"
    log "Shutting down"
    [ -n "${VITE_PID}" ] && kill "${VITE_PID}" 2>/dev/null || true
    [ -n "${STREAMLIT_PID}" ] && kill "${STREAMLIT_PID}" 2>/dev/null || true
    [ -n "${FLASK_PID}" ] && kill "${FLASK_PID}" 2>/dev/null || true
    [ -n "${OLLAMA_PID}" ] && kill "${OLLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ -n "${FLASK_PID}" ]; then
    wait "${FLASK_PID}"
elif [ -n "${VITE_PID}" ]; then
    wait "${VITE_PID}"
else
    while true; do sleep 60; done
fi
