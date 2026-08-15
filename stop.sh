#!/usr/bin/env bash
# ============================================================================
# FrameVitals — emergency stop
#
# Forcefully shuts down every FrameVitals development process, even if
# run.sh's Ctrl+C trap was bypassed (closed terminal, suspend, force-quit, etc.).
#
# Three-pass kill:
#   1. By port    — every listener on :5055 :5173 :8501 :11434
#   2. By name    — any python/node/streamlit/ollama process whose argv
#                   mentions this project root
#   3. SIGKILL    — anything still alive after a 2-second grace period
#
# Usage:
#     ./stop.sh                # stop everything
#     ./stop.sh --keep-ollama  # leave the Ollama daemon running
#     ./stop.sh --quiet        # only print what was actually killed
# ============================================================================

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── flags ─────────────────────────────────────────────────────────────────────
KEEP_OLLAMA=0
QUIET=0
for arg in "$@"; do
    case "$arg" in
        --keep-ollama) KEEP_OLLAMA=1 ;;
        --quiet|-q)    QUIET=1 ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

# ── colours ──────────────────────────────────────────────────────────────────
_T='\033[38;5;86m'
_R='\033[0m'
_G='\033[38;5;84m'
_Y='\033[38;5;221m'
_E='\033[38;5;203m'
_D='\033[2m'

log()  { [ "$QUIET" = "0" ] && printf "${_T}▸${_R}  %s\n" "$*"; }
ok()   { printf "${_G}✓${_R}  %s\n" "$*"; }
warn() { [ "$QUIET" = "0" ] && printf "${_Y}!${_R}  %s\n" "$*"; }
fail() { printf "${_E}✗${_R}  %s\n" "$*" >&2; }

KILLED_TOTAL=0

# ── pass 1: kill by port ─────────────────────────────────────────────────────

kill_port() {
    local port="$1" label="$2" sig="${3:-TERM}"
    local pids
    pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -z "$pids" ]; then
        return 0
    fi
    for pid in $pids; do
        if kill "-${sig}" "$pid" 2>/dev/null; then
            ok "killed ${label} (pid ${pid}, port ${port}, SIG${sig})"
            KILLED_TOTAL=$((KILLED_TOTAL + 1))
        fi
    done
}

log "Pass 1 — terminating by port"
kill_port 5173  "Vite"
kill_port 8501  "Streamlit"
kill_port 5055  "Flask"
if [ "$KEEP_OLLAMA" = "0" ]; then
    kill_port 11434 "Ollama"
fi

# ── pass 2: kill by command pattern ──────────────────────────────────────────
# Catch processes that aren't bound to a port yet (mid-startup) or that
# slipped past the lsof scan because they bound late.

log "Pass 2 — terminating by command pattern"

kill_pattern() {
    local pattern="$1" label="$2"
    local pids
    # Match processes whose full command line contains both the project root
    # and the pattern. Avoids killing unrelated python/node processes.
    pids=$(pgrep -f "${ROOT_DIR}.*${pattern}" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        return 0
    fi
    for pid in $pids; do
        if kill -TERM "$pid" 2>/dev/null; then
            ok "killed ${label} (pid ${pid})"
            KILLED_TOTAL=$((KILLED_TOTAL + 1))
        fi
    done
}

kill_pattern "app\.py"          "Flask app.py"
kill_pattern "streamlit_app"    "Streamlit app"
kill_pattern "vite"             "Vite dev server"
kill_pattern "node.*frontend"   "Node frontend process"

# Ollama daemon doesn't include the project root in its argv
if [ "$KEEP_OLLAMA" = "0" ]; then
    pids=$(pgrep -f "ollama serve" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            if kill -TERM "$pid" 2>/dev/null; then
                ok "killed Ollama serve (pid ${pid})"
                KILLED_TOTAL=$((KILLED_TOTAL + 1))
            fi
        done
    fi
fi

# ── pass 3: SIGKILL the survivors ────────────────────────────────────────────

sleep 2

log "Pass 3 — force-killing survivors"

force_kill_port() {
    local port="$1" label="$2"
    local pids
    pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -z "$pids" ]; then
        return 0
    fi
    for pid in $pids; do
        if kill -KILL "$pid" 2>/dev/null; then
            warn "force-killed ${label} (pid ${pid}, port ${port}, SIGKILL)"
            KILLED_TOTAL=$((KILLED_TOTAL + 1))
        fi
    done
}

force_kill_port 5173  "Vite"
force_kill_port 8501  "Streamlit"
force_kill_port 5055  "Flask"
[ "$KEEP_OLLAMA" = "0" ] && force_kill_port 11434 "Ollama"

# ── pass 4: housekeeping ─────────────────────────────────────────────────────

# Some Streamlit installs leave a stale lock file behind that prevents the
# next start. Same for Vite's optimize cache when killed mid-bundle.
rm -f /tmp/.streamlit-* 2>/dev/null
rm -rf "${ROOT_DIR}/frontend/node_modules/.vite" 2>/dev/null

# ── summary ──────────────────────────────────────────────────────────────────

echo ""
if [ "$KILLED_TOTAL" -eq 0 ]; then
    printf "${_D}Nothing was running. Project is already stopped.${_R}\n"
else
    printf "${_T}FrameVitals fully stopped. ${KILLED_TOTAL} process(es) terminated.${_R}\n"
fi

# Final port check — anything still listening is a problem
LEAKS=""
for port_label in "5055:Flask" "5173:Vite" "8501:Streamlit"; do
    port="${port_label%%:*}"
    label="${port_label##*:}"
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        LEAKS="${LEAKS} ${label}(${port})"
    fi
done

if [ -n "$LEAKS" ]; then
    fail "Ports still in use:${LEAKS}"
    fail "Manual fix: lsof -nP -iTCP:<port> -sTCP:LISTEN"
    exit 1
fi
exit 0
