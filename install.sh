#!/usr/bin/env bash
# ============================================================================
# DataLens AI — One-shot installer
# Sets up the Python venv, Python deps, frontend npm deps, and (optionally)
# pulls the local Ollama models the agent uses.
#
# Safe to re-run: skips work that's already done.
#
# Usage:
#     ./install.sh                # full install
#     SKIP_OLLAMA=1 ./install.sh  # skip pulling Ollama models
#     PYTHON=python3.12 ./install.sh   # override the Python interpreter
# ============================================================================

set -e   # exit on first error
set -u   # error on unset vars

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ----- helpers ---------------------------------------------------------------

log()   { printf "\033[1;36m▶\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m✓\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m!\033[0m %s\n" "$*"; }
fail()  { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }

have()  { command -v "$1" >/dev/null 2>&1; }

# ----- 1. Python venv --------------------------------------------------------

PYTHON_BIN="${PYTHON:-}"
if [ -z "${PYTHON_BIN}" ]; then
    if have python3.14; then PYTHON_BIN="python3.14"
    elif have python3.13; then PYTHON_BIN="python3.13"
    elif have python3.12; then PYTHON_BIN="python3.12"
    elif have python3.11; then PYTHON_BIN="python3.11"
    elif have python3;    then PYTHON_BIN="python3"
    else fail "No Python 3 interpreter found. Install Python 3.11+ first."
    fi
fi

VENV_DIR="${ROOT_DIR}/venv"
VENV_PY="${VENV_DIR}/bin/python"

if [ -x "${VENV_PY}" ]; then
    ok "venv already present at ${VENV_DIR}"
else
    log "Creating venv with ${PYTHON_BIN} → ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    ok "venv created"
fi

# Sanity-check the interpreter (handles relocated venvs whose pip script has a stale shebang)
if ! "${VENV_PY}" -c "import sys" >/dev/null 2>&1; then
    fail "venv interpreter is broken at ${VENV_PY}. Delete the venv folder and re-run."
fi

# ----- 2. Python dependencies ------------------------------------------------

log "Upgrading pip in the venv"
"${VENV_PY}" -m pip install --upgrade pip --quiet
ok "pip upgraded"

log "Installing Python dependencies (this can take a few minutes the first time)"
"${VENV_PY}" -m pip install -r requirements.txt
ok "Python dependencies installed"

# fpdf2 sometimes ships alongside the legacy `fpdf` package, which shadows it.
# Cleanly remove the legacy one if present.
if "${VENV_PY}" -c "import fpdf, sys; sys.exit(0 if fpdf.__version__.startswith('2.') else 1)" 2>/dev/null; then
    ok "fpdf2 active in the venv"
else
    warn "Legacy PyFPDF detected — replacing with fpdf2"
    "${VENV_PY}" -m pip uninstall -y fpdf >/dev/null 2>&1 || true
    "${VENV_PY}" -m pip install --force-reinstall --no-deps fpdf2 --quiet
    ok "fpdf2 reinstalled"
fi

# ----- 3. Smoke-test the imports --------------------------------------------

log "Smoke-testing critical Python imports"
"${VENV_PY}" - <<'PY'
mods = [
    "pandas", "numpy", "matplotlib", "seaborn", "streamlit", "ollama",
    "flask", "scipy", "statsmodels", "pingouin",
    "sklearn", "xgboost", "lightgbm", "imblearn", "optuna", "pyod", "shap",
    "plotly", "missingno", "fpdf", "jinja2", "pydantic", "joblib", "loguru",
    "pytest", "hypothesis",
]
import importlib
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m} ({exc})")
if missing:
    print("MISSING:", ", ".join(missing))
    raise SystemExit(1)
print(f"OK: all {len(mods)} libraries import")
PY
ok "All critical Python libraries import"

# ----- 4. Frontend dependencies (npm) ---------------------------------------

if [ -d "${ROOT_DIR}/frontend" ]; then
    if have npm; then
        log "Installing frontend npm dependencies"
        if [ -d "${ROOT_DIR}/frontend/node_modules" ] && [ -f "${ROOT_DIR}/frontend/package-lock.json" ]; then
            (cd "${ROOT_DIR}/frontend" && npm install --silent)
        else
            (cd "${ROOT_DIR}/frontend" && npm install)
        fi
        ok "Frontend dependencies installed"
    else
        warn "npm not found — skipping frontend install. Install Node 18+ to enable the React dashboard."
    fi
else
    warn "frontend/ folder not found — skipping"
fi

# ----- 5. Ollama models (optional) ------------------------------------------

if [ "${SKIP_OLLAMA:-0}" = "1" ]; then
    warn "SKIP_OLLAMA=1 set — skipping Ollama model pulls"
elif ! have ollama; then
    warn "ollama not installed. Install from https://ollama.com to enable the local LLM agent."
    warn "  (the project still works — it falls back to a deterministic writer when Ollama is offline.)"
else
    log "Pulling Ollama models needed by the agent"

    OLLAMA_CHAT_MODEL="${OLLAMA_MODEL:-qwen3:4b}"
    OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

    log "Pulling chat model: ${OLLAMA_CHAT_MODEL}"
    if ollama pull "${OLLAMA_CHAT_MODEL}"; then
        ok "Pulled ${OLLAMA_CHAT_MODEL}"
    else
        warn "Failed to pull ${OLLAMA_CHAT_MODEL}. The agent will use OpenRouter or the heuristic fallback."
    fi

    log "Pulling embedding model: ${OLLAMA_EMBED_MODEL}"
    if ollama pull "${OLLAMA_EMBED_MODEL}"; then
        ok "Pulled ${OLLAMA_EMBED_MODEL}"
    else
        warn "Failed to pull ${OLLAMA_EMBED_MODEL}. RAG will fall back to TF-IDF."
    fi
fi

# ----- 6. Done ---------------------------------------------------------------

mkdir -p uploads cleaned reports outputs static/charts logs

cat <<EOF

──────────────────────────────────────────────────────────────────
✅  Install complete.

Next:
    ./run.sh             # start Flask + React in one go
    SKIP_FRONTEND=1 ./run.sh   # only Flask
    SKIP_OLLAMA=1 ./run.sh     # don't try to start Ollama

Useful URLs (after run.sh):
    http://127.0.0.1:5055/    Flask live console
    http://127.0.0.1:5173/    React dashboard (when frontend is running)

Health check:
    curl http://127.0.0.1:5055/api/health

EOF
