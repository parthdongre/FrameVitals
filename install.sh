#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

log() { printf "\033[1;36m▶\033[0m %s\n" "$*"; }
ok() { printf "\033[1;32m✓\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!\033[0m %s\n" "$*"; }
fail() { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

PYTHON_BIN="${PYTHON:-}"
if [ -z "${PYTHON_BIN}" ]; then
    if have python3.13; then PYTHON_BIN="python3.13"
    elif have python3.12; then PYTHON_BIN="python3.12"
    elif have python3.11; then PYTHON_BIN="python3.11"
    elif have python3; then PYTHON_BIN="python3"
    else fail "Python 3.11+ is required."
    fi
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("FrameVitals requires Python 3.11+")
print(f"Using Python {sys.version.split()[0]}")
PY

VENV_DIR="${ROOT_DIR}/.venv"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    log "Creating virtual environment at .venv"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
log "Installing FrameVitals with all optional features and development tools"
"${VENV_PY}" -m pip install --upgrade pip
"${VENV_PY}" -m pip install -e ".[all,dev]"

log "Verifying Python package and CLI"
"${VENV_PY}" -c "import framevitals; print('FrameVitals', framevitals.__version__)"
"${VENV_DIR}/bin/framevitals" --version

if [ -d frontend ]; then
    if have npm; then
        log "Installing frontend dependencies"
        (
            cd frontend
            if [ -f package-lock.json ]; then
                npm ci
            else
                npm install
            fi
        )
        ok "Frontend dependencies installed"
    else
        warn "npm not found; skipping the optional React dashboard setup"
    fi
fi

ok "FrameVitals development environment is ready"
printf "\nActivate it with:\n  source .venv/bin/activate\n\n"
printf "Run the package:\n  framevitals --version\n  framevitals analyze demo_datasets/<file>.csv --mode quick\n\n"
printf "Run the web applications:\n  ./run.sh\n"
