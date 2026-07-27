#!/usr/bin/env bash
# Flec one-command setup for any host (macOS dev or ARM64 embedded Linux).
#
# Usage:
#   cp .env.example .env          # fill in HUGGING_FACE_HUB_TOKEN
#   bash scripts/setup.sh
#
# What it does:
#   1. Creates .env from .env.example if missing
#   2. Checks system dependencies (espeak-ng, libgl1)
#   3. Creates a Python 3.11 venv and installs all Python dependencies
#   4. Downloads all AI models (models already present are skipped)
#   5. Downloads fingertip reference fixtures for yolo26n-pose fine-tune
#
# HuggingFace token:
#   Set HUGGING_FACE_HUB_TOKEN in your .env file.
#   Required for: HaGRID dataset (fingertip fixtures) and BLIP-2 model weights.
#   How to get a read token:
#     1. Go to https://huggingface.co/settings/tokens
#     2. Click "New token"
#     3. Name: e.g. "flec-dev"
#     4. Type: "Read" (not "Write" — read access is sufficient)
#     5. Click "Generate a token" and copy it into .env:
#          HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#
#   Without a token: HaGRID images fall back to the sbercloud mirror or
#   synthetic placeholders, and BLIP-2 is attempted unauthenticated (may be
#   rate-limited on large files).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}[OK]${RESET}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
err()  { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
step() { echo -e "\n${GREEN}──${RESET} $*"; }

# ── 1. .env ───────────────────────────────────────────────────────────────────
step "Environment file"
if [ ! -f .env ]; then
  cp .env.example .env
  warn ".env created from .env.example — open it and fill in HUGGING_FACE_HUB_TOKEN"
else
  ok ".env already exists"
fi

# Load .env into the current shell so subsequent python calls inherit the vars
set -a
# shellcheck disable=SC1091
source .env 2>/dev/null || true
set +a

HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [ -z "$HF_TOKEN" ]; then
  warn "HUGGING_FACE_HUB_TOKEN is not set in .env"
  warn "  → HaGRID fixtures will fall back to sbercloud mirror or synthetic placeholders"
  warn "  → BLIP-2 download will be attempted unauthenticated (may fail on large files)"
  warn "  Get a free read token at: https://huggingface.co/settings/tokens"
else
  ok "HUGGING_FACE_HUB_TOKEN is set (${#HF_TOKEN} chars)"
fi

# ── 2. System dependencies ────────────────────────────────────────────────────
step "System dependencies"
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
  if command -v brew &>/dev/null; then
    if ! brew list espeak &>/dev/null 2>&1 && ! brew list espeak-ng &>/dev/null 2>&1; then
      echo "  Installing espeak-ng via Homebrew..."
      brew install espeak
    else
      ok "espeak-ng present (macOS)"
    fi
  else
    warn "Homebrew not found — install espeak-ng manually: https://brew.sh"
  fi
elif [ "$OS" = "Linux" ]; then
  MISSING_PKGS=""
  for pkg in espeak-ng libgl1 libglib2.0-0; do
    if ! dpkg -l "$pkg" &>/dev/null 2>&1; then
      MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
  done
  if [ -n "$MISSING_PKGS" ]; then
    echo "  Installing:$MISSING_PKGS"
    sudo apt-get update -qq
    # shellcheck disable=SC2086
    sudo apt-get install -y --no-install-recommends $MISSING_PKGS
  else
    ok "System packages present (Linux)"
  fi
else
  warn "Unrecognised OS ($OS) — ensure espeak-ng and libgl are installed manually"
fi

# ── 3. Python venv ────────────────────────────────────────────────────────────
step "Python virtual environment"
PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" &>/dev/null; then
  # Fallback: try uv, then python3
  if command -v uv &>/dev/null; then
    PYTHON="uv python"
  elif command -v python3 &>/dev/null; then
    PYTHON="python3"
  else
    err "Python 3.11 not found. Install it and re-run."
    exit 1
  fi
fi

if [ ! -d .venv ]; then
  echo "  Creating .venv with $PYTHON..."
  if command -v uv &>/dev/null; then
    uv venv --python 3.11 .venv
  else
    "$PYTHON" -m venv .venv
  fi
  ok ".venv created"
else
  ok ".venv already exists"
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

# ── 4. Python dependencies ────────────────────────────────────────────────────
step "Python dependencies"
pip install --upgrade pip "setuptools<81" --quiet
# openai-whisper builds from sdist and needs setuptools<81 at build time
pip install --no-build-isolation openai-whisper --quiet
pip install -r requirements.txt --quiet
pip install -e . --quiet
ok "Python dependencies installed"

# ── 5. AI models ──────────────────────────────────────────────────────────────
step "AI model downloads"
echo "  Running scripts/download_models.py..."
python scripts/download_models.py

# ── 6. Fingertip reference fixtures ──────────────────────────────────────────
step "Fingertip reference fixtures (HaGRID / yolo26n-pose training set)"
if [ -n "$HF_TOKEN" ]; then
  echo "  Running scripts/download_fingertip_fixtures.py --source hf ..."
  python scripts/download_fingertip_fixtures.py --source hf
else
  echo "  No HF token — attempting direct download (sbercloud mirror)..."
  python scripts/download_fingertip_fixtures.py --source auto
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}  Flec setup complete.${RESET}"
echo ""
echo "  Activate env:   source .venv/bin/activate"
echo "  Run dev:        python -m flec.main --mode dev"
echo "  Run tests:      pytest"
echo "  Dry-run check:  python -m flec.main --dry-run"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
