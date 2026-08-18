#!/usr/bin/env bash
#
# Install kotel-python-ocr on a Raspberry Pi:
#   - creates the Python venv (deps from pyproject.toml)
#   - installs ssocr (for temperature OCR) if missing
#   - renders the systemd units with THIS repo's path and the right user
#   - stops/disables any previous install of these units, then enables the timer
#
# Usage (run from anywhere, as root):
#   sudo scripts/install.sh [--user NAME] [--no-calibrator]
#
# --user NAME        run the services as NAME (default: owner of the repo dir)
# --no-calibrator    don't enable the ROI calibrator web service
#
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root:  sudo $0 $*" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_USER=""
WITH_CAL=1
while [ $# -gt 0 ]; do
    case "$1" in
        --user) SERVICE_USER="${2:-}"; shift 2 ;;
        --no-calibrator) WITH_CAL=0; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[ -n "$SERVICE_USER" ] || SERVICE_USER="$(stat -c '%U' "$REPO_DIR")"
id "$SERVICE_USER" >/dev/null 2>&1 || { echo "No such user: $SERVICE_USER" >&2; exit 1; }

VENV_PY="$REPO_DIR/.venv/bin/python"

echo "==> Repo:  $REPO_DIR"
echo "==> User:  $SERVICE_USER"

# --- 1. Python venv + dependencies ------------------------------------------
# Validate it actually runs here (a .venv copied from another machine has the
# wrong-architecture python and fails with "Exec format error").
if [ -x "$VENV_PY" ] && "$VENV_PY" -c 'import sys' >/dev/null 2>&1; then
    echo "==> Virtualenv already present, skipping"
else
    [ -e "$REPO_DIR/.venv" ] && { echo "==> Removing unusable .venv"; rm -rf "$REPO_DIR/.venv"; }
    echo "==> Creating virtualenv"
    if command -v uv >/dev/null 2>&1; then
        sudo -u "$SERVICE_USER" sh -c "cd '$REPO_DIR' && uv sync"
    else
        python3 -m venv "$REPO_DIR/.venv"
        "$VENV_PY" -m pip install --upgrade pip
        # Same version caps as pyproject.toml so a Pi install matches dev.
        "$VENV_PY" -m pip install \
            "flask>=3.1.3,<4" "numpy>=2.4.4,<3" "opencv-python>=4.13.0.92,<6" \
            "paho-mqtt>=2.1.0,<3" "requests>=2.33.1,<3"
        chown -R "$SERVICE_USER":"$SERVICE_USER" "$REPO_DIR/.venv"
    fi
fi

# --- 2. ssocr (temperature OCR) ---------------------------------------------
if ! command -v ssocr >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "==> Installing ssocr"
        apt-get update -qq && apt-get install -y ssocr \
            || echo "WARN: ssocr install failed; temperature OCR will be unavailable"
    else
        echo "WARN: ssocr not found and apt-get unavailable; install it manually for temperature OCR"
    fi
fi

# --- 3. Stop/disable any previous install of these units --------------------
UNITS="boilerocr-mqtt-oneshot.timer boilerocr-mqtt-oneshot.service roi-calibrator.service"
echo "==> Removing any previous units"
for u in $UNITS; do
    systemctl stop "$u" 2>/dev/null || true
    systemctl disable "$u" 2>/dev/null || true
    rm -f "/etc/systemd/system/$u"
done

# --- 4. Render + install units for this repo/user ---------------------------
echo "==> Installing systemd units"
for tmpl in "$REPO_DIR"/systemd/*.service "$REPO_DIR"/systemd/*.timer; do
    name="$(basename "$tmpl")"
    sed -e "s#/home/admin/boilerocr#$REPO_DIR#g" \
        -e "s#^User=admin#User=$SERVICE_USER#" \
        "$tmpl" > "/etc/systemd/system/$name"
    echo "    installed $name"
done

# --- 5. Enable ---------------------------------------------------------------
systemctl daemon-reload
systemctl enable --now boilerocr-mqtt-oneshot.timer
echo "==> Enabled boilerocr-mqtt-oneshot.timer"
if [ "$WITH_CAL" -eq 1 ]; then
    systemctl enable --now roi-calibrator.service
    echo "==> Enabled roi-calibrator.service (http://<pi>:5001)"
fi

echo "==> Done."
echo "    Check:  systemctl status boilerocr-mqtt-oneshot.timer"
echo "    Logs:   journalctl -u boilerocr-mqtt-oneshot.service -f"
