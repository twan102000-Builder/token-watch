#!/usr/bin/env bash
# ==============================================================================
# Token Watch macOS launchd Service Management Script
# Idempotently manages Token Watch Hub & Sidecar user agents on Apple Silicon
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"

HUB_PLIST_SRC="${SCRIPT_DIR}/com.tokenwatch.hub.plist"
SIDECAR_PLIST_SRC="${SCRIPT_DIR}/com.tokenwatch.sidecar.plist"

HUB_PLIST_DEST="${LAUNCH_AGENTS_DIR}/com.tokenwatch.hub.plist"
SIDECAR_PLIST_DEST="${LAUNCH_AGENTS_DIR}/com.tokenwatch.sidecar.plist"

mkdir -p "${LAUNCH_AGENTS_DIR}"
mkdir -p "${WORKSPACE_ROOT}/data"

usage() {
    echo "Usage: $0 {install|start|stop|restart|status|logs|uninstall}"
    echo ""
    echo "Commands:"
    echo "  install   - Symlink plists to ~/Library/LaunchAgents and load services"
    echo "  start     - Start Token Watch Hub and Sidecar services"
    echo "  stop      - Stop Token Watch Hub and Sidecar services"
    echo "  restart   - Restart both services"
    echo "  status    - Check running status and HTTP endpoints"
    echo "  logs      - Tail active stdout and stderr logs"
    echo "  uninstall - Unload and remove plists from ~/Library/LaunchAgents"
    exit 1
}

cmd_install() {
    echo "[+] Configuring and installing Token Watch launchd plists into ${LAUNCH_AGENTS_DIR}..."

    # Detect Python executable (prefer workspace .venv)
    if [[ -f "${WORKSPACE_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${WORKSPACE_ROOT}/.venv/bin/python"
        PATH_ENV="${WORKSPACE_ROOT}/.venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
        PATH_ENV="$(dirname "${PYTHON_BIN}"):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    else
        PYTHON_BIN="/usr/bin/python3"
        PATH_ENV="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    fi

    # Render templates directly into LaunchAgents destination
    sed -e "s|{{WORKSPACE_ROOT}}|${WORKSPACE_ROOT}|g" \
        -e "s|{{PYTHON_BIN}}|${PYTHON_BIN}|g" \
        -e "s|{{PATH_ENV}}|${PATH_ENV}|g" \
        "${HUB_PLIST_SRC}" > "${HUB_PLIST_DEST}"

    sed -e "s|{{WORKSPACE_ROOT}}|${WORKSPACE_ROOT}|g" \
        -e "s|{{PYTHON_BIN}}|${PYTHON_BIN}|g" \
        -e "s|{{PATH_ENV}}|${PATH_ENV}|g" \
        "${SIDECAR_PLIST_SRC}" > "${SIDECAR_PLIST_DEST}"

    cmd_start
    echo "[✓] Installation complete."
}

cmd_start() {
    echo "[+] Starting Token Watch services via launchctl..."
    launchctl unload "${HUB_PLIST_DEST}" 2>/dev/null || true
    launchctl unload "${SIDECAR_PLIST_DEST}" 2>/dev/null || true

    launchctl load "${HUB_PLIST_DEST}"
    launchctl load "${SIDECAR_PLIST_DEST}"
    echo "[✓] Token Watch services loaded."
}

cmd_stop() {
    echo "[+] Stopping Token Watch services..."
    launchctl unload "${SIDECAR_PLIST_DEST}" 2>/dev/null || true
    launchctl unload "${HUB_PLIST_DEST}" 2>/dev/null || true
    echo "[✓] Services stopped."
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

cmd_status() {
    echo "=== Token Watch launchd Service Status ==="
    launchctl list | grep -E "com.tokenwatch" || echo "No active com.tokenwatch services registered in launchctl."
    echo ""
    echo "=== HTTP Endpoint Checks ==="
    if curl -s -m 2 http://127.0.0.1:9000/ > /dev/null 2>&1; then
        echo "[✓] Hub (port 9000): ONLINE"
        curl -s http://127.0.0.1:9000/ | python3 -m json.tool 2>/dev/null || true
    else
        echo "[✗] Hub (port 9000): OFFLINE or unreachable"
    fi

    if curl -s -m 2 http://127.0.0.1:8080/openapi.json > /dev/null 2>&1; then
        echo "[✓] Sidecar (port 8080): ONLINE"
    else
        echo "[✗] Sidecar (port 8080): OFFLINE or unreachable"
    fi
}

cmd_logs() {
    echo "=== Tailing Token Watch logs (Ctrl+C to stop) ==="
    tail -f \
        "${WORKSPACE_ROOT}/data/tokenwatch-hub.log" \
        "${WORKSPACE_ROOT}/data/tokenwatch-hub.err.log" \
        "${WORKSPACE_ROOT}/data/tokenwatch-sidecar.log" \
        "${WORKSPACE_ROOT}/data/tokenwatch-sidecar.err.log" 2>/dev/null || {
            echo "Logs have not been generated yet. Start the services first."
        }
}

cmd_uninstall() {
    echo "[+] Unloading and removing Token Watch services..."
    cmd_stop
    rm -f "${HUB_PLIST_DEST}" "${SIDECAR_PLIST_DEST}"
    echo "[✓] Uninstalled."
}

ACTION="${1:-}"
case "${ACTION}" in
    install)   cmd_install ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    logs)      cmd_logs ;;
    uninstall) cmd_uninstall ;;
    *)         usage ;;
esac
