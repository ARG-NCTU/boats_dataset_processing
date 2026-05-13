#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP_DIR="$ROOT_DIR/sam3_hil"

STAMP_GUI="sam3_hil"
STAMP_SERVER="stamp-server"
CVAT_SAM="nuclio-nuclio-pth-facebookresearch-sam-vit-h"
LABELME_GPU="argmm"

usage() {
    cat <<EOF
Usage: ./switch_tool.sh <tool>

Tools:
  stamp         Stop CVAT SAM and LabelMe GPU, start STAMP server, then launch STAMP GUI
  stamp-server  Stop CVAT SAM and LabelMe GPU, start only STAMP server
  cvat          Stop STAMP and LabelMe GPU, start CVAT SAM
  labelme       Stop STAMP and CVAT SAM, then launch LabelMe GPU container shell
  stop-all      Stop all known GPU tool containers
  status        Show known containers and GPU usage

Known GPU containers:
  $STAMP_GUI
  $STAMP_SERVER
  $CVAT_SAM
  $LABELME_GPU
EOF
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "$1"
}

container_running() {
    docker ps --format '{{.Names}}' | grep -Fxq "$1"
}

stop_container() {
    local name="$1"
    if container_running "$name"; then
        echo "[stop] $name"
        docker stop "$name" >/dev/null
    else
        echo "[skip] $name is not running"
    fi
}

start_existing_container() {
    local name="$1"
    if container_running "$name"; then
        echo "[skip] $name is already running"
    elif container_exists "$name"; then
        echo "[start] $name"
        docker start "$name" >/dev/null
    else
        echo "[warn] $name does not exist; use its normal run script to create it"
        return 1
    fi
}

restart_existing_container() {
    local name="$1"
    if container_exists "$name"; then
        echo "[restart] $name"
        docker restart "$name" >/dev/null
    else
        echo "[warn] $name does not exist; use its normal run script to create it"
        return 1
    fi
}

container_host_port() {
    local name="$1"
    local container_port="${2:-8080/tcp}"
    docker port "$name" "$container_port" 2>/dev/null \
        | sed -n 's/.*:\([0-9][0-9]*\)$/\1/p' \
        | head -n 1
}

wait_container_healthy() {
    local name="$1"
    local timeout_seconds="${2:-120}"
    local started_at
    local status
    started_at="$(date +%s)"

    while true; do
        status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$name" 2>/dev/null || true)"
        if [ "$status" = "healthy" ] || [ "$status" = "running" ]; then
            echo "[ready] $name status=$status"
            return 0
        fi

        if [ $(( $(date +%s) - started_at )) -ge "$timeout_seconds" ]; then
            echo "[error] Timed out waiting for $name health. Current status=$status"
            docker logs "$name" --tail 80 || true
            return 1
        fi

        echo "[wait] $name status=$status"
        sleep 3
    done
}

wait_http_port() {
    local url="$1"
    local timeout_seconds="${2:-120}"
    local started_at
    started_at="$(date +%s)"

    while true; do
        if curl -sS --max-time 2 "$url" >/dev/null 2>&1; then
            echo "[ready] $url"
            return 0
        fi

        if [ $(( $(date +%s) - started_at )) -ge "$timeout_seconds" ]; then
            echo "[error] Timed out waiting for $url"
            return 1
        fi

        echo "[wait] $url"
        sleep 3
    done
}

start_cvat_sam() {
    restart_existing_container "$CVAT_SAM" || {
        echo "[hint] If the CVAT SAM function was removed, redeploy it from tools/cvat."
        return 1
    }

    wait_container_healthy "$CVAT_SAM" 180

    local port
    port="$(container_host_port "$CVAT_SAM" "8080/tcp")"
    if [ -z "$port" ]; then
        echo "[error] Cannot find host port for $CVAT_SAM:8080/tcp"
        docker ps --filter "name=^/$CVAT_SAM$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
        return 1
    fi

    echo "[info] CVAT SAM host URL: http://127.0.0.1:$port/"
    wait_http_port "http://127.0.0.1:$port/" 180
}

start_stamp_server() {
    echo "[start] $STAMP_SERVER via docker compose"
    docker compose -f "$STAMP_DIR/docker-compose.yml" --project-directory "$STAMP_DIR" up -d stamp-server
}

show_status() {
    echo "=== Docker containers ==="
    docker ps -a --filter "name=^/($STAMP_GUI|$STAMP_SERVER|$CVAT_SAM|$LABELME_GPU)$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}'
    echo
    echo "=== GPU usage ==="
    nvidia-smi
}

tool="${1:-}"

case "$tool" in
    stamp)
        stop_container "$CVAT_SAM"
        stop_container "$LABELME_GPU"
        start_stamp_server
        echo "[launch] STAMP GUI"
        cd "$STAMP_DIR"
        exec bash -lc "source docker/docker_run.sh run"
        ;;

    stamp-server)
        stop_container "$CVAT_SAM"
        stop_container "$LABELME_GPU"
        stop_container "$STAMP_GUI"
        start_stamp_server
        show_status
        ;;

    cvat)
        stop_container "$STAMP_GUI"
        stop_container "$STAMP_SERVER"
        stop_container "$LABELME_GPU"
        start_cvat_sam
        show_status
        ;;

    labelme)
        stop_container "$STAMP_GUI"
        stop_container "$STAMP_SERVER"
        stop_container "$CVAT_SAM"
        echo "[launch] LabelMe GPU container shell via gpu_run.sh"
        cd "$ROOT_DIR"
        exec bash -lc "source gpu_run.sh"
        ;;

    stop-all)
        stop_container "$STAMP_GUI"
        stop_container "$STAMP_SERVER"
        stop_container "$CVAT_SAM"
        stop_container "$LABELME_GPU"
        show_status
        ;;

    status)
        show_status
        ;;

    ""|help|--help|-h)
        usage
        ;;

    *)
        echo "[error] Unknown tool: $tool"
        echo
        usage
        exit 1
        ;;
esac
