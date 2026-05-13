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

start_stamp_server() {
    echo "[start] $STAMP_SERVER via docker compose"
    docker compose -f "$STAMP_DIR/docker-compose.yml" --project-directory "$STAMP_DIR" up -d stamp-server
}

show_status() {
    echo "=== Docker containers ==="
    docker ps -a --filter "name=^/($STAMP_GUI|$STAMP_SERVER|$CVAT_SAM|$LABELME_GPU)$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
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
        start_existing_container "$CVAT_SAM" || {
            echo "[hint] If the CVAT SAM function was removed, redeploy it from tools/cvat."
            exit 1
        }
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
