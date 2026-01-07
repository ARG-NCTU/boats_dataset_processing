#!/usr/bin/env bash
# =============================================================================
# HIL-AA Maritime Annotation System - Docker Run Script
# 可以用 source 或 ./ 執行
# =============================================================================

# 設定 Docker 映像名稱
IMAGE_NAME="sam3_hil:latest"
CONTAINER_NAME="sam3_hil"

# 取得腳本目錄（支援 source 和 ./）
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# =============================================================================
# 檢查是否有已運行的容器
# =============================================================================
CONTAINER_ID=$(docker ps -aqf "name=^/${CONTAINER_NAME}$")
if [ -n "$CONTAINER_ID" ]; then
    echo -e "${GREEN}✅ 附加到已運行的容器: $CONTAINER_NAME${NC}"
    xhost +local:docker 2>/dev/null
    docker exec --privileged -e DISPLAY=${DISPLAY} -it ${CONTAINER_ID} bash
    xhost -local:docker 2>/dev/null
    return 0 2>/dev/null || exit 0
fi

# =============================================================================
# 設定 X11
# =============================================================================
info "Setting up X11 forwarding..."
xhost +local:docker 2>/dev/null || warn "xhost command failed"

if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    warn "DISPLAY not set, using :0"
fi
info "Using DISPLAY=$DISPLAY"

# =============================================================================
# 檢查 GPU
# =============================================================================
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        info "NVIDIA Docker support detected."
        GPU_FLAG="--gpus all"
    else
        warn "NVIDIA Docker runtime not available. Running in CPU mode."
    fi
else
    warn "NVIDIA driver not found. Running in CPU mode."
fi

# =============================================================================
# 處理命令
# =============================================================================
CMD="${1:-shell}"

case "$CMD" in
    build)
        info "Building Docker image: $IMAGE_NAME"
        docker build \
            -t "$IMAGE_NAME" \
            -f "$SCRIPT_DIR/Dockerfile" \
            --build-arg UID=$(id -u) \
            --build-arg GID=$(id -g) \
            "$PROJECT_ROOT"
        info "Build complete!"
        ;;
    
    shell|run|test)
        # 決定要執行的命令
        if [ "$CMD" = "shell" ]; then
            EXEC_CMD="bash"
        elif [ "$CMD" = "test" ]; then
            shift
            EXEC_CMD="pytest tests/ -v $@"
        else
            shift
            EXEC_CMD="${@:-python main.py}"
        fi
        
        info "Starting container: $CONTAINER_NAME"
        info "Command: $EXEC_CMD"
        
        docker run -it --rm \
            --name "$CONTAINER_NAME" \
            $GPU_FLAG \
            -e DISPLAY="$DISPLAY" \
            -e QT_X11_NO_MITSHM=1 \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            -v "$HOME/.Xauthority:/home/appuser/.Xauthority:rw" \
            -v "$PROJECT_ROOT/src:/app/src:rw" \
            -v "$PROJECT_ROOT/configs:/app/configs:rw" \
            -v "$PROJECT_ROOT/data:/app/data:rw" \
            -v "$PROJECT_ROOT/tests:/app/tests:rw" \
            -v "$PROJECT_ROOT/models:/app/models:rw" \
            -v "$PROJECT_ROOT/output:/app/output:rw" \
            -v "$PROJECT_ROOT/third_party:/app/third_party:ro" \
            --network host \
            --ipc host \
            --shm-size=8g \
            "$IMAGE_NAME" \
            $EXEC_CMD
        ;;
    
    help|--help|-h)
        echo "Usage: source docker/docker_run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  build    Build the Docker image"
        echo "  shell    Start interactive bash (default)"
        echo "  run      Run the application"
        echo "  test     Run pytest"
        echo "  help     Show this message"
        ;;
    
    *)
        warn "Unknown command: $CMD"
        echo "Use 'help' for usage information"
        ;;
esac