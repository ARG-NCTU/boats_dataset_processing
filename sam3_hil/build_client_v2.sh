#!/bin/bash
# =============================================================================
# STAMP Client v2 build script
# =============================================================================
#
# Usage:
#   ./build_client_v2.sh          # Build with stamp_client_v2.spec
#   ./build_client_v2.sh clean    # Remove build artifacts
#   ./build_client_v2.sh test     # Run built client
#   ./build_client_v2.sh deps     # Install client build dependencies
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_DIR="dist/STAMP_Client"
APP_BIN="$APP_DIR/STAMP_Client"
SPEC_FILE="stamp_client_v2.spec"

case "${1:-build}" in
    clean)
        info "Cleaning build files..."
        rm -rf build/ dist/ __pycache__/
        find . -name "*.pyc" -delete
        find . -name "__pycache__" -type d -delete
        info "Done!"
        ;;

    deps)
        info "Installing v2 client build dependencies..."
        pip install \
            pyinstaller \
            PyQt6 \
            numpy \
            pillow \
            opencv-python-headless \
            requests \
            loguru \
            websocket-client \
            cython \
            pycocotools \
            pydantic \
            pydantic-settings \
            pandas \
            pyarrow \
            datasets \
            mcap
        info "Done!"
        ;;

    test)
        if [ -f "$APP_BIN" ]; then
            info "Running STAMP_Client..."
            "$APP_BIN"
        else
            error "STAMP_Client not found. Run './build_client_v2.sh' first."
        fi
        ;;

    build|*)
        info "=============================================="
        info "Building STAMP Client v2"
        info "=============================================="

        if ! command -v pyinstaller &> /dev/null; then
            error "PyInstaller not found. Run: ./build_client_v2.sh deps"
        fi

        [ -f "main_server.py" ] || error "main_server.py not found"
        [ -f "$SPEC_FILE" ] || error "$SPEC_FILE not found"
        [ -f "configs/config.py" ] || error "configs/config.py not found"
        [ -d "src" ] || error "src/ directory not found"
        [ -f "src/gui/export_paths.py" ] || error "src/gui/export_paths.py not found"
        [ -f "src/gui/canvas_viewport.py" ] || error "src/gui/canvas_viewport.py not found"

        info "Cleaning previous build..."
        rm -rf build/ dist/

        info "Running PyInstaller with $SPEC_FILE..."
        pyinstaller "$SPEC_FILE" --noconfirm

        if [ -f "$APP_BIN" ]; then
            info "=============================================="
            info "Build successful!"
            info "=============================================="
            info "Output: $APP_DIR/"
            info ""
            info "To run:"
            info "  ./$APP_BIN"
            info ""
            info "To distribute:"
            info "  tar -czvf STAMP_Client_Linux.tar.gz -C dist STAMP_Client"
            info "=============================================="

            SIZE=$(du -sh "$APP_DIR" | cut -f1)
            info "Package size: $SIZE"
        else
            error "Build failed."
        fi
        ;;
esac
