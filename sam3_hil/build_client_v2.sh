#!/bin/bash
# =============================================================================
# STAMP Client v2 build script
# =============================================================================
#
# Usage:
#   ./build_client_v2.sh          # Build
#   ./build_client_v2.sh clean    # Clean
#   ./build_client_v2.sh test     # Run built client
#   ./build_client_v2.sh deps     # Install dependencies
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

case "${1:-build}" in
    clean)
        info "Cleaning build files..."
        rm -rf build/ dist/ __pycache__/
        find . -name "*.pyc" -delete
        find . -name "__pycache__" -type d -delete
        info "Done!"
        ;;
    
    deps)
        info "Installing dependencies..."
        pip install pyinstaller PyQt6 numpy pillow opencv-python requests \
                    loguru websocket-client cython pycocotools pydantic-settings \
                    pandas pyarrow datasets mcap
        info "Done!"
        ;;
    
    test)
        if [ -f "dist/STAMP_Client/STAMP_Client" ]; then
            info "Running STAMP_Client..."
            ./dist/STAMP_Client/STAMP_Client
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
        [ -f "stamp_client_v2.spec" ] || error "stamp_client_v2.spec not found"
        [ -d "configs" ] || error "configs/ directory not found"
        [ -d "src" ] || error "src/ directory not found"
        [ -f "src/gui/export_paths.py" ] || error "src/gui/export_paths.py not found"
        [ -f "src/gui/canvas_viewport.py" ] || error "src/gui/canvas_viewport.py not found"
        
        info "Cleaning previous build..."
        rm -rf build/ dist/
        
        info "Running PyInstaller..."
        pyinstaller stamp_client_v2.spec --noconfirm
        
        if [ -f "dist/STAMP_Client/STAMP_Client" ]; then
            info "=============================================="
            info "Build successful!"
            info "=============================================="
            info "Output: dist/STAMP_Client/"
            info ""
            info "To run:"
            info "  ./dist/STAMP_Client/STAMP_Client"
            info ""
            info "To distribute:"
            info "  tar -czvf STAMP_Client_Linux.tar.gz -C dist STAMP_Client"
            info "=============================================="
            
            SIZE=$(du -sh dist/STAMP_Client | cut -f1)
            info "Package size: $SIZE"
        else
            error "Build failed."
        fi
        ;;
esac
