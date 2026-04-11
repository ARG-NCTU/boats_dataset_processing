#!/bin/bash
# =============================================================================
# STAMP Client 打包腳本
# =============================================================================
#
# 用法：
#   ./build_client.sh          # 打包
#   ./build_client.sh clean    # 清理
#   ./build_client.sh test     # 測試執行
#   ./build_client.sh deps     # 安裝依賴
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
                    loguru websocket-client cython pycocotools pydantic-settings
        info "Done!"
        ;;
    
    test)
        if [ -f "dist/STAMP_Client/STAMP_Client" ]; then
            info "Running STAMP_Client..."
            ./dist/STAMP_Client/STAMP_Client
        else
            error "STAMP_Client not found. Run './build_client.sh' first."
        fi
        ;;
    
    build|*)
        info "=============================================="
        info "Building STAMP Client"
        info "=============================================="
        
        # 檢查 PyInstaller
        if ! command -v pyinstaller &> /dev/null; then
            error "PyInstaller not found. Run: ./build_client.sh deps"
        fi
        
        # 檢查必要檔案
        [ -f "main_server.py" ] || error "main_server.py not found"
        [ -f "stamp_client.spec" ] || error "stamp_client.spec not found"
        [ -d "configs" ] || error "configs/ directory not found"
        [ -d "src" ] || error "src/ directory not found"
        
        # 清理
        info "Cleaning previous build..."
        rm -rf build/ dist/
        
        # 打包
        info "Running PyInstaller..."
        pyinstaller stamp_client.spec --noconfirm
        
        # 檢查結果
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
