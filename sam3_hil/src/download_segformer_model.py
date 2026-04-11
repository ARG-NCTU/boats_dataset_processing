#!/usr/bin/env python3
"""
SegFormer Model Download Script
===============================

Downloads the SegFormer model for maritime horizon detection.

Usage:
    # 在 Docker 內執行（推薦）
    python download_segformer_model.py --output /app/models/Segformer/segformer_model
    
    # 或使用預設路徑
    python download_segformer_model.py
    
    # 在 host 上使用 venv
    python3 -m venv ~/venv
    ~/venv/bin/pip install transformers
    ~/venv/bin/python download_segformer_model.py --output ~/sam3_hil/models/Segformer/segformer_model
"""

import argparse
import os
from pathlib import Path


def get_default_save_dir():
    """智能選擇預設儲存路徑"""
    # 優先順序：
    # 1. Docker 環境：/app/models/Segformer/segformer_model
    # 2. Host 環境：~/sam3_hil/models/Segformer/segformer_model
    
    if Path("/app").exists() and os.environ.get("USER") == "appuser":
        # Docker 環境
        return Path("/app/models/Segformer/segformer_model")
    else:
        # Host 環境
        return Path.home() / "sam3_hil" / "models" / "Segformer" / "segformer_model"


def main():
    parser = argparse.ArgumentParser(description="Download SegFormer model for maritime horizon detection")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for the model (default: auto-detect)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if model exists"
    )
    args = parser.parse_args()
    
    # 決定儲存路徑
    if args.output:
        save_dir = Path(args.output)
    else:
        save_dir = get_default_save_dir()
    
    # 建立資料夾
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 目標資料夾: {save_dir}")
    print()
    
    # 檢查是否已經下載
    if (save_dir / "config.json").exists() and not args.force:
        print("⚠️  模型已存在！")
        print("   使用 --force 或 -f 強制重新下載")
        print()
        print("📋 現有檔案:")
        for f in sorted(save_dir.iterdir()):
            size = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name}: {size:.2f} MB")
        return
    
    # 導入 transformers
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    except ImportError:
        print("❌ 錯誤: 找不到 transformers 模組")
        print()
        print("📋 安裝方式:")
        print("   Docker 內: pip install transformers")
        print("   Host (Ubuntu 24.04):")
        print("     python3 -m venv ~/venv")
        print("     ~/venv/bin/pip install transformers")
        print("     ~/venv/bin/python download_segformer_model.py")
        return
    
    # 模型名稱
    model_name = "Wilbur1240/segformer-b0-finetuned-ade-512-512-finetune-mastr1325-v2"
    
    print(f"⏳ 下載模型: {model_name}")
    print("   這可能需要幾分鐘...")
    print()
    
    # 下載 ImageProcessor（新版 API）
    print("⏳ 下載 ImageProcessor...")
    try:
        processor = SegformerImageProcessor.from_pretrained(model_name)
        processor.save_pretrained(str(save_dir))
        print("✅ ImageProcessor 下載完成")
    except Exception as e:
        print(f"❌ ImageProcessor 下載失敗: {e}")
        return
    
    # 下載 Model
    print("⏳ 下載 Model...")
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        model.save_pretrained(str(save_dir))
        print("✅ Model 下載完成")
    except Exception as e:
        print(f"❌ Model 下載失敗: {e}")
        return
    
    print()
    print("=" * 60)
    print(f"✅ 模型已下載至: {save_dir}")
    print()
    print("📋 下載的檔案:")
    total_size = 0
    for f in sorted(save_dir.iterdir()):
        size = f.stat().st_size / (1024 * 1024)
        total_size += size
        print(f"   {f.name}: {size:.2f} MB")
    print(f"   ─────────────────────")
    print(f"   Total: {total_size:.2f} MB")
    print("=" * 60)
    
    # 顯示 label map
    print()
    print("📊 Label Map:")
    print(f"   {model.config.id2label}")
    
    # 提示下一步
    print()
    print("📌 下一步:")
    print(f"   確保 maritime_roi.py 的模型路徑指向: {save_dir}")


if __name__ == "__main__":
    main()
