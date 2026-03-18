# Track-Anything 環境架設指南

本文件記錄在 NYCU Assistive Robotic Group 實驗室環境下架設 Track-Anything 作為 HIL-AA 系統 baseline 比較工具的完整流程。

## 環境資訊

| 項目 | 版本/規格 |
|------|-----------|
| 電腦 | arg-4090 (電腦 B) |
| GPU | NVIDIA RTX 4090 |
| OS | Ubuntu 24.04 |
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |

## 安裝步驟

### 1. 加入為 Git Submodule

```bash
cd ~/boats_dataset_processing
mkdir -p tools && cd tools
git submodule add https://github.com/gaomingqi/Track-Anything.git
cd Track-Anything
```

### 2. 建立虛擬環境並安裝依賴

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 修改 CUDA 設備編號

原始碼預設使用 `cuda:3`，需改為 `cuda:0`：

```bash
sed -i 's/cuda:3/cuda:0/g' app.py
```

### 4. 修復 Inpainting 模組問題

Track-Anything 的 inpainting 功能需要 `mmcv`，但在 Python 3.12 + PyTorch 2.10 環境下無法安裝。由於 baseline 比較只需追蹤功能，我們跳過 inpainting：

編輯 `track_anything.py`，將第 20 行：

```python
# 原本：
self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint, args.device)

# 改成：
try:
    self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint, args.device)
except Exception as e:
    print(f"Warning: Inpainting disabled ({e})")
    self.baseinpainter = None
```

### 5. 降級 Gradio 版本

Track-Anything 是 2023 年的專案，使用舊版 Gradio API。新版 Gradio 會報錯，需降級：

```bash
pip install gradio==3.50.2
```

### 6. 啟動服務

```bash
python app.py --device cuda:0 --sam_model_type vit_h
```

成功後會顯示：
```
Running on local URL:  http://0.0.0.0:12212
```

用瀏覽器訪問 `http://localhost:12212` 或 `http://<電腦IP>:12212`。

## 遇到的問題與解決方案

### 問題 1：mmcv 安裝失敗 - `ModuleNotFoundError: No module named 'pkg_resources'`

**原因**：`setuptools 82.0` 移除了 `pkg_resources` 模組，導致 mmcv 的 setup.py 無法執行。這是 mmcv 的已知 bug ([GitHub Issue #3325](https://github.com/open-mmlab/mmcv/issues/3325))。

**嘗試的解法**：
1. ❌ `pip install setuptools` - 已安裝但問題依舊
2. ❌ `pip install mmcv --no-build-isolation` - 仍然失敗
3. ❌ `pip install "setuptools<82"` - 降級到 81.0.0 後仍失敗（pip 的 build isolation 環境會重新下載最新 setuptools）
4. ❌ `pip install openmim && mim install mmcv` - mim 本身也依賴 pkg_resources

**最終解法**：跳過 inpainting 模組（見步驟 4）。對於 baseline 比較實驗，追蹤功能已足夠。

### 問題 2：`OSError: CUDA_HOME environment variable is not set`

**原因**：mmcv 嘗試從源碼編譯 CUDA 擴展，但系統沒有設定 `CUDA_HOME`。

**解法**：由於我們跳過了 mmcv，此問題自動解決。若需要安裝 mmcv，需執行：
```bash
export CUDA_HOME=/usr/local/cuda
```

### 問題 3：Gradio API 不相容 - `TypeError: Video.__init__() got an unexpected keyword argument 'autosize'`

**原因**：Track-Anything 使用 Gradio 3.x API，但 pip 預設安裝 Gradio 6.x，API 已大幅改變。

**解法**：降級 Gradio（見步驟 5）。

### 問題 4：numpy 版本衝突警告

降級 Gradio 後會出現警告：
```
opencv-python 4.13.0.92 requires numpy>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
```

**影響**：目前測試下不影響 Track-Anything 的追蹤功能，可忽略。

## 使用說明

### 基本操作流程

1. **上傳影片**：在左上角上傳要處理的影片
2. **選擇模板幀**：使用滑桿選擇要標註的起始幀
3. **標記物件**：
   - 左鍵點擊：正向標記（這是我要追蹤的物件）
   - 右鍵點擊：負向標記（這不是我要追蹤的物件）
4. **執行追蹤**：點擊 "Track" 按鈕開始追蹤
5. **查看結果**：追蹤完成後可播放輸出影片

### SAM 模型選擇

| 模型 | VRAM 需求 | 精度 |
|------|-----------|------|
| `vit_h` | ~8GB | 最高 |
| `vit_l` | ~6GB | 高 |
| `vit_b` | ~4GB | 中 |

若 VRAM 不足，可改用較小模型：
```bash
python app.py --device cuda:0 --sam_model_type vit_b
```

### 啟動參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--device` | 運算設備 | `cuda:0` |
| `--sam_model_type` | SAM 模型大小 | `vit_h` |
| `--port` | Web UI 埠號 | `12212` |

## 與 CVAT 的 GPU 衝突

Track-Anything 和 CVAT 的 SAM 容器都需要 GPU。同時運行會發生衝突。

**使用前**：
```bash
# 停止 CVAT 的 SAM 容器
docker stop nuclio-nuclio-pth-facebookresearch-sam-vit-h
```

**使用後**：
```bash
# 重新啟動 CVAT 的 SAM 容器
docker start nuclio-nuclio-pth-facebookresearch-sam-vit-h
```

## 檔案位置

```
~/boats_dataset_processing/
└── tools/
    └── Track-Anything/          # Track-Anything 主目錄
        ├── venv/                 # Python 虛擬環境
        ├── app.py                # Gradio Web UI 主程式
        ├── track_anything.py     # 追蹤邏輯（已修改跳過 inpainting）
        ├── track_anything.py.bak # 原始備份
        └── checkpoints/          # 模型權重（首次執行自動下載）
```

## 用於 Baseline 比較

Track-Anything 作為 HIL-AA 的 baseline 比較對象，主要比較：

| 比較項目 | Track-Anything | HIL-AA |
|----------|----------------|--------|
| 核心模型 | SAM + XMem | SAM3 |
| 互動方式 | 點擊標記 | 點擊標記 + Presence Score 驅動 |
| 主動學習 | 無 | 有（Confidence-Driven） |
| 效率指標 | 手動記錄 | ActionLogger 自動記錄 TEO/EOR/FCR |

## 版本紀錄

| 日期 | 變更 |
|------|------|
| 2025-03-18 | 初始架設，解決 mmcv/Gradio 相容性問題 |

## 參考資料

- [Track-Anything GitHub](https://github.com/gaomingqi/Track-Anything)
- [mmcv Issue #3325 - setuptools 82.0 相容性](https://github.com/open-mmlab/mmcv/issues/3325)
- [mmcv Issue #3263 - Python 3.12 相容性](https://github.com/open-mmlab/mmcv/issues/3263)
