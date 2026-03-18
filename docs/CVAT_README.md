# CVAT 環境架設指南

本文件記錄在 NYCU Assistive Robotic Group 實驗室環境下架設 CVAT 作為 HIL-AA 系統 Ground Truth 標註工具的完整流程。

## 環境資訊

| 項目 | 版本/規格 |
|------|-----------|
| 電腦 | arg-4090 |
| GPU | NVIDIA RTX 4090 |
| OS | Ubuntu 24.04 |
| Docker | Docker Compose v2 |
| CVAT | 最新版（Git clone） |

## 安裝步驟

### 1. 建立目錄並下載 CVAT

將 CVAT 用 submodule 加入，避免 Git 衝突：

```bash
cd ~/boats_dataset_processing

# 建立 tools 資料夾
mkdir -p tools
cd tools

# 用 submodule 加入 CVAT
git submodule add https://github.com/cvat-ai/cvat.git

# 回到根目錄 commit
cd ..
git add .
git commit -m "Add CVAT as submodule"
```

最終結構：
```
boats_dataset_processing/
├── sam3_hil/              ← HIL-AA 系統
├── tools/
│   ├── cvat/              ← CVAT (submodule)
│   └── Track-Anything/    ← Track-Anything (submodule)
└── .gitmodules
```

### 2. 啟動 CVAT

```bash
cd ~/boats_dataset_processing/tools/cvat
docker compose up -d
```

首次啟動需要下載 Docker images，約需 5-10 分鐘。

### 3. 建立管理員帳號

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

依提示輸入：
- Username：你的帳號
- Email：隨便填
- Password：你的密碼（打字不會顯示，這是正常的）

### 4. 開啟瀏覽器

```
http://localhost:8080
```

用剛才建立的帳號密碼登入。

## 啟用 SAM 功能

CVAT 支援內建 SAM，可以實現半自動標註。

### 步驟 1：啟動 Nuclio（AI 服務）

```bash
cd ~/boats_dataset_processing/tools/cvat
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

### 步驟 2：安裝 nuctl

```bash
wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64
chmod +x nuctl-1.13.0-linux-amd64
sudo mv nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl
```

### 步驟 3：部署 SAM 模型

```bash
cd ~/boats_dataset_processing/tools/cvat

# GPU 版本
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam/
```

這會下載 SAM 模型並建立 Docker image，約需 10-15 分鐘。

成功後會顯示：
```
 NAMESPACE | NAME                           | PROJECT | STATE | REPLICAS | NODE PORT 
 nuclio    | pth-facebookresearch-sam-vit-h | cvat    | ready | 1/1      | 32768
```

### 步驟 4：在 CVAT 使用 SAM

- 進入標註畫面
- 點左側工具列的 **Magic Wand（魔術棒）** 圖示
- 或按 **`I`** 鍵啟用 AI 輔助
- 點擊物件，SAM 會自動產生 Mask

## 使用說明

### 建立專案

1. 進入 CVAT 主頁
2. 點擊 **Create new project**
3. 輸入專案名稱（例如 `taichung_port_gt`）
4. 設定 Labels：
   - 點擊 **Add label**
   - 輸入 `ship`
   - 選擇 **Polygon** 類型

### 建立任務並上傳影片

1. 進入專案頁面
2. 點擊 **Create new task**
3. 輸入任務名稱
4. 上傳 mp4 影片（CVAT 支援直接上傳影片）
5. 點擊 **Submit**

### 標註操作

| 操作 | 說明 |
|------|------|
| 點擊物件 | SAM 自動產生 Mask |
| 修改 Mask | 點擊邊界拖曳調整 |
| 下一幀 | `D` 鍵 |
| 上一幀 | `A` 鍵 |
| 儲存 | `Ctrl+S` |

### 匯出標註

1. 點擊任務右上角的選單
2. 選擇 **Export task annotations**
3. 選擇格式：
   - **COCO 1.0**：標準 COCO JSON
   - **Segmentation mask 1.1**：Mask 圖片
   - **CVAT for images 1.1**：CVAT XML

## 遇到的問題與解決方案

### 問題 1：SAM 無法連線 - `Error: connect ECONNREFUSED host.docker.internal:32768`

**原因**：GPU 記憶體不足。sam3_hil 容器佔用了大量 VRAM，導致 CVAT 的 SAM 無法載入。

**診斷**：
```bash
docker ps | grep sam
docker logs $(docker ps -aqf "name=nuclio-nuclio-pth-facebookresearch-sam") --tail 50
```

錯誤訊息：
```
torch.OutOfMemoryError: CUDA out of memory.
```

**解法**：停止 sam3_hil 容器，釋放 GPU 記憶體：
```bash
docker stop sam3_hil
docker restart nuclio-nuclio-pth-facebookresearch-sam-vit-h
```

### 問題 2：在 Git repo 內 clone 另一個 repo

**原因**：直接 clone 會造成 nested Git 問題，外層 Git 會把內層整個資料夾當成普通檔案追蹤。

**解法**：使用 Git submodule（見安裝步驟 1）。

## GPU 衝突管理

CVAT 的 SAM 和 sam3_hil 容器都需要 GPU，無法同時運行。

### 使用 CVAT SAM 前

```bash
# 停止 sam3_hil
docker stop sam3_hil

# 確認 CVAT SAM 正常
docker restart nuclio-nuclio-pth-facebookresearch-sam-vit-h
```

### 使用 HIL-AA 前

```bash
# 停止 CVAT 的 SAM
docker stop nuclio-nuclio-pth-facebookresearch-sam-vit-h

# 啟動 sam3_hil
docker start sam3_hil
```

## 常用指令速查

### CVAT 服務管理

```bash
# 啟動 CVAT
cd ~/boats_dataset_processing/tools/cvat
docker compose up -d

# 停止 CVAT
docker compose down

# 停止並清除資料
docker compose down -v

# 查看狀態
docker compose ps

# 查看 log
docker logs cvat_server --tail 50
```

### SAM 服務管理

```bash
# 啟動 serverless（含 SAM）
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

# 查看 SAM 狀態
docker ps | grep sam

# 查看 SAM log
docker logs nuclio-nuclio-pth-facebookresearch-sam-vit-h --tail 50

# 重啟 SAM
docker restart nuclio-nuclio-pth-facebookresearch-sam-vit-h
```

## 檔案位置

```
~/boats_dataset_processing/
├── sam3_hil/                          # HIL-AA 系統
├── tools/
│   ├── cvat/                          # CVAT 主目錄 (submodule)
│   │   ├── docker-compose.yml
│   │   ├── components/
│   │   │   └── serverless/
│   │   │       └── docker-compose.serverless.yml
│   │   └── serverless/
│   │       └── pytorch/
│   │           └── facebookresearch/
│   │               └── sam/           # SAM 設定
│   └── Track-Anything/                # Track-Anything (submodule)
└── .gitmodules
```

## 用於 Ground Truth 標註

CVAT 在 HIL-AA 專案中的角色：

1. **標註 Ground Truth**：用 SAM 輔助標註測試影片的正確 Mask
2. **計算 mIoU**：與 HIL-AA、Track-Anything 的輸出比較品質
3. **Baseline 比較**：作為傳統標註工具的代表

### 標註工作流程

```
步驟 1：建立專案 + 任務
    │
    ↓
步驟 2：用 SAM 輔助標註物件
    │  - 點擊物件產生 Mask
    │  - 手動修正邊界
    │
    ↓
步驟 3：逐幀檢查
    │  - 按 D 下一幀
    │  - 確認 Mask 正確
    │
    ↓
步驟 4：匯出為 COCO JSON
    │
    ↓
步驟 5：與其他工具輸出比較 mIoU
```

## 參考資料

- [CVAT GitHub](https://github.com/cvat-ai/cvat)
- [CVAT Documentation](https://docs.cvat.ai/)
- [CVAT Serverless Tutorial](https://docs.cvat.ai/docs/administration/advanced/installation_automatic_annotation/)
- [Nuclio GitHub](https://github.com/nuclio/nuclio)
