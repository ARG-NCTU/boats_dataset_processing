# STAMP — SAM Tracking Annotation with Minimal Processing

> Leveraging SAM 3 Presence Confidence for Efficient Human-in-the-Loop Video Annotation

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![SAM 3](https://img.shields.io/badge/Model-SAM%203-orange.svg)](https://github.com/facebookresearch/sam3)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Core Concept

STAMP transforms annotation from a labor-intensive process into an efficient human-in-the-loop workflow:

```
SAM 3 Zero-Shot   →   Presence Score   →   Confidence        →   Human Review
Discovery               Extraction          Classification         (if needed)
                                            HIGH / UNC / LOW
```

1. **SAM 3** detects all objects matching text prompts (e.g., `"ship, buoy"`)
2. **Presence Score** classifies each detection's confidence
3. **HIGH** (≥ 0.80) → auto-accept, no human needed
4. **UNCERTAIN** (0.50–0.80) → queue for human review
5. **LOW** (< 0.50) → auto-reject or flag
6. Human reviewer refines with simple point clicks

**Result:** the human role shifts from *annotator* to *reviewer*.

---

## System Architecture

STAMP uses a **Client-Server** architecture:

```
Client Machine(s)                    Server (Docker + GPU)
┌─────────────────┐                ┌──────────────────────┐
│ STAMP Client    │                │  Docker              │
│ (PyQt6 GUI)     │     HTTP       │  ┌────────────────┐  │
│        ↓        │ ─────────────► │  │  FastAPI        │  │
│  api_client.py  │  192.168.x.x   │  │  (Uvicorn)      │  │
└─────────────────┘     :8000      │  │       ↓         │  │
                         WebSocket │  │  Task Manager   │  │
Another Client           ◄───────► │  │       ↓         │  │
┌─────────────────┐                │  │  SAM3 (GPU)     │  │
│ STAMP Client    │     HTTP       │  └────────────────┘  │
│ (PyQt6 GUI)     │ ─────────────► │                      │
└─────────────────┘                └──────────────────────┘
```

- **Server**: Docker container running FastAPI + SAM3 on GPU
- **Client**: lightweight PyQt6 desktop app, no GPU required
- Multiple annotators can share a single GPU server

### Core Modules

| # | Module | Function | Type |
|---|--------|----------|------|
| 1 | Zero-Shot Discovery | Text-prompt object detection via SAM3 PCS | Core |
| 2 | Confidence-Guided Review | THREE-level classification (HIGH/UNC/LOW) | Core |
| 3 | Temporal Tracking | Keyframe propagation via SAM3 video predictor | Core |
| 4 | Interactive Refinement | Point-prompt mask correction | Core |
| 5 | Optimized GUI | Timeline + keyboard shortcuts + batch operations | Core |
| 6 | Export Module | COCO / Labelme / HuggingFace Parquet output | Core |
| 7 | ActionLogger | Efficiency metrics tracking (TEO, EOR, CPO, SPF) | Core |
| 8 | Jitter Detection | Tracking failure detection (scene-dependent, optional) | Auxiliary |
| 9 | Maritime ROI | Horizon detection for sky filtering (scene-dependent, optional) | Auxiliary |

---

## Installation

There are two ways to use STAMP:

### Option A: Pre-built Client (Recommended for Annotators)

Download the pre-built executable. No Python, GPU, or Docker needed on the client machine.

**Requirements:**
- Ubuntu 22.04+ (x86_64)
- `libxcb-cursor0` system package

```bash
# 1. Install system dependency
sudo apt install libxcb-cursor0

# 2. Extract the package
tar -xzvf STAMP_Client_Linux.tar.gz

# 3. Run
./STAMP_Client/STAMP_Client
```

On startup, select **Remote Server** mode and enter the server URL (e.g., `http://192.168.0.114:8000`).

> **Note:** The server must already be running. See [Server Setup](#server-setup) below.

### Option B: Clone and Build (For Self-hosted Deployment)

For cases where no pre-built executable is available, or you need to build on a different machine.

**Requirements:**
- Python 3.10+
- Git

```bash
# 1. Clone with submodules
git clone --recursive git@github.com:ARG-NCTU/boats_dataset_processing.git
cd boats_dataset_processing/sam3_hil

# 2. Create virtual environment and install dependencies
sudo apt install libxcb-cursor0
python3 -m venv venv
source venv/bin/activate
pip install pyinstaller
pip install -r requirements-client.txt
pip install cython pycocotools pydantic-settings

# 3. Build
./build_client.sh

# 4. Deactivate virtual environment
deactivate

# 5. Run
./dist/STAMP_Client/STAMP_Client
```

To distribute the built package:

```bash
tar -czvf STAMP_Client_Linux.tar.gz -C dist STAMP_Client
```

> **Important:** PyInstaller binaries are tied to the glibc version of the build machine. Always build on the machine with the **oldest** glibc for maximum compatibility. Check with `ldd --version`.

### Server Setup

The server runs inside Docker with GPU access. Set this up on the machine with the NVIDIA GPU.

**Requirements:**
- NVIDIA GPU (24GB+ VRAM recommended)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- HuggingFace account with [SAM 3 access](https://huggingface.co/facebook/sam3)

```bash
# 1. Clone (if not already done)
git clone --recursive git@github.com:ARG-NCTU/boats_dataset_processing.git
cd boats_dataset_processing/sam3_hil

# 2. Set HuggingFace token (add to bashrc for persistence)
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc

# 3. Start the server
docker compose up

# Server will be available at http://<server-ip>:8000
# API docs at http://<server-ip>:8000/docs
```

To verify the server is running:

```bash
curl http://<server-ip>:8000/health
# Expected: {"status":"healthy","sam3":"loaded","task_manager":"ready"}
```

### Development Mode (For Developers)

To run the GUI directly inside the Docker container (for development and debugging):

```bash
# 1. Start Docker container
source docker/docker_run.sh

# 2. Run inside the container
python main_server.py
```

---

## Usage

### Workflow

1. **Start the server** (`docker compose up`)
2. Open the client, connect to the server
3. Load a video or image folder
4. Enter a text prompt (e.g., `"ship, boat"`) and run detection
5. SAM3 detects objects and classifies confidence
6. Review flagged frames (yellow/red on timeline)
7. Refine masks with point clicks if needed
8. Export annotations (COCO / Labelme / HuggingFace Parquet)

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `←` / `→` | Previous / Next frame |
| `Home` / `End` | First / Last frame |
| `Tab` | Jump to next review-needed frame |
| `Ctrl+O` | Open video |
| `Ctrl+Q` | Quit |

### Mouse Controls

| Input | Action |
|-------|--------|
| Left Click | Positive point — "this IS the object" |
| Right Click | Negative point — "this is NOT the object" |

### Timeline Color Coding

| Color | Meaning |
|-------|---------|
| Green | HIGH confidence — auto-accepted |
| Yellow | UNCERTAIN — needs review |
| Red | LOW confidence or jitter detected |
| Blue | User-edited frame |

---

## Export Formats

| Format | Files | Use Case |
|--------|-------|----------|
| COCO JSON | `coco/annotations/instances_{split}.json` | Object detection / segmentation training |
| Labelme JSON | `json_image/*.json` + images | Per-frame review and correction |
| HuggingFace Parquet | `parquet/instances_{split}.parquet` | HuggingFace Datasets upload |

All formats support automatic train/val/test split (default 80/10/10).

---

## Project Structure

```
sam3_hil/
├── server/                     # FastAPI backend (runs in Docker)
│   ├── main.py                 # Server entry point
│   ├── task_manager.py         # Job queue + GPU management
│   └── routes/                 # API endpoints
├── src/
│   ├── api_client.py           # HTTP/WebSocket client
│   ├── core/
│   │   ├── sam3_engine.py      # SAM3 inference wrapper
│   │   ├── confidence_analyzer.py  # Presence Score classification
│   │   ├── action_logger.py    # Efficiency metrics (TEO, EOR, CPO, SPF)
│   │   ├── exporter.py         # Multi-format export
│   │   ├── video_loader.py     # Video/image I/O with LRU cache
│   │   ├── jitter_detector.py  # Temporal quality control (optional)
│   │   └── maritime_roi.py     # Horizon detection (optional)
│   └── gui/
│       ├── main_window_server.py   # Main GUI window
│       ├── interactive_canvas.py   # Point-click refinement
│       ├── timeline_widget.py      # Confidence timeline
│       ├── startup_dialog.py       # Mode selection dialog
│       └── server_workers/         # Background workers for server communication
├── configs/config.py           # Pydantic-based configuration
├── docker/
│   ├── Dockerfile.server       # Server Docker image
│   └── docker_run.sh           # Legacy standalone launcher
├── main_server.py              # Client entry point
├── docker-compose.yml          # Server deployment
├── build_client.sh             # PyInstaller build script
├── stamp_client.spec           # PyInstaller spec
├── requirements.txt            # Full dependencies
├── requirements-client.txt     # Client-only dependencies
└── third_party/sam3/           # SAM3 Git submodule
```

---

## VRAM Reference

SAM3 video processing is memory-intensive. Empirical formula:

```
VRAM (GB) ≈ 2.0 + 0.034 × frames × (resolution / 1080p)
```

| Resolution | 200 frames | 400 frames | 500 frames | Max safe (24GB GPU) |
|------------|------------|------------|------------|---------------------|
| 1920×1080  | ~8.8 GB    | ~15.6 GB   | ~19.0 GB   | ~580 frames         |
| 1280×720   | ~3.9 GB    | ~6.9 GB    | ~8.5 GB    | ~1300 frames        |
| 960×540    | ~2.2 GB    | ~3.8 GB    | ~4.8 GB    | ~2900 frames        |

For videos exceeding VRAM limits:

```bash
# Trim frames
ffmpeg -i input.mp4 -vframes 500 output_500f.mp4

# Reduce resolution
ffmpeg -i input.mp4 -vf "scale=1280:720" output_720p.mp4
```

---

## Configuration

Key parameters (adjustable in `configs/config.py` or via environment variables):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `high_threshold` | 0.80 | Score ≥ this → auto-accept |
| `low_threshold` | 0.50 | Score < this → auto-reject / flag |
| `iou_threshold` | 0.85 | Jitter detection: IoU below this triggers review |
| `area_change_threshold` | 0.15 | Jitter detection: area change above 15% triggers review |

---

## Troubleshooting

**Server won't start / SAM3 not loading:**
```bash
docker compose logs -f              # Check server logs
curl http://localhost:8000/status    # Check SAM3 and GPU status
```

**Client can't connect:**
```bash
# On server machine, find IP:
hostname -I
# On client, verify connectivity:
curl http://<server-ip>:8000/health
# If blocked, open port:
sudo ufw allow 8000
```

**CUDA out of memory:**
```bash
# Restart the server to clear GPU state
docker compose restart
# Or reduce video size before processing
```

**Client crashes on startup (glibc error):**
The pre-built client requires glibc ≥ the version on the build machine. Rebuild on the target machine or a machine with older glibc.

**HuggingFace authentication:**
```bash
echo $HF_TOKEN                      # Verify token is set
huggingface-cli login                # Manual login alternative
```

---

## Efficiency Metrics

STAMP tracks the following metrics via ActionLogger:

| Metric | Full Name | Definition |
|--------|-----------|------------|
| **TEO** | Total Edit Operations | Total count of all edit actions (primary workload metric) |
| **EOR** | Edit Operation Rate | TEO / total frames |
| **FCR** | Frame Coverage Rate | Unique edited frames / total frames |
| **CPO** | Clicks Per Object | Total clicks / unique objects |
| **SPF** | Seconds Per Frame | Total annotation time / total frames |

---

## References

- [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719)
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — ICLR 2025
- [Cutie: Putting the Object Back into Video Object Segmentation](https://arxiv.org/abs/2310.12982) — CVPR 2024
- [XMem++: Production-level Video Segmentation](https://arxiv.org/abs/2307.15958) — ICCV 2023

---

## Citation

```bibtex
@mastersthesis{ShihSTAMP2026,
  title   = {Efficiency-Driven Semi-Automated Data Engine:
             Leveraging SAM 3 Presence Confidence for
             Human-in-the-Loop Video Annotation},
  author  = {Shih, Chih-Yuan},
  school  = {National Yang Ming Chiao Tung University},
  year    = {2026},
  note    = {Assistive Robotics Group}
}
```

---

## License

MIT License — Adam Shih @ NYCU Assistive Robotics Group
