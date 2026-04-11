# STAMP Annotation System

> **SAM Tracking Annotation with Minimal Processing:**  
> Leveraging SAM 3 Presence Confidence for Efficient Maritime Video Annotation

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![SAM 3](https://img.shields.io/badge/Model-SAM%203-orange.svg)](https://github.com/facebookresearch/sam3)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Core Concept

**「用 AI 信心分數來指揮人類，達成極致的標註效率。」**

This system transforms annotation from a labor-intensive process into an efficient human-in-the-loop workflow:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  SAM 3      │ →  │  Presence    │ →  │  Confidence     │ →  │  Human       │
│  Zero-Shot  │    │  Score       │    │  Classification │    │  Review      │
│  Discovery  │    │  Extraction  │    │  HIGH/UNC/LOW   │    │  (if needed) │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
```

1. **SAM 3** searches for all objects matching text prompts (e.g., "ship, buoy")
2. **Presence Score** classifies each detection's confidence
3. **HIGH confidence** → Auto-accept (no human needed)
4. **UNCERTAIN** → Queue for human review
5. **LOW confidence** → Auto-reject or flag for review
6. **Human reviewer** uses simple clicks to refine

**Result:** Human role shifts from "annotator" to "reviewer", achieving **5-10x efficiency improvement**.

---

## 📊 System Architecture

### Six Core Modules

| Module | Name | Function | Status |
|--------|------|----------|--------|
| 1 | **Zero-Shot Discovery** | SAM3 text prompt detection | ✅ Complete |
| 2 | **Confidence-Driven Active Learning** | Presence Score classification (HIGH/UNC/LOW) | ✅ Complete |
| 3 | **Temporal Tracking + Jitter Detection** | Track objects across frames, detect failures | ✅ Complete |
| 4 | **Interactive Refinement** | Point prompts for mask correction | ✅ Complete |
| 5 | **Optimized GUI** | PyQt6 interface with Timeline visualization | ✅ Complete |
| 6 | **Export Module** | Labelme / COCO / HuggingFace formats | ✅ Complete |
| 7 | **Maritime ROI** (Optional) | Horizon detection to filter sky regions | ✅ Complete |

### Confidence Classification

```
┌────────────────────────────────────────────────────────────┐
│                    Presence Score                          │
├──────────────┬──────────────────────┬─────────────────────┤
│  HIGH ≥ 0.80 │  0.50 ≤ UNCERTAIN    │  LOW < 0.50         │
│  🟢 Auto     │  🟡 Human Review     │  🔴 Auto-reject     │
│  Accept      │  Required            │  or Flag            │
└──────────────┴──────────────────────┴─────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU with 24GB+ VRAM (RTX 4090 recommended)
- **Software**: Docker with NVIDIA Container Toolkit
- **Display**: X11 server (Linux) or XQuartz (macOS)
- **Account**: HuggingFace with SAM 3 access

### Clone Repository

```bash
# Clone with submodules
git clone --recursive git@github.com:ARG-NCTU/boats_dataset_processing.git
cd boats_dataset_processing/sam3_hil

# Or initialize submodules if already cloned
git submodule update --init --recursive
```

### HuggingFace Setup

```bash
# 1. Get token from: https://huggingface.co/settings/tokens
# 2. Request SAM 3 access: https://huggingface.co/facebook/sam3
# 3. Set environment variable:
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### Build & Run

```bash
# Build Docker image
./docker/docker_run.sh build

# Run interactive shell
./docker/docker_run.sh shell

# Inside container - launch GUI
python /app/src/gui/main_window.py
```

### Demo

```bash
# Inside container
# 1. Extract a frame from video
ffmpeg -i /app/third_party/sam3/assets/videos/bedroom.mp4 -vframes 1 /app/data/output/bedroom_frame.jpg

# 2. Run SAM 3 demo
python /app/src/demo.py --image /app/data/output/bedroom_frame.jpg --prompt "bed"
```

---

## 📁 Project Structure

```
sam3_hil/
├── docker/
│   ├── Dockerfile              # Container definition
│   └── docker_run.sh           # Launch script with X11 forwarding
├── src/
│   ├── demo.py                 # SAM 3 demo script
│   ├── core/
│   │   ├── video_loader.py         # Video loading with LRU cache
│   │   ├── sam3_engine.py          # SAM 3 API wrapper + Tracker API
│   │   ├── confidence_analyzer.py  # Presence Score analysis + HIR
│   │   ├── jitter_detector.py      # Temporal quality control
│   │   ├── exporter.py             # Multi-format export
│   │   └── maritime_roi.py         # Horizon detection (Traditional/SegFormer)
│   ├── gui/
│   │   ├── main_window.py              # Main GUI (basic version)
│   │   ├── main_window_with_maritime_roi.py  # GUI with Maritime ROI
│   │   ├── interactive_canvas.py       # Click-to-refine canvas
│   │   └── timeline_widget.py          # Jitter visualization
│   └── utils/
├── tests/
│   └── test_vram_estimation.py     # VRAM testing tools
├── data/
│   ├── video/                  # Input videos
│   └── output/                 # Exported annotations
├── models/
│   └── Segformer/              # SegFormer model for horizon detection
├── third_party/
│   └── sam3/                   # SAM 3 submodule
├── requirements.txt
└── README.md
```

---

## 💾 VRAM Requirements

### Empirical Formula

```
VRAM (GB) ≈ 2.0 + 0.034 × frames × (resolution / 1080p)
```

### Reference Table (RTX 4090, 24GB)

| Resolution | 200 frames | 400 frames | 500 frames | 600 frames |
|------------|------------|------------|------------|------------|
| 1920×1080  | ~8.8 GB    | ~15.6 GB   | ~19.0 GB   | ~22.4 GB ⚠️ |
| 1280×720   | ~3.9 GB    | ~6.9 GB    | ~8.5 GB    | ~10.0 GB   |
| 960×540    | ~2.2 GB    | ~3.8 GB    | ~4.8 GB    | ~5.6 GB    |

### Safe Limits (24GB GPU)

| Resolution | Max Frames | Video Duration @30fps |
|------------|------------|----------------------|
| 1080p      | ~580       | ~19 seconds          |
| 720p       | ~1300      | ~43 seconds          |
| 540p       | ~2900      | ~97 seconds          |

### Processing Long Videos

For videos exceeding VRAM limits:

```bash
# Option 1: Trim video
ffmpeg -i input.mp4 -vframes 500 output_500f.mp4

# Option 2: Reduce resolution
ffmpeg -i input.mp4 -vf "scale=1280:720" output_720p.mp4

# Option 3: Segment processing (recommended for long videos)
# Process in 500-frame chunks and merge results
```

---

## ⚙️ Configuration

### Confidence Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HIGH_THRESHOLD` | 0.80 | Score ≥ this → auto-accept (green) |
| `LOW_THRESHOLD` | 0.50 | Score < this → auto-reject (red) |
| `JITTER_IOU_THRESHOLD` | 0.85 | IoU < this → tracking failure |
| `JITTER_AREA_THRESHOLD` | 0.15 | Area change > 15% → flag for review |

### Environment Variables

```bash
# Override defaults
export HIL_CONF_HIGH_THRESHOLD=0.85
export HIL_CONF_LOW_THRESHOLD=0.55
export HIL_SAM3_MOCK_MODE=true  # Development mode (no GPU)
```

---

## 🎮 GUI Controls

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play / Pause video |
| `←` / `→` | Previous / Next frame |
| `Home` / `End` | First / Last frame |
| `Tab` | Jump to next review-needed frame |
| `Enter` | Confirm current annotation |
| `Ctrl+O` | Open video |
| `Ctrl+S` | Save annotations |
| `Ctrl+Q` | Quit |

### Mouse Controls

| Input | Action |
|-------|--------|
| **Left Click** | Add positive point (this IS the object) |
| **Right Click** | Add negative point (this is NOT the object) |
| **Scroll** | Zoom in/out |

### Timeline Color Coding

| Color | Meaning |
|-------|---------|
| 🟢 Green | HIGH confidence - auto-accepted |
| 🟡 Yellow | UNCERTAIN - needs review |
| 🔴 Red | LOW confidence or Jitter detected |
| 🔵 Blue | User-edited frame |

---

## 📊 Performance Metrics

### Target vs Traditional Methods

| Metric | Traditional | HIL-AA Target | Description |
|--------|-------------|---------------|-------------|
| **SPF** | 60-120s | < 5s | Seconds per Frame |
| **CPO** | 20+ | < 2 | Clicks per Object |
| **HIR** | 100% | < 15% | Human Intervention Rate |
| **mIoU** | 95%+ | > 90% | Annotation Quality |

### Key Formulas

```python
# Human Intervention Rate
HIR = frames_needing_review / total_frames × 100%

# Annotation Speedup
Speedup = traditional_time / hil_aa_time
```

---

## 🔬 Key Innovations

1. **Zero-Shot Discovery**: SAM 3 text prompts find all objects automatically
2. **Confidence-Driven Active Learning**: Presence Score determines human involvement
3. **Temporal Tracking + Jitter Detection**: Monitor shape changes to catch tracking failures
4. **Interactive Refinement**: Simple click corrections, not polygon editing
5. **Maritime ROI**: Horizon detection (Traditional + SegFormer) reduces sky false positives
6. **Optimized GUI**: Color-coded timeline + keyboard shortcuts + batch operations

---

## 🌊 Maritime ROI Feature

### Horizon Detection Methods

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **Traditional** | Fast | Moderate | Real-time, clear horizon |
| **SegFormer** | Slow | High | Complex scenes, accurate filtering |
| **Auto** | Adaptive | High | Fallback strategy |

### Usage

```python
from core.maritime_roi import MaritimeROI

# Initialize
roi = MaritimeROI(method='segformer', segformer_model_path='models/Segformer/segformer_model')

# Detect horizon
horizon = roi.detect_horizon(frame)
# Returns: HorizonResult(slope=0.027, center=(960, 522), valid=True, method_used='segformer')

# Get sky box for filtering
sky_box = roi.get_sky_box_xyxy(frame, horizon)
# Returns: [0, 0, 1920, 544] (pixels to exclude)
```

---

## 📤 Export Formats

### Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| **Labelme** | `.json` | Manual review, polygon editing |
| **COCO** | `.json` | Object detection training |
| **HuggingFace** | Dataset | Direct upload to HF Hub |

### Export Example

```python
from core.exporter import Exporter

exporter = Exporter(output_dir='./output')

# Export to Labelme format
exporter.export_labelme(results, video_loader, 'annotations/')

# Export to COCO format
exporter.export_coco(results, video_loader, 'coco_annotations.json')
```

---

## ⚠️ Troubleshooting

### Submodule is empty
```bash
git submodule update --init --recursive
```

### HuggingFace authentication error
```bash
# Check token
echo $HF_TOKEN

# Manual login
huggingface-cli login
```

### CUDA out of memory
```bash
# Check current VRAM usage
nvidia-smi

# Solutions:
# 1. Reduce frame count: ffmpeg -i input.mp4 -vframes 400 output.mp4
# 2. Reduce resolution: ffmpeg -i input.mp4 -vf "scale=1280:720" output.mp4
# 3. Restart application to clear GPU memory
```

### GUI not displaying (X11 error)
```bash
# On host machine
xhost +local:docker

# Then restart container
./docker/docker_run.sh shell
```

### SegFormer model not found
```bash
# Inside container
python /app/src/download_segformer_model.py -o /app/models/Segformer/segformer_model
```

---

## 📚 References

### SAM 3
- [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719)
- [Meta SAM 3 Blog](https://ai.meta.com/blog/segment-anything-model-3/)

### Related Work
- **SAM 2** (Meta, ICLR 2025) - 8.4× annotation speedup
- **Cutie** (CVPR 2024) - 88.0% J&F on DAVIS 2017
- **XMem++** (ICCV 2023) - 2× annotation efficiency
- **MaskAL** (2022) - 82% labeling data reduction

### Maritime Datasets
- **MarineInst20M** (ECCV 2024) - 19.2M masks
- **M2SODAI** (2023) - RGB + hyperspectral maritime detection

---

## 📜 License

MIT License - Adam @ NYCU Assistive Robotics Group

---

## 📖 Citation

If you use this system in your research, please cite:

```bibtex
@mastersthesis{AdamShihSTAMP,
  title={基於 SAM 3 語義置信度與主動學習之海事影像人機協作標註系統},
  author={Adam},
  school={National Yang Ming Chiao Tung University},
  year={2026},
  note={Assistive Robotics Group}
}
```

---

**Thesis Title:**  
基於 SAM 3 語義置信度與主動學習之海事影像人機協作標註系統

**English Title:**  
Human-in-the-Loop Active Annotation System for Video Using SAM 3 Semantic Confidence and Active Learning
