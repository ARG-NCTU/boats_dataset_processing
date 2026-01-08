# HIL-AA Maritime Annotation System

> **Efficiency-Driven Semi-Automated Data Engine:**  
> Leveraging SAM 3 Presence Confidence for Maritime Video Annotation

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![SAM 3](https://img.shields.io/badge/Model-SAM%203-orange.svg)](https://github.com/facebookresearch/sam3)

## 🎯 Core Concept

**「用 AI 信心分數來指揮人類，達成極致的標註效率。」**

This system transforms annotation from a labor-intensive process into an efficient human-in-the-loop workflow:

1. **SAM 3** searches for all objects matching text prompts (e.g., "ship, buoy")
2. **Presence Score** classifies each detection's confidence
3. **High confidence** → Auto-save (no human needed)
4. **Low confidence** → Queue for human review
5. **Human reviewer** uses simple clicks to refine

**Result:** Human role shifts from "annotator" to "reviewer", achieving **5-10x efficiency improvement**.

## 🚀 Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU support)
- X11 display server (Linux) or XQuartz (macOS)
- HuggingFace account with SAM 3 access

### Clone Repository

```bash
# Clone with submodules (recommended)
git clone --recursive git@github.com:ARG-NCTU/boats_dataset_processing.git

# Or if already cloned, initialize submodules
cd boats_dataset_processing
git submodule update --init --recursive
```

### HuggingFace Setup

SAM 3 model requires HuggingFace authentication:

1. Get your token from: https://huggingface.co/settings/tokens
2. Request access to SAM 3 model: https://huggingface.co/facebook/sam3
3. Set environment variable:

```bash
# Add to ~/.bashrc (permanent)
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### Build & Run

```bash
cd sam3_hil

# Build Docker image
./docker/docker_run.sh build

# Run interactive shell
./docker/docker_run.sh shell

# Run tests
./docker/docker_run.sh test
```

### Demo

```bash
# Inside container - extract a frame from video
ffmpeg -i /app/third_party/sam3/assets/videos/bedroom.mp4 -vframes 1 /app/data/output/bedroom_frame.jpg

# Run SAM 3 image demo
python /app/src/demo.py --image /app/data/output/bedroom_frame.jpg --prompt "bed"

# Output saved to: /app/data/output/
```

## 📁 Project Structure

```
sam3_hil/
├── docker/
│   ├── Dockerfile          # Container definition
│   └── docker_run.sh       # Launch script with X11 forwarding
├── src/
│   ├── demo.py             # SAM 3 demo script
│   ├── core/               # Backend modules
│   │   ├── video_loader.py
│   │   ├── horizon_detector.py
│   │   ├── sam3_engine.py
│   │   ├── confidence_analyzer.py
│   │   └── temporal_tracker.py
│   ├── gui/                # PyQt6 interface
│   │   ├── main_window.py
│   │   ├── video_canvas.py
│   │   ├── timeline_widget.py
│   │   └── control_panel.py
│   └── utils/
│       └── export_manager.py
├── configs/
│   └── config.py           # All configurable parameters
├── tests/
├── data/
│   ├── image/              # Input images
│   ├── video/              # Input videos
│   ├── bag/                # Input rosbags
│   └── output/             # Exported annotations
├── models/                 # Model cache (gitignored)
├── third_party/
│   └── sam3/               # SAM 3 submodule
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

All parameters are centralized in `configs/config.py`. Override via environment variables:

```bash
# .env file or export directly
export HIL_CONF_HIGH_THRESHOLD=0.9     # Auto-save threshold
export HIL_CONF_LOW_THRESHOLD=0.7      # Review threshold
export HIL_SAM3_MOCK_MODE=true         # Development mode
```

### Key Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HIGH_THRESHOLD` | 0.9 | Score ≥ this → auto-save (green) |
| `LOW_THRESHOLD` | 0.7 | Score < this → needs review (red) |
| `JITTER_THRESHOLD` | 0.15 | Shape change > 15% → tracking failure |

## 🎮 Controls

| Input | Action |
|-------|--------|
| **Left Click** | Add positive point (this IS the object) |
| **Right Click** | Add negative point (this is NOT the object) |
| **Tab** | Jump to next red (review-needed) frame |
| **Enter** | Confirm current annotation |
| **Space** | Play/Pause video |
| **←/→** | Previous/Next frame |
| **Ctrl+Z** | Undo last point |

## 📊 Expected Performance

| Metric | Traditional | Our Target |
|--------|------------|------------|
| **SPF** (Seconds per Frame) | 60-120s | < 5s |
| **CPO** (Clicks per Object) | 20+ | < 2 |
| **HIR** (Human Intervention Rate) | 100% | < 15% |
| **mIoU** (Annotation Quality) | 95%+ | > 90% |

## 🔬 Key Innovations

1. **Zero-Shot Discovery**: SAM 3 text prompts find all objects automatically
2. **Confidence-Driven AL**: Presence Score determines human involvement
3. **Temporal Tracking + Jitter**: Monitor shape changes to catch tracking failures
4. **Interactive Refinement**: Simple click corrections, not polygon editing
5. **Maritime ROI**: Horizon detection reduces sky false positives
6. **Optimized GUI**: Color-coded timeline + keyboard shortcuts

## ⚠️ Troubleshooting

### Submodule is empty
```bash
git submodule update --init --recursive
```

### HuggingFace authentication error
```bash
# Check if token is set
echo $HF_TOKEN

# Or login manually inside container
huggingface-cli login
```

### CUDA out of memory
- Use image mode instead of video mode (RTX 4070 8GB limitation)
- Reduce batch size in config

## 📚 References

- [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719)
- [Meta SAM 3 Blog](https://ai.meta.com/blog/segment-anything-model-3/)

## 📜 License

MIT License - Sonic @ NYCU Maritime Robotics Lab

---

**Thesis Title:**  
基於 SAM 3 語義置信度與主動學習之海事影像人機協作標註系統