# HIL-AA Maritime Annotation System

> **Efficiency-Driven Semi-Automated Data Engine:**  
> Leveraging SAM 3 Presence Confidence for Maritime Video Annotation

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
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

### Build & Run

```bash
# Clone the repository
cd boat_dataset_processing/sam3_hil

# Build Docker image
./docker/run_docker.sh build

# Run the application
./docker/run_docker.sh run

# Run with mock SAM 3 (no GPU needed)
./docker/run_docker.sh run python main.py --mock

# Run tests
./docker/run_docker.sh test

# Interactive shell
./docker/run_docker.sh shell
```

### Test X11 & GPU

```bash
# Inside container
python main.py test-x11    # Test X11 display
python main.py test-gpu    # Test CUDA availability
```

## 📁 Project Structure

```
sam3_hil/
├── docker/
│   ├── Dockerfile          # Container definition
│   └── run_docker.sh       # Launch script with X11 forwarding
├── src/
│   ├── config.py           # All configurable parameters
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
├── tests/
├── data/                   # Input videos (gitignored)
├── models/                 # SAM 3 checkpoints (gitignored)
├── output/                 # Exported annotations
├── main.py                 # Entry point
├── requirements.txt
└── .env.example            # Environment variable template
```

## ⚙️ Configuration

All parameters are centralized in `src/config.py`. Override via environment variables:

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

## 📚 References

- [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719)
- [Meta SAM 3 Blog](https://ai.meta.com/blog/segment-anything-model-3/)

## 📜 License

MIT License - Sonic @ NYCU Maritime Robotics Lab

---

**Thesis Title:**  
基於 SAM 3 語義置信度與主動學習之海事影像人機協作標註系統