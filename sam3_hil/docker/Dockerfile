# =============================================================================
# HIL-AA Maritime Annotation System - Dockerfile
# SAM 3 + PyQt6 + X11 GUI + CUDA Support
# =============================================================================

FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

LABEL version="1.0"

# =============================================================================
# Environment Variables
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# X11 Display (will be overridden by run_docker.sh)
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    # Python (Removed python3-pip from here to avoid distutils error)
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    # Qt/PyQt6 dependencies
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libegl1 \
    libgl1 \
    libglx0 \
    libdbus-1-3 \
    # X11
    x11-apps \
    # Video processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Utilities
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Setup
# =============================================================================
# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip manually for Python 3.12 (Fixes 'No module named distutils' error)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Upgrade pip and tools
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Working Directory
# =============================================================================
WORKDIR /app

# =============================================================================
# Python Dependencies
# =============================================================================
# FIX: Torch 2.7.0 does not exist yet, using stable 2.5.1
RUN pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# SAM 3 Installation (from GitHub)
# =============================================================================
# Option 2: GitHub install (more reliable for now)
# Ensure your local directory third_party/sam3 exists, or comment this out if not ready
COPY third_party/sam3 /tmp/sam3
RUN pip install /tmp/sam3

# =============================================================================
# Create non-root user for X11 compatibility
# =============================================================================
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser \
    && useradd -m -u ${UID} -g ${GID} appuser \
    && chown -R appuser:appuser /app

# =============================================================================
# Application Code
# =============================================================================
COPY --chown=appuser:appuser . /app/

# =============================================================================
# Model Cache Directory
# =============================================================================
RUN mkdir -p /app/models && chown -R appuser:appuser /app/models
ENV SAM3_MODEL_PATH=/app/models

# =============================================================================
# Switch to non-root user
# =============================================================================
USER appuser

# =============================================================================
# Entry Point
# =============================================================================
CMD ["python", "src/main.py"]