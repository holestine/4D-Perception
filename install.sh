#!/usr/bin/env bash
# Install 4D-Perception dependencies and set up OpenPCDet.
#
# Detects the system CUDA version and installs the matching PyTorch and spconv
# builds, then clones OpenPCDet at the tested commit, applies the compatibility
# patch, and builds the CUDA extensions.
#
# Requirements: conda environment with Python 3.10 already active, CUDA toolkit
# on PATH (nvcc accessible), and a CUDA-capable GPU.
#
# Usage:
#   conda create -n 4d python=3.10 -y && conda activate 4d
#   bash install.sh

set -euo pipefail

OPENPCDET_COMMIT="233f849"
OPENPCDET_REPO="https://github.com/open-mmlab/OpenPCDet.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. Detect CUDA version from nvcc
# ---------------------------------------------------------------------------
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install the CUDA toolkit and ensure it is on PATH."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
echo "Detected CUDA $CUDA_VERSION"

# GPU compute capability decides which PyTorch wheels actually run: the cu128
# wheels only ship kernels for sm_75 (Turing) and newer, so Pascal cards like
# the GTX 1080 (sm_61) need cu126 or older even when the toolkit is 12.8+.
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
if [[ -n "$COMPUTE_CAP" ]]; then
    echo "Detected GPU compute capability $COMPUTE_CAP"
else
    echo "WARNING: could not query GPU compute capability (old driver?); assuming a modern GPU."
fi
pre_turing() { [[ -n "$COMPUTE_CAP" ]] && awk -v c="$COMPUTE_CAP" 'BEGIN { exit !(c < 7.5) }'; }

# ---------------------------------------------------------------------------
# 2. Select PyTorch index URL and spconv package based on CUDA version.
#    spconv has no cu128 build; cu124 is binary-compatible with CUDA 12.8.
# ---------------------------------------------------------------------------
if   [[ $CUDA_MAJOR -ge 12 && $CUDA_MINOR -ge 8 ]] && pre_turing; then
    echo "Pre-Turing GPU (sm < 7.5): cu128 wheels would not run — using cu126 wheels instead."
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    TORCH_SPEC="torch==2.7.0+cu126"   # last wheel family with Pascal/Volta kernels
    SPCONV_PKG="spconv-cu124"
elif [[ $CUDA_MAJOR -ge 12 && $CUDA_MINOR -ge 8 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    TORCH_SPEC="torch==2.7.0+cu128"   # tested version for Blackwell / sm_120
    SPCONV_PKG="spconv-cu124"
elif [[ $CUDA_MAJOR -ge 12 && $CUDA_MINOR -ge 4 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    TORCH_SPEC="torch"
    SPCONV_PKG="spconv-cu124"
elif [[ $CUDA_MAJOR -ge 12 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    TORCH_SPEC="torch"
    SPCONV_PKG="spconv-cu121"
elif [[ $CUDA_MAJOR -eq 11 && $CUDA_MINOR -ge 8 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    TORCH_SPEC="torch"
    SPCONV_PKG="spconv-cu118"
else
    echo "ERROR: CUDA 11.8 or newer is required (detected $CUDA_VERSION)."
    exit 1
fi

echo "PyTorch: $TORCH_SPEC  |  spconv: $SPCONV_PKG"

# ---------------------------------------------------------------------------
# 3. Install PyTorch, spconv, and the rest of the project dependencies
# ---------------------------------------------------------------------------
pip install "$TORCH_SPEC" torchvision torchaudio --index-url "$TORCH_INDEX"
pip install "$SPCONV_PKG"
pip install -r "$SCRIPT_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 4. Clone OpenPCDet, check out the tested commit, apply compatibility patch
# ---------------------------------------------------------------------------
if [[ -d "$SCRIPT_DIR/OpenPCDet/.git" ]]; then
    echo "OpenPCDet directory already exists — skipping clone."
else
    git clone "$OPENPCDET_REPO" "$SCRIPT_DIR/OpenPCDet"
fi

git -C "$SCRIPT_DIR/OpenPCDet" checkout "$OPENPCDET_COMMIT"
git -C "$SCRIPT_DIR/OpenPCDet" apply "$SCRIPT_DIR/openpcdet.patch"

# ---------------------------------------------------------------------------
# 5. Build OpenPCDet CUDA extensions for the local GPU architecture
# ---------------------------------------------------------------------------
# Note for NVIDIA Blackwell (RTX 5080, sm_120): nvcc 12.8 and its companion
# compiler cicc must be on PATH. In a conda environment you may need to symlink
# them from the base env:
#   ln -s /path/to/base/bin/nvcc  $CONDA_PREFIX/bin/nvcc
#   ln -s /path/to/base/bin/cicc  $CONDA_PREFIX/bin/cicc
cd "$SCRIPT_DIR/OpenPCDet"
pip install -r requirements.txt
python setup.py develop
cd "$SCRIPT_DIR"

echo ""
echo "Installation complete. Run 'python main.py' to verify."
