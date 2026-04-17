#!/bin/bash
# Build torchcodec 0.10.0 from source (CPU-only) for ROCm.
#
# Why CPU-only: torchcodec's GPU decode path uses NVIDIA NVDEC (CUDA).
# AMD GPUs have VCN hardware but torchcodec has no VA-API backend yet,
# so CPU libavcodec is the only working path on ROCm.
# This is fine for training — video I/O is not the bottleneck.
set -e

TORCHCODEC_VERSION="v0.10.0"
BUILD_DIR="/tmp/torchcodec"

echo "[torchcodec] Cloning ${TORCHCODEC_VERSION} ..."
rm -rf "${BUILD_DIR}"
git clone --depth 1 --branch "${TORCHCODEC_VERSION}" \
    https://github.com/pytorch/torchcodec.git "${BUILD_DIR}"

pip install -q pybind11

echo "[torchcodec] Configuring (CPU-only, ENABLE_CUDA=OFF) ..."
mkdir -p "${BUILD_DIR}/build" && cd "${BUILD_DIR}/build"
cmake "${BUILD_DIR}" \
    -DENABLE_CUDA= \
    -DTorch_DIR="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch" \
    -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
    -DTORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON \
    -DCMAKE_BUILD_TYPE=Release

echo "[torchcodec] Building ($(nproc) jobs) ..."
cmake --build . -j"$(nproc)" && cmake --install .

echo "[torchcodec] Installing Python package ..."
pip install --no-build-isolation -e "${BUILD_DIR}"

echo "[torchcodec] Verifying ..."
python -c "import torchcodec; print('torchcodec OK:', torchcodec.__version__)"
echo "[torchcodec] Done."
