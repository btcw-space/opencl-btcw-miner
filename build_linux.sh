#!/bin/bash
# =============================================================================
# BTCW OpenCL GPU Miner - Linux Build Script
# =============================================================================
# Copyright (c) 2026 btcw.space <btcw.space@proton.me>
# Copyright (c) 2026 btcw.space. All rights reserved.
#
# Requirements:
#   sudo apt install build-essential ocl-icd-opencl-dev
#
# For NVIDIA GPUs:
#   sudo apt install nvidia-driver-560-server nvidia-opencl-icd
#
# For AMD GPUs:
#   Install ROCm or AMDGPU-PRO driver with OpenCL support
#
# Usage:
#   chmod +x build_linux.sh
#   ./build_linux.sh
#
# Then run:
#   cd release
#   ./btcw_opencl_miner 1          # auto-tune work size
#   ./btcw_opencl_miner 1 524288   # manual work size override
# =============================================================================

set -e

echo ""
echo "===== BTCW OpenCL GPU Miner - Linux Build ====="
echo ""

# Output directory
OUT_DIR="release"
EXE_NAME="btcw_opencl_miner"

# Create output directories
mkdir -p "${OUT_DIR}/kernels"

# Compile
echo "Compiling opencl_miner_linux.cpp ..."
g++ -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
    -o "${OUT_DIR}/${EXE_NAME}" \
    opencl_miner_linux.cpp \
    -lOpenCL -lrt -lpthread

echo "Compilation successful!"
echo ""

# Copy kernel files
echo "Copying OpenCL kernel files..."
cp -v kernels/secp256k1_field.cl "${OUT_DIR}/kernels/"
cp -v kernels/secp256k1_point.cl "${OUT_DIR}/kernels/"
cp -v mining_kernel.cl "${OUT_DIR}/kernels/"

echo ""
echo "===== Build Complete ====="
echo ""
echo "Output: ${OUT_DIR}/${EXE_NAME}"
echo "Kernels: ${OUT_DIR}/kernels/"
echo ""
echo "To run:"
echo "  cd ${OUT_DIR}"
echo "  ./${EXE_NAME} 1"
echo ""
echo "Make sure the BTCW node is running with: bitcoinpowd -emergencymining=1"
echo ""
echo "Tip: Run in tmux or screen for persistent sessions:"
echo "  tmux new -s miner"
echo "  cd ${OUT_DIR} && ./${EXE_NAME} 1"
echo "  # Press Ctrl+B then D to detach"
echo "  # tmux attach -t miner  to reattach"
echo ""
