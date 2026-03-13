#!/bin/bash
# =============================================================================
# BTCW Metal GPU Miner - macOS Build Script
# =============================================================================
# Compiles Metal shaders (.metal -> .air -> .metallib) and host code (.mm).
# Requires Xcode Command Line Tools (xcrun, metal, metallib, clang++).
# =============================================================================
set -e

echo "===== BTCW Metal GPU Miner - macOS Build ====="
echo ""

OUT_DIR="release"
EXE_NAME="btcw_metal_miner"

mkdir -p "${OUT_DIR}" build

# -------------------------------------------------------------------------
# Step 1: Compile Metal shaders to AIR (intermediate representation)
# -------------------------------------------------------------------------
# mining_kernel.metal #includes secp256k1_point.metal which #includes
# secp256k1_field.metal, so compiling the top-level file pulls in everything.
# We compile field.metal separately first as a quick syntax check.

echo "[1/3] Compiling Metal shaders..."

xcrun -sdk macosx metal -c kernels/secp256k1_field.metal -I kernels/ \
    -o build/field.air 2>&1 || { echo "FAILED: secp256k1_field.metal"; exit 1; }
echo "      secp256k1_field.metal -> OK"

xcrun -sdk macosx metal -c kernels/mining_kernel.metal -I kernels/ \
    -o build/mining.air 2>&1 || { echo "FAILED: mining_kernel.metal"; exit 1; }
echo "      mining_kernel.metal   -> OK (includes point + field)"

# -------------------------------------------------------------------------
# Step 2: Link AIR files into a Metal library (.metallib)
# -------------------------------------------------------------------------
# Only mining.air is needed since it contains all functions via #include.

echo "[2/3] Linking metallib..."

xcrun -sdk macosx metallib build/mining.air \
    -o "${OUT_DIR}/mining.metallib" 2>&1 || { echo "FAILED: metallib linking"; exit 1; }
echo "      mining.metallib -> OK"

# -------------------------------------------------------------------------
# Step 3: Compile host code (Objective-C++ with Metal + Foundation frameworks)
# -------------------------------------------------------------------------

echo "[3/3] Compiling host code..."

clang++ -O2 -std=c++17 -fobjc-arc \
    -framework Metal -framework Foundation -framework CoreGraphics \
    -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
    metal_miner_macos.mm -o "${OUT_DIR}/${EXE_NAME}"

echo "      ${EXE_NAME} -> OK"

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------

echo ""
echo "===== Build successful! ====="
echo ""
echo "  Executable:  ${OUT_DIR}/${EXE_NAME}"
echo "  Metal lib:   ${OUT_DIR}/mining.metallib"
echo ""
echo "To run:"
echo "  cd ${OUT_DIR}"
echo "  ./${EXE_NAME} [gpu_number] [work_size]"
echo ""
echo "Examples:"
echo "  ./${EXE_NAME}          # auto-tune, default GPU"
echo "  ./${EXE_NAME} 1        # GPU 1 (informational on Mac)"
echo "  ./${EXE_NAME} 1 524288 # GPU 1, manual work size"
