# BTCW OpenCL GPU Miner

OpenCL GPU miner for **Bitcoin-PoW (BTCW)**. Works with both **NVIDIA** and **AMD** GPUs on Windows and Linux.

## Quick Start (Pre-built Windows Binary)

1. Download the latest release from `x64/Release/`
2. Make sure your BTCW node is running:
   ```
   bitcoin-pow-qt.exe -emergencymining=1
   ```
3. Have at least **1 UTXO** in your wallet. Open `Window -> Console` and type:
   ```
   generate
   ```
4. Wait for the node to begin its mining process, then start the miner:
   ```
   cd x64\Release
   BTCW_OPENCL_MINER.exe 1
   ```

The `1` selects GPU device #1. Use `2`, `3`, etc. for additional GPUs.

## GPU Compatibility

| Vendor | Platform | Status |
|--------|----------|--------|
| NVIDIA | CUDA OpenCL runtime | Fully supported |
| AMD | ROCm / AMDGPU-PRO OpenCL | Fully supported |
| Intel | Intel OpenCL runtime | Untested |

This miner uses **OpenCL**, which is the cross-platform GPU compute standard. Unlike CUDA-only miners, it runs on any GPU that provides an OpenCL driver.

## Building from Source

### Windows

**Prerequisites:**
- Visual Studio 2022 Build Tools (or full VS)
- OpenCL SDK (included with CUDA Toolkit, AMD APP SDK, or Intel OpenCL SDK)
- [PDCurses](https://github.com/wmcbrine/PDCurses) (for console UI)

**Build PDCurses first:**
```
git clone https://github.com/wmcbrine/PDCurses.git C:\PDCurses
cd C:\PDCurses\wincon
nmake -f Makefile.vc clean
nmake -f Makefile.vc
```

**Build the miner:**

Open an **x64 Native Tools Command Prompt for VS 2022**, then:
```
cd path\to\opengl-btcw-miner
build.bat
```

Edit `build.bat` to adjust `CUDA_PATH` and `PDCURSES_PATH` for your system.

### Linux

**Prerequisites:**
```bash
sudo apt install build-essential ocl-icd-opencl-dev
```

For NVIDIA:
```bash
sudo apt install nvidia-driver-560-server nvidia-opencl-icd
```

For AMD:
```bash
# Install ROCm or AMDGPU-PRO driver with OpenCL support
```

**Build:**
```bash
chmod +x build_linux.sh
./build_linux.sh
```

**Run:**
```bash
cd release
./btcw_opencl_miner 1
```

## Usage

```
BTCW_OPENCL_MINER.exe <gpu_number> [work_size]
```

| Argument | Description |
|----------|-------------|
| `gpu_number` | 1-based GPU index (required) |
| `work_size` | Manual global work size override (optional, auto-tuned by default) |

### Multi-GPU Mining

Run one instance per GPU in separate terminals:
```
BTCW_OPENCL_MINER.exe 1
BTCW_OPENCL_MINER.exe 2
BTCW_OPENCL_MINER.exe 3
```

### Typical Output (Mining)
```
BTCW.SPACE OpenCL GPU Miner v26.5.4

Hash found - NONCE: 0707070707070707

Hash no sig low64: 6bfb0091a42a80d9

CONNECTED TO BTCW NODE WALLET

=======================================================
Device: NVIDIA GeForce RTX 4080 SUPER
-------------------------------------------------------
Hashrate: 18415616.000000 H/s
Power: 245 W | Temp: 62 C | Fan: 55%
=======================================================
```

### Typical Output (Not Mining)
```
Hash no sig low64: 0000000000000000

!!! NOT CONNECTED TO BTCW NODE WALLET !!!
Make sure your wallet has at least 1 utxo.
```

## Important Notes

- **Do NOT start the miner before the node has begun mining.** Run `generate` in the node console first.
- **Only run one BTCW node per machine** when using this miner. Multiple nodes sharing the same shared memory will cause nonce conflicts.
- The miner communicates with the node via shared memory (IPC). The node must be running on the same machine.
- Kernel files (`.cl`) are compiled by the OpenCL runtime on first launch and cached in `kernel_cache.bin`. Delete the cache if you update kernel files.

## Project Structure

```
opengl-btcw-miner/
├── opencl_miner.cpp          # Windows host code (PDCurses UI)
├── opencl_miner_linux.cpp    # Linux host code (headless stdout)
├── mining_kernel.cl           # Main OpenCL mining kernel
├── kernels/
│   ├── secp256k1_field.cl     # secp256k1 field arithmetic
│   └── secp256k1_point.cl     # secp256k1 point operations
├── ecmult_gen_table.bin       # Pre-computed EC multiplication table
├── extract_table.py           # Table generation utility
├── build.bat                  # Windows build script
├── build_linux.sh             # Linux build script
└── x64/Release/               # Pre-built Windows binary
    ├── BTCW_OPENCL_MINER.exe
    └── kernels/
        ├── mining_kernel.cl
        ├── secp256k1_field.cl
        └── secp256k1_point.cl
```

## License

Copyright (c) 2026 0x369d. All rights reserved. Unauthorized copying or distribution is prohibited.
