@echo off
REM =============================================================================
REM BTCW OpenCL GPU Miner - Build Script for Windows
REM =============================================================================
REM Copyright (c) 2026 0x369d <0x369d@gmail.com>
REM Copyright (c) 2026 0x369d. All rights reserved.
REM
REM Requirements:
REM   - Visual Studio 2022 Build Tools (or full VS)
REM   - OpenCL headers and library (from CUDA Toolkit, AMD SDK, or Intel SDK)
REM   - PDCurses (built for wincon)
REM =============================================================================

REM --- Configuration (adjust these paths for your system) ---
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set PDCURSES_PATH=C:\PDCurses
REM OpenCL headers and lib from CUDA Toolkit (works for all GPU vendors)
set OPENCL_INC=%CUDA_PATH%\include
set OPENCL_LIB=%CUDA_PATH%\lib\x64

REM PDCurses
set PDCURSES_INC=%PDCURSES_PATH%
set PDCURSES_LIB=%PDCURSES_PATH%\wincon

REM Output
set OUT_DIR=x64\Release
set EXE_NAME=BTCW_OPENCL_MINER.exe

echo.
echo ===== BTCW OpenCL GPU Miner Build =====
echo.

REM Create output directory
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM Create kernels output directory
if not exist "%OUT_DIR%\kernels" mkdir "%OUT_DIR%\kernels"

REM --- Compile ---
echo Compiling opencl_miner.cpp ...
cl.exe /nologo /O2 /EHsc /std:c++17 /DWIN32 /D_WINDOWS /DNDEBUG /I"%OPENCL_INC%" /I"%PDCURSES_INC%" /I"%PDCURSES_INC%\wincon" opencl_miner.cpp /Fe"%OUT_DIR%\%EXE_NAME%" /Fo"%OUT_DIR%\opencl_miner.obj" /link /LIBPATH:"%OPENCL_LIB%" /LIBPATH:"%PDCURSES_LIB%" OpenCL.lib pdcurses.lib user32.lib advapi32.lib

if %ERRORLEVEL% neq 0 (
    echo.
    echo BUILD FAILED!
    echo.
    pause
    exit /b 1
)

echo.
echo Compilation successful!
echo.

REM --- Copy kernel files ---
echo Copying OpenCL kernel files...

REM Copy local kernel source files
copy /Y "kernels\secp256k1_field.cl" "%OUT_DIR%\kernels\" >nul 2>nul
copy /Y "kernels\secp256k1_point.cl" "%OUT_DIR%\kernels\" >nul 2>nul

REM Copy mining kernel
copy /Y "mining_kernel.cl" "%OUT_DIR%\kernels\" >nul

echo.
echo ===== Build Complete =====
echo.
echo Output: %OUT_DIR%\%EXE_NAME%
echo Kernels: %OUT_DIR%\kernels\
echo.
echo To run:
echo   cd %OUT_DIR%
echo   %EXE_NAME% 1
echo.
echo Make sure the BTCW node is running with: bitcoin-pow-qt.exe -emergencymining=1
echo.
