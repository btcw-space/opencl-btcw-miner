// =============================================================================
// BTCW OpenCL GPU Miner - Windows Host Code
// =============================================================================
// Copyright (c) 2026 btcw.space <btcw.space@proton.me>
//
// OpenCL GPU miner for Bitcoin-PoW (BTCW). Compatible with NVIDIA and AMD GPUs.
// Communicates with the Bitcoin-PoW node via Windows shared memory (IPC).
// Uses PDCurses for console UI.
//
// Copyright (c) 2026 btcw.space. All rights reserved.
// =============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <thread>

// Windows shared memory
#include <windows.h>

// PDCurses
#include <curses.h>

// OpenCL
#include <CL/cl.h>

// =============================================================================
// NVML (NVIDIA Management Library) — dynamic loading for GPU telemetry
// =============================================================================
// Loaded at runtime from nvml.dll so no build-time dependency is needed.
// If the dll isn't present (AMD GPU), telemetry gracefully shows "N/A".

typedef enum {
    NVML_SUCCESS_V = 0
} nvml_return_t;

typedef void* nvmlDevice_t;

// Function pointer types
typedef nvml_return_t (*pfn_nvmlInit)(void);
typedef nvml_return_t (*pfn_nvmlShutdown)(void);
typedef nvml_return_t (*pfn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*);
typedef nvml_return_t (*pfn_nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int*);
typedef nvml_return_t (*pfn_nvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int*);
typedef nvml_return_t (*pfn_nvmlDeviceGetFanSpeed)(nvmlDevice_t, unsigned int*);

// Global NVML state
static HMODULE g_nvml_dll = NULL;
static bool    g_nvml_ok  = false;
static nvmlDevice_t g_nvml_device = NULL;

static pfn_nvmlInit                     p_nvmlInit = nullptr;
static pfn_nvmlShutdown                 p_nvmlShutdown = nullptr;
static pfn_nvmlDeviceGetHandleByIndex   p_nvmlDeviceGetHandleByIndex = nullptr;
static pfn_nvmlDeviceGetPowerUsage      p_nvmlDeviceGetPowerUsage = nullptr;
static pfn_nvmlDeviceGetTemperature     p_nvmlDeviceGetTemperature = nullptr;
static pfn_nvmlDeviceGetFanSpeed        p_nvmlDeviceGetFanSpeed = nullptr;

static void nvml_init(unsigned int gpu_index = 0) {
    g_nvml_dll = LoadLibraryA("nvml.dll");
    if (!g_nvml_dll) {
        std::cout << "NVML: nvml.dll not found (non-NVIDIA GPU?) - GPU telemetry disabled." << std::endl;
        return;
    }
    p_nvmlInit                   = (pfn_nvmlInit)GetProcAddress(g_nvml_dll, "nvmlInit_v2");
    p_nvmlShutdown               = (pfn_nvmlShutdown)GetProcAddress(g_nvml_dll, "nvmlShutdown");
    p_nvmlDeviceGetHandleByIndex = (pfn_nvmlDeviceGetHandleByIndex)GetProcAddress(g_nvml_dll, "nvmlDeviceGetHandleByIndex_v2");
    p_nvmlDeviceGetPowerUsage    = (pfn_nvmlDeviceGetPowerUsage)GetProcAddress(g_nvml_dll, "nvmlDeviceGetPowerUsage");
    p_nvmlDeviceGetTemperature   = (pfn_nvmlDeviceGetTemperature)GetProcAddress(g_nvml_dll, "nvmlDeviceGetTemperature");
    p_nvmlDeviceGetFanSpeed      = (pfn_nvmlDeviceGetFanSpeed)GetProcAddress(g_nvml_dll, "nvmlDeviceGetFanSpeed");

    if (!p_nvmlInit || !p_nvmlDeviceGetHandleByIndex || !p_nvmlDeviceGetPowerUsage) {
        std::cout << "NVML: Failed to resolve required functions - GPU telemetry disabled." << std::endl;
        FreeLibrary(g_nvml_dll);
        g_nvml_dll = NULL;
        return;
    }

    if (p_nvmlInit() != NVML_SUCCESS_V) {
        std::cout << "NVML: nvmlInit failed - GPU telemetry disabled." << std::endl;
        FreeLibrary(g_nvml_dll);
        g_nvml_dll = NULL;
        return;
    }

    if (p_nvmlDeviceGetHandleByIndex(gpu_index, &g_nvml_device) != NVML_SUCCESS_V) {
        std::cout << "NVML: Could not get device handle for GPU " << gpu_index << " - GPU telemetry disabled." << std::endl;
        p_nvmlShutdown();
        FreeLibrary(g_nvml_dll);
        g_nvml_dll = NULL;
        return;
    }

    g_nvml_ok = true;
    std::cout << "NVML: GPU telemetry enabled (power, temperature, fan speed)." << std::endl;
}

static void nvml_shutdown() {
    if (g_nvml_dll) {
        if (p_nvmlShutdown) p_nvmlShutdown();
        FreeLibrary(g_nvml_dll);
        g_nvml_dll = NULL;
    }
    g_nvml_ok = false;
}

// Query GPU stats — safe to call even if NVML is not available
struct GpuStats {
    bool  available;
    float power_w;      // watts
    int   temp_c;       // celsius
    int   fan_pct;      // 0-100%
    bool  has_temp;
    bool  has_fan;
};

static GpuStats nvml_query() {
    GpuStats s = {};
    if (!g_nvml_ok) return s;
    s.available = true;

    unsigned int power_mw = 0;
    if (p_nvmlDeviceGetPowerUsage(g_nvml_device, &power_mw) == NVML_SUCCESS_V) {
        s.power_w = power_mw / 1000.0f;
    }

    if (p_nvmlDeviceGetTemperature) {
        unsigned int temp = 0;
        // 0 = NVML_TEMPERATURE_GPU
        if (p_nvmlDeviceGetTemperature(g_nvml_device, 0, &temp) == NVML_SUCCESS_V) {
            s.temp_c = (int)temp;
            s.has_temp = true;
        }
    }

    if (p_nvmlDeviceGetFanSpeed) {
        unsigned int fan = 0;
        if (p_nvmlDeviceGetFanSpeed(g_nvml_device, &fan) == NVML_SUCCESS_V) {
            s.fan_pct = (int)fan;
            s.has_fan = true;
        }
    }

    return s;
}

// =============================================================================
// Constants (must match the BTCW node's miner.cpp)
// =============================================================================

#define SHM_NAME "/shared_mem"

static const int CTX_SIZE_BYTES          = 8 * 20; // 160
static const int KEY_SIZE_BYTES          = 32;
static const int HASH_NO_SIG_SIZE_BYTES  = 32;
static const int TOTAL_BYTES_SEND        = CTX_SIZE_BYTES + KEY_SIZE_BYTES + HASH_NO_SIG_SIZE_BYTES;
static const int NONCE_SIZE_BYTES        = 8;

static const uint64_t SENTINEL_NONCE = 0x0707070707070707ULL;

// =============================================================================
// SharedData structure (must match node's definition)
// =============================================================================

struct SharedData {
    volatile uint64_t nonce;
    volatile uint8_t  data[TOTAL_BYTES_SEND]; // [key 32B][ctx 160B][hash_no_sig 32B]
};

// =============================================================================
// OpenCL Helper Functions
// =============================================================================

static const char* cl_err_str(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                        return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:               return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:          return "CL_COMPILER_NOT_AVAILABLE";
        case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";
        case CL_INVALID_PLATFORM:                return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE:           return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_PROGRAM:                 return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:      return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:             return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL:                  return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:               return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:               return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:          return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:          return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:           return "CL_INVALID_GLOBAL_OFFSET";
        case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";
        default:                                 return "UNKNOWN_CL_ERROR";
    }
}

#define CL_CHECK(call) do { \
    cl_int _err = (call); \
    if (_err != CL_SUCCESS) { \
        std::cerr << "OpenCL error " << cl_err_str(_err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

// Read a text file into a string
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        return "";
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    int gpu_num = 0;
    int user_work_size = 0; // 0 = auto-tune
    if (argc >= 2) {
        gpu_num = atoi(argv[1]);
    }
    if (argc >= 3) {
        user_work_size = atoi(argv[2]);
    }

    // -----------------------------------------------------------------
    // 1. Enumerate OpenCL platforms and devices
    // -----------------------------------------------------------------
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    std::cout << "BTCW.SPACE OpenCL GPU Miner v26.5.4" << std::endl;
    std::cout << "Found " << num_platforms << " OpenCL platform(s):" << std::endl;

    // Collect all GPU devices across all platforms
    struct DeviceEntry {
        cl_platform_id platform;
        cl_device_id   device;
        std::string    name;
        std::string    vendor;
        cl_uint        compute_units;
        cl_ulong       global_mem;
        size_t         max_wg_size;
    };
    std::vector<DeviceEntry> all_devices;

    for (cl_uint pi = 0; pi < num_platforms; pi++) {
        char plat_name[256] = {};
        clGetPlatformInfo(platforms[pi], CL_PLATFORM_NAME, sizeof(plat_name), plat_name, nullptr);
        std::cout << "  Platform " << pi << ": " << plat_name << std::endl;

        cl_uint num_devs = 0;
        clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devs);
        if (num_devs == 0) continue;

        std::vector<cl_device_id> devs(num_devs);
        clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_GPU, num_devs, devs.data(), nullptr);

        for (cl_uint di = 0; di < num_devs; di++) {
            DeviceEntry e;
            e.platform = platforms[pi];
            e.device   = devs[di];

            char name[256] = {}, vendor[256] = {};
            clGetDeviceInfo(devs[di], CL_DEVICE_NAME, sizeof(name), name, nullptr);
            clGetDeviceInfo(devs[di], CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
            clGetDeviceInfo(devs[di], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(e.compute_units), &e.compute_units, nullptr);
            clGetDeviceInfo(devs[di], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(e.global_mem), &e.global_mem, nullptr);
            clGetDeviceInfo(devs[di], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(e.max_wg_size), &e.max_wg_size, nullptr);

            e.name   = name;
            e.vendor = vendor;

            std::cout << "    Device " << all_devices.size() << ": " << name
                      << " (" << vendor << ")"
                      << " CUs=" << e.compute_units
                      << " Mem=" << (e.global_mem / (1024 * 1024)) << "MB"
                      << " MaxWG=" << e.max_wg_size
                      << std::endl;

            all_devices.push_back(e);
        }
    }

    if (all_devices.empty()) {
        std::cerr << "No OpenCL GPU devices found." << std::endl;
        return 1;
    }

    // Select device (gpu_num is 1-based from user, 0 means first)
    int dev_idx = (gpu_num > 0) ? (gpu_num - 1) : 0;
    if (dev_idx >= (int)all_devices.size()) {
        std::cerr << "GPU " << gpu_num << " not found. Only " << all_devices.size() << " GPUs available." << std::endl;
        return 1;
    }

    DeviceEntry& sel = all_devices[dev_idx];
    std::cout << "\nUsing device " << dev_idx << ": " << sel.name << std::endl;

    // -----------------------------------------------------------------
    // 2. Create OpenCL context, command queue
    // -----------------------------------------------------------------
    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &sel.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << cl_err_str(err) << std::endl;
        return 1;
    }

    cl_command_queue queue = clCreateCommandQueue(context, sel.device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue: " << cl_err_str(err) << std::endl;
        return 1;
    }

    // -----------------------------------------------------------------
    // 3. Load and compile OpenCL kernel
    // -----------------------------------------------------------------
    // We need to concatenate: secp256k1_field.cl + secp256k1_point.cl + mining_kernel.cl
    std::string field_src = read_file("kernels/secp256k1_field.cl");
    std::string point_src = read_file("kernels/secp256k1_point.cl");
    std::string mine_src  = read_file("kernels/mining_kernel.cl");

    if (field_src.empty() || point_src.empty() || mine_src.empty()) {
        std::cerr << "Failed to load kernel source files." << std::endl;
        std::cerr << "Make sure the 'kernels/' directory contains:" << std::endl;
        std::cerr << "  secp256k1_field.cl" << std::endl;
        std::cerr << "  secp256k1_point.cl" << std::endl;
        std::cerr << "  mining_kernel.cl" << std::endl;
        return 1;
    }

    // Remove the #include directives from point.cl since we're concatenating
    // (secp256k1_point.cl includes secp256k1_field.cl)
    // We concatenate in order: field -> point (minus include) -> mining kernel
    std::string combined_src = field_src + "\n" + point_src + "\n" + mine_src;

    // Build options: use standard CL1.2 only (no aggressive math flags for crypto correctness)
    std::string build_opts_str = "-cl-std=CL1.2";
    const char* build_opts = build_opts_str.c_str();

    // --- Binary cache: skip compilation if cached binary exists and source hasn't changed ---
    std::string cache_bin_path = "kernels/kernel_cache.bin";
    std::string cache_hash_path = "kernels/kernel_cache.hash";
    
    // Simple hash of source to detect changes: use source length + first/last 1KB
    auto compute_source_hash = [&]() -> std::string {
        std::ostringstream oss;
        oss << combined_src.length();
        if (combined_src.length() > 2048) {
            oss << combined_src.substr(0, 1024) << combined_src.substr(combined_src.length() - 1024);
        } else {
            oss << combined_src;
        }
        // Simple hash: sum all bytes
        uint64_t h = 0;
        for (char c : oss.str()) h = h * 131 + (unsigned char)c;
        char buf[32];
        snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
        return std::string(buf);
    };
    
    std::string src_hash = compute_source_hash();
    cl_program program = nullptr;
    bool used_cache = false;
    
    // Try loading cached binary
    {
        std::ifstream hash_file(cache_hash_path);
        std::string cached_hash;
        if (hash_file.is_open() && std::getline(hash_file, cached_hash) && cached_hash == src_hash) {
            std::ifstream bin_file(cache_bin_path, std::ios::binary | std::ios::ate);
            if (bin_file.is_open()) {
                size_t bin_size = (size_t)bin_file.tellg();
                if (bin_size > 0) {
                    bin_file.seekg(0);
                    std::vector<unsigned char> bin_data(bin_size);
                    bin_file.read(reinterpret_cast<char*>(bin_data.data()), bin_size);
                    if (bin_file.good()) {
                        const unsigned char* bin_ptr = bin_data.data();
                        cl_int bin_status = CL_SUCCESS;
                        program = clCreateProgramWithBinary(context, 1, &sel.device,
                            &bin_size, &bin_ptr, &bin_status, &err);
                        if (err == CL_SUCCESS && bin_status == CL_SUCCESS) {
                            err = clBuildProgram(program, 1, &sel.device, nullptr, nullptr, nullptr);
                            if (err == CL_SUCCESS) {
                                std::cout << "Loaded cached OpenCL kernel binary." << std::endl;
                                used_cache = true;
                            } else {
                                clReleaseProgram(program);
                                program = nullptr;
                            }
                        } else if (program) {
                            clReleaseProgram(program);
                            program = nullptr;
                        }
                    }
                }
            }
        }
    }
    
    // Compile from source if no cache hit
    if (!used_cache) {
        const char* src_ptr = combined_src.c_str();
        size_t src_len = combined_src.length();
        program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create program: " << cl_err_str(err) << std::endl;
            return 1;
        }
        std::cout << "Compiling OpenCL kernel (first run - will be cached for next time)..." << std::endl;
        err = clBuildProgram(program, 1, &sel.device, build_opts, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, sel.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(program, sel.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "OpenCL build failed:\n" << log.data() << std::endl;
            return 1;
        }
        std::cout << "OpenCL kernel compiled successfully." << std::endl;
        
        // Save binary to cache
        size_t bin_size = 0;
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, nullptr);
        if (bin_size > 0) {
            std::vector<unsigned char> bin_data(bin_size);
            unsigned char* bin_ptr = bin_data.data();
            clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &bin_ptr, nullptr);
            std::ofstream bin_out(cache_bin_path, std::ios::binary);
            if (bin_out.is_open()) {
                bin_out.write(reinterpret_cast<char*>(bin_data.data()), bin_size);
                bin_out.close();
                std::ofstream hash_out(cache_hash_path);
                hash_out << src_hash;
                hash_out.close();
                std::cout << "Kernel binary cached (" << bin_size << " bytes)." << std::endl;
            }
        }
    }

    cl_kernel kernel = clCreateKernel(program, "btcw_mine", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel 'btcw_mine': " << cl_err_str(err) << std::endl;
        return 1;
    }

    cl_kernel precomp_kernel = clCreateKernel(program, "precompute_ecmult_gen_table", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel 'precompute_ecmult_gen_table': " << cl_err_str(err) << std::endl;
        return 1;
    }

    // -----------------------------------------------------------------
    // 4. Allocate GPU buffers
    // -----------------------------------------------------------------
    // Precomputed generator table: 128 groups x 4 entries x 8 ulongs = 4096 ulongs = 32KB
    cl_mem d_ecmult_table  = clCreateBuffer(context, CL_MEM_READ_WRITE, 4096 * sizeof(cl_ulong), nullptr, &err);
    cl_mem d_key_data      = clCreateBuffer(context, CL_MEM_READ_ONLY,  KEY_SIZE_BYTES, nullptr, &err);
    cl_mem d_hash_no_sig   = clCreateBuffer(context, CL_MEM_READ_ONLY,  HASH_NO_SIG_SIZE_BYTES, nullptr, &err);
    cl_mem d_result_nonce  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_ulong), nullptr, &err);
    cl_mem d_result_found  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    cl_mem d_hashrate_ctr  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);

    if (!d_ecmult_table || !d_key_data || !d_hash_no_sig || !d_result_nonce || !d_result_found || !d_hashrate_ctr) {
        std::cerr << "Failed to allocate GPU buffers." << std::endl;
        return 1;
    }

    // -----------------------------------------------------------------
    // 4b. Precompute ecmult generator table on GPU (runs once at startup)
    // -----------------------------------------------------------------
    std::cout << "Precomputing ecmult generator table..." << std::endl;
    CL_CHECK(clSetKernelArg(precomp_kernel, 0, sizeof(cl_mem), &d_ecmult_table));
    size_t precomp_global = 1, precomp_local = 1;
    CL_CHECK(clEnqueueNDRangeKernel(queue, precomp_kernel, 1, nullptr, &precomp_global, &precomp_local, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    std::cout << "Ecmult table ready (32KB)." << std::endl;
    clReleaseKernel(precomp_kernel);

    // -----------------------------------------------------------------
    // 4c. ECDSA Diagnostic: sign a known test vector on GPU and display result
    // -----------------------------------------------------------------
    {
        std::cout << "Running ECDSA diagnostic..." << std::endl;

        cl_kernel diag_kernel = clCreateKernel(program, "diagnostic_ecdsa_sign", &err);
        if (err != CL_SUCCESS) {
            std::cerr << "WARNING: diagnostic kernel not found, skipping." << std::endl;
        } else {
            // Test vector: private key = 1, message = all-zeros
            // These are passed as big-endian 32-byte arrays
            uint8_t test_sk[32] = {};
            uint8_t test_msg[32] = {};
            test_sk[31] = 0x01;  // private key = 1 (big-endian)
            // message = 0x00...00

            cl_mem d_diag_sk      = clCreateBuffer(context, CL_MEM_READ_ONLY,  32, nullptr, &err);
            cl_mem d_diag_msg     = clCreateBuffer(context, CL_MEM_READ_ONLY,  32, nullptr, &err);
            cl_mem d_diag_sig     = clCreateBuffer(context, CL_MEM_READ_WRITE, 73, nullptr, &err);
            cl_mem d_diag_sig_len = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &err);

            CL_CHECK(clEnqueueWriteBuffer(queue, d_diag_sk,  CL_TRUE, 0, 32, test_sk,  0, nullptr, nullptr));
            CL_CHECK(clEnqueueWriteBuffer(queue, d_diag_msg, CL_TRUE, 0, 32, test_msg, 0, nullptr, nullptr));
            cl_int zero_len = 0;
            CL_CHECK(clEnqueueWriteBuffer(queue, d_diag_sig_len, CL_TRUE, 0, sizeof(cl_int), &zero_len, 0, nullptr, nullptr));

            CL_CHECK(clSetKernelArg(diag_kernel, 0, sizeof(cl_mem), &d_diag_sk));
            CL_CHECK(clSetKernelArg(diag_kernel, 1, sizeof(cl_mem), &d_diag_msg));
            CL_CHECK(clSetKernelArg(diag_kernel, 2, sizeof(cl_mem), &d_diag_sig));
            CL_CHECK(clSetKernelArg(diag_kernel, 3, sizeof(cl_mem), &d_diag_sig_len));
            CL_CHECK(clSetKernelArg(diag_kernel, 4, sizeof(cl_mem), &d_ecmult_table));

            size_t diag_global = 1, diag_local = 1;
            CL_CHECK(clEnqueueNDRangeKernel(queue, diag_kernel, 1, nullptr, &diag_global, &diag_local, 0, nullptr, nullptr));
            CL_CHECK(clFinish(queue));

            uint8_t sig_buf[73] = {};
            cl_int sig_len = 0;
            CL_CHECK(clEnqueueReadBuffer(queue, d_diag_sig_len, CL_TRUE, 0, sizeof(cl_int), &sig_len, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueReadBuffer(queue, d_diag_sig, CL_TRUE, 0, 73, sig_buf, 0, nullptr, nullptr));

            std::cout << "ECDSA Diagnostic (sk=1, msg=0):" << std::endl;
            std::cout << "  Sig length: " << sig_len << std::endl;
            std::cout << "  DER hex: ";
            for (int i = 0; i < sig_len; i++) {
                char hex[4];
                snprintf(hex, sizeof(hex), "%02x", sig_buf[i]);
                std::cout << hex;
            }
            std::cout << std::endl;

            // Validate DER structure
            bool der_ok = (sig_len >= 70 && sig_len <= 72 &&
                           sig_buf[0] == 0x30 &&
                           sig_buf[2] == 0x02);
            std::cout << "  DER structure: " << (der_ok ? "OK" : "INVALID") << std::endl;

            // Second test: different message to ensure determinism works
            uint8_t test_msg2[32] = {};
            test_msg2[31] = 0x01;
            CL_CHECK(clEnqueueWriteBuffer(queue, d_diag_msg, CL_TRUE, 0, 32, test_msg2, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueWriteBuffer(queue, d_diag_sig_len, CL_TRUE, 0, sizeof(cl_int), &zero_len, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueNDRangeKernel(queue, diag_kernel, 1, nullptr, &diag_global, &diag_local, 0, nullptr, nullptr));
            CL_CHECK(clFinish(queue));

            uint8_t sig_buf2[73] = {};
            cl_int sig_len2 = 0;
            CL_CHECK(clEnqueueReadBuffer(queue, d_diag_sig_len, CL_TRUE, 0, sizeof(cl_int), &sig_len2, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueReadBuffer(queue, d_diag_sig, CL_TRUE, 0, 73, sig_buf2, 0, nullptr, nullptr));

            std::cout << "ECDSA Diagnostic (sk=1, msg=1):" << std::endl;
            std::cout << "  Sig length: " << sig_len2 << std::endl;
            std::cout << "  DER hex: ";
            for (int i = 0; i < sig_len2; i++) {
                char hex[4];
                snprintf(hex, sizeof(hex), "%02x", sig_buf2[i]);
                std::cout << hex;
            }
            std::cout << std::endl;

            bool sigs_differ = (sig_len != sig_len2) ||
                               (memcmp(sig_buf, sig_buf2, sig_len) != 0);
            std::cout << "  Different from test 1: " << (sigs_differ ? "YES (good)" : "NO (BAD - same sig for different messages!)") << std::endl;

            clReleaseMemObject(d_diag_sk);
            clReleaseMemObject(d_diag_msg);
            clReleaseMemObject(d_diag_sig);
            clReleaseMemObject(d_diag_sig_len);
            clReleaseKernel(diag_kernel);

            // Build hex strings from GPU output
            char gpu_hex1[150] = {};
            char gpu_hex2[150] = {};
            for (int i = 0; i < sig_len; i++)  snprintf(gpu_hex1 + i*2, 3, "%02x", sig_buf[i]);
            for (int i = 0; i < sig_len2; i++) snprintf(gpu_hex2 + i*2, 3, "%02x", sig_buf2[i]);

            // ---- Known-good reference signatures (BTCW's own libsecp256k1 + RFC 6979) ----
            // Generated by verify_sigs.exe using the BTCW node's secp256k1 library.
            // These are the ONLY correct deterministic signatures for these inputs.
            // sk=1 (big-endian 0x00..01), msg=0x00..00 (32 zero bytes):
            const char* ref_sig1 = "3045022100a0b37f8fba683cc68f6574cd43b39f0343a50008bf6ccea9d13231d9e7e2e1e4022011edc8d307254296264aebfc3dc76cd8b668373a072fd64665b50000e9fcce52";
            // sk=1 (big-endian 0x00..01), msg=0x00..01 (31 zero bytes + 0x01):
            const char* ref_sig2 = "304402206673ffad2147741f04772b6f921f0ba6af0c1e77fc439e65c36dedf4092e889802204c1a971652e0ada880120ef8025e709fff2080c4a39aae068d12eed009b68c89";

            bool sig1_match = (strcmp(gpu_hex1, ref_sig1) == 0);
            bool sig2_match = (strcmp(gpu_hex2, ref_sig2) == 0);

            // Determine overall status
            bool structural_ok = der_ok && sigs_differ;
            bool exact_ok = sig1_match && sig2_match;

            // ---- Write diagnostic log file ----
            {
                FILE* diagf = fopen("miner_diagnostic.log", "w");
                if (diagf) {
                    time_t now = time(nullptr);
                    char timebuf[64];
                    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
                    fprintf(diagf, "============================================================\n");
                    fprintf(diagf, "  BTCW OpenCL Miner - ECDSA Diagnostic Report\n");
                    fprintf(diagf, "  Timestamp: %s\n", timebuf);
                    fprintf(diagf, "============================================================\n\n");

                    fprintf(diagf, "Test 1: sk = 0x00..01 (32 bytes), msg = 0x00..00 (32 bytes)\n");
                    fprintf(diagf, "  GPU sig (%d bytes): %s\n", sig_len, gpu_hex1);
                    fprintf(diagf, "  Reference        : %s\n", ref_sig1);
                    fprintf(diagf, "  Exact match      : %s\n\n", sig1_match ? "YES" : "*** NO ***");

                    fprintf(diagf, "Test 2: sk = 0x00..01 (32 bytes), msg = 0x00..01 (32 bytes)\n");
                    fprintf(diagf, "  GPU sig (%d bytes): %s\n", sig_len2, gpu_hex2);
                    fprintf(diagf, "  Reference        : %s\n", ref_sig2);
                    fprintf(diagf, "  Exact match      : %s\n\n", sig2_match ? "YES" : "*** NO ***");

                    fprintf(diagf, "------------------------------------------------------------\n");
                    fprintf(diagf, "  DER structure valid : %s\n", der_ok ? "PASS" : "FAIL");
                    fprintf(diagf, "  Signatures differ   : %s\n", sigs_differ ? "PASS" : "FAIL");
                    fprintf(diagf, "  Test 1 exact match  : %s\n", sig1_match ? "PASS" : "FAIL");
                    fprintf(diagf, "  Test 2 exact match  : %s\n", sig2_match ? "PASS" : "FAIL");
                    fprintf(diagf, "------------------------------------------------------------\n\n");

                    if (exact_ok) {
                        fprintf(diagf, "===== RESULT: PASS =====\n");
                        fprintf(diagf, "GPU ECDSA signatures are cryptographically correct.\n");
                        fprintf(diagf, "The miner should be able to find and submit valid blocks.\n");
                    } else if (structural_ok) {
                        fprintf(diagf, "===== RESULT: WARNING =====\n");
                        fprintf(diagf, "DER structure is valid but signatures don't match the reference.\n");
                        fprintf(diagf, "This means the GPU is producing WRONG signatures.\n");
                        fprintf(diagf, "Candidates will be rejected by the node.\n");
                    } else {
                        fprintf(diagf, "===== RESULT: FAIL =====\n");
                        fprintf(diagf, "GPU ECDSA is fundamentally broken.\n");
                    }
                    fclose(diagf);
                    std::cout << "Diagnostic written to miner_diagnostic.log" << std::endl;
                }
            }

            if (!der_ok) {
                std::cerr << "FATAL: GPU ECDSA produces invalid DER signatures. Cannot mine." << std::endl;
                return 1;
            }

            if (!exact_ok) {
                std::cerr << "WARNING: GPU signatures don't match reference. Check miner_diagnostic.log" << std::endl;
                std::cerr << "  GPU sig1: " << gpu_hex1 << std::endl;
                std::cerr << "  Expected: " << ref_sig1 << std::endl;
                // Don't exit — let the user decide. But log clearly.
            } else {
                std::cout << "ECDSA diagnostic PASSED - signatures match reference exactly." << std::endl;
            }
        }
    }

    // -----------------------------------------------------------------
    // 4d. Scalar Arithmetic Diagnostic: verify mul/inv on GPU
    // -----------------------------------------------------------------
    {
        cl_kernel scalar_diag = clCreateKernel(program, "diagnostic_scalar_ops", &err);
        if (err != CL_SUCCESS) {
            std::cerr << "WARNING: scalar diagnostic kernel not found, skipping." << std::endl;
        } else {
            const int SOUT_SIZE = 193;
            cl_mem d_sout = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SOUT_SIZE, nullptr, &err);
            clSetKernelArg(scalar_diag, 0, sizeof(cl_mem), &d_sout);
            size_t one = 1;
            clEnqueueNDRangeKernel(queue, scalar_diag, 1, nullptr, &one, &one, 0, nullptr, nullptr);
            clFinish(queue);
            uint8_t sout[193];
            clEnqueueReadBuffer(queue, d_sout, CL_TRUE, 0, SOUT_SIZE, sout, 0, nullptr, nullptr);

            FILE* sf = fopen("miner_scalar_diag.log", "w");
            if (sf) {
                fprintf(sf, "=== Scalar Arithmetic Diagnostic ===\n\n");

                // Test 1: mul(3,7) = 21
                uint64_t t1[4]; memcpy(t1, sout + 0, 32);
                fprintf(sf, "Test 1: mul(3, 7)\n");
                fprintf(sf, "  Result: [%016llx, %016llx, %016llx, %016llx]\n",
                    (unsigned long long)t1[0], (unsigned long long)t1[1],
                    (unsigned long long)t1[2], (unsigned long long)t1[3]);
                fprintf(sf, "  Expect: [0000000000000015, 0000000000000000, 0000000000000000, 0000000000000000]\n");
                fprintf(sf, "  Status: %s\n\n", (sout[128] & 0x01) ? "PASS" : "FAIL");

                // Test 2: inv(2)
                uint64_t t2[4]; memcpy(t2, sout + 32, 32);
                fprintf(sf, "Test 2: inv(2)\n");
                fprintf(sf, "  Result: [%016llx, %016llx, %016llx, %016llx]\n",
                    (unsigned long long)t2[0], (unsigned long long)t2[1],
                    (unsigned long long)t2[2], (unsigned long long)t2[3]);
                fprintf(sf, "  Expect: [dfe92f46681b20a1, 5d576e7357a4501d, ffffffffffffffff, 7fffffffffffffff]\n");
                fprintf(sf, "  Status: %s\n\n", (sout[128] & 0x02) ? "PASS" : "FAIL");

                // Test 3: 2 * inv(2) = 1
                uint64_t t3[4]; memcpy(t3, sout + 64, 32);
                fprintf(sf, "Test 3: mul(2, inv(2))\n");
                fprintf(sf, "  Result: [%016llx, %016llx, %016llx, %016llx]\n",
                    (unsigned long long)t3[0], (unsigned long long)t3[1],
                    (unsigned long long)t3[2], (unsigned long long)t3[3]);
                fprintf(sf, "  Expect: [0000000000000001, 0000000000000000, 0000000000000000, 0000000000000000]\n");
                fprintf(sf, "  Status: %s\n\n", (sout[128] & 0x04) ? "PASS" : "FAIL");

                // Test 4: (n-1)*(n-1) = 1
                uint64_t t4[4]; memcpy(t4, sout + 96, 32);
                fprintf(sf, "Test 4: mul(n-1, n-1)\n");
                fprintf(sf, "  Result: [%016llx, %016llx, %016llx, %016llx]\n",
                    (unsigned long long)t4[0], (unsigned long long)t4[1],
                    (unsigned long long)t4[2], (unsigned long long)t4[3]);
                fprintf(sf, "  Expect: [0000000000000001, 0000000000000000, 0000000000000000, 0000000000000000]\n");
                fprintf(sf, "  Status: %s\n\n", (sout[128] & 0x08) ? "PASS" : "FAIL");

                // Test 5: 2^256 mod n via 8 chained squarings
                uint64_t t5[4]; memcpy(t5, sout + 129, 32);
                fprintf(sf, "Test 5: 2^256 mod n (8 chained squarings of 2)\n");
                fprintf(sf, "  Result: [%016llx, %016llx, %016llx, %016llx]\n",
                    (unsigned long long)t5[0], (unsigned long long)t5[1],
                    (unsigned long long)t5[2], (unsigned long long)t5[3]);
                fprintf(sf, "  Expect: [402da1732fc9bebf, 4551231950b75fc4, 0000000000000001, 0000000000000000]\n");
                fprintf(sf, "  Status: %s\n\n", (sout[128] & 0x10) ? "PASS" : "FAIL");

                // Test 6: (2^128)^2 = 2^256 mod n
                uint64_t t6[4]; memcpy(t6, sout + 161, 32);
                fprintf(sf, "Test 6: (2^128)^2 mod n (single squaring)\n");
                fprintf(sf, "  Result: [%016llx, %016llx, %016llx, %016llx]\n",
                    (unsigned long long)t6[0], (unsigned long long)t6[1],
                    (unsigned long long)t6[2], (unsigned long long)t6[3]);
                fprintf(sf, "  Expect: [402da1732fc9bebf, 4551231950b75fc4, 0000000000000001, 0000000000000000]\n");
                fprintf(sf, "  Status: %s\n\n", (sout[128] & 0x20) ? "PASS" : "FAIL");

                uint8_t flags = sout[128];
                if (flags == 0x3F) {
                    fprintf(sf, "=== ALL SCALAR TESTS PASSED ===\n");
                    std::cout << "Scalar arithmetic diagnostic: ALL PASSED. See miner_scalar_diag.log" << std::endl;
                } else {
                    fprintf(sf, "=== SOME TESTS FAILED (flags=0x%02x) ===\n", flags);
                    std::cerr << "WARNING: Scalar arithmetic tests FAILED (flags=0x"
                              << std::hex << (int)flags << std::dec << "). See miner_scalar_diag.log" << std::endl;
                }
                fclose(sf);
            }
            clReleaseMemObject(d_sout);
            clReleaseKernel(scalar_diag);
        }
    }

    // -----------------------------------------------------------------
    // 5. Set up Windows shared memory (IPC with BTCW node)
    // -----------------------------------------------------------------
    HANDLE hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(SharedData),
        SHM_NAME
    );

    if (hMapFile == NULL) {
        std::cerr << "Could not create file mapping: " << GetLastError() << std::endl;
        return 1;
    }

    SharedData* shared_data = (SharedData*)MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0, 0,
        sizeof(SharedData)
    );

    if (shared_data == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return 1;
    }

    // -----------------------------------------------------------------
    // 6. Host-side data buffers
    // -----------------------------------------------------------------
    uint8_t h_key_data[KEY_SIZE_BYTES] = {};
    uint8_t h_hash_no_sig[HASH_NO_SIG_SIZE_BYTES] = {};

    // -----------------------------------------------------------------
    // 7a. NVML GPU telemetry init (dynamic — non-fatal if unavailable)
    // -----------------------------------------------------------------
    nvml_init((unsigned int)gpu_num);

    // -----------------------------------------------------------------
    // 7b. PDCurses console UI
    // -----------------------------------------------------------------
    initscr();
    noecho();
    curs_set(FALSE);

    int prev_y, prev_x, curr_y, curr_x;
    getmaxyx(stdscr, prev_y, prev_x);
    mvprintw(0, 0, "BTCW.SPACE OpenCL GPU Miner v26.5.4\n");
    refresh();

    // -----------------------------------------------------------------
    // 8. Mining loop
    // -----------------------------------------------------------------
    volatile uint64_t nonce_prev = 1234;
    uint64_t hash_no_sig_low64 = 0;
    int block_transitions = 0;
    int candidates_found = 0;   // count of GPU candidate nonces sent to node
    bool first_sentinel_seen = false;
    bool was_connected = false;
    auto disconnect_start = std::chrono::steady_clock::now();
    bool disconnect_timing = false;
    const int DISCONNECT_SECONDS = 3;
    auto session_start = std::chrono::steady_clock::now();

    // Track previous hash_no_sig to detect block transitions reliably
    // (more reliable than catching sentinel in shared_data->nonce which gets overwritten)
    uint8_t prev_hash_no_sig[HASH_NO_SIG_SIZE_BYTES] = {};
    bool have_initial_hash = false;

    // Mining parameters — auto-tune or use manual override
    // Clamp local work size to device maximum (important for AMD compatibility)
    size_t LOCAL_WORK_SIZE  = (sel.max_wg_size < 256) ? sel.max_wg_size : 256;
    size_t GLOBAL_WORK_SIZE;
    char worksize_info[128];

    if (user_work_size > 0) {
        // Manual override from command line
        GLOBAL_WORK_SIZE = (size_t)user_work_size;
        // Round up to nearest multiple of LOCAL_WORK_SIZE
        if (GLOBAL_WORK_SIZE % LOCAL_WORK_SIZE != 0) {
            GLOBAL_WORK_SIZE = ((GLOBAL_WORK_SIZE / LOCAL_WORK_SIZE) + 1) * LOCAL_WORK_SIZE;
        }
        snprintf(worksize_info, sizeof(worksize_info), "Work size: %zu (manual override)", GLOBAL_WORK_SIZE);
    } else {
        // Auto-tune: saturate all CUs with 32x oversubscription to hide latency
        // (AMD RDNA reports WGPs not CUs, so needs higher multiplier)
        GLOBAL_WORK_SIZE = (size_t)sel.compute_units * LOCAL_WORK_SIZE * 32;
        // Clamp to reasonable bounds
        if (GLOBAL_WORK_SIZE < 65536)   GLOBAL_WORK_SIZE = 65536;    // minimum floor
        if (GLOBAL_WORK_SIZE > 4194304) GLOBAL_WORK_SIZE = 4194304;  // 4M cap
        snprintf(worksize_info, sizeof(worksize_info), "Work size: %zu (auto-tuned for %u CUs)", GLOBAL_WORK_SIZE, sel.compute_units);
    }

    std::cout << worksize_info << std::endl;

    uint64_t nonce_base = 0;
    cl_uint gpu_num_cl = (cl_uint)gpu_num;

    uint32_t throttle = 0;

    while (true) {
        int changeCount = 0;
        const int durationSeconds = 2;
        auto startTime = std::chrono::steady_clock::now();

        while (std::chrono::steady_clock::now() - startTime < std::chrono::seconds(durationSeconds)) {
            if ((throttle % 3) == 0) {
                // Read data from shared memory (written by BTCW node)
                memcpy(h_key_data,    const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[0])),   KEY_SIZE_BYTES);
                memcpy(h_hash_no_sig, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), HASH_NO_SIG_SIZE_BYTES);

                // --- Detect block transitions via hash_no_sig data change ---
                // This is more reliable than catching SENTINEL in shared_data->nonce
                // because the host continuously writes nonces there (overwriting sentinel).
                if (!have_initial_hash) {
                    // First time reading — save as baseline
                    memcpy(prev_hash_no_sig, h_hash_no_sig, HASH_NO_SIG_SIZE_BYTES);
                    have_initial_hash = true;
                    first_sentinel_seen = true;
                    mvprintw(2, 0, "GPU initialized - waiting for block data...                            \n");
                } else if (memcmp(h_hash_no_sig, prev_hash_no_sig, HASH_NO_SIG_SIZE_BYTES) != 0) {
                    // hash_no_sig changed — node sent new block data (block transition)
                    memcpy(prev_hash_no_sig, h_hash_no_sig, HASH_NO_SIG_SIZE_BYTES);
                    block_transitions++;

                    // Reset nonce search for the new block
                    nonce_base = 0;

                    // Write SENTINEL to shared_data->nonce to acknowledge new block
                    shared_data->nonce = SENTINEL_NONCE;
                    nonce_prev = SENTINEL_NONCE;

                    time_t now = time(nullptr);
                    struct tm* lt = localtime(&now);
                    char timebuf[16];
                    strftime(timebuf, sizeof(timebuf), "%H:%M:%S", lt);
                    mvprintw(2, 0, "New block data from node at %s (block #%d this session)                \n", timebuf, block_transitions);
                }

                // Upload to GPU
                CL_CHECK(clEnqueueWriteBuffer(queue, d_key_data,    CL_FALSE, 0, KEY_SIZE_BYTES, h_key_data, 0, nullptr, nullptr));
                CL_CHECK(clEnqueueWriteBuffer(queue, d_hash_no_sig, CL_FALSE, 0, HASH_NO_SIG_SIZE_BYTES, h_hash_no_sig, 0, nullptr, nullptr));
            }
            throttle++;

            // Reset result buffers
            cl_ulong zero_nonce = 0;
            cl_uint  zero_found = 0;
            cl_uint  zero_ctr   = 0;
            CL_CHECK(clEnqueueWriteBuffer(queue, d_result_nonce,  CL_FALSE, 0, sizeof(cl_ulong), &zero_nonce, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueWriteBuffer(queue, d_result_found,  CL_FALSE, 0, sizeof(cl_uint),  &zero_found, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueWriteBuffer(queue, d_hashrate_ctr,  CL_FALSE, 0, sizeof(cl_uint),  &zero_ctr,   0, nullptr, nullptr));

            // Set kernel arguments (matches btcw_mine signature)
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &d_key_data));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &d_hash_no_sig));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &d_result_nonce));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &d_result_found));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &d_hashrate_ctr));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &nonce_base));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint),  &gpu_num_cl));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem),   &d_ecmult_table));

            // Launch kernel
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &GLOBAL_WORK_SIZE, &LOCAL_WORK_SIZE, 0, nullptr, nullptr));

            // Read results back
            cl_ulong result_nonce = 0;
            cl_uint  result_found = 0;
            cl_uint  hashrate_ctr = 0;
            CL_CHECK(clEnqueueReadBuffer(queue, d_result_found, CL_TRUE, 0, sizeof(cl_uint),  &result_found, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueReadBuffer(queue, d_result_nonce, CL_TRUE, 0, sizeof(cl_ulong), &result_nonce, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueReadBuffer(queue, d_hashrate_ctr, CL_TRUE, 0, sizeof(cl_uint),  &hashrate_ctr, 0, nullptr, nullptr));

            // Count completed batches for hashrate
            changeCount += hashrate_ctr;

            // --- Console UI (same as CUDA version) ---
            getmaxyx(stdscr, curr_y, curr_x);
            if (curr_y != prev_y || curr_x != prev_x) {
                clear();
                prev_y = curr_y;
                prev_x = curr_x;
                mvprintw(0, 0, "Bitcoin-PoW OpenCL GPU Miner v26.5.4\n");
                mvprintw(1, 0, "Powered by btcw.space\n");
            }

            // Belt-and-suspenders: also check shared_data->nonce for sentinel
            // BEFORE we overwrite it (node may have written SENTINEL between iterations)
            if (nonce_prev != shared_data->nonce) {
                uint64_t shm_nonce = shared_data->nonce;
                if (shm_nonce == SENTINEL_NONCE) {
                    // Node wrote sentinel — a new block arrived via shared memory signal
                    // (block transition will also be caught by hash_no_sig change above)
                    nonce_prev = SENTINEL_NONCE;
                }
            }

            // Only write to shared_data->nonce when a real solution is found.
            // CRITICAL: Do NOT write nonce_base here — that overwrites the SENTINEL
            // value the node uses to know we're still mining. The CUDA kernel only
            // writes to nonce4host when it finds a valid hash, and so must we.
            if (result_found) {
                shared_data->nonce = result_nonce;
                nonce_prev = result_nonce;
                candidates_found++;

                // Log candidate to file for debugging
                {
                    FILE* logf = fopen("miner_candidates.log", "a");
                    if (logf) {
                        time_t now = time(nullptr);
                        char timebuf[64];
                        strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
                        fprintf(logf, "[%s] CANDIDATE #%d  nonce=%016llx\n",
                                timebuf, candidates_found, (unsigned long long)result_nonce);
                        fflush(logf);
                        fclose(logf);
                    }
                }
            }

            // Display current mining nonce and candidate count
            mvprintw(2, 0, "Mining - NONCE: %016llx  Candidates sent: %d                      \n",
                     nonce_prev, candidates_found);

            // Advance nonce base for next batch (4 nonces per thread)
            nonce_base += GLOBAL_WORK_SIZE * 4;

            // Connection status
            memcpy(&hash_no_sig_low64, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), 8);
            mvprintw(4, 0, "Hash no sig low64: %016llx\n", hash_no_sig_low64);

            if (hash_no_sig_low64 == 0) {
                if (!disconnect_timing) {
                    disconnect_start = std::chrono::steady_clock::now();
                    disconnect_timing = true;
                }
                auto elapsed = std::chrono::steady_clock::now() - disconnect_start;
                if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= DISCONNECT_SECONDS) {
                    mvprintw(6, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");
                    mvprintw(7, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");
                    mvprintw(8, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");

                    // Try to re-open shared memory
                    shared_data = (SharedData*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedData));
                    if (shared_data == NULL) {
                        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
                    }
                    Sleep(1000);
                } else if (was_connected) {
                    mvprintw(6, 0, "Waiting for new block data...                                                              \n");
                    mvprintw(7, 0, "                                                                                            \n");
                    mvprintw(8, 0, "                                                                                            \n");
                }
            } else {
                disconnect_timing = false;
                was_connected = true;
                mvprintw(6, 0, "CONNECTED TO BTCW NODE WALLET                                                              \n");
                mvprintw(7, 0, "                                                                                            \n");
                mvprintw(8, 0, "                                                                                            \n");
            }
        }

        // Calculate hashrate
        double rate = static_cast<double>(changeCount) / durationSeconds;
        if (hash_no_sig_low64 == 0) rate = 0;

        // Calculate uptime
        auto uptime = std::chrono::steady_clock::now() - session_start;
        auto uptime_sec = std::chrono::duration_cast<std::chrono::seconds>(uptime).count();
        int up_h = (int)(uptime_sec / 3600);
        int up_m = (int)((uptime_sec % 3600) / 60);
        int up_s = (int)(uptime_sec % 60);

        // Query GPU telemetry via NVML
        GpuStats gpu = nvml_query();

        // Status bar
        mvprintw(curr_y - 10, 0, "=======================================================\n");
        mvprintw(curr_y - 9, 0, "Device: %s\n", sel.name.c_str());
        mvprintw(curr_y - 8, 0, "%s\n", worksize_info);
        mvprintw(curr_y - 7, 0, "-------------------------------------------------------\n");
        double rate_mh = rate / 1000000.0;
        mvprintw(curr_y - 6, 0, "Hashrate: %.0f H/s  (%.2f MH/s)\n", rate, rate_mh);
        if (gpu.available) {
            char gpu_line[128];
            int pos = snprintf(gpu_line, sizeof(gpu_line), "GPU: %.1f W", gpu.power_w);
            if (gpu.has_temp)
                pos += snprintf(gpu_line + pos, sizeof(gpu_line) - pos, "  |  %d C", gpu.temp_c);
            if (gpu.has_fan)
                pos += snprintf(gpu_line + pos, sizeof(gpu_line) - pos, "  |  Fan %d%%", gpu.fan_pct);
            mvprintw(curr_y - 5, 0, "%s\n", gpu_line);
        } else {
            mvprintw(curr_y - 5, 0, "GPU: N/A (NVML not available)                          \n");
        }
        mvprintw(curr_y - 4, 0, "Block transitions: %d\n", block_transitions);
        mvprintw(curr_y - 3, 0, "Uptime: %02d:%02d:%02d\n", up_h, up_m, up_s);
        mvprintw(curr_y - 2, 0, "=======================================================\n");
        refresh();
    }

    // Cleanup (unreachable in normal operation, but good practice)
    clReleaseMemObject(d_ecmult_table);
    clReleaseMemObject(d_key_data);
    clReleaseMemObject(d_hash_no_sig);
    clReleaseMemObject(d_result_nonce);
    clReleaseMemObject(d_result_found);
    clReleaseMemObject(d_hashrate_ctr);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    UnmapViewOfFile(shared_data);
    CloseHandle(hMapFile);

    nvml_shutdown();
    endwin();
    return 0;
}
