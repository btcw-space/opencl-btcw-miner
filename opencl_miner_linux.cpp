// =============================================================================
// BTCW OpenCL GPU Miner - Linux Headless Host Code
// =============================================================================
// Copyright (c) 2026 0x369d <0x369d@gmail.com>
//
// Linux port of the BTCW OpenCL miner. Compatible with NVIDIA and AMD GPUs.
// Uses POSIX shared memory and plain stdout for headless SSH operation.
//
// Usage:  ./btcw_opencl_miner [gpu_number] [work_size]
//   gpu_number: 1-based GPU index (default: first GPU)
//   work_size:  manual global work size override (default: auto-tune)
//
// Copyright (c) 2026 0x369d. All rights reserved.
// =============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <thread>
#include <atomic>

// POSIX shared memory
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>

// OpenCL
#include <CL/cl.h>

// =============================================================================
// Signal handling for clean shutdown
// =============================================================================

static std::atomic<bool> g_running{true};

static void signal_handler(int sig) {
    (void)sig;
    g_running.store(false);
}

// =============================================================================
// Timestamp helper
// =============================================================================

static void print_timestamp() {
    time_t now = time(nullptr);
    struct tm* lt = localtime(&now);
    char buf[16];
    strftime(buf, sizeof(buf), "%H:%M:%S", lt);
    printf("[%s] ", buf);
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
        fprintf(stderr, "OpenCL error %s at %s:%d\n", cl_err_str(_err), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

// Read a text file into a string
static std::string read_file_to_string(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open file: %s\n", path.c_str());
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
    // Install signal handlers for clean shutdown
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

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
        fprintf(stderr, "No OpenCL platforms found.\n");
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    print_timestamp();
    printf("BTCW.SPACE OpenCL GPU Miner v26.5.4\n");
    print_timestamp();
    printf("Found %u OpenCL platform(s):\n", num_platforms);

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
        printf("  Platform %u: %s\n", pi, plat_name);

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

            printf("    Device %zu: %s (%s) CUs=%u Mem=%luMB MaxWG=%zu\n",
                   all_devices.size(), name, vendor,
                   e.compute_units, (unsigned long)(e.global_mem / (1024 * 1024)), e.max_wg_size);

            all_devices.push_back(e);
        }
    }

    if (all_devices.empty()) {
        fprintf(stderr, "No OpenCL GPU devices found.\n");
        return 1;
    }

    // Select device (gpu_num is 1-based from user, 0 means first)
    int dev_idx = (gpu_num > 0) ? (gpu_num - 1) : 0;
    if (dev_idx >= (int)all_devices.size()) {
        fprintf(stderr, "GPU %d not found. Only %zu GPUs available.\n", gpu_num, all_devices.size());
        return 1;
    }

    DeviceEntry& sel = all_devices[dev_idx];
    print_timestamp();
    printf("Using device %d: %s (%u CUs, %luMB)\n",
           dev_idx, sel.name.c_str(), sel.compute_units,
           (unsigned long)(sel.global_mem / (1024 * 1024)));

    // -----------------------------------------------------------------
    // 2. Create OpenCL context, command queue
    // -----------------------------------------------------------------
    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &sel.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL context: %s\n", cl_err_str(err));
        return 1;
    }

    cl_command_queue queue = clCreateCommandQueue(context, sel.device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue: %s\n", cl_err_str(err));
        return 1;
    }

    // -----------------------------------------------------------------
    // 3. Load and compile OpenCL kernel (with binary caching)
    // -----------------------------------------------------------------
    std::string field_src = read_file_to_string("kernels/secp256k1_field.cl");
    std::string point_src = read_file_to_string("kernels/secp256k1_point.cl");
    std::string mine_src  = read_file_to_string("kernels/mining_kernel.cl");

    if (field_src.empty() || point_src.empty() || mine_src.empty()) {
        fprintf(stderr, "Failed to load kernel source files.\n");
        fprintf(stderr, "Make sure the 'kernels/' directory contains:\n");
        fprintf(stderr, "  secp256k1_field.cl\n");
        fprintf(stderr, "  secp256k1_point.cl\n");
        fprintf(stderr, "  mining_kernel.cl\n");
        return 1;
    }

    // Concatenate in order: field -> point -> mining kernel
    std::string combined_src = field_src + "\n" + point_src + "\n" + mine_src;

    // --- Kernel binary caching ---
    // Hash the source + device name to create a cache key. If a cached binary
    // exists and matches, load it directly to skip the expensive compile step.
    const std::string cache_dir  = "kernel_cache";
    const std::string cache_file = cache_dir + "/btcw_mine_" + sel.name + ".bin";

    // Simple source hash: sum of all bytes (fast, sufficient for invalidation)
    auto source_hash = [&]() -> uint64_t {
        uint64_t h = 0;
        for (unsigned char c : combined_src) h = h * 131 + c;
        return h;
    }();

    cl_program program = nullptr;
    bool loaded_from_cache = false;

    // Try loading cached binary
    {
        std::ifstream cf(cache_file, std::ios::binary);
        if (cf.good()) {
            // Read stored hash
            uint64_t stored_hash = 0;
            cf.read(reinterpret_cast<char*>(&stored_hash), sizeof(stored_hash));
            if (cf.good() && stored_hash == source_hash) {
                // Read binary
                std::string bin_data((std::istreambuf_iterator<char>(cf)),
                                      std::istreambuf_iterator<char>());
                if (!bin_data.empty()) {
                    const unsigned char* bin_ptr = reinterpret_cast<const unsigned char*>(bin_data.data());
                    size_t bin_len = bin_data.size();
                    cl_int bin_status = CL_SUCCESS;
                    program = clCreateProgramWithBinary(context, 1, &sel.device,
                                                       &bin_len, &bin_ptr, &bin_status, &err);
                    if (err == CL_SUCCESS && bin_status == CL_SUCCESS) {
                        err = clBuildProgram(program, 1, &sel.device, nullptr, nullptr, nullptr);
                        if (err == CL_SUCCESS) {
                            loaded_from_cache = true;
                            print_timestamp();
                            printf("Loaded cached kernel binary (instant start).\n");
                        } else {
                            clReleaseProgram(program);
                            program = nullptr;
                        }
                    } else {
                        if (program) { clReleaseProgram(program); program = nullptr; }
                    }
                }
            }
        }
    }

    // Fall back to source compilation if cache miss/invalid
    if (!loaded_from_cache) {
        const char* src_ptr = combined_src.c_str();
        size_t src_len = combined_src.length();

        program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create program: %s\n", cl_err_str(err));
            return 1;
        }

        // Build options: use standard CL1.2 only (no aggressive math flags for crypto correctness)
        const char* build_opts = "-cl-std=CL1.2";

        print_timestamp();
        printf("Compiling OpenCL kernel (first run – may take 30-60s)...\n");

        err = clBuildProgram(program, 1, &sel.device, build_opts, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, sel.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(program, sel.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "OpenCL build failed:\n%s\n", log.data());
            return 1;
        }

        print_timestamp();
        printf("OpenCL kernel compiled successfully.\n");

        // Save binary to cache for next run
        size_t bin_size = 0;
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr);
        if (bin_size > 0) {
            std::vector<unsigned char> binary(bin_size);
            unsigned char* bin_ptr = binary.data();
            clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(bin_ptr), &bin_ptr, nullptr);
            // Create cache directory (ignore error if exists)
            mkdir(cache_dir.c_str(), 0755);
            std::ofstream cf(cache_file, std::ios::binary | std::ios::trunc);
            if (cf.good()) {
                cf.write(reinterpret_cast<const char*>(&source_hash), sizeof(source_hash));
                cf.write(reinterpret_cast<const char*>(binary.data()), bin_size);
                print_timestamp();
                printf("Cached kernel binary for instant startup next time.\n");
            }
        }
    }

    cl_kernel kernel = clCreateKernel(program, "btcw_mine", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel 'btcw_mine': %s\n", cl_err_str(err));
        return 1;
    }

    cl_kernel precomp_kernel = clCreateKernel(program, "precompute_ecmult_gen_table", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel 'precompute_ecmult_gen_table': %s\n", cl_err_str(err));
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
        fprintf(stderr, "Failed to allocate GPU buffers.\n");
        return 1;
    }

    // -----------------------------------------------------------------
    // 4b. Precompute ecmult generator table on GPU (runs once at startup)
    // -----------------------------------------------------------------
    print_timestamp();
    printf("Precomputing ecmult generator table...\n");
    CL_CHECK(clSetKernelArg(precomp_kernel, 0, sizeof(cl_mem), &d_ecmult_table));
    size_t precomp_global = 1, precomp_local = 1;
    CL_CHECK(clEnqueueNDRangeKernel(queue, precomp_kernel, 1, nullptr, &precomp_global, &precomp_local, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    print_timestamp();
    printf("Ecmult table ready (32KB).\n");
    clReleaseKernel(precomp_kernel);

    // -----------------------------------------------------------------
    // 5. Set up POSIX shared memory (IPC with BTCW node)
    // -----------------------------------------------------------------
    int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        // Try creating it (miner may start before node)
        shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            fprintf(stderr, "Could not open/create shared memory '%s': %s\n", SHM_NAME, strerror(errno));
            return 1;
        }
        if (ftruncate(shm_fd, sizeof(SharedData)) == -1) {
            fprintf(stderr, "Could not set shared memory size: %s\n", strerror(errno));
            close(shm_fd);
            return 1;
        }
    }

    SharedData* shared_data = (SharedData*)mmap(nullptr, sizeof(SharedData),
        PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    if (shared_data == MAP_FAILED) {
        fprintf(stderr, "Could not mmap shared memory: %s\n", strerror(errno));
        close(shm_fd);
        return 1;
    }

    print_timestamp();
    printf("Shared memory '%s' mapped successfully.\n", SHM_NAME);

    // -----------------------------------------------------------------
    // 6. Host-side data buffers
    // -----------------------------------------------------------------
    uint8_t h_key_data[KEY_SIZE_BYTES] = {};
    uint8_t h_hash_no_sig[HASH_NO_SIG_SIZE_BYTES] = {};

    // -----------------------------------------------------------------
    // 7. Mining state
    // -----------------------------------------------------------------
    volatile uint64_t nonce_prev = 1234;
    uint64_t hash_no_sig_low64 = 0;
    int block_transitions = 0;
    bool first_sentinel_seen = false;
    bool was_connected = false;
    auto disconnect_start = std::chrono::steady_clock::now();
    bool disconnect_timing = false;
    const int DISCONNECT_SECONDS = 3;
    auto session_start = std::chrono::steady_clock::now();

    // Track previous hash_no_sig to detect block transitions reliably
    uint8_t prev_hash_no_sig[HASH_NO_SIG_SIZE_BYTES] = {};
    bool have_initial_hash = false;

    // Mining parameters — auto-tune or use manual override
    // Clamp local work size to device maximum (important for AMD compatibility)
    size_t LOCAL_WORK_SIZE  = (sel.max_wg_size < 256) ? sel.max_wg_size : 256;
    size_t GLOBAL_WORK_SIZE;
    char worksize_info[128];

    if (user_work_size > 0) {
        GLOBAL_WORK_SIZE = (size_t)user_work_size;
        if (GLOBAL_WORK_SIZE % LOCAL_WORK_SIZE != 0) {
            GLOBAL_WORK_SIZE = ((GLOBAL_WORK_SIZE / LOCAL_WORK_SIZE) + 1) * LOCAL_WORK_SIZE;
        }
        snprintf(worksize_info, sizeof(worksize_info), "Work size: %zu (manual override)", GLOBAL_WORK_SIZE);
    } else {
        GLOBAL_WORK_SIZE = (size_t)sel.compute_units * LOCAL_WORK_SIZE * 8;
        if (GLOBAL_WORK_SIZE < 32768)   GLOBAL_WORK_SIZE = 32768;
        if (GLOBAL_WORK_SIZE > 2097152) GLOBAL_WORK_SIZE = 2097152;
        snprintf(worksize_info, sizeof(worksize_info), "Work size: %zu (auto-tuned for %u CUs)", GLOBAL_WORK_SIZE, sel.compute_units);
    }

    print_timestamp();
    printf("%s\n", worksize_info);

    uint64_t nonce_base = 0;
    cl_uint gpu_num_cl = (cl_uint)gpu_num;

    uint32_t throttle = 0;
    bool connection_status_printed = false;

    // -----------------------------------------------------------------
    // 8. Mining loop
    // -----------------------------------------------------------------
    while (g_running.load()) {
        int changeCount = 0;
        const int durationSeconds = 2;
        auto startTime = std::chrono::steady_clock::now();

        while (g_running.load() &&
               std::chrono::steady_clock::now() - startTime < std::chrono::seconds(durationSeconds)) {

            if ((throttle % 3) == 0) {
                // Read data from shared memory (written by BTCW node)
                memcpy(h_key_data,    const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[0])),   KEY_SIZE_BYTES);
                memcpy(h_hash_no_sig, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), HASH_NO_SIG_SIZE_BYTES);

                // --- Detect block transitions via hash_no_sig data change ---
                if (!have_initial_hash) {
                    memcpy(prev_hash_no_sig, h_hash_no_sig, HASH_NO_SIG_SIZE_BYTES);
                    have_initial_hash = true;
                    first_sentinel_seen = true;
                    print_timestamp();
                    printf("GPU initialized - waiting for block data...\n");
                } else if (memcmp(h_hash_no_sig, prev_hash_no_sig, HASH_NO_SIG_SIZE_BYTES) != 0) {
                    memcpy(prev_hash_no_sig, h_hash_no_sig, HASH_NO_SIG_SIZE_BYTES);
                    block_transitions++;

                    nonce_base = 0;

                    shared_data->nonce = SENTINEL_NONCE;
                    nonce_prev = SENTINEL_NONCE;

                    print_timestamp();
                    printf("New block data from node (block #%d this session)\n", block_transitions);
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

            changeCount += hashrate_ctr;

            // Check shared_data->nonce for sentinel BEFORE we overwrite it
            if (nonce_prev != shared_data->nonce) {
                uint64_t shm_nonce = shared_data->nonce;
                if (shm_nonce == SENTINEL_NONCE) {
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
            }

            // Advance nonce base for next batch (4 nonces per thread)
            nonce_base += GLOBAL_WORK_SIZE * 4;

            // Connection status (only print on change to avoid log spam)
            memcpy(&hash_no_sig_low64, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), 8);

            if (hash_no_sig_low64 == 0) {
                if (!disconnect_timing) {
                    disconnect_start = std::chrono::steady_clock::now();
                    disconnect_timing = true;
                }
                auto elapsed = std::chrono::steady_clock::now() - disconnect_start;
                if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= DISCONNECT_SECONDS) {
                    if (connection_status_printed != false || !was_connected) {
                        print_timestamp();
                        printf("!!! NOT CONNECTED TO BTCW NODE WALLET !!! Make sure your wallet has at least 1 utxo.\n");
                        connection_status_printed = false;
                    }
                    // Try to re-mmap shared memory
                    SharedData* new_data = (SharedData*)mmap(nullptr, sizeof(SharedData),
                        PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
                    if (new_data != MAP_FAILED) {
                        if (shared_data != MAP_FAILED) {
                            munmap((void*)shared_data, sizeof(SharedData));
                        }
                        shared_data = new_data;
                    }
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                } else if (was_connected) {
                    // Brief disconnection during block transition — don't spam
                }
            } else {
                if (!was_connected || !connection_status_printed) {
                    print_timestamp();
                    printf("Connected to BTCW node wallet\n");
                    connection_status_printed = true;
                }
                disconnect_timing = false;
                was_connected = true;
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

        double rate_mh = rate / 1000000.0;

        // Print status line
        print_timestamp();
        printf("Mining | %.2f MH/s | Nonce: %016llx | Blocks: %d | Up: %02d:%02d:%02d\n",
               rate_mh, (unsigned long long)nonce_prev, block_transitions, up_h, up_m, up_s);
        fflush(stdout);
    }

    // -----------------------------------------------------------------
    // Clean shutdown
    // -----------------------------------------------------------------
    print_timestamp();
    printf("Shutting down...\n");

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

    munmap((void*)shared_data, sizeof(SharedData));
    close(shm_fd);

    print_timestamp();
    printf("Goodbye.\n");
    return 0;
}
