// =============================================================================
// BTCW Metal GPU Miner - macOS Host Code
// =============================================================================
// Copyright (c) 2026 btcw.space <btcw.space@proton.me>
//
// macOS Metal port of the BTCW OpenCL miner. Designed for Apple Silicon Macs
// with unified memory (StorageModeShared = zero-copy GPU access).
// Uses POSIX shared memory and plain stdout for headless SSH operation.
//
// Usage:  ./btcw_metal_miner [gpu_number] [work_size]
//   gpu_number: informational only (Mac has 1 GPU), passed to kernel for nonce partitioning
//   work_size:  manual global work size override (default: auto-tune)
//
// Copyright (c) 2026 btcw.space. All rights reserved.
// =============================================================================

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <unistd.h>
#include <cerrno>
#include <thread>

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
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    int gpu_num = 0;
    int user_work_size = 0;
    if (argc >= 2) {
        gpu_num = atoi(argv[1]);
    }
    if (argc >= 3) {
        user_work_size = atoi(argv[2]);
    }

    @autoreleasepool {

    // -----------------------------------------------------------------
    // 1. Get Metal device
    // -----------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "No Metal-capable GPU found on this Mac.\n");
        return 1;
    }

    print_timestamp();
    printf("BTCW.SPACE Metal GPU Miner v26.5.4\n");
    print_timestamp();
    printf("Metal device: %s\n", [[device name] UTF8String]);
    print_timestamp();
    printf("Recommended max working set: %luMB\n",
           (unsigned long)([device recommendedMaxWorkingSetSize] / (1024 * 1024)));

    // -----------------------------------------------------------------
    // 2. Create command queue
    // -----------------------------------------------------------------
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        fprintf(stderr, "Failed to create Metal command queue.\n");
        return 1;
    }

    // -----------------------------------------------------------------
    // 3. Load pre-compiled Metal shader library
    // -----------------------------------------------------------------
    NSError* err = nil;
    NSURL* libURL = [NSURL fileURLWithPath:@"mining.metallib"];
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&err];
    if (!library) {
        fprintf(stderr, "Failed to load mining.metallib: %s\n",
                [[err localizedDescription] UTF8String]);
        fprintf(stderr, "Make sure mining.metallib is in the current directory.\n");
        return 1;
    }
    print_timestamp();
    printf("Metal shader library loaded (instant start — no runtime compilation).\n");

    // -----------------------------------------------------------------
    // 4. Create compute pipeline states
    // -----------------------------------------------------------------
    id<MTLFunction> mineFunc = [library newFunctionWithName:@"btcw_mine"];
    if (!mineFunc) {
        fprintf(stderr, "Kernel function 'btcw_mine' not found in metallib.\n");
        return 1;
    }
    id<MTLComputePipelineState> minePipeline =
        [device newComputePipelineStateWithFunction:mineFunc error:&err];
    if (!minePipeline) {
        fprintf(stderr, "Failed to create mine pipeline: %s\n",
                [[err localizedDescription] UTF8String]);
        return 1;
    }

    id<MTLFunction> precompFunc = [library newFunctionWithName:@"precompute_ecmult_gen_table"];
    if (!precompFunc) {
        fprintf(stderr, "Kernel function 'precompute_ecmult_gen_table' not found in metallib.\n");
        return 1;
    }
    id<MTLComputePipelineState> precompPipeline =
        [device newComputePipelineStateWithFunction:precompFunc error:&err];
    if (!precompPipeline) {
        fprintf(stderr, "Failed to create precompute pipeline: %s\n",
                [[err localizedDescription] UTF8String]);
        return 1;
    }

    // -----------------------------------------------------------------
    // 5. Allocate GPU buffers (StorageModeShared = zero-copy on Apple Silicon)
    // -----------------------------------------------------------------
    id<MTLBuffer> keyBuf         = [device newBufferWithLength:KEY_SIZE_BYTES          options:MTLResourceStorageModeShared];
    id<MTLBuffer> hashBuf        = [device newBufferWithLength:HASH_NO_SIG_SIZE_BYTES  options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultNonceBuf = [device newBufferWithLength:sizeof(uint64_t)         options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultFoundBuf = [device newBufferWithLength:sizeof(uint32_t)         options:MTLResourceStorageModeShared];
    id<MTLBuffer> hashrateBuf    = [device newBufferWithLength:sizeof(uint32_t)         options:MTLResourceStorageModeShared];
    id<MTLBuffer> nonceBaseBuf   = [device newBufferWithLength:sizeof(uint64_t)         options:MTLResourceStorageModeShared];
    id<MTLBuffer> gpuNumBuf      = [device newBufferWithLength:sizeof(uint32_t)         options:MTLResourceStorageModeShared];
    id<MTLBuffer> ecmultTableBuf = [device newBufferWithLength:4096 * sizeof(uint64_t)  options:MTLResourceStorageModeShared];

    if (!keyBuf || !hashBuf || !resultNonceBuf || !resultFoundBuf ||
        !hashrateBuf || !nonceBaseBuf || !gpuNumBuf || !ecmultTableBuf) {
        fprintf(stderr, "Failed to allocate Metal buffers.\n");
        return 1;
    }

    *(uint32_t*)[gpuNumBuf contents] = (uint32_t)gpu_num;

    // -----------------------------------------------------------------
    // 5b. Precompute ecmult generator table on GPU (runs once at startup)
    // -----------------------------------------------------------------
    print_timestamp();
    printf("Precomputing ecmult generator table...\n");
    {
        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:precompPipeline];
        [encoder setBuffer:ecmultTableBuf offset:0 atIndex:0];
        MTLSize gridSize = MTLSizeMake(1, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "Precompute kernel failed: %s\n",
                    [[[cmdBuf error] localizedDescription] UTF8String]);
            return 1;
        }
    }
    print_timestamp();
    printf("Ecmult table ready (32KB).\n");

    // -----------------------------------------------------------------
    // 6. Set up POSIX shared memory (IPC with BTCW node)
    // -----------------------------------------------------------------
    int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            fprintf(stderr, "Could not open/create shared memory '%s': %s\n",
                    SHM_NAME, strerror(errno));
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
    // 7. Mining state (mirrors OpenCL version exactly)
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

    uint8_t prev_hash_no_sig[HASH_NO_SIG_SIZE_BYTES] = {};
    bool have_initial_hash = false;

    // -----------------------------------------------------------------
    // 8. Auto-tune work size
    // -----------------------------------------------------------------
    NSUInteger maxThreadsPerGroup = minePipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger LOCAL_WORK_SIZE = (maxThreadsPerGroup < 256) ? maxThreadsPerGroup : 256;
    NSUInteger GLOBAL_WORK_SIZE;
    char worksize_info[128];

    if (user_work_size > 0) {
        GLOBAL_WORK_SIZE = (NSUInteger)user_work_size;
        if (GLOBAL_WORK_SIZE % LOCAL_WORK_SIZE != 0) {
            GLOBAL_WORK_SIZE = ((GLOBAL_WORK_SIZE / LOCAL_WORK_SIZE) + 1) * LOCAL_WORK_SIZE;
        }
        snprintf(worksize_info, sizeof(worksize_info),
                 "Work size: %lu (manual override)", (unsigned long)GLOBAL_WORK_SIZE);
    } else {
        GLOBAL_WORK_SIZE = LOCAL_WORK_SIZE * 32 * 128;
        if (GLOBAL_WORK_SIZE < 65536)   GLOBAL_WORK_SIZE = 65536;
        if (GLOBAL_WORK_SIZE > 4194304) GLOBAL_WORK_SIZE = 4194304;
        snprintf(worksize_info, sizeof(worksize_info),
                 "Work size: %lu (auto-tuned for Metal)", (unsigned long)GLOBAL_WORK_SIZE);
    }

    print_timestamp();
    printf("Max threads per threadgroup: %lu\n", (unsigned long)maxThreadsPerGroup);
    print_timestamp();
    printf("%s\n", worksize_info);

    uint64_t nonce_base = 0;
    uint32_t throttle = 0;
    bool connection_status_printed = false;

    // -----------------------------------------------------------------
    // 9. Mining loop
    // -----------------------------------------------------------------
    while (g_running.load()) {
        int changeCount = 0;
        const int durationSeconds = 2;
        auto startTime = std::chrono::steady_clock::now();

        while (g_running.load() &&
               std::chrono::steady_clock::now() - startTime < std::chrono::seconds(durationSeconds)) {

            if ((throttle % 3) == 0) {
                // Read key data from shared memory directly into Metal buffer (zero-copy)
                memcpy([keyBuf contents],
                       const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[0])),
                       KEY_SIZE_BYTES);

                // Read hash_no_sig into temp buffer for block transition comparison
                uint8_t h_hash_no_sig[HASH_NO_SIG_SIZE_BYTES];
                memcpy(h_hash_no_sig,
                       const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])),
                       HASH_NO_SIG_SIZE_BYTES);
                memcpy([hashBuf contents], h_hash_no_sig, HASH_NO_SIG_SIZE_BYTES);

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
            }
            throttle++;

            // Zero result buffers (zero-copy write)
            memset([resultNonceBuf contents], 0, sizeof(uint64_t));
            memset([resultFoundBuf contents], 0, sizeof(uint32_t));
            memset([hashrateBuf contents],    0, sizeof(uint32_t));

            // Set nonce base for this batch
            *(uint64_t*)[nonceBaseBuf contents] = nonce_base;

            // Dispatch mining kernel
            @autoreleasepool {
                id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                [encoder setComputePipelineState:minePipeline];
                [encoder setBuffer:keyBuf         offset:0 atIndex:0];
                [encoder setBuffer:hashBuf        offset:0 atIndex:1];
                [encoder setBuffer:resultNonceBuf offset:0 atIndex:2];
                [encoder setBuffer:resultFoundBuf offset:0 atIndex:3];
                [encoder setBuffer:hashrateBuf    offset:0 atIndex:4];
                [encoder setBuffer:nonceBaseBuf   offset:0 atIndex:5];
                [encoder setBuffer:gpuNumBuf      offset:0 atIndex:6];
                [encoder setBuffer:ecmultTableBuf offset:0 atIndex:7];

                MTLSize gridSize = MTLSizeMake(GLOBAL_WORK_SIZE, 1, 1);
                MTLSize threadgroupSize = MTLSizeMake(LOCAL_WORK_SIZE, 1, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];
            }

            // Read results (zero-copy — just dereference buffer contents pointer)
            uint32_t result_found = *(uint32_t*)[resultFoundBuf contents];
            uint64_t result_nonce = *(uint64_t*)[resultNonceBuf contents];
            uint32_t hashrate_ctr = *(uint32_t*)[hashrateBuf contents];

            changeCount += hashrate_ctr;

            // Check shared_data->nonce for sentinel BEFORE we overwrite it
            if (nonce_prev != shared_data->nonce) {
                uint64_t shm_nonce = shared_data->nonce;
                if (shm_nonce == SENTINEL_NONCE) {
                    nonce_prev = SENTINEL_NONCE;
                }
            }

            // Only write to shared_data->nonce when a real solution is found
            if (result_found) {
                shared_data->nonce = result_nonce;
                nonce_prev = result_nonce;
            }

            // Advance nonce base for next batch (4 nonces per thread)
            nonce_base += GLOBAL_WORK_SIZE * 4;

            // Connection status (only print on change to avoid log spam)
            memcpy(&hash_no_sig_low64,
                   const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), 8);

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
                    // Re-mmap shared memory in case node restarted
                    SharedData* new_data = (SharedData*)mmap(nullptr, sizeof(SharedData),
                        PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
                    if (new_data != MAP_FAILED) {
                        if (shared_data != MAP_FAILED) {
                            munmap((void*)shared_data, sizeof(SharedData));
                        }
                        shared_data = new_data;
                    }
                    std::this_thread::sleep_for(std::chrono::seconds(1));
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

    munmap((void*)shared_data, sizeof(SharedData));
    close(shm_fd);

    print_timestamp();
    printf("Goodbye.\n");

    } // @autoreleasepool
    return 0;
}
