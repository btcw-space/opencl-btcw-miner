// =============================================================================
// BTCW OpenCL Mining Kernel
// =============================================================================
// Copyright (c) 2026 btcw.space <btcw.space@proton.me>
//
// Implements the Bitcoin-PoW Stage 2 mining algorithm:
//   1. Pick nonce from upper 64-bit space (GPU partitioned)
//   2. mud = hash_no_sig + nonce  (uint256 LE addition)
//   3. ECDSA sign(seckey, mud_LE_bytes): RFC 6979 nonce k, R = k*G, DER encode
//   4. preimage = nonce(8 LE) || CompactSize(sig_len)(1) || DER_sig(N)
//   5. hashPoW = double-SHA256(preimage)
//   6. Check trailing bytes [28-31] for difficulty (reversed byte order)
//
// Copyright (c) 2026 btcw.space. All rights reserved.
// =============================================================================

// =============================================================================
// secp256k1 OpenCL primitives (field + point operations)
// These files are concatenated during kernel build.
// =============================================================================

// --- secp256k1_field.cl inlined (field arithmetic) ---
// We rely on the build system to prepend secp256k1_field.cl and secp256k1_point.cl
// via clCreateProgramWithSource with multiple source strings.
// The field and point types/functions are thus available here.

// =============================================================================
// SHA-256 Implementation (for GPU)
// =============================================================================

typedef struct {
    uint state[8];
    uchar buf[64];
    uint bytes;
} SHA256_CTX;

__constant uint SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define SHA_Ch(x,y,z)    ((z) ^ ((x) & ((y) ^ (z))))
#define SHA_Maj(x,y,z)   (((x) & (y)) | ((z) & ((x) | (y))))
#define SHA_Sigma0(x)    (rotate((x), 30U) ^ rotate((x), 19U) ^ rotate((x), 10U))
#define SHA_Sigma1(x)    (rotate((x), 26U) ^ rotate((x), 21U) ^ rotate((x), 7U))
#define SHA_sigma0(x)    (rotate((x), 25U) ^ rotate((x), 14U) ^ ((x) >> 3))
#define SHA_sigma1(x)    (rotate((x), 15U) ^ rotate((x), 13U) ^ ((x) >> 10))

static inline uint read_be32(const uchar* p) {
    return ((uint)p[0] << 24) | ((uint)p[1] << 16) | ((uint)p[2] << 8) | (uint)p[3];
}

static inline void write_be32(uchar* p, uint v) {
    p[0] = (uchar)(v >> 24);
    p[1] = (uchar)(v >> 16);
    p[2] = (uchar)(v >> 8);
    p[3] = (uchar)(v);
}

static inline void sha256_transform(uint* s, const uchar* buf) {
    uint a = s[0], b = s[1], c = s[2], d = s[3];
    uint e = s[4], f = s[5], g = s[6], h = s[7];
    uint w[64];

    #pragma unroll
    for (int i = 0; i < 16; i++)
        w[i] = read_be32(&buf[i * 4]);

    #pragma unroll
    for (int i = 16; i < 64; i++)
        w[i] = SHA_sigma1(w[i-2]) + w[i-7] + SHA_sigma0(w[i-15]) + w[i-16];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint t1 = h + SHA_Sigma1(e) + SHA_Ch(e, f, g) + SHA256_K[i] + w[i];
        uint t2 = SHA_Sigma0(a) + SHA_Maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s[0] += a; s[1] += b; s[2] += c; s[3] += d;
    s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

static inline void sha256_init(SHA256_CTX* ctx) {
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    ctx->bytes = 0;
}

static inline void sha256_update(SHA256_CTX* ctx, const uchar* data, uint len) {
    uint bufsize = ctx->bytes & 0x3F;
    ctx->bytes += len;
    while (len >= 64 - bufsize) {
        uint chunk = 64 - bufsize;
        for (uint i = 0; i < chunk; i++)
            ctx->buf[bufsize + i] = data[i];
        data += chunk;
        len -= chunk;
        sha256_transform(ctx->state, ctx->buf);
        bufsize = 0;
    }
    for (uint i = 0; i < len; i++)
        ctx->buf[bufsize + i] = data[i];
}

static inline void sha256_final(SHA256_CTX* ctx, uchar* out32) {
    uint bufsize = ctx->bytes & 0x3F;
    // Pad
    ctx->buf[bufsize++] = 0x80;
    if (bufsize > 56) {
        for (uint i = bufsize; i < 64; i++) ctx->buf[i] = 0;
        sha256_transform(ctx->state, ctx->buf);
        bufsize = 0;
    }
    for (uint i = bufsize; i < 56; i++) ctx->buf[i] = 0;
    // Length in bits (big-endian)
    ulong bitlen = (ulong)ctx->bytes * 8;
    write_be32(&ctx->buf[56], (uint)(bitlen >> 32));
    write_be32(&ctx->buf[60], (uint)(bitlen));
    sha256_transform(ctx->state, ctx->buf);
    for (int i = 0; i < 8; i++)
        write_be32(&out32[i * 4], ctx->state[i]);
}

// Double SHA-256
static inline void double_sha256(const uchar* data, uint len, uchar* out32) {
    SHA256_CTX ctx;
    uchar mid[32];
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, mid);
    sha256_init(&ctx);
    sha256_update(&ctx, mid, 32);
    sha256_final(&ctx, out32);
}

// =============================================================================
// HMAC-SHA256
// =============================================================================

typedef struct {
    SHA256_CTX inner;
    SHA256_CTX outer;
} HMAC_SHA256_CTX;

static inline void hmac_sha256_init(HMAC_SHA256_CTX* hmac, const uchar* key, uint keylen) {
    uchar rkey[64];
    for (int i = 0; i < 64; i++) rkey[i] = 0;

    if (keylen <= 64) {
        for (uint i = 0; i < keylen; i++) rkey[i] = key[i];
    } else {
        SHA256_CTX tmp;
        sha256_init(&tmp);
        sha256_update(&tmp, key, keylen);
        sha256_final(&tmp, rkey);
    }

    uchar opad[64], ipad[64];
    for (int i = 0; i < 64; i++) {
        opad[i] = rkey[i] ^ 0x5c;
        ipad[i] = rkey[i] ^ 0x36;
    }

    sha256_init(&hmac->outer);
    sha256_update(&hmac->outer, opad, 64);

    sha256_init(&hmac->inner);
    sha256_update(&hmac->inner, ipad, 64);
}

static inline void hmac_sha256_update(HMAC_SHA256_CTX* hmac, const uchar* data, uint len) {
    sha256_update(&hmac->inner, data, len);
}

static inline void hmac_sha256_final(HMAC_SHA256_CTX* hmac, uchar* out32) {
    uchar tmp[32];
    sha256_final(&hmac->inner, tmp);
    sha256_update(&hmac->outer, tmp, 32);
    sha256_final(&hmac->outer, out32);
}

// =============================================================================
// RFC 6979 Deterministic Nonce Generation
// =============================================================================
// Input: secret_key (32 bytes), message (32 bytes)
// Output: nonce32 (32 bytes)
// Follows the procedure from Bitcoin's libsecp256k1

// Forward declarations (defined later, needed for message mod n reduction)
static inline void scalar_set_b32(Scalar* s, const uchar* b32);
static inline void scalar_get_b32(uchar* b32, const Scalar* s);
static inline void scalar_reduce(Scalar* s);

static inline void rfc6979_generate_k(const uchar* seckey32, const uchar* msg32, uchar* nonce32) {
    uchar v[32], k[32];
    uchar keydata[64]; // seckey || msg_mod_n
    HMAC_SHA256_CTX hmac;

    // Copy seckey into keydata
    for (int i = 0; i < 32; i++) keydata[i] = seckey32[i];

    // Reduce message mod n before using as HMAC-DRBG input
    // (matches CUDA's nonce_function_rfc6979: scalar_set_b32 + scalar_get_b32)
    Scalar msg_tmp;
    scalar_set_b32(&msg_tmp, msg32);
    scalar_reduce(&msg_tmp);
    uchar msgmod32[32];
    scalar_get_b32(msgmod32, &msg_tmp);
    for (int i = 0; i < 32; i++) keydata[32 + i] = msgmod32[i];

    // Initialize: V = 0x01...01, K = 0x00...00  (RFC 6979 3.2.b, 3.2.c)
    for (int i = 0; i < 32; i++) { v[i] = 0x01; k[i] = 0x00; }

    // Step d: K = HMAC_K(V || 0x00 || keydata)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    uchar zero = 0x00;
    hmac_sha256_update(&hmac, &zero, 1);
    hmac_sha256_update(&hmac, keydata, 64);
    hmac_sha256_final(&hmac, k);

    // V = HMAC_K(V)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    hmac_sha256_final(&hmac, v);

    // Step f: K = HMAC_K(V || 0x01 || keydata)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    uchar one = 0x01;
    hmac_sha256_update(&hmac, &one, 1);
    hmac_sha256_update(&hmac, keydata, 64);
    hmac_sha256_final(&hmac, k);

    // V = HMAC_K(V)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    hmac_sha256_final(&hmac, v);

    // Step h: generate candidate
    // V = HMAC_K(V)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    hmac_sha256_final(&hmac, v);

    // Output the first valid nonce candidate
    for (int i = 0; i < 32; i++) nonce32[i] = v[i];
}

// =============================================================================
// Scalar Arithmetic (mod curve order n)
// =============================================================================
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

// Scalar type is already defined in secp256k1_point.cl

// Load big-endian 32 bytes into Scalar (little-endian limbs)
static inline void scalar_set_b32(Scalar* s, const uchar* b32) {
    s->limbs[3] = ((ulong)b32[0] << 56) | ((ulong)b32[1] << 48) | ((ulong)b32[2] << 40) | ((ulong)b32[3] << 32) |
                   ((ulong)b32[4] << 24) | ((ulong)b32[5] << 16) | ((ulong)b32[6] << 8)  | (ulong)b32[7];
    s->limbs[2] = ((ulong)b32[8] << 56) | ((ulong)b32[9] << 48) | ((ulong)b32[10] << 40) | ((ulong)b32[11] << 32) |
                   ((ulong)b32[12] << 24) | ((ulong)b32[13] << 16) | ((ulong)b32[14] << 8)  | (ulong)b32[15];
    s->limbs[1] = ((ulong)b32[16] << 56) | ((ulong)b32[17] << 48) | ((ulong)b32[18] << 40) | ((ulong)b32[19] << 32) |
                   ((ulong)b32[20] << 24) | ((ulong)b32[21] << 16) | ((ulong)b32[22] << 8)  | (ulong)b32[23];
    s->limbs[0] = ((ulong)b32[24] << 56) | ((ulong)b32[25] << 48) | ((ulong)b32[26] << 40) | ((ulong)b32[27] << 32) |
                   ((ulong)b32[28] << 24) | ((ulong)b32[29] << 16) | ((ulong)b32[30] << 8)  | (ulong)b32[31];
}

// Store Scalar to big-endian 32 bytes
static inline void scalar_get_b32(uchar* b32, const Scalar* s) {
    b32[0]  = (uchar)(s->limbs[3] >> 56); b32[1]  = (uchar)(s->limbs[3] >> 48);
    b32[2]  = (uchar)(s->limbs[3] >> 40); b32[3]  = (uchar)(s->limbs[3] >> 32);
    b32[4]  = (uchar)(s->limbs[3] >> 24); b32[5]  = (uchar)(s->limbs[3] >> 16);
    b32[6]  = (uchar)(s->limbs[3] >> 8);  b32[7]  = (uchar)(s->limbs[3]);
    b32[8]  = (uchar)(s->limbs[2] >> 56); b32[9]  = (uchar)(s->limbs[2] >> 48);
    b32[10] = (uchar)(s->limbs[2] >> 40); b32[11] = (uchar)(s->limbs[2] >> 32);
    b32[12] = (uchar)(s->limbs[2] >> 24); b32[13] = (uchar)(s->limbs[2] >> 16);
    b32[14] = (uchar)(s->limbs[2] >> 8);  b32[15] = (uchar)(s->limbs[2]);
    b32[16] = (uchar)(s->limbs[1] >> 56); b32[17] = (uchar)(s->limbs[1] >> 48);
    b32[18] = (uchar)(s->limbs[1] >> 40); b32[19] = (uchar)(s->limbs[1] >> 32);
    b32[20] = (uchar)(s->limbs[1] >> 24); b32[21] = (uchar)(s->limbs[1] >> 16);
    b32[22] = (uchar)(s->limbs[1] >> 8);  b32[23] = (uchar)(s->limbs[1]);
    b32[24] = (uchar)(s->limbs[0] >> 56); b32[25] = (uchar)(s->limbs[0] >> 48);
    b32[26] = (uchar)(s->limbs[0] >> 40); b32[27] = (uchar)(s->limbs[0] >> 32);
    b32[28] = (uchar)(s->limbs[0] >> 24); b32[29] = (uchar)(s->limbs[0] >> 16);
    b32[30] = (uchar)(s->limbs[0] >> 8);  b32[31] = (uchar)(s->limbs[0]);
}

// n in little-endian limbs
#define N_LIMB0 0xBFD25E8CD0364141UL
#define N_LIMB1 0xBAAEDCE6AF48A03BUL
#define N_LIMB2 0xFFFFFFFFFFFFFFFEUL
#define N_LIMB3 0xFFFFFFFFFFFFFFFFUL

// Check if scalar >= n
static inline int scalar_check_overflow(const Scalar* s) {
    if (s->limbs[3] > N_LIMB3) return 1;
    if (s->limbs[3] < N_LIMB3) return 0;
    if (s->limbs[2] > N_LIMB2) return 1;
    if (s->limbs[2] < N_LIMB2) return 0;
    if (s->limbs[1] > N_LIMB1) return 1;
    if (s->limbs[1] < N_LIMB1) return 0;
    if (s->limbs[0] >= N_LIMB0) return 1;
    return 0;
}

// Reduce scalar mod n (simple subtraction if >= n)
static inline void scalar_reduce(Scalar* s) {
    if (!scalar_check_overflow(s)) return;
    ulong borrow = 0;
    ulong d0 = sub_with_borrow(s->limbs[0], N_LIMB0, 0, &borrow);
    ulong d1 = sub_with_borrow(s->limbs[1], N_LIMB1, borrow, &borrow);
    ulong d2 = sub_with_borrow(s->limbs[2], N_LIMB2, borrow, &borrow);
    ulong d3 = sub_with_borrow(s->limbs[3], N_LIMB3, borrow, &borrow);
    s->limbs[0] = d0; s->limbs[1] = d1;
    s->limbs[2] = d2; s->limbs[3] = d3;
}

// Scalar addition: r = (a + b) mod n
static inline void scalar_add_mod_n(Scalar* r, const Scalar* a, const Scalar* b) {
    ulong carry = 0;
    r->limbs[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    r->limbs[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    r->limbs[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    r->limbs[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);
    // If carry or >= n, reduce
    if (carry || scalar_check_overflow(r)) {
        ulong borrow = 0;
        r->limbs[0] = sub_with_borrow(r->limbs[0], N_LIMB0, 0, &borrow);
        r->limbs[1] = sub_with_borrow(r->limbs[1], N_LIMB1, borrow, &borrow);
        r->limbs[2] = sub_with_borrow(r->limbs[2], N_LIMB2, borrow, &borrow);
        r->limbs[3] = sub_with_borrow(r->limbs[3], N_LIMB3, borrow, &borrow);
    }
}

// =============================================================================
// Scalar Reduction: reduce 512-bit product mod n
// =============================================================================
// Direct port of libsecp256k1's secp256k1_scalar_reduce_512.
// Uses a 192-bit accumulator (c0, c1, c2) to avoid any possibility of
// carry overflow — the same approach used by the proven CUDA miner.
//
// Reduces in three passes: 512→385 bits, 385→258 bits, 258→256 bits.

// --- 192-bit accumulator macros (ported from CUDA secp256k1) ---
// These use the same carry-propagation logic as libsecp256k1.
// c2:c1:c0 forms the 192-bit running sum.

#define ACC_MULADD_FAST(a, b) { \
    ulong2 _m = mul64_full((a), (b)); \
    c0 += _m.x; \
    ulong _th = _m.y + ((c0 < _m.x) ? 1UL : 0UL); \
    c1 += _th; \
}

#define ACC_MULADD(a, b) { \
    ulong2 _m = mul64_full((a), (b)); \
    c0 += _m.x; \
    ulong _th = _m.y + ((c0 < _m.x) ? 1UL : 0UL); \
    c1 += _th; \
    c2 += (c1 < _th) ? 1UL : 0UL; \
}

#define ACC_SUMADD_FAST(a) { \
    c0 += (a); \
    c1 += (c0 < (a)) ? 1UL : 0UL; \
}

#define ACC_SUMADD(a) { \
    c0 += (a); \
    ulong _over = (c0 < (a)) ? 1UL : 0UL; \
    c1 += _over; \
    c2 += (c1 < _over) ? 1UL : 0UL; \
}

#define ACC_EXTRACT(n) { \
    (n) = c0; \
    c0 = c1; \
    c1 = c2; \
    c2 = 0; \
}

#define ACC_EXTRACT_FAST(n) { \
    (n) = c0; \
    c0 = c1; \
    c1 = 0; \
}

static inline void scalar_reduce_512(Scalar* r, const ulong* l) {
    // nc = 2^256 - n (the complement)
    const ulong NC0 = 0x402DA1732FC9BEBFUL;  // ~N_LIMB0 + 1
    const ulong NC1 = 0x4551231950B75FC4UL;  // ~N_LIMB1

    ulong n0 = l[4], n1 = l[5], n2 = l[6], n3 = l[7];
    ulong m0, m1, m2, m3, m4, m5;
    ulong m6;
    ulong p0, p1, p2, p3;
    ulong p4;

    // --- Pass 1: Reduce 512 bits into 385 ---
    // m[0..6] = l[0..3] + n[0..3] * NC
    ulong c0, c1, c2;

    c0 = l[0]; c1 = 0; c2 = 0;
    ACC_MULADD_FAST(n0, NC0);
    ACC_EXTRACT_FAST(m0);
    ACC_SUMADD_FAST(l[1]);
    ACC_MULADD(n1, NC0);
    ACC_MULADD(n0, NC1);
    ACC_EXTRACT(m1);
    ACC_SUMADD(l[2]);
    ACC_MULADD(n2, NC0);
    ACC_MULADD(n1, NC1);
    ACC_SUMADD(n0);
    ACC_EXTRACT(m2);
    ACC_SUMADD(l[3]);
    ACC_MULADD(n3, NC0);
    ACC_MULADD(n2, NC1);
    ACC_SUMADD(n1);
    ACC_EXTRACT(m3);
    ACC_MULADD(n3, NC1);
    ACC_SUMADD(n2);
    ACC_EXTRACT(m4);
    ACC_SUMADD_FAST(n3);
    ACC_EXTRACT_FAST(m5);
    m6 = c0;

    // --- Pass 2: Reduce 385 bits into 258 ---
    // p[0..4] = m[0..3] + m[4..6] * NC
    c0 = m0; c1 = 0; c2 = 0;
    ACC_MULADD_FAST(m4, NC0);
    ACC_EXTRACT_FAST(p0);
    ACC_SUMADD_FAST(m1);
    ACC_MULADD(m5, NC0);
    ACC_MULADD(m4, NC1);
    ACC_EXTRACT(p1);
    ACC_SUMADD(m2);
    ACC_MULADD(m6, NC0);
    ACC_MULADD(m5, NC1);
    ACC_SUMADD(m4);
    ACC_EXTRACT(p2);
    ACC_SUMADD_FAST(m3);
    ACC_MULADD_FAST(m6, NC1);
    ACC_SUMADD_FAST(m5);
    ACC_EXTRACT_FAST(p3);
    p4 = c0 + m6;

    // --- Pass 3: Reduce 258 bits into 256 ---
    // r[0..3] = p[0..3] + p4 * NC
    // Emulate uint128_t with hi:lo pairs
    ulong2 t;
    ulong carry;

    t = mul64_full(p4, NC0);
    ulong r0 = p0 + t.x;
    carry = t.y + ((r0 < p0) ? 1UL : 0UL);

    t = mul64_full(p4, NC1);
    ulong sum1 = p1 + t.x;
    ulong c_1 = (sum1 < p1) ? 1UL : 0UL;
    ulong r1 = sum1 + carry;
    ulong c_2 = (r1 < sum1) ? 1UL : 0UL;
    carry = t.y + c_1 + c_2;

    ulong sum2 = p2 + p4;
    ulong c_3 = (sum2 < p2) ? 1UL : 0UL;
    ulong r2 = sum2 + carry;
    ulong c_4 = (r2 < sum2) ? 1UL : 0UL;
    carry = c_3 + c_4;

    ulong r3 = p3 + carry;
    // Detect ACTUAL carry from the addition (not a bit of the result!)
    // The old code used (r3 >> 63) which is WRONG: it checks bit 63, but
    // valid scalars in [2^255, n-1) have bit 63 set. That caused false
    // reductions (subtracting n from values < n), corrupting ~50% of results.
    ulong final_carry = (r3 < p3) ? 1UL : 0UL;

    r->limbs[0] = r0;
    r->limbs[1] = r1;
    r->limbs[2] = r2;
    r->limbs[3] = r3;

    // Reduce: matches CUDA's secp256k1_scalar_reduce(r, c + check_overflow(r))
    // If final_carry=1 or result >= n, subtract n. If both, subtract n twice.
    ulong red = final_carry + (ulong)(scalar_check_overflow(r) ? 1 : 0);
    while (red > 0) {
        ulong borrow = 0;
        r->limbs[0] = sub_with_borrow(r->limbs[0], N_LIMB0, 0, &borrow);
        r->limbs[1] = sub_with_borrow(r->limbs[1], N_LIMB1, borrow, &borrow);
        r->limbs[2] = sub_with_borrow(r->limbs[2], N_LIMB2, borrow, &borrow);
        r->limbs[3] = sub_with_borrow(r->limbs[3], N_LIMB3, borrow, &borrow);
        red--;
    }
}

// Scalar multiplication: r = (a * b) mod n
// Direct port of libsecp256k1's secp256k1_scalar_mul_512 + reduce.
// Uses column-accumulation with a 192-bit accumulator (c2:c1:c0) to avoid
// any possibility of carry overflow. This is the same approach as the CUDA miner.
static inline void scalar_mul_mod_n(Scalar* r, const Scalar* a, const Scalar* b) {
    ulong product[8];
    ulong c0, c1, c2;

    ulong a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    ulong b0 = b->limbs[0], b1 = b->limbs[1], b2 = b->limbs[2], b3 = b->limbs[3];

    // Column 0: a0*b0
    c0 = 0; c1 = 0; c2 = 0;
    ACC_MULADD_FAST(a0, b0);
    ACC_EXTRACT_FAST(product[0]);

    // Column 1: a0*b1 + a1*b0
    ACC_MULADD(a0, b1);
    ACC_MULADD(a1, b0);
    ACC_EXTRACT(product[1]);

    // Column 2: a0*b2 + a1*b1 + a2*b0
    ACC_MULADD(a0, b2);
    ACC_MULADD(a1, b1);
    ACC_MULADD(a2, b0);
    ACC_EXTRACT(product[2]);

    // Column 3: a0*b3 + a1*b2 + a2*b1 + a3*b0
    ACC_MULADD(a0, b3);
    ACC_MULADD(a1, b2);
    ACC_MULADD(a2, b1);
    ACC_MULADD(a3, b0);
    ACC_EXTRACT(product[3]);

    // Column 4: a1*b3 + a2*b2 + a3*b1
    ACC_MULADD(a1, b3);
    ACC_MULADD(a2, b2);
    ACC_MULADD(a3, b1);
    ACC_EXTRACT(product[4]);

    // Column 5: a2*b3 + a3*b2
    ACC_MULADD(a2, b3);
    ACC_MULADD(a3, b2);
    ACC_EXTRACT(product[5]);

    // Column 6: a3*b3
    ACC_MULADD_FAST(a3, b3);
    ACC_EXTRACT_FAST(product[6]);

    product[7] = c0;

    scalar_reduce_512(r, product);
}

// Scalar squaring: r = a² mod n
// Uses the same 192-bit accumulator approach as scalar_mul_mod_n.
// Simply calls mul with both operands the same — the compiler can optimize.
static inline void scalar_sqr_mod_n(Scalar* r, const Scalar* a) {
    scalar_mul_mod_n(r, a, a);
}

// =============================================================================
// Scalar inversion mod n — Direct port of libsecp256k1 modinv64 (safegcd)
// =============================================================================
// This is the SAME algorithm used by the CUDA miner. It uses the safegcd
// (divsteps) algorithm which only needs basic integer arithmetic (add/sub/shift),
// NOT exponentiation. This avoids the 255-iteration multiply loop that was
// causing OpenCL compiler issues.
//
// Requires: 128-bit signed integer emulation (since OpenCL has no int128_t).

// --- Signed 128-bit integer emulation ---
typedef struct { long lo; long hi; } s128;  // signed 128-bit as (hi:lo)

static inline s128 s128_mul(long a, long b) {
    // Signed 64x64→128 multiply
    // Split into unsigned multiply + sign correction
    ulong au = (ulong)a, bu = (ulong)b;
    ulong lo = au * bu;
    long hi = (long)mul_hi(au, bu);
    // Sign correction: if a < 0, subtract b from high; if b < 0, subtract a from high
    if (a < 0) hi -= b;
    if (b < 0) hi -= a;
    s128 r; r.lo = (long)lo; r.hi = hi;
    return r;
}

static inline void s128_accum_mul(s128* acc, long a, long b) {
    s128 prod = s128_mul(a, b);
    ulong old_lo = (ulong)acc->lo;
    acc->lo += prod.lo;
    ulong carry = ((ulong)acc->lo < old_lo) ? 1UL : 0UL;
    acc->hi += prod.hi + (long)carry;
}

static inline void s128_rshift(s128* r, int n) {
    // Arithmetic right shift by n (0 < n < 64)
    r->lo = (long)(((ulong)r->lo >> n) | ((ulong)r->hi << (64 - n)));
    r->hi = r->hi >> n;  // arithmetic shift
}

static inline ulong s128_to_u64(const s128* a) { return (ulong)a->lo; }
static inline long  s128_to_i64(const s128* a) { return a->lo; }

// --- Signed62 representation (5 limbs of 62 bits each) ---
typedef struct { long v[5]; } Signed62;

typedef struct { long u, v, q, r; } Trans2x2;

typedef struct {
    Signed62 modulus;
    ulong modulus_inv62;
} ModInfo;

// n in signed62 limbs
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Converted to signed62:
//   v[0] = 0x3FD25E8CD0364141  (low 62 bits of n)
//   v[1] = 0x2ABB739ABD2280EE
//   v[2] = -0x15  = 0xFFFFFFFFFFFFFFEB (as signed)
//   v[3] = 0
//   v[4] = 256  (just encodes the bit length)
// modulus_inv62 = 0x34F20099AA774EC1

static inline long modinv64_divsteps_59(long zeta, ulong f0, ulong g0, Trans2x2* t) {
    ulong u = 8, v = 0, q = 0, r = 8;
    ulong f = f0, g = g0, x, y, z;
    ulong mask1, mask2;
    long c1;
    ulong c2;

    for (int i = 3; i < 62; ++i) {
        c1 = zeta >> 63;
        mask1 = (ulong)c1;
        c2 = g & 1;
        mask2 = (ulong)(-(long)c2);
        x = (f ^ mask1) - mask1;
        y = (u ^ mask1) - mask1;
        z = (v ^ mask1) - mask1;
        g += x & mask2;
        q += y & mask2;
        r += z & mask2;
        mask1 &= mask2;
        zeta = (zeta ^ (long)mask1) - 1;
        f += g & mask1;
        u += q & mask1;
        v += r & mask1;
        g >>= 1;
        u <<= 1;
        v <<= 1;
    }
    t->u = (long)u;
    t->v = (long)v;
    t->q = (long)q;
    t->r = (long)r;
    return zeta;
}

static inline void modinv64_update_de_62(Signed62* d, Signed62* e, const Trans2x2* t,
                                   long mod0, long mod1, long mod2, long mod3, long mod4,
                                   ulong mod_inv62) {
    const ulong M62 = 0x3FFFFFFFFFFFFFFFUL;
    const long d0=d->v[0], d1=d->v[1], d2=d->v[2], d3=d->v[3], d4=d->v[4];
    const long e0=e->v[0], e1=e->v[1], e2=e->v[2], e3=e->v[3], e4=e->v[4];
    const long u=t->u, v_=t->v, q=t->q, r_=t->r;
    long md, me;
    long sd = d4 >> 63;
    long se = e4 >> 63;
    md = (u & sd) + (v_ & se);
    me = (q & sd) + (r_ & se);
    s128 cd = s128_mul(u, d0); s128_accum_mul(&cd, v_, e0);
    s128 ce = s128_mul(q, d0); s128_accum_mul(&ce, r_, e0);
    md -= (long)((mod_inv62 * s128_to_u64(&cd) + (ulong)md) & M62);
    me -= (long)((mod_inv62 * s128_to_u64(&ce) + (ulong)me) & M62);
    s128_accum_mul(&cd, mod0, md);
    s128_accum_mul(&ce, mod0, me);
    s128_rshift(&cd, 62);
    s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d1); s128_accum_mul(&cd, v_, e1);
    s128_accum_mul(&ce, q, d1); s128_accum_mul(&ce, r_, e1);
    if (mod1) { s128_accum_mul(&cd, mod1, md); s128_accum_mul(&ce, mod1, me); }
    d->v[0] = (long)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[0] = (long)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d2); s128_accum_mul(&cd, v_, e2);
    s128_accum_mul(&ce, q, d2); s128_accum_mul(&ce, r_, e2);
    if (mod2) { s128_accum_mul(&cd, mod2, md); s128_accum_mul(&ce, mod2, me); }
    d->v[1] = (long)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[1] = (long)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d3); s128_accum_mul(&cd, v_, e3);
    s128_accum_mul(&ce, q, d3); s128_accum_mul(&ce, r_, e3);
    if (mod3) { s128_accum_mul(&cd, mod3, md); s128_accum_mul(&ce, mod3, me); }
    d->v[2] = (long)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[2] = (long)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d4); s128_accum_mul(&cd, v_, e4);
    s128_accum_mul(&ce, q, d4); s128_accum_mul(&ce, r_, e4);
    s128_accum_mul(&cd, mod4, md);
    s128_accum_mul(&ce, mod4, me);
    d->v[3] = (long)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[3] = (long)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    d->v[4] = s128_to_i64(&cd);
    e->v[4] = s128_to_i64(&ce);
}

static inline void modinv64_update_fg_62(Signed62* f, Signed62* g, const Trans2x2* t) {
    const ulong M62 = 0x3FFFFFFFFFFFFFFFUL;
    const long f0=f->v[0], f1=f->v[1], f2=f->v[2], f3=f->v[3], f4=f->v[4];
    const long g0=g->v[0], g1=g->v[1], g2=g->v[2], g3=g->v[3], g4=g->v[4];
    const long u=t->u, v_=t->v, q=t->q, r_=t->r;
    s128 cf = s128_mul(u, f0); s128_accum_mul(&cf, v_, g0);
    s128 cg = s128_mul(q, f0); s128_accum_mul(&cg, r_, g0);
    s128_rshift(&cf, 62);
    s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f1); s128_accum_mul(&cf, v_, g1);
    s128_accum_mul(&cg, q, f1); s128_accum_mul(&cg, r_, g1);
    f->v[0] = (long)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[0] = (long)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f2); s128_accum_mul(&cf, v_, g2);
    s128_accum_mul(&cg, q, f2); s128_accum_mul(&cg, r_, g2);
    f->v[1] = (long)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[1] = (long)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f3); s128_accum_mul(&cf, v_, g3);
    s128_accum_mul(&cg, q, f3); s128_accum_mul(&cg, r_, g3);
    f->v[2] = (long)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[2] = (long)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f4); s128_accum_mul(&cf, v_, g4);
    s128_accum_mul(&cg, q, f4); s128_accum_mul(&cg, r_, g4);
    f->v[3] = (long)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[3] = (long)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    f->v[4] = s128_to_i64(&cf);
    g->v[4] = s128_to_i64(&cg);
}

static inline void modinv64_normalize_62(Signed62* r, long sign,
                                   long mod0, long mod1, long mod2, long mod3, long mod4) {
    const long M62 = (long)0x3FFFFFFFFFFFFFFFUL;
    long r0=r->v[0], r1=r->v[1], r2=r->v[2], r3=r->v[3], r4=r->v[4];
    long cond_add, cond_negate;

    // Step 1: if negative, add modulus; then negate if requested
    cond_add = r4 >> 63;
    r0 += mod0 & cond_add;
    r1 += mod1 & cond_add;
    r2 += mod2 & cond_add;
    r3 += mod3 & cond_add;
    r4 += mod4 & cond_add;
    cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    // Propagate
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    // Step 2: if still negative, add modulus again
    cond_add = r4 >> 63;
    r0 += mod0 & cond_add;
    r1 += mod1 & cond_add;
    r2 += mod2 & cond_add;
    r3 += mod3 & cond_add;
    r4 += mod4 & cond_add;
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    r->v[0]=r0; r->v[1]=r1; r->v[2]=r2; r->v[3]=r3; r->v[4]=r4;
}

static inline void scalar_inverse_mod_n(Scalar* r, const Scalar* a) {
    const ulong M62 = 0x3FFFFFFFFFFFFFFFUL;

    // Modulus constants (n in signed62)
    const long mod0 = 0x3FD25E8CD0364141L;
    const long mod1 = 0x2ABB739ABD2280EEL;
    const long mod2 = -0x15L;
    const long mod3 = 0L;
    const long mod4 = 256L;
    const ulong mod_inv62 = 0x34F20099AA774EC1UL;

    // Convert scalar to signed62
    ulong a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    Signed62 s;
    s.v[0] = (long)( a0                  & M62);
    s.v[1] = (long)((a0 >> 62 | a1 << 2) & M62);
    s.v[2] = (long)((a1 >> 60 | a2 << 4) & M62);
    s.v[3] = (long)((a2 >> 58 | a3 << 6) & M62);
    s.v[4] = (long)( a3 >> 56);

    // Run modinv64: d=0, e=1, f=modulus, g=x, zeta=-1
    Signed62 d; d.v[0]=0; d.v[1]=0; d.v[2]=0; d.v[3]=0; d.v[4]=0;
    Signed62 e; e.v[0]=1; e.v[1]=0; e.v[2]=0; e.v[3]=0; e.v[4]=0;
    Signed62 f; f.v[0]=mod0; f.v[1]=mod1; f.v[2]=mod2; f.v[3]=mod3; f.v[4]=mod4;
    Signed62 g = s;
    long zeta = -1L;

    // 10 iterations of 59 divsteps = 590 total (sufficient for 256-bit)
    for (int i = 0; i < 10; i++) {
        Trans2x2 t;
        zeta = modinv64_divsteps_59(zeta, (ulong)f.v[0], (ulong)g.v[0], &t);
        modinv64_update_de_62(&d, &e, &t, mod0, mod1, mod2, mod3, mod4, mod_inv62);
        modinv64_update_fg_62(&f, &g, &t);
    }

    // Normalize and convert back
    modinv64_normalize_62(&d, f.v[4], mod0, mod1, mod2, mod3, mod4);

    // Convert signed62 back to scalar (4 × 64-bit limbs)
    ulong d0 = (ulong)d.v[0], d1 = (ulong)d.v[1], d2 = (ulong)d.v[2];
    ulong d3 = (ulong)d.v[3], d4 = (ulong)d.v[4];
    r->limbs[0] = d0      | d1 << 62;
    r->limbs[1] = d1 >> 2 | d2 << 60;
    r->limbs[2] = d2 >> 4 | d3 << 58;
    r->limbs[3] = d3 >> 6 | d4 << 56;
}

// Scalar negation: r = n - a
static inline void scalar_negate(Scalar* r, const Scalar* a) {
    if (scalar_is_zero(a)) {
        *r = *a;
        return;
    }
    ulong borrow = 0;
    r->limbs[0] = sub_with_borrow(N_LIMB0, a->limbs[0], 0, &borrow);
    r->limbs[1] = sub_with_borrow(N_LIMB1, a->limbs[1], borrow, &borrow);
    r->limbs[2] = sub_with_borrow(N_LIMB2, a->limbs[2], borrow, &borrow);
    r->limbs[3] = sub_with_borrow(N_LIMB3, a->limbs[3], borrow, &borrow);
}

// Check if scalar is "high" (s > n/2)
static inline int scalar_is_high(const Scalar* s) {
    // n/2 = 7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
    // Compare s > n/2
    Scalar half_n;
    half_n.limbs[3] = 0x7FFFFFFFFFFFFFFFUL;
    half_n.limbs[2] = 0xFFFFFFFFFFFFFFFFUL;
    half_n.limbs[1] = 0x5D576E7357A4501DUL;
    half_n.limbs[0] = 0xDFE92F46681B20A0UL;
    
    if (s->limbs[3] > half_n.limbs[3]) return 1;
    if (s->limbs[3] < half_n.limbs[3]) return 0;
    if (s->limbs[2] > half_n.limbs[2]) return 1;
    if (s->limbs[2] < half_n.limbs[2]) return 0;
    if (s->limbs[1] > half_n.limbs[1]) return 1;
    if (s->limbs[1] < half_n.limbs[1]) return 0;
    if (s->limbs[0] > half_n.limbs[0]) return 1;
    return 0;
}

// =============================================================================
// ECDSA Signing
// =============================================================================
// Given: secret key, message hash (both as 32-byte big-endian)
// Returns: DER-encoded signature in sig_out, length in sig_len
// Uses RFC 6979 deterministic nonce

// DER encode a scalar (big-endian 32 bytes) into DER integer format
// Returns number of bytes written
static inline int der_encode_integer(uchar* out, const uchar* val32) {
    int start = 0;
    // Skip leading zeros
    while (start < 32 && val32[start] == 0) start++;
    if (start == 32) {
        // Zero value
        out[0] = 0x02; out[1] = 0x01; out[2] = 0x00;
        return 3;
    }
    int need_pad = (val32[start] & 0x80) ? 1 : 0;
    int len = 32 - start + need_pad;
    out[0] = 0x02;
    out[1] = (uchar)len;
    int pos = 2;
    if (need_pad) out[pos++] = 0x00;
    for (int i = start; i < 32; i++) out[pos++] = val32[i];
    return pos;
}

// Full DER signature encoding: 0x30 <total_len> <r_der> <s_der>
static inline int der_encode_signature(uchar* out, const uchar* r32, const uchar* s32) {
    uchar r_der[35], s_der[35];
    int r_len = der_encode_integer(r_der, r32);
    int s_len = der_encode_integer(s_der, s32);
    
    out[0] = 0x30;
    out[1] = (uchar)(r_len + s_len);
    int pos = 2;
    for (int i = 0; i < r_len; i++) out[pos++] = r_der[i];
    for (int i = 0; i < s_len; i++) out[pos++] = s_der[i];
    return pos;
}

// =============================================================================
// Precomputed Generator Table - 2-bit windowed lookup for k*G
// =============================================================================
// Table layout: 128 groups x 4 entries = 512 affine points
//   table[group * 4 + value] = value * (4^group) * G
//   Each entry: 8 x uint64 (x[4 LE limbs] + y[4 LE limbs]) = 64 bytes
//   Entry for value=0 is unused (infinity); we skip it during lookup.
// Total: 512 * 64 = 32768 bytes = 4096 ulong values
//
// This replaces the per-thread wNAF computation (~256 doubles + ~85 adds)
// with ~96 mixed affine-Jacobian additions (no doubles at all).
// =============================================================================

// Load an affine point from the precomputed table buffer
static inline void ecmult_table_load_point(AffinePoint* p, __global const ulong* table, int index) {
    int off = index * 8;  // 8 ulongs per entry
    p->x.limbs[0] = table[off + 0];
    p->x.limbs[1] = table[off + 1];
    p->x.limbs[2] = table[off + 2];
    p->x.limbs[3] = table[off + 3];
    p->y.limbs[0] = table[off + 4];
    p->y.limbs[1] = table[off + 5];
    p->y.limbs[2] = table[off + 6];
    p->y.limbs[3] = table[off + 7];
}

// Fast scalar multiplication k*G using precomputed 2-bit windowed table
// Replaces the naive wNAF approach with 128 iterations of table lookup + mixed add
static inline void scalar_mul_generator_precomp(JacobianPoint* r, const Scalar* k, __global const ulong* table) {
    point_set_infinity(r);
    
    AffinePoint ap;
    
    for (int group = 0; group < 128; group++) {
        // Extract 2-bit value from scalar k at position (group * 2)
        int limb_idx = (group * 2) / 64;
        int bit_idx  = (group * 2) % 64;
        int value    = (int)((k->limbs[limb_idx] >> bit_idx) & 3UL);
        
        if (value != 0) {
            // Load affine point from table: table[group * 4 + value]
            ecmult_table_load_point(&ap, table, group * 4 + value);
            // Mixed affine-Jacobian addition (cheaper than full Jacobian add)
            point_add_mixed_impl(r, r, &ap);
        }
    }
}

// =============================================================================
// ECDSA signing function (precomputed table k*G + optimized inverse)
// =============================================================================
// Uses scalar_mul_generator_precomp (2-bit windowed table, ~8x faster) for k*G
// and scalar_inverse_mod_n (nibble-based, faster) for k^(-1).
static inline int ecdsa_sign(const uchar* seckey32, const uchar* msg32, uchar* sig_out,
                      __global const ulong* ecmult_table) {
    // 1. Generate deterministic nonce k via RFC 6979
    uchar nonce32[32];
    rfc6979_generate_k(seckey32, msg32, nonce32);
    
    // 2. Load nonce as scalar k
    Scalar k;
    scalar_set_b32(&k, nonce32);
    if (scalar_is_zero(&k)) return 0;
    scalar_reduce(&k);
    if (scalar_is_zero(&k)) return 0;
    
    // 3. Compute R = k * G using precomputed table (fast path)
    JacobianPoint R_jac;
    scalar_mul_generator_precomp(&R_jac, &k, ecmult_table);
    
    // 4. Convert R to affine to get R.x
    FieldElement z_inv, z_inv2, rx_fe;
    field_inv_impl(&z_inv, &R_jac.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&rx_fe, &R_jac.x, &z_inv2);
    
    // 5. r = R.x mod n
    uchar rx_bytes[32];
    rx_bytes[0]  = (uchar)(rx_fe.limbs[3] >> 56); rx_bytes[1]  = (uchar)(rx_fe.limbs[3] >> 48);
    rx_bytes[2]  = (uchar)(rx_fe.limbs[3] >> 40); rx_bytes[3]  = (uchar)(rx_fe.limbs[3] >> 32);
    rx_bytes[4]  = (uchar)(rx_fe.limbs[3] >> 24); rx_bytes[5]  = (uchar)(rx_fe.limbs[3] >> 16);
    rx_bytes[6]  = (uchar)(rx_fe.limbs[3] >> 8);  rx_bytes[7]  = (uchar)(rx_fe.limbs[3]);
    rx_bytes[8]  = (uchar)(rx_fe.limbs[2] >> 56); rx_bytes[9]  = (uchar)(rx_fe.limbs[2] >> 48);
    rx_bytes[10] = (uchar)(rx_fe.limbs[2] >> 40); rx_bytes[11] = (uchar)(rx_fe.limbs[2] >> 32);
    rx_bytes[12] = (uchar)(rx_fe.limbs[2] >> 24); rx_bytes[13] = (uchar)(rx_fe.limbs[2] >> 16);
    rx_bytes[14] = (uchar)(rx_fe.limbs[2] >> 8);  rx_bytes[15] = (uchar)(rx_fe.limbs[2]);
    rx_bytes[16] = (uchar)(rx_fe.limbs[1] >> 56); rx_bytes[17] = (uchar)(rx_fe.limbs[1] >> 48);
    rx_bytes[18] = (uchar)(rx_fe.limbs[1] >> 40); rx_bytes[19] = (uchar)(rx_fe.limbs[1] >> 32);
    rx_bytes[20] = (uchar)(rx_fe.limbs[1] >> 24); rx_bytes[21] = (uchar)(rx_fe.limbs[1] >> 16);
    rx_bytes[22] = (uchar)(rx_fe.limbs[1] >> 8);  rx_bytes[23] = (uchar)(rx_fe.limbs[1]);
    rx_bytes[24] = (uchar)(rx_fe.limbs[0] >> 56); rx_bytes[25] = (uchar)(rx_fe.limbs[0] >> 48);
    rx_bytes[26] = (uchar)(rx_fe.limbs[0] >> 40); rx_bytes[27] = (uchar)(rx_fe.limbs[0] >> 32);
    rx_bytes[28] = (uchar)(rx_fe.limbs[0] >> 24); rx_bytes[29] = (uchar)(rx_fe.limbs[0] >> 16);
    rx_bytes[30] = (uchar)(rx_fe.limbs[0] >> 8);  rx_bytes[31] = (uchar)(rx_fe.limbs[0]);
    
    Scalar sig_r;
    scalar_set_b32(&sig_r, rx_bytes);
    scalar_reduce(&sig_r);
    if (scalar_is_zero(&sig_r)) return 0;
    
    // 6. s = k^(-1) * (msg + r * seckey) mod n
    Scalar sec, msg_scalar, n_val, sig_s;
    scalar_set_b32(&sec, seckey32);
    scalar_reduce(&sec);           // Ensure sec < n (matches CUDA scalar_set_b32 reduce)
    scalar_set_b32(&msg_scalar, msg32);
    scalar_reduce(&msg_scalar);    // Ensure msg < n (matches CUDA scalar_set_b32 reduce)
    
    scalar_mul_mod_n(&n_val, &sig_r, &sec);
    scalar_add_mod_n(&n_val, &n_val, &msg_scalar);
    // k_inv using OPTIMIZED nibble-based inverse (uses scalar_sqr_mod_n)
    Scalar k_inv;
    scalar_inverse_mod_n(&k_inv, &k);
    scalar_mul_mod_n(&sig_s, &k_inv, &n_val);
    
    if (scalar_is_zero(&sig_s)) return 0;
    
    // 7. Normalize s
    if (scalar_is_high(&sig_s)) {
        scalar_negate(&sig_s, &sig_s);
    }
    
    // 8. DER encode
    uchar r_bytes[32], s_bytes[32];
    scalar_get_b32(r_bytes, &sig_r);
    scalar_get_b32(s_bytes, &sig_s);
    
    return der_encode_signature(sig_out, r_bytes, s_bytes);
}

// =============================================================================
// uint256 Addition (for mud = hash_no_sig + nonce)
// =============================================================================
// Both are 32-byte little-endian arrays interpreted as 4x64-bit limbs

typedef struct {
    ulong limbs[4]; // little-endian
} uint256_t;

static inline void uint256_add(uint256_t* r, const uint256_t* a, const uint256_t* b) {
    ulong carry = 0;
    r->limbs[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    r->limbs[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    r->limbs[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    r->limbs[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);
    // Overflow is ignored (wraps around, matching Bitcoin uint256 behavior)
}

// =============================================================================
// Mining Kernel
// =============================================================================
// Each work item tries a different nonce.
// Matches the CUDA kernel's algorithm exactly:
//   1. mud = hash_no_sig + nonce  (uint256 LE addition)
//   2. ECDSA sign(seckey, mud_LE_bytes) -> DER signature
//   3. preimage = nonce(8 bytes LE) || vchSig_len(1 byte) || DER_sig(N bytes)
//   4. hashPoW = double-SHA256(preimage)
//   5. Check trailing bytes [28-31] for difficulty (reversed byte order)
//
// Shared memory layout (from node via host):
//   key_data[32]      - secret key
//   hash_no_sig[32]   - block hash without signature
//
// Output:
//   result_nonce[1]   - winning nonce (0 if not found)
//   result_found[1]   - flag: 1 if a winning nonce was found
//   hashrate_ctr[1]   - atomic counter for hashrate measurement

// Number of nonces each thread processes per kernel invocation.
// Amortizes global memory loads (key, hash_no_sig) across multiple hashes.
#define NONCES_PER_THREAD 4

__kernel void btcw_mine(
    __global const uchar* key_data,       // 32 bytes: secret key
    __global const uchar* hash_no_sig,    // 32 bytes: block hash without signature
    __global volatile ulong* result_nonce,// output: winning nonce
    __global volatile uint* result_found, // output: 1 if found
    __global volatile uint* hashrate_ctr, // output: hashrate counter
    const ulong nonce_base,               // base nonce for this batch
    const uint gpu_num,                   // GPU number for nonce partitioning
    __global const ulong* ecmult_table    // precomputed generator table (32KB)
) {
    uint gid = get_global_id(0);
    
    // Load secret key ONCE (amortized across NONCES_PER_THREAD nonces)
    uchar seckey[32];
    for (int i = 0; i < 32; i++) seckey[i] = key_data[i];
    
    // Load hash_no_sig ONCE into uint256 (little-endian from node)
    uint256_t h_no_sig;
    h_no_sig.limbs[0] = ((ulong)hash_no_sig[0])       | ((ulong)hash_no_sig[1] << 8)  |
                         ((ulong)hash_no_sig[2] << 16)  | ((ulong)hash_no_sig[3] << 24) |
                         ((ulong)hash_no_sig[4] << 32)  | ((ulong)hash_no_sig[5] << 40) |
                         ((ulong)hash_no_sig[6] << 48)  | ((ulong)hash_no_sig[7] << 56);
    h_no_sig.limbs[1] = ((ulong)hash_no_sig[8])        | ((ulong)hash_no_sig[9] << 8)  |
                         ((ulong)hash_no_sig[10] << 16) | ((ulong)hash_no_sig[11] << 24) |
                         ((ulong)hash_no_sig[12] << 32) | ((ulong)hash_no_sig[13] << 40) |
                         ((ulong)hash_no_sig[14] << 48) | ((ulong)hash_no_sig[15] << 56);
    h_no_sig.limbs[2] = ((ulong)hash_no_sig[16])       | ((ulong)hash_no_sig[17] << 8)  |
                         ((ulong)hash_no_sig[18] << 16) | ((ulong)hash_no_sig[19] << 24) |
                         ((ulong)hash_no_sig[20] << 32) | ((ulong)hash_no_sig[21] << 40) |
                         ((ulong)hash_no_sig[22] << 48) | ((ulong)hash_no_sig[23] << 56);
    h_no_sig.limbs[3] = ((ulong)hash_no_sig[24])       | ((ulong)hash_no_sig[25] << 8)  |
                         ((ulong)hash_no_sig[26] << 16) | ((ulong)hash_no_sig[27] << 24) |
                         ((ulong)hash_no_sig[28] << 32) | ((ulong)hash_no_sig[29] << 40) |
                         ((ulong)hash_no_sig[30] << 48) | ((ulong)hash_no_sig[31] << 56);
    
    // GPU number mask for nonce space partitioning (upper byte)
    ulong gpu_mask = ((ulong)(gpu_num & 0xFF) << 56);
    
    // Multi-nonce loop: process NONCES_PER_THREAD nonces per thread
    for (int iter = 0; iter < NONCES_PER_THREAD; iter++) {
        ulong nonce = nonce_base + (ulong)gid * NONCES_PER_THREAD + (ulong)iter;
        nonce = (nonce & 0x00FFFFFFFFFFFFFFUL) | gpu_mask;
        
        // mud = hash_no_sig + nonce  (uint256 addition, little-endian)
        uint256_t nonce_256;
        nonce_256.limbs[0] = nonce;
        nonce_256.limbs[1] = 0;
        nonce_256.limbs[2] = 0;
        nonce_256.limbs[3] = 0;
        
        uint256_t mud;
        uint256_add(&mud, &h_no_sig, &nonce_256);
        
        // Serialize mud as little-endian bytes (matching CUDA's memcpy from uint256)
        uchar mud_bytes[32];
        mud_bytes[0]  = (uchar)(mud.limbs[0]);       mud_bytes[1]  = (uchar)(mud.limbs[0] >> 8);
        mud_bytes[2]  = (uchar)(mud.limbs[0] >> 16);  mud_bytes[3]  = (uchar)(mud.limbs[0] >> 24);
        mud_bytes[4]  = (uchar)(mud.limbs[0] >> 32);  mud_bytes[5]  = (uchar)(mud.limbs[0] >> 40);
        mud_bytes[6]  = (uchar)(mud.limbs[0] >> 48);  mud_bytes[7]  = (uchar)(mud.limbs[0] >> 56);
        mud_bytes[8]  = (uchar)(mud.limbs[1]);        mud_bytes[9]  = (uchar)(mud.limbs[1] >> 8);
        mud_bytes[10] = (uchar)(mud.limbs[1] >> 16);  mud_bytes[11] = (uchar)(mud.limbs[1] >> 24);
        mud_bytes[12] = (uchar)(mud.limbs[1] >> 32);  mud_bytes[13] = (uchar)(mud.limbs[1] >> 40);
        mud_bytes[14] = (uchar)(mud.limbs[1] >> 48);  mud_bytes[15] = (uchar)(mud.limbs[1] >> 56);
        mud_bytes[16] = (uchar)(mud.limbs[2]);        mud_bytes[17] = (uchar)(mud.limbs[2] >> 8);
        mud_bytes[18] = (uchar)(mud.limbs[2] >> 16);  mud_bytes[19] = (uchar)(mud.limbs[2] >> 24);
        mud_bytes[20] = (uchar)(mud.limbs[2] >> 32);  mud_bytes[21] = (uchar)(mud.limbs[2] >> 40);
        mud_bytes[22] = (uchar)(mud.limbs[2] >> 48);  mud_bytes[23] = (uchar)(mud.limbs[2] >> 56);
        mud_bytes[24] = (uchar)(mud.limbs[3]);        mud_bytes[25] = (uchar)(mud.limbs[3] >> 8);
        mud_bytes[26] = (uchar)(mud.limbs[3] >> 16);  mud_bytes[27] = (uchar)(mud.limbs[3] >> 24);
        mud_bytes[28] = (uchar)(mud.limbs[3] >> 32);  mud_bytes[29] = (uchar)(mud.limbs[3] >> 40);
        mud_bytes[30] = (uchar)(mud.limbs[3] >> 48);  mud_bytes[31] = (uchar)(mud.limbs[3] >> 56);
        
        // ECDSA sign(seckey, mud_LE) -> DER signature (using precomputed table for k*G)
        uchar der_sig[73]; // Max DER signature size
        int sig_len = ecdsa_sign(seckey, mud_bytes, der_sig, ecmult_table);
        
        if (sig_len == 0) continue; // Signing failed (extremely rare)
        
        // Construct preimage matching CUDA's CDataStream serialization:
        //   ss << nonce << vchBlockSig
        //   = nonce (8 bytes LE) || CompactSize(sig_len) (1 byte) || DER_sig (sig_len bytes)
        uchar preimage[82]; // 8 + 1 + 73 max
        preimage[0] = (uchar)(nonce);
        preimage[1] = (uchar)(nonce >> 8);
        preimage[2] = (uchar)(nonce >> 16);
        preimage[3] = (uchar)(nonce >> 24);
        preimage[4] = (uchar)(nonce >> 32);
        preimage[5] = (uchar)(nonce >> 40);
        preimage[6] = (uchar)(nonce >> 48);
        preimage[7] = (uchar)(nonce >> 56);
        preimage[8] = (uchar)sig_len;  // CompactSize length byte (70-72)
        for (int i = 0; i < sig_len; i++) preimage[9 + i] = der_sig[i];
        
        // hashPoW = double-SHA256(nonce || vchSig_len || signature)
        uchar hashPoW[32];
        double_sha256(preimage, 9 + sig_len, hashPoW);
        
        // Check trailing bytes of SHA-256 output (matching CUDA's difficulty check).
        // BTCW displays hashes in reversed byte order, so "leading zeros" in the
        // displayed hash correspond to trailing zeros in the SHA-256 output bytes.
        // hashPoW[31]==0 && hashPoW[30]==0 -> ~16-bit filter for hashrate counting
        if ((hashPoW[31] == 0) && (hashPoW[30] == 0)) {
            // Update hashrate counter (only ~1 in 65536 hashes pass here)
            // This matches CUDA's nonce4hashrate behavior
            
            // Check deeper: ~30-bit filter for block candidate submission
            if ((hashPoW[29] == 0) && ((hashPoW[28] & 0xFC) == 0)) {
                *result_nonce = nonce;
                *result_found = 1;
            }
        }
    }
    
    // Count all NONCES_PER_THREAD hashes (one atomic per thread, not per nonce)
    atomic_add(hashrate_ctr, NONCES_PER_THREAD);
}

// =============================================================================
// Precompute Kernel: builds the ecmult_gen table on-GPU at startup
// =============================================================================
// Runs ONCE with a single work item. Computes:
//   table[group * 4 + value] = value * (4^group) * G   (as affine point)
// for group = 0..127, value = 0..3.
// Each entry: 8 ulongs = {x[4 LE limbs], y[4 LE limbs]} = 64 bytes.
// Total: 512 entries * 64 bytes = 32768 bytes = 4096 ulongs.

// Helper: convert Jacobian point to affine and write 8 ulongs to table
static inline void write_affine_to_table(__global ulong* table, int index,
                                   const JacobianPoint* jac) {
    FieldElement z_inv, z_inv2, z_inv3, ax, ay;
    field_inv_impl(&z_inv, &jac->z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv, &z_inv2);
    field_mul_impl(&ax, &jac->x, &z_inv2);
    field_mul_impl(&ay, &jac->y, &z_inv3);
    
    int off = index * 8;
    table[off + 0] = ax.limbs[0];
    table[off + 1] = ax.limbs[1];
    table[off + 2] = ax.limbs[2];
    table[off + 3] = ax.limbs[3];
    table[off + 4] = ay.limbs[0];
    table[off + 5] = ay.limbs[1];
    table[off + 6] = ay.limbs[2];
    table[off + 7] = ay.limbs[3];
}

__kernel void precompute_ecmult_gen_table(
    __global ulong* table_out  // 4096 ulongs = 32768 bytes
) {
    // Only use global_id 0
    if (get_global_id(0) != 0) return;
    
    // Start with generator point G
    AffinePoint G_aff;
    get_generator(&G_aff);
    JacobianPoint base;
    point_from_affine(&base, &G_aff);
    
    for (int group = 0; group < 128; group++) {
        int base_idx = group * 4;
        
        // Entry 0: infinity (all zeros)
        for (int i = 0; i < 8; i++) table_out[base_idx * 8 + i] = 0;
        
        // Entry 1: base = (4^group) * G
        write_affine_to_table(table_out, base_idx + 1, &base);
        
        // Entry 2: 2 * base = point_double(base)
        JacobianPoint p2;
        point_double_impl(&p2, &base);
        write_affine_to_table(table_out, base_idx + 2, &p2);
        
        // Entry 3: 3 * base = base + 2*base
        JacobianPoint p3;
        point_add_impl(&p3, &base, &p2);
        write_affine_to_table(table_out, base_idx + 3, &p3);
        
        // Advance base: base = 4 * base (double twice for next group)
        JacobianPoint tmp;
        point_double_impl(&tmp, &base);
        point_double_impl(&base, &tmp);
    }
}

// =============================================================================
// Diagnostic Kernel: verify ECDSA signing on GPU
// =============================================================================
// Signs a caller-provided (seckey, message) pair and writes the DER signature
// to the output buffer.  Runs a single work-item.
// The host code can then compare this against a known-good reference.
__kernel void diagnostic_ecdsa_sign(
    __global const uchar*  seckey32,    // 32 bytes
    __global const uchar*  msg32,       // 32 bytes
    __global uchar*        sig_out,     // 73 bytes max DER
    __global int*          sig_len_out, // 1 int
    __global const ulong*  ecmult_table // precomputed generator table
) {
    if (get_global_id(0) != 0) return;

    uchar sk[32], m[32];
    for (int i = 0; i < 32; i++) { sk[i] = seckey32[i]; m[i] = msg32[i]; }

    uchar der[73];
    int len = ecdsa_sign(sk, m, der, ecmult_table);

    *sig_len_out = len;
    for (int i = 0; i < len; i++) sig_out[i] = der[i];
}

// =============================================================================
// Diagnostic Kernel: test scalar arithmetic in isolation
// =============================================================================
// Writes results as hex limbs (little-endian) to output buffer.
// Tests:
//   [0..31]   = mul(3, 7)                    → expect {21, 0, 0, 0}
//   [32..63]  = inv(2)                       → expect (n+1)/2
//   [64..95]  = mul(2, inv(2))               → expect {1, 0, 0, 0}
//   [96..127] = mul(n-1, n-1)                → expect {1, 0, 0, 0}
//   [128]     = pass/fail byte (0xFF = all pass, else bit flags)
__kernel void diagnostic_scalar_ops(
    __global uchar* output     // 129 bytes
) {
    if (get_global_id(0) != 0) return;

    Scalar a, b, r;
    uchar flags = 0;

    // Helper: write scalar limbs (LE 64-bit) to output
    #define WRITE_SCALAR(off, s) \
        for (int _i = 0; _i < 4; _i++) \
            for (int _j = 0; _j < 8; _j++) \
                output[(off) + _i * 8 + _j] = (uchar)((s).limbs[_i] >> (_j * 8));

    // Test 1: 3 * 7 = 21
    a.limbs[0] = 3; a.limbs[1] = 0; a.limbs[2] = 0; a.limbs[3] = 0;
    b.limbs[0] = 7; b.limbs[1] = 0; b.limbs[2] = 0; b.limbs[3] = 0;
    scalar_mul_mod_n(&r, &a, &b);
    WRITE_SCALAR(0, r);
    if (r.limbs[0] == 21 && r.limbs[1] == 0 && r.limbs[2] == 0 && r.limbs[3] == 0)
        flags |= 0x01;

    // Test 2: inverse(2) → should be (n+1)/2
    // (n+1)/2 = 7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A1
    a.limbs[0] = 2; a.limbs[1] = 0; a.limbs[2] = 0; a.limbs[3] = 0;
    scalar_inverse_mod_n(&r, &a);
    WRITE_SCALAR(32, r);
    if (r.limbs[0] == 0xDFE92F46681B20A1UL &&
        r.limbs[1] == 0x5D576E7357A4501DUL &&
        r.limbs[2] == 0xFFFFFFFFFFFFFFFFUL &&
        r.limbs[3] == 0x7FFFFFFFFFFFFFFFUL)
        flags |= 0x02;

    // Test 3: 2 * inv(2) = 1
    Scalar two;
    two.limbs[0] = 2; two.limbs[1] = 0; two.limbs[2] = 0; two.limbs[3] = 0;
    Scalar identity;
    scalar_mul_mod_n(&identity, &two, &r);
    WRITE_SCALAR(64, identity);
    if (identity.limbs[0] == 1 && identity.limbs[1] == 0 &&
        identity.limbs[2] == 0 && identity.limbs[3] == 0)
        flags |= 0x04;

    // Test 4: (n-1) * (n-1) = 1  (since n-1 ≡ -1 mod n, (-1)^2 = 1)
    a.limbs[0] = N_LIMB0 - 1; // Wait, n-1 limbs: subtract 1 from n
    // n = {N_LIMB0, N_LIMB1, N_LIMB2, N_LIMB3}
    // n-1 = {N_LIMB0-1, N_LIMB1, N_LIMB2, N_LIMB3}  (since N_LIMB0 > 0)
    a.limbs[0] = 0xBFD25E8CD0364140UL;
    a.limbs[1] = 0xBAAEDCE6AF48A03BUL;
    a.limbs[2] = 0xFFFFFFFFFFFFFFFEUL;
    a.limbs[3] = 0xFFFFFFFFFFFFFFFFUL;
    scalar_mul_mod_n(&r, &a, &a);
    WRITE_SCALAR(96, r);
    if (r.limbs[0] == 1 && r.limbs[1] == 0 && r.limbs[2] == 0 && r.limbs[3] == 0)
        flags |= 0x08;

    // Test 5: 2^256 mod n via 8 chained squarings
    // 2^(2^8) = 2^256 mod n = 2^256 - n = NC
    // Expected: {0x402DA1732FC9BEBF, 0x4551231950B75FC4, 1, 0}
    a.limbs[0] = 2; a.limbs[1] = 0; a.limbs[2] = 0; a.limbs[3] = 0;
    for (int _sq = 0; _sq < 8; _sq++) {
        scalar_mul_mod_n(&r, &a, &a);
        a = r;
    }
    // a is now 2^256 mod n
    // Write at offset 129 (output[129..160])
    for (int _i = 0; _i < 4; _i++)
        for (int _j = 0; _j < 8; _j++)
            output[129 + _i * 8 + _j] = (uchar)(a.limbs[_i] >> (_j * 8));
    if (a.limbs[0] == 0x402DA1732FC9BEBFUL &&
        a.limbs[1] == 0x4551231950B75FC4UL &&
        a.limbs[2] == 1 && a.limbs[3] == 0)
        flags |= 0x10;

    // Test 6: single squaring of a mid-range value
    // (2^128)^2 = 2^256 mod n = NC
    a.limbs[0] = 0; a.limbs[1] = 0; a.limbs[2] = 1; a.limbs[3] = 0;  // 2^128
    scalar_mul_mod_n(&r, &a, &a);
    // Write at offset 161 (output[161..192])
    for (int _i = 0; _i < 4; _i++)
        for (int _j = 0; _j < 8; _j++)
            output[161 + _i * 8 + _j] = (uchar)(r.limbs[_i] >> (_j * 8));
    if (r.limbs[0] == 0x402DA1732FC9BEBFUL &&
        r.limbs[1] == 0x4551231950B75FC4UL &&
        r.limbs[2] == 1 && r.limbs[3] == 0)
        flags |= 0x20;

    output[128] = flags;
    #undef WRITE_SCALAR
}

