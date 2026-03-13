// =============================================================================
// BTCW Metal Mining Kernel
// =============================================================================
// Translated from mining_kernel.cl (OpenCL) to Metal Shading Language (MSL).
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

#include <metal_stdlib>
#include <metal_integer>
#include <metal_atomic>
using namespace metal;
#include "secp256k1_point.metal"

// =============================================================================
// SHA-256 Implementation (for GPU)
// =============================================================================

struct SHA256_CTX {
    uint32_t state[8];
    uint8_t buf[64];
    uint32_t bytes;
};

constant uint32_t SHA256_K[64] = {
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

static inline uint32_t rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

#define SHA_Ch(x,y,z)    ((z) ^ ((x) & ((y) ^ (z))))
#define SHA_Maj(x,y,z)   (((x) & (y)) | ((z) & ((x) | (y))))
#define SHA_Sigma0(x)    (rotl32((x), 30U) ^ rotl32((x), 19U) ^ rotl32((x), 10U))
#define SHA_Sigma1(x)    (rotl32((x), 26U) ^ rotl32((x), 21U) ^ rotl32((x), 7U))
#define SHA_sigma0(x)    (rotl32((x), 25U) ^ rotl32((x), 14U) ^ ((x) >> 3))
#define SHA_sigma1(x)    (rotl32((x), 15U) ^ rotl32((x), 13U) ^ ((x) >> 10))

static inline uint32_t read_be32(thread const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static inline void write_be32(thread uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v);
}

static inline void sha256_transform(thread uint32_t* s, thread const uint8_t* buf) {
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3];
    uint32_t e = s[4], f = s[5], g = s[6], h = s[7];
    uint32_t w[64];

    #pragma unroll
    for (int i = 0; i < 16; i++)
        w[i] = read_be32(&buf[i * 4]);

    #pragma unroll
    for (int i = 16; i < 64; i++)
        w[i] = SHA_sigma1(w[i-2]) + w[i-7] + SHA_sigma0(w[i-15]) + w[i-16];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + SHA_Sigma1(e) + SHA_Ch(e, f, g) + SHA256_K[i] + w[i];
        uint32_t t2 = SHA_Sigma0(a) + SHA_Maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s[0] += a; s[1] += b; s[2] += c; s[3] += d;
    s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

static inline void sha256_init(thread SHA256_CTX* ctx) {
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

static inline void sha256_update(thread SHA256_CTX* ctx, thread const uint8_t* data, uint32_t len) {
    uint32_t bufsize = ctx->bytes & 0x3F;
    ctx->bytes += len;
    while (len >= 64 - bufsize) {
        uint32_t chunk = 64 - bufsize;
        for (uint32_t i = 0; i < chunk; i++)
            ctx->buf[bufsize + i] = data[i];
        data += chunk;
        len -= chunk;
        sha256_transform(ctx->state, ctx->buf);
        bufsize = 0;
    }
    for (uint32_t i = 0; i < len; i++)
        ctx->buf[bufsize + i] = data[i];
}

static inline void sha256_final(thread SHA256_CTX* ctx, thread uint8_t* out32) {
    uint32_t bufsize = ctx->bytes & 0x3F;
    ctx->buf[bufsize++] = 0x80;
    if (bufsize > 56) {
        for (uint32_t i = bufsize; i < 64; i++) ctx->buf[i] = 0;
        sha256_transform(ctx->state, ctx->buf);
        bufsize = 0;
    }
    for (uint32_t i = bufsize; i < 56; i++) ctx->buf[i] = 0;
    uint64_t bitlen = (uint64_t)ctx->bytes * 8;
    write_be32(&ctx->buf[56], (uint32_t)(bitlen >> 32));
    write_be32(&ctx->buf[60], (uint32_t)(bitlen));
    sha256_transform(ctx->state, ctx->buf);
    for (int i = 0; i < 8; i++)
        write_be32(&out32[i * 4], ctx->state[i]);
}

static inline void double_sha256(thread const uint8_t* data, uint32_t len, thread uint8_t* out32) {
    SHA256_CTX ctx;
    uint8_t mid[32];
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

struct HMAC_SHA256_CTX {
    SHA256_CTX inner;
    SHA256_CTX outer;
};

static inline void hmac_sha256_init(thread HMAC_SHA256_CTX* hmac, thread const uint8_t* key, uint32_t keylen) {
    uint8_t rkey[64];
    for (int i = 0; i < 64; i++) rkey[i] = 0;

    if (keylen <= 64) {
        for (uint32_t i = 0; i < keylen; i++) rkey[i] = key[i];
    } else {
        SHA256_CTX tmp;
        sha256_init(&tmp);
        sha256_update(&tmp, key, keylen);
        sha256_final(&tmp, rkey);
    }

    uint8_t opad[64], ipad[64];
    for (int i = 0; i < 64; i++) {
        opad[i] = rkey[i] ^ 0x5c;
        ipad[i] = rkey[i] ^ 0x36;
    }

    sha256_init(&hmac->outer);
    sha256_update(&hmac->outer, opad, 64);

    sha256_init(&hmac->inner);
    sha256_update(&hmac->inner, ipad, 64);
}

static inline void hmac_sha256_update(thread HMAC_SHA256_CTX* hmac, thread const uint8_t* data, uint32_t len) {
    sha256_update(&hmac->inner, data, len);
}

static inline void hmac_sha256_final(thread HMAC_SHA256_CTX* hmac, thread uint8_t* out32) {
    uint8_t tmp[32];
    sha256_final(&hmac->inner, tmp);
    sha256_update(&hmac->outer, tmp, 32);
    sha256_final(&hmac->outer, out32);
}

// =============================================================================
// RFC 6979 Deterministic Nonce Generation
// =============================================================================

static inline void scalar_set_b32(thread Scalar* s, thread const uint8_t* b32);
static inline void scalar_get_b32(thread uint8_t* b32, thread const Scalar* s);
static inline void scalar_reduce(thread Scalar* s);

static inline void rfc6979_generate_k(thread const uint8_t* seckey32, thread const uint8_t* msg32, thread uint8_t* nonce32) {
    uint8_t v[32], k[32];
    uint8_t keydata[64];
    HMAC_SHA256_CTX hmac;

    for (int i = 0; i < 32; i++) keydata[i] = seckey32[i];

    Scalar msg_tmp;
    scalar_set_b32(&msg_tmp, msg32);
    scalar_reduce(&msg_tmp);
    uint8_t msgmod32[32];
    scalar_get_b32(msgmod32, &msg_tmp);
    for (int i = 0; i < 32; i++) keydata[32 + i] = msgmod32[i];

    for (int i = 0; i < 32; i++) { v[i] = 0x01; k[i] = 0x00; }

    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    uint8_t zero = 0x00;
    hmac_sha256_update(&hmac, &zero, 1);
    hmac_sha256_update(&hmac, keydata, 64);
    hmac_sha256_final(&hmac, k);

    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    hmac_sha256_final(&hmac, v);

    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    uint8_t one = 0x01;
    hmac_sha256_update(&hmac, &one, 1);
    hmac_sha256_update(&hmac, keydata, 64);
    hmac_sha256_final(&hmac, k);

    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    hmac_sha256_final(&hmac, v);

    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    hmac_sha256_final(&hmac, v);

    for (int i = 0; i < 32; i++) nonce32[i] = v[i];
}

// =============================================================================
// Scalar Arithmetic (mod curve order n)
// =============================================================================

static inline void scalar_set_b32(thread Scalar* s, thread const uint8_t* b32) {
    s->limbs[3] = ((uint64_t)b32[0] << 56) | ((uint64_t)b32[1] << 48) | ((uint64_t)b32[2] << 40) | ((uint64_t)b32[3] << 32) |
                   ((uint64_t)b32[4] << 24) | ((uint64_t)b32[5] << 16) | ((uint64_t)b32[6] << 8)  | (uint64_t)b32[7];
    s->limbs[2] = ((uint64_t)b32[8] << 56) | ((uint64_t)b32[9] << 48) | ((uint64_t)b32[10] << 40) | ((uint64_t)b32[11] << 32) |
                   ((uint64_t)b32[12] << 24) | ((uint64_t)b32[13] << 16) | ((uint64_t)b32[14] << 8)  | (uint64_t)b32[15];
    s->limbs[1] = ((uint64_t)b32[16] << 56) | ((uint64_t)b32[17] << 48) | ((uint64_t)b32[18] << 40) | ((uint64_t)b32[19] << 32) |
                   ((uint64_t)b32[20] << 24) | ((uint64_t)b32[21] << 16) | ((uint64_t)b32[22] << 8)  | (uint64_t)b32[23];
    s->limbs[0] = ((uint64_t)b32[24] << 56) | ((uint64_t)b32[25] << 48) | ((uint64_t)b32[26] << 40) | ((uint64_t)b32[27] << 32) |
                   ((uint64_t)b32[28] << 24) | ((uint64_t)b32[29] << 16) | ((uint64_t)b32[30] << 8)  | (uint64_t)b32[31];
}

static inline void scalar_get_b32(thread uint8_t* b32, thread const Scalar* s) {
    b32[0]  = (uint8_t)(s->limbs[3] >> 56); b32[1]  = (uint8_t)(s->limbs[3] >> 48);
    b32[2]  = (uint8_t)(s->limbs[3] >> 40); b32[3]  = (uint8_t)(s->limbs[3] >> 32);
    b32[4]  = (uint8_t)(s->limbs[3] >> 24); b32[5]  = (uint8_t)(s->limbs[3] >> 16);
    b32[6]  = (uint8_t)(s->limbs[3] >> 8);  b32[7]  = (uint8_t)(s->limbs[3]);
    b32[8]  = (uint8_t)(s->limbs[2] >> 56); b32[9]  = (uint8_t)(s->limbs[2] >> 48);
    b32[10] = (uint8_t)(s->limbs[2] >> 40); b32[11] = (uint8_t)(s->limbs[2] >> 32);
    b32[12] = (uint8_t)(s->limbs[2] >> 24); b32[13] = (uint8_t)(s->limbs[2] >> 16);
    b32[14] = (uint8_t)(s->limbs[2] >> 8);  b32[15] = (uint8_t)(s->limbs[2]);
    b32[16] = (uint8_t)(s->limbs[1] >> 56); b32[17] = (uint8_t)(s->limbs[1] >> 48);
    b32[18] = (uint8_t)(s->limbs[1] >> 40); b32[19] = (uint8_t)(s->limbs[1] >> 32);
    b32[20] = (uint8_t)(s->limbs[1] >> 24); b32[21] = (uint8_t)(s->limbs[1] >> 16);
    b32[22] = (uint8_t)(s->limbs[1] >> 8);  b32[23] = (uint8_t)(s->limbs[1]);
    b32[24] = (uint8_t)(s->limbs[0] >> 56); b32[25] = (uint8_t)(s->limbs[0] >> 48);
    b32[26] = (uint8_t)(s->limbs[0] >> 40); b32[27] = (uint8_t)(s->limbs[0] >> 32);
    b32[28] = (uint8_t)(s->limbs[0] >> 24); b32[29] = (uint8_t)(s->limbs[0] >> 16);
    b32[30] = (uint8_t)(s->limbs[0] >> 8);  b32[31] = (uint8_t)(s->limbs[0]);
}

#define N_LIMB0 0xBFD25E8CD0364141ULL
#define N_LIMB1 0xBAAEDCE6AF48A03BULL
#define N_LIMB2 0xFFFFFFFFFFFFFFFEULL
#define N_LIMB3 0xFFFFFFFFFFFFFFFFULL

static inline int scalar_check_overflow(thread const Scalar* s) {
    if (s->limbs[3] > N_LIMB3) return 1;
    if (s->limbs[3] < N_LIMB3) return 0;
    if (s->limbs[2] > N_LIMB2) return 1;
    if (s->limbs[2] < N_LIMB2) return 0;
    if (s->limbs[1] > N_LIMB1) return 1;
    if (s->limbs[1] < N_LIMB1) return 0;
    if (s->limbs[0] >= N_LIMB0) return 1;
    return 0;
}

static inline void scalar_reduce(thread Scalar* s) {
    if (!scalar_check_overflow(s)) return;
    uint64_t borrow = 0;
    uint64_t d0 = sub_with_borrow(s->limbs[0], N_LIMB0, 0, &borrow);
    uint64_t d1 = sub_with_borrow(s->limbs[1], N_LIMB1, borrow, &borrow);
    uint64_t d2 = sub_with_borrow(s->limbs[2], N_LIMB2, borrow, &borrow);
    uint64_t d3 = sub_with_borrow(s->limbs[3], N_LIMB3, borrow, &borrow);
    s->limbs[0] = d0; s->limbs[1] = d1;
    s->limbs[2] = d2; s->limbs[3] = d3;
}

static inline void scalar_add_mod_n(thread Scalar* r, thread const Scalar* a, thread const Scalar* b) {
    uint64_t carry = 0;
    r->limbs[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    r->limbs[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    r->limbs[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    r->limbs[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);
    if (carry || scalar_check_overflow(r)) {
        uint64_t borrow = 0;
        r->limbs[0] = sub_with_borrow(r->limbs[0], N_LIMB0, 0, &borrow);
        r->limbs[1] = sub_with_borrow(r->limbs[1], N_LIMB1, borrow, &borrow);
        r->limbs[2] = sub_with_borrow(r->limbs[2], N_LIMB2, borrow, &borrow);
        r->limbs[3] = sub_with_borrow(r->limbs[3], N_LIMB3, borrow, &borrow);
    }
}

// =============================================================================
// Scalar Reduction: reduce 512-bit product mod n
// =============================================================================
// 192-bit accumulator macros — mul64_full returns via output params

#define ACC_MULADD_FAST(a, b) { \
    uint64_t _m_lo, _m_hi; \
    mul64_full((a), (b), _m_lo, _m_hi); \
    c0 += _m_lo; \
    uint64_t _th = _m_hi + ((c0 < _m_lo) ? 1ULL : 0ULL); \
    c1 += _th; \
}

#define ACC_MULADD(a, b) { \
    uint64_t _m_lo, _m_hi; \
    mul64_full((a), (b), _m_lo, _m_hi); \
    c0 += _m_lo; \
    uint64_t _th = _m_hi + ((c0 < _m_lo) ? 1ULL : 0ULL); \
    c1 += _th; \
    c2 += (c1 < _th) ? 1ULL : 0ULL; \
}

#define ACC_SUMADD_FAST(a) { \
    c0 += (a); \
    c1 += (c0 < (a)) ? 1ULL : 0ULL; \
}

#define ACC_SUMADD(a) { \
    c0 += (a); \
    uint64_t _over = (c0 < (a)) ? 1ULL : 0ULL; \
    c1 += _over; \
    c2 += (c1 < _over) ? 1ULL : 0ULL; \
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

static inline void scalar_reduce_512(thread Scalar* r, thread const uint64_t* l) {
    const uint64_t NC0 = 0x402DA1732FC9BEBFULL;
    const uint64_t NC1 = 0x4551231950B75FC4ULL;

    uint64_t n0 = l[4], n1 = l[5], n2 = l[6], n3 = l[7];
    uint64_t m0, m1, m2, m3, m4, m5;
    uint64_t m6;
    uint64_t p0, p1, p2, p3;
    uint64_t p4;

    // --- Pass 1: Reduce 512 bits into 385 ---
    uint64_t c0, c1, c2;

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
    uint64_t t_lo, t_hi;
    uint64_t carry;

    mul64_full(p4, NC0, t_lo, t_hi);
    uint64_t r0 = p0 + t_lo;
    carry = t_hi + ((r0 < p0) ? 1ULL : 0ULL);

    mul64_full(p4, NC1, t_lo, t_hi);
    uint64_t sum1 = p1 + t_lo;
    uint64_t c_1 = (sum1 < p1) ? 1ULL : 0ULL;
    uint64_t r1 = sum1 + carry;
    uint64_t c_2 = (r1 < sum1) ? 1ULL : 0ULL;
    carry = t_hi + c_1 + c_2;

    uint64_t sum2 = p2 + p4;
    uint64_t c_3 = (sum2 < p2) ? 1ULL : 0ULL;
    uint64_t r2 = sum2 + carry;
    uint64_t c_4 = (r2 < sum2) ? 1ULL : 0ULL;
    carry = c_3 + c_4;

    uint64_t r3 = p3 + carry;
    uint64_t final_carry = (r3 < p3) ? 1ULL : 0ULL;

    r->limbs[0] = r0;
    r->limbs[1] = r1;
    r->limbs[2] = r2;
    r->limbs[3] = r3;

    uint64_t red = final_carry + (uint64_t)(scalar_check_overflow(r) ? 1 : 0);
    while (red > 0) {
        uint64_t borrow = 0;
        r->limbs[0] = sub_with_borrow(r->limbs[0], N_LIMB0, 0, &borrow);
        r->limbs[1] = sub_with_borrow(r->limbs[1], N_LIMB1, borrow, &borrow);
        r->limbs[2] = sub_with_borrow(r->limbs[2], N_LIMB2, borrow, &borrow);
        r->limbs[3] = sub_with_borrow(r->limbs[3], N_LIMB3, borrow, &borrow);
        red--;
    }
}

static inline void scalar_mul_mod_n(thread Scalar* r, thread const Scalar* a, thread const Scalar* b) {
    uint64_t product[8];
    uint64_t c0, c1, c2;

    uint64_t a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    uint64_t b0 = b->limbs[0], b1 = b->limbs[1], b2 = b->limbs[2], b3 = b->limbs[3];

    c0 = 0; c1 = 0; c2 = 0;
    ACC_MULADD_FAST(a0, b0);
    ACC_EXTRACT_FAST(product[0]);

    ACC_MULADD(a0, b1);
    ACC_MULADD(a1, b0);
    ACC_EXTRACT(product[1]);

    ACC_MULADD(a0, b2);
    ACC_MULADD(a1, b1);
    ACC_MULADD(a2, b0);
    ACC_EXTRACT(product[2]);

    ACC_MULADD(a0, b3);
    ACC_MULADD(a1, b2);
    ACC_MULADD(a2, b1);
    ACC_MULADD(a3, b0);
    ACC_EXTRACT(product[3]);

    ACC_MULADD(a1, b3);
    ACC_MULADD(a2, b2);
    ACC_MULADD(a3, b1);
    ACC_EXTRACT(product[4]);

    ACC_MULADD(a2, b3);
    ACC_MULADD(a3, b2);
    ACC_EXTRACT(product[5]);

    ACC_MULADD_FAST(a3, b3);
    ACC_EXTRACT_FAST(product[6]);

    product[7] = c0;

    scalar_reduce_512(r, product);
}

static inline void scalar_sqr_mod_n(thread Scalar* r, thread const Scalar* a) {
    scalar_mul_mod_n(r, a, a);
}

// =============================================================================
// Scalar inversion mod n — safegcd (modinv64 divsteps)
// =============================================================================

struct s128 { int64_t lo; int64_t hi; };

static inline s128 s128_mul(int64_t a, int64_t b) {
    uint64_t au = (uint64_t)a, bu = (uint64_t)b;
    uint64_t lo = au * bu;
    int64_t hi = (int64_t)mulhi(au, bu);
    if (a < 0) hi -= b;
    if (b < 0) hi -= a;
    s128 r; r.lo = (int64_t)lo; r.hi = hi;
    return r;
}

static inline void s128_accum_mul(thread s128* acc, int64_t a, int64_t b) {
    s128 prod = s128_mul(a, b);
    uint64_t old_lo = (uint64_t)acc->lo;
    acc->lo += prod.lo;
    uint64_t carry_val = ((uint64_t)acc->lo < old_lo) ? 1ULL : 0ULL;
    acc->hi += prod.hi + (int64_t)carry_val;
}

static inline void s128_rshift(thread s128* r, int n) {
    r->lo = (int64_t)(((uint64_t)r->lo >> n) | ((uint64_t)r->hi << (64 - n)));
    r->hi = r->hi >> n;
}

static inline uint64_t s128_to_u64(thread const s128* a) { return (uint64_t)a->lo; }
static inline int64_t  s128_to_i64(thread const s128* a) { return a->lo; }

// --- Signed62 representation (5 limbs of 62 bits each) ---
struct Signed62 { int64_t v[5]; };

struct Trans2x2 { int64_t u, v, q, r; };

struct ModInfo {
    Signed62 modulus;
    uint64_t modulus_inv62;
};

static inline int64_t modinv64_divsteps_59(int64_t zeta, uint64_t f0, uint64_t g0, thread Trans2x2* t) {
    uint64_t u = 8, v = 0, q = 0, r = 8;
    uint64_t f = f0, g = g0, x, y, z;
    uint64_t mask1, mask2;
    int64_t c1;
    uint64_t c2;

    for (int i = 3; i < 62; ++i) {
        c1 = zeta >> 63;
        mask1 = (uint64_t)c1;
        c2 = g & 1;
        mask2 = (uint64_t)(-(int64_t)c2);
        x = (f ^ mask1) - mask1;
        y = (u ^ mask1) - mask1;
        z = (v ^ mask1) - mask1;
        g += x & mask2;
        q += y & mask2;
        r += z & mask2;
        mask1 &= mask2;
        zeta = (zeta ^ (int64_t)mask1) - 1;
        f += g & mask1;
        u += q & mask1;
        v += r & mask1;
        g >>= 1;
        u <<= 1;
        v <<= 1;
    }
    t->u = (int64_t)u;
    t->v = (int64_t)v;
    t->q = (int64_t)q;
    t->r = (int64_t)r;
    return zeta;
}

static inline void modinv64_update_de_62(thread Signed62* d, thread Signed62* e, thread const Trans2x2* t,
                                   int64_t mod0, int64_t mod1, int64_t mod2, int64_t mod3, int64_t mod4,
                                   uint64_t mod_inv62) {
    const uint64_t M62 = 0x3FFFFFFFFFFFFFFFULL;
    const int64_t d0=d->v[0], d1=d->v[1], d2=d->v[2], d3=d->v[3], d4=d->v[4];
    const int64_t e0=e->v[0], e1=e->v[1], e2=e->v[2], e3=e->v[3], e4=e->v[4];
    const int64_t u=t->u, v_=t->v, q=t->q, r_=t->r;
    int64_t md, me;
    int64_t sd = d4 >> 63;
    int64_t se = e4 >> 63;
    md = (u & sd) + (v_ & se);
    me = (q & sd) + (r_ & se);
    s128 cd = s128_mul(u, d0); s128_accum_mul(&cd, v_, e0);
    s128 ce = s128_mul(q, d0); s128_accum_mul(&ce, r_, e0);
    md -= (int64_t)((mod_inv62 * s128_to_u64(&cd) + (uint64_t)md) & M62);
    me -= (int64_t)((mod_inv62 * s128_to_u64(&ce) + (uint64_t)me) & M62);
    s128_accum_mul(&cd, mod0, md);
    s128_accum_mul(&ce, mod0, me);
    s128_rshift(&cd, 62);
    s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d1); s128_accum_mul(&cd, v_, e1);
    s128_accum_mul(&ce, q, d1); s128_accum_mul(&ce, r_, e1);
    if (mod1) { s128_accum_mul(&cd, mod1, md); s128_accum_mul(&ce, mod1, me); }
    d->v[0] = (int64_t)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[0] = (int64_t)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d2); s128_accum_mul(&cd, v_, e2);
    s128_accum_mul(&ce, q, d2); s128_accum_mul(&ce, r_, e2);
    if (mod2) { s128_accum_mul(&cd, mod2, md); s128_accum_mul(&ce, mod2, me); }
    d->v[1] = (int64_t)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[1] = (int64_t)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d3); s128_accum_mul(&cd, v_, e3);
    s128_accum_mul(&ce, q, d3); s128_accum_mul(&ce, r_, e3);
    if (mod3) { s128_accum_mul(&cd, mod3, md); s128_accum_mul(&ce, mod3, me); }
    d->v[2] = (int64_t)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[2] = (int64_t)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d4); s128_accum_mul(&cd, v_, e4);
    s128_accum_mul(&ce, q, d4); s128_accum_mul(&ce, r_, e4);
    s128_accum_mul(&cd, mod4, md);
    s128_accum_mul(&ce, mod4, me);
    d->v[3] = (int64_t)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[3] = (int64_t)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    d->v[4] = s128_to_i64(&cd);
    e->v[4] = s128_to_i64(&ce);
}

static inline void modinv64_update_fg_62(thread Signed62* f, thread Signed62* g, thread const Trans2x2* t) {
    const uint64_t M62 = 0x3FFFFFFFFFFFFFFFULL;
    const int64_t f0=f->v[0], f1=f->v[1], f2=f->v[2], f3=f->v[3], f4=f->v[4];
    const int64_t g0=g->v[0], g1=g->v[1], g2=g->v[2], g3=g->v[3], g4=g->v[4];
    const int64_t u=t->u, v_=t->v, q=t->q, r_=t->r;
    s128 cf = s128_mul(u, f0); s128_accum_mul(&cf, v_, g0);
    s128 cg = s128_mul(q, f0); s128_accum_mul(&cg, r_, g0);
    s128_rshift(&cf, 62);
    s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f1); s128_accum_mul(&cf, v_, g1);
    s128_accum_mul(&cg, q, f1); s128_accum_mul(&cg, r_, g1);
    f->v[0] = (int64_t)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[0] = (int64_t)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f2); s128_accum_mul(&cf, v_, g2);
    s128_accum_mul(&cg, q, f2); s128_accum_mul(&cg, r_, g2);
    f->v[1] = (int64_t)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[1] = (int64_t)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f3); s128_accum_mul(&cf, v_, g3);
    s128_accum_mul(&cg, q, f3); s128_accum_mul(&cg, r_, g3);
    f->v[2] = (int64_t)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[2] = (int64_t)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f4); s128_accum_mul(&cf, v_, g4);
    s128_accum_mul(&cg, q, f4); s128_accum_mul(&cg, r_, g4);
    f->v[3] = (int64_t)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[3] = (int64_t)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    f->v[4] = s128_to_i64(&cf);
    g->v[4] = s128_to_i64(&cg);
}

static inline void modinv64_normalize_62(thread Signed62* r, int64_t sign,
                                   int64_t mod0, int64_t mod1, int64_t mod2, int64_t mod3, int64_t mod4) {
    const int64_t M62 = (int64_t)0x3FFFFFFFFFFFFFFFULL;
    int64_t r0=r->v[0], r1=r->v[1], r2=r->v[2], r3=r->v[3], r4=r->v[4];
    int64_t cond_add, cond_negate;

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
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

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

static inline void scalar_inverse_mod_n(thread Scalar* r, thread const Scalar* a) {
    const uint64_t M62 = 0x3FFFFFFFFFFFFFFFULL;

    const int64_t mod0 = 0x3FD25E8CD0364141LL;
    const int64_t mod1 = 0x2ABB739ABD2280EELL;
    const int64_t mod2 = -0x15LL;
    const int64_t mod3 = 0LL;
    const int64_t mod4 = 256LL;
    const uint64_t mod_inv62 = 0x34F20099AA774EC1ULL;

    uint64_t a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    Signed62 s;
    s.v[0] = (int64_t)( a0                  & M62);
    s.v[1] = (int64_t)((a0 >> 62 | a1 << 2) & M62);
    s.v[2] = (int64_t)((a1 >> 60 | a2 << 4) & M62);
    s.v[3] = (int64_t)((a2 >> 58 | a3 << 6) & M62);
    s.v[4] = (int64_t)( a3 >> 56);

    Signed62 d; d.v[0]=0; d.v[1]=0; d.v[2]=0; d.v[3]=0; d.v[4]=0;
    Signed62 e; e.v[0]=1; e.v[1]=0; e.v[2]=0; e.v[3]=0; e.v[4]=0;
    Signed62 f; f.v[0]=mod0; f.v[1]=mod1; f.v[2]=mod2; f.v[3]=mod3; f.v[4]=mod4;
    Signed62 g = s;
    int64_t zeta = -1LL;

    for (int i = 0; i < 10; i++) {
        Trans2x2 t;
        zeta = modinv64_divsteps_59(zeta, (uint64_t)f.v[0], (uint64_t)g.v[0], &t);
        modinv64_update_de_62(&d, &e, &t, mod0, mod1, mod2, mod3, mod4, mod_inv62);
        modinv64_update_fg_62(&f, &g, &t);
    }

    modinv64_normalize_62(&d, f.v[4], mod0, mod1, mod2, mod3, mod4);

    uint64_t d0 = (uint64_t)d.v[0], d1 = (uint64_t)d.v[1], d2 = (uint64_t)d.v[2];
    uint64_t d3 = (uint64_t)d.v[3], d4 = (uint64_t)d.v[4];
    r->limbs[0] = d0      | d1 << 62;
    r->limbs[1] = d1 >> 2 | d2 << 60;
    r->limbs[2] = d2 >> 4 | d3 << 58;
    r->limbs[3] = d3 >> 6 | d4 << 56;
}

static inline void scalar_negate(thread Scalar* r, thread const Scalar* a) {
    if (scalar_is_zero(a)) {
        *r = *a;
        return;
    }
    uint64_t borrow = 0;
    r->limbs[0] = sub_with_borrow(N_LIMB0, a->limbs[0], 0, &borrow);
    r->limbs[1] = sub_with_borrow(N_LIMB1, a->limbs[1], borrow, &borrow);
    r->limbs[2] = sub_with_borrow(N_LIMB2, a->limbs[2], borrow, &borrow);
    r->limbs[3] = sub_with_borrow(N_LIMB3, a->limbs[3], borrow, &borrow);
}

static inline int scalar_is_high(thread const Scalar* s) {
    Scalar half_n;
    half_n.limbs[3] = 0x7FFFFFFFFFFFFFFFULL;
    half_n.limbs[2] = 0xFFFFFFFFFFFFFFFFULL;
    half_n.limbs[1] = 0x5D576E7357A4501DULL;
    half_n.limbs[0] = 0xDFE92F46681B20A0ULL;
    
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

static inline int der_encode_integer(thread uint8_t* out, thread const uint8_t* val32) {
    int start = 0;
    while (start < 32 && val32[start] == 0) start++;
    if (start == 32) {
        out[0] = 0x02; out[1] = 0x01; out[2] = 0x00;
        return 3;
    }
    int need_pad = (val32[start] & 0x80) ? 1 : 0;
    int len = 32 - start + need_pad;
    out[0] = 0x02;
    out[1] = (uint8_t)len;
    int pos = 2;
    if (need_pad) out[pos++] = 0x00;
    for (int i = start; i < 32; i++) out[pos++] = val32[i];
    return pos;
}

static inline int der_encode_signature(thread uint8_t* out, thread const uint8_t* r32, thread const uint8_t* s32) {
    uint8_t r_der[35], s_der[35];
    int r_len = der_encode_integer(r_der, r32);
    int s_len = der_encode_integer(s_der, s32);
    
    out[0] = 0x30;
    out[1] = (uint8_t)(r_len + s_len);
    int pos = 2;
    for (int i = 0; i < r_len; i++) out[pos++] = r_der[i];
    for (int i = 0; i < s_len; i++) out[pos++] = s_der[i];
    return pos;
}

// =============================================================================
// Precomputed Generator Table - 2-bit windowed lookup for k*G
// =============================================================================

static inline void ecmult_table_load_point(thread AffinePoint* p, device const uint64_t* table, int index) {
    int off = index * 8;
    p->x.limbs[0] = table[off + 0];
    p->x.limbs[1] = table[off + 1];
    p->x.limbs[2] = table[off + 2];
    p->x.limbs[3] = table[off + 3];
    p->y.limbs[0] = table[off + 4];
    p->y.limbs[1] = table[off + 5];
    p->y.limbs[2] = table[off + 6];
    p->y.limbs[3] = table[off + 7];
}

static inline void scalar_mul_generator_precomp(thread JacobianPoint* r, thread const Scalar* k, device const uint64_t* table) {
    point_set_infinity(r);
    
    AffinePoint ap;
    
    for (int group = 0; group < 128; group++) {
        int limb_idx = (group * 2) / 64;
        int bit_idx  = (group * 2) % 64;
        int value    = (int)((k->limbs[limb_idx] >> bit_idx) & 3ULL);
        
        if (value != 0) {
            ecmult_table_load_point(&ap, table, group * 4 + value);
            point_add_mixed_impl(r, r, &ap);
        }
    }
}

// =============================================================================
// ECDSA signing function (precomputed table k*G + optimized inverse)
// =============================================================================
static inline int ecdsa_sign(thread const uint8_t* seckey32, thread const uint8_t* msg32, thread uint8_t* sig_out,
                      device const uint64_t* ecmult_table) {
    uint8_t nonce32[32];
    rfc6979_generate_k(seckey32, msg32, nonce32);
    
    Scalar k;
    scalar_set_b32(&k, nonce32);
    if (scalar_is_zero(&k)) return 0;
    scalar_reduce(&k);
    if (scalar_is_zero(&k)) return 0;
    
    JacobianPoint R_jac;
    scalar_mul_generator_precomp(&R_jac, &k, ecmult_table);
    
    FieldElement z_inv, z_inv2, rx_fe;
    field_inv_impl(&z_inv, &R_jac.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&rx_fe, &R_jac.x, &z_inv2);
    
    uint8_t rx_bytes[32];
    rx_bytes[0]  = (uint8_t)(rx_fe.limbs[3] >> 56); rx_bytes[1]  = (uint8_t)(rx_fe.limbs[3] >> 48);
    rx_bytes[2]  = (uint8_t)(rx_fe.limbs[3] >> 40); rx_bytes[3]  = (uint8_t)(rx_fe.limbs[3] >> 32);
    rx_bytes[4]  = (uint8_t)(rx_fe.limbs[3] >> 24); rx_bytes[5]  = (uint8_t)(rx_fe.limbs[3] >> 16);
    rx_bytes[6]  = (uint8_t)(rx_fe.limbs[3] >> 8);  rx_bytes[7]  = (uint8_t)(rx_fe.limbs[3]);
    rx_bytes[8]  = (uint8_t)(rx_fe.limbs[2] >> 56); rx_bytes[9]  = (uint8_t)(rx_fe.limbs[2] >> 48);
    rx_bytes[10] = (uint8_t)(rx_fe.limbs[2] >> 40); rx_bytes[11] = (uint8_t)(rx_fe.limbs[2] >> 32);
    rx_bytes[12] = (uint8_t)(rx_fe.limbs[2] >> 24); rx_bytes[13] = (uint8_t)(rx_fe.limbs[2] >> 16);
    rx_bytes[14] = (uint8_t)(rx_fe.limbs[2] >> 8);  rx_bytes[15] = (uint8_t)(rx_fe.limbs[2]);
    rx_bytes[16] = (uint8_t)(rx_fe.limbs[1] >> 56); rx_bytes[17] = (uint8_t)(rx_fe.limbs[1] >> 48);
    rx_bytes[18] = (uint8_t)(rx_fe.limbs[1] >> 40); rx_bytes[19] = (uint8_t)(rx_fe.limbs[1] >> 32);
    rx_bytes[20] = (uint8_t)(rx_fe.limbs[1] >> 24); rx_bytes[21] = (uint8_t)(rx_fe.limbs[1] >> 16);
    rx_bytes[22] = (uint8_t)(rx_fe.limbs[1] >> 8);  rx_bytes[23] = (uint8_t)(rx_fe.limbs[1]);
    rx_bytes[24] = (uint8_t)(rx_fe.limbs[0] >> 56); rx_bytes[25] = (uint8_t)(rx_fe.limbs[0] >> 48);
    rx_bytes[26] = (uint8_t)(rx_fe.limbs[0] >> 40); rx_bytes[27] = (uint8_t)(rx_fe.limbs[0] >> 32);
    rx_bytes[28] = (uint8_t)(rx_fe.limbs[0] >> 24); rx_bytes[29] = (uint8_t)(rx_fe.limbs[0] >> 16);
    rx_bytes[30] = (uint8_t)(rx_fe.limbs[0] >> 8);  rx_bytes[31] = (uint8_t)(rx_fe.limbs[0]);
    
    Scalar sig_r;
    scalar_set_b32(&sig_r, rx_bytes);
    scalar_reduce(&sig_r);
    if (scalar_is_zero(&sig_r)) return 0;
    
    Scalar sec, msg_scalar, n_val, sig_s;
    scalar_set_b32(&sec, seckey32);
    scalar_reduce(&sec);
    scalar_set_b32(&msg_scalar, msg32);
    scalar_reduce(&msg_scalar);
    
    scalar_mul_mod_n(&n_val, &sig_r, &sec);
    scalar_add_mod_n(&n_val, &n_val, &msg_scalar);
    Scalar k_inv;
    scalar_inverse_mod_n(&k_inv, &k);
    scalar_mul_mod_n(&sig_s, &k_inv, &n_val);
    
    if (scalar_is_zero(&sig_s)) return 0;
    
    if (scalar_is_high(&sig_s)) {
        scalar_negate(&sig_s, &sig_s);
    }
    
    uint8_t r_bytes[32], s_bytes[32];
    scalar_get_b32(r_bytes, &sig_r);
    scalar_get_b32(s_bytes, &sig_s);
    
    return der_encode_signature(sig_out, r_bytes, s_bytes);
}

// =============================================================================
// uint256 Addition (for mud = hash_no_sig + nonce)
// =============================================================================

struct uint256_t {
    uint64_t limbs[4];
};

static inline void uint256_add(thread uint256_t* r, thread const uint256_t* a, thread const uint256_t* b) {
    uint64_t carry = 0;
    r->limbs[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    r->limbs[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    r->limbs[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    r->limbs[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);
}

// =============================================================================
// Mining Kernel
// =============================================================================

#define NONCES_PER_THREAD 4

kernel void btcw_mine(
    device const uint8_t* key_data [[buffer(0)]],
    device const uint8_t* hash_no_sig [[buffer(1)]],
    device uint64_t* result_nonce [[buffer(2)]],
    device atomic_uint* result_found [[buffer(3)]],
    device atomic_uint* hashrate_ctr [[buffer(4)]],
    constant uint64_t& nonce_base [[buffer(5)]],
    constant uint32_t& gpu_num [[buffer(6)]],
    device const uint64_t* ecmult_table [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint8_t seckey[32];
    for (int i = 0; i < 32; i++) seckey[i] = key_data[i];
    
    uint256_t h_no_sig;
    h_no_sig.limbs[0] = ((uint64_t)hash_no_sig[0])       | ((uint64_t)hash_no_sig[1] << 8)  |
                         ((uint64_t)hash_no_sig[2] << 16)  | ((uint64_t)hash_no_sig[3] << 24) |
                         ((uint64_t)hash_no_sig[4] << 32)  | ((uint64_t)hash_no_sig[5] << 40) |
                         ((uint64_t)hash_no_sig[6] << 48)  | ((uint64_t)hash_no_sig[7] << 56);
    h_no_sig.limbs[1] = ((uint64_t)hash_no_sig[8])        | ((uint64_t)hash_no_sig[9] << 8)  |
                         ((uint64_t)hash_no_sig[10] << 16) | ((uint64_t)hash_no_sig[11] << 24) |
                         ((uint64_t)hash_no_sig[12] << 32) | ((uint64_t)hash_no_sig[13] << 40) |
                         ((uint64_t)hash_no_sig[14] << 48) | ((uint64_t)hash_no_sig[15] << 56);
    h_no_sig.limbs[2] = ((uint64_t)hash_no_sig[16])       | ((uint64_t)hash_no_sig[17] << 8)  |
                         ((uint64_t)hash_no_sig[18] << 16) | ((uint64_t)hash_no_sig[19] << 24) |
                         ((uint64_t)hash_no_sig[20] << 32) | ((uint64_t)hash_no_sig[21] << 40) |
                         ((uint64_t)hash_no_sig[22] << 48) | ((uint64_t)hash_no_sig[23] << 56);
    h_no_sig.limbs[3] = ((uint64_t)hash_no_sig[24])       | ((uint64_t)hash_no_sig[25] << 8)  |
                         ((uint64_t)hash_no_sig[26] << 16) | ((uint64_t)hash_no_sig[27] << 24) |
                         ((uint64_t)hash_no_sig[28] << 32) | ((uint64_t)hash_no_sig[29] << 40) |
                         ((uint64_t)hash_no_sig[30] << 48) | ((uint64_t)hash_no_sig[31] << 56);
    
    uint64_t gpu_mask = ((uint64_t)(gpu_num & 0xFF) << 56);
    
    for (int iter = 0; iter < NONCES_PER_THREAD; iter++) {
        uint64_t nonce = nonce_base + (uint64_t)gid * NONCES_PER_THREAD + (uint64_t)iter;
        nonce = (nonce & 0x00FFFFFFFFFFFFFFULL) | gpu_mask;
        
        uint256_t nonce_256;
        nonce_256.limbs[0] = nonce;
        nonce_256.limbs[1] = 0;
        nonce_256.limbs[2] = 0;
        nonce_256.limbs[3] = 0;
        
        uint256_t mud;
        uint256_add(&mud, &h_no_sig, &nonce_256);
        
        uint8_t mud_bytes[32];
        mud_bytes[0]  = (uint8_t)(mud.limbs[0]);       mud_bytes[1]  = (uint8_t)(mud.limbs[0] >> 8);
        mud_bytes[2]  = (uint8_t)(mud.limbs[0] >> 16);  mud_bytes[3]  = (uint8_t)(mud.limbs[0] >> 24);
        mud_bytes[4]  = (uint8_t)(mud.limbs[0] >> 32);  mud_bytes[5]  = (uint8_t)(mud.limbs[0] >> 40);
        mud_bytes[6]  = (uint8_t)(mud.limbs[0] >> 48);  mud_bytes[7]  = (uint8_t)(mud.limbs[0] >> 56);
        mud_bytes[8]  = (uint8_t)(mud.limbs[1]);        mud_bytes[9]  = (uint8_t)(mud.limbs[1] >> 8);
        mud_bytes[10] = (uint8_t)(mud.limbs[1] >> 16);  mud_bytes[11] = (uint8_t)(mud.limbs[1] >> 24);
        mud_bytes[12] = (uint8_t)(mud.limbs[1] >> 32);  mud_bytes[13] = (uint8_t)(mud.limbs[1] >> 40);
        mud_bytes[14] = (uint8_t)(mud.limbs[1] >> 48);  mud_bytes[15] = (uint8_t)(mud.limbs[1] >> 56);
        mud_bytes[16] = (uint8_t)(mud.limbs[2]);        mud_bytes[17] = (uint8_t)(mud.limbs[2] >> 8);
        mud_bytes[18] = (uint8_t)(mud.limbs[2] >> 16);  mud_bytes[19] = (uint8_t)(mud.limbs[2] >> 24);
        mud_bytes[20] = (uint8_t)(mud.limbs[2] >> 32);  mud_bytes[21] = (uint8_t)(mud.limbs[2] >> 40);
        mud_bytes[22] = (uint8_t)(mud.limbs[2] >> 48);  mud_bytes[23] = (uint8_t)(mud.limbs[2] >> 56);
        mud_bytes[24] = (uint8_t)(mud.limbs[3]);        mud_bytes[25] = (uint8_t)(mud.limbs[3] >> 8);
        mud_bytes[26] = (uint8_t)(mud.limbs[3] >> 16);  mud_bytes[27] = (uint8_t)(mud.limbs[3] >> 24);
        mud_bytes[28] = (uint8_t)(mud.limbs[3] >> 32);  mud_bytes[29] = (uint8_t)(mud.limbs[3] >> 40);
        mud_bytes[30] = (uint8_t)(mud.limbs[3] >> 48);  mud_bytes[31] = (uint8_t)(mud.limbs[3] >> 56);
        
        uint8_t der_sig[73];
        int sig_len = ecdsa_sign(seckey, mud_bytes, der_sig, ecmult_table);
        
        if (sig_len == 0) continue;
        
        uint8_t preimage[82];
        preimage[0] = (uint8_t)(nonce);
        preimage[1] = (uint8_t)(nonce >> 8);
        preimage[2] = (uint8_t)(nonce >> 16);
        preimage[3] = (uint8_t)(nonce >> 24);
        preimage[4] = (uint8_t)(nonce >> 32);
        preimage[5] = (uint8_t)(nonce >> 40);
        preimage[6] = (uint8_t)(nonce >> 48);
        preimage[7] = (uint8_t)(nonce >> 56);
        preimage[8] = (uint8_t)sig_len;
        for (int i = 0; i < sig_len; i++) preimage[9 + i] = der_sig[i];
        
        uint8_t hashPoW[32];
        double_sha256(preimage, 9 + sig_len, hashPoW);
        
        if ((hashPoW[31] == 0) && (hashPoW[30] == 0)) {
            if ((hashPoW[29] == 0) && ((hashPoW[28] & 0xFC) == 0)) {
                *result_nonce = nonce;
                atomic_store_explicit(result_found, 1u, memory_order_relaxed);
            }
        }
    }
    
    atomic_fetch_add_explicit(hashrate_ctr, (uint32_t)NONCES_PER_THREAD, memory_order_relaxed);
}

// =============================================================================
// Precompute Kernel: builds the ecmult_gen table on-GPU at startup
// =============================================================================

static inline void write_affine_to_table(device uint64_t* table, int index,
                                   thread const JacobianPoint* jac) {
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

kernel void precompute_ecmult_gen_table(
    device uint64_t* table_out [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    
    AffinePoint G_aff;
    get_generator(&G_aff);
    JacobianPoint base;
    point_from_affine(&base, &G_aff);
    
    for (int group = 0; group < 128; group++) {
        int base_idx = group * 4;
        
        for (int i = 0; i < 8; i++) table_out[base_idx * 8 + i] = 0;
        
        write_affine_to_table(table_out, base_idx + 1, &base);
        
        JacobianPoint p2;
        point_double_impl(&p2, &base);
        write_affine_to_table(table_out, base_idx + 2, &p2);
        
        JacobianPoint p3;
        point_add_impl(&p3, &base, &p2);
        write_affine_to_table(table_out, base_idx + 3, &p3);
        
        JacobianPoint tmp;
        point_double_impl(&tmp, &base);
        point_double_impl(&base, &tmp);
    }
}

// =============================================================================
// Diagnostic Kernel: verify ECDSA signing on GPU
// =============================================================================
kernel void diagnostic_ecdsa_sign(
    device const uint8_t*  seckey32    [[buffer(0)]],
    device const uint8_t*  msg32       [[buffer(1)]],
    device uint8_t*        sig_out     [[buffer(2)]],
    device int*            sig_len_out [[buffer(3)]],
    device const uint64_t* ecmult_table [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint8_t sk[32], m[32];
    for (int i = 0; i < 32; i++) { sk[i] = seckey32[i]; m[i] = msg32[i]; }

    uint8_t der[73];
    int len = ecdsa_sign(sk, m, der, ecmult_table);

    *sig_len_out = len;
    for (int i = 0; i < len; i++) sig_out[i] = der[i];
}

// =============================================================================
// Diagnostic Kernel: test scalar arithmetic in isolation
// =============================================================================
kernel void diagnostic_scalar_ops(
    device uint8_t* output [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    Scalar a, b, r;
    uint8_t flags = 0;

    #define WRITE_SCALAR(off, s) \
        for (int _i = 0; _i < 4; _i++) \
            for (int _j = 0; _j < 8; _j++) \
                output[(off) + _i * 8 + _j] = (uint8_t)((s).limbs[_i] >> (_j * 8));

    a.limbs[0] = 3; a.limbs[1] = 0; a.limbs[2] = 0; a.limbs[3] = 0;
    b.limbs[0] = 7; b.limbs[1] = 0; b.limbs[2] = 0; b.limbs[3] = 0;
    scalar_mul_mod_n(&r, &a, &b);
    WRITE_SCALAR(0, r);
    if (r.limbs[0] == 21 && r.limbs[1] == 0 && r.limbs[2] == 0 && r.limbs[3] == 0)
        flags |= 0x01;

    a.limbs[0] = 2; a.limbs[1] = 0; a.limbs[2] = 0; a.limbs[3] = 0;
    scalar_inverse_mod_n(&r, &a);
    WRITE_SCALAR(32, r);
    if (r.limbs[0] == 0xDFE92F46681B20A1ULL &&
        r.limbs[1] == 0x5D576E7357A4501DULL &&
        r.limbs[2] == 0xFFFFFFFFFFFFFFFFULL &&
        r.limbs[3] == 0x7FFFFFFFFFFFFFFFULL)
        flags |= 0x02;

    Scalar two;
    two.limbs[0] = 2; two.limbs[1] = 0; two.limbs[2] = 0; two.limbs[3] = 0;
    Scalar identity;
    scalar_mul_mod_n(&identity, &two, &r);
    WRITE_SCALAR(64, identity);
    if (identity.limbs[0] == 1 && identity.limbs[1] == 0 &&
        identity.limbs[2] == 0 && identity.limbs[3] == 0)
        flags |= 0x04;

    a.limbs[0] = 0xBFD25E8CD0364140ULL;
    a.limbs[1] = 0xBAAEDCE6AF48A03BULL;
    a.limbs[2] = 0xFFFFFFFFFFFFFFFEULL;
    a.limbs[3] = 0xFFFFFFFFFFFFFFFFULL;
    scalar_mul_mod_n(&r, &a, &a);
    WRITE_SCALAR(96, r);
    if (r.limbs[0] == 1 && r.limbs[1] == 0 && r.limbs[2] == 0 && r.limbs[3] == 0)
        flags |= 0x08;

    a.limbs[0] = 2; a.limbs[1] = 0; a.limbs[2] = 0; a.limbs[3] = 0;
    for (int _sq = 0; _sq < 8; _sq++) {
        scalar_mul_mod_n(&r, &a, &a);
        a = r;
    }
    for (int _i = 0; _i < 4; _i++)
        for (int _j = 0; _j < 8; _j++)
            output[129 + _i * 8 + _j] = (uint8_t)(a.limbs[_i] >> (_j * 8));
    if (a.limbs[0] == 0x402DA1732FC9BEBFULL &&
        a.limbs[1] == 0x4551231950B75FC4ULL &&
        a.limbs[2] == 1 && a.limbs[3] == 0)
        flags |= 0x10;

    a.limbs[0] = 0; a.limbs[1] = 0; a.limbs[2] = 1; a.limbs[3] = 0;
    scalar_mul_mod_n(&r, &a, &a);
    for (int _i = 0; _i < 4; _i++)
        for (int _j = 0; _j < 8; _j++)
            output[161 + _i * 8 + _j] = (uint8_t)(r.limbs[_i] >> (_j * 8));
    if (r.limbs[0] == 0x402DA1732FC9BEBFULL &&
        r.limbs[1] == 0x4551231950B75FC4ULL &&
        r.limbs[2] == 1 && r.limbs[3] == 0)
        flags |= 0x20;

    output[128] = flags;
    #undef WRITE_SCALAR
}
