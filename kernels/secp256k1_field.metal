#ifndef SECP256K1_FIELD_METAL_H
#define SECP256K1_FIELD_METAL_H

#include <metal_stdlib>
#include <metal_integer>
#include <metal_atomic>
using namespace metal;

// =============================================================================
// secp256k1 Metal - Field Arithmetic
// =============================================================================
// Copyright (c) 2026 btcw.space <btcw.space@proton.me>
//
// secp256k1 field: F_p where p = 2^256 - 2^32 - 977
// Little-endian 256-bit integers using 4x64-bit limbs
//
// Copyright (c) 2026 btcw.space. All rights reserved.
// =============================================================================

// Field prime p = 2^256 - 0x1000003D1
// In 64-bit limbs (little-endian):
// p = {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}

// Constants
#define SECP256K1_P0 0xFFFFFFFEFFFFFC2FULL
#define SECP256K1_P1 0xFFFFFFFFFFFFFFFFULL
#define SECP256K1_P2 0xFFFFFFFFFFFFFFFFULL
#define SECP256K1_P3 0xFFFFFFFFFFFFFFFFULL

// K = 2^32 + 977 = 0x1000003D1 (for fast reduction)
#define SECP256K1_K 0x1000003D1ULL

// =============================================================================
// 64-bit Multiplication Helpers
// =============================================================================

static inline void mul64_full(uint64_t a, uint64_t b, thread uint64_t& lo, thread uint64_t& hi) {
    lo = a * b;
    hi = mulhi(a, b);
}

// Add with carry: result = a + b + carry_in, returns new carry
static inline uint64_t add_with_carry(uint64_t a, uint64_t b, uint64_t carry_in, thread uint64_t* carry_out) {
    uint64_t sum = a + b;
    uint64_t c1 = (sum < a) ? (uint64_t)1 : (uint64_t)0;
    sum += carry_in;
    uint64_t c2 = (sum < carry_in) ? (uint64_t)1 : (uint64_t)0;
    *carry_out = c1 + c2;
    return sum;
}

// Subtract with borrow: result = a - b - borrow_in, returns new borrow
static inline uint64_t sub_with_borrow(uint64_t a, uint64_t b, uint64_t borrow_in, thread uint64_t* borrow_out) {
    uint64_t diff = a - b;
    uint64_t b1 = (a < b) ? (uint64_t)1 : (uint64_t)0;
    uint64_t temp = diff;
    diff -= borrow_in;
    uint64_t b2 = (temp < borrow_in) ? (uint64_t)1 : (uint64_t)0;
    *borrow_out = b1 + b2;
    return diff;
}

// =============================================================================
// Field Element Type (256-bit)
// =============================================================================

struct FieldElement {
    uint64_t limbs[4];  // Little-endian: limbs[0] is LSB
};

// =============================================================================
// Field Reduction: r = a mod p
// Uses the fact that p = 2^256 - K where K = 0x1000003D1
// So 2^256 ≡ K (mod p), meaning we can reduce by replacing high bits with K*high
// =============================================================================

static inline void field_reduce(thread FieldElement* r, thread const uint64_t* a8) {
    // a8 is 512-bit number (8 limbs), reduce to 256-bit mod p
    // Since p = 2^256 - K, we have: a mod p = a_low + K * a_high (mod p)

    uint64_t carry = 0;
    uint64_t temp[5];

    // First reduction: fold a[4..7] into a[0..3] using K
    // temp = a[0..3] + K * a[4..7]

    uint64_t prod_lo, prod_hi;

    // limb 0: a[0] + K * a[4]
    mul64_full(SECP256K1_K, a8[4], prod_lo, prod_hi);
    temp[0] = a8[0] + prod_lo;
    carry = (temp[0] < a8[0]) ? (uint64_t)1 : (uint64_t)0;
    carry += prod_hi;

    // limb 1: a[1] + K * a[5] + carry
    mul64_full(SECP256K1_K, a8[5], prod_lo, prod_hi);
    temp[1] = a8[1] + carry;
    uint64_t c1 = (temp[1] < carry) ? (uint64_t)1 : (uint64_t)0;
    temp[1] += prod_lo;
    c1 += (temp[1] < prod_lo) ? (uint64_t)1 : (uint64_t)0;
    carry = c1 + prod_hi;

    // limb 2: a[2] + K * a[6] + carry
    mul64_full(SECP256K1_K, a8[6], prod_lo, prod_hi);
    temp[2] = a8[2] + carry;
    c1 = (temp[2] < carry) ? (uint64_t)1 : (uint64_t)0;
    temp[2] += prod_lo;
    c1 += (temp[2] < prod_lo) ? (uint64_t)1 : (uint64_t)0;
    carry = c1 + prod_hi;

    // limb 3: a[3] + K * a[7] + carry
    mul64_full(SECP256K1_K, a8[7], prod_lo, prod_hi);
    temp[3] = a8[3] + carry;
    c1 = (temp[3] < carry) ? (uint64_t)1 : (uint64_t)0;
    temp[3] += prod_lo;
    c1 += (temp[3] < prod_lo) ? (uint64_t)1 : (uint64_t)0;
    temp[4] = c1 + prod_hi;

    // Second reduction: if temp[4] > 0, fold it in
    if (temp[4] != 0) {
        mul64_full(SECP256K1_K, temp[4], prod_lo, prod_hi);
        temp[0] += prod_lo;
        carry = (temp[0] < prod_lo) ? (uint64_t)1 : (uint64_t)0;
        carry += prod_hi;

        temp[1] += carry;
        carry = (temp[1] < carry) ? (uint64_t)1 : (uint64_t)0;

        temp[2] += carry;
        carry = (temp[2] < carry) ? (uint64_t)1 : (uint64_t)0;

        temp[3] += carry;
    }

    // Final reduction: if result >= p, subtract p
    uint64_t borrow = 0;
    uint64_t diff[4];

    diff[0] = sub_with_borrow(temp[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(temp[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(temp[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(temp[3], SECP256K1_P3, borrow, &borrow);

    // Branchless selection
    uint64_t mask = (borrow == 0) ? ~(uint64_t)0 : (uint64_t)0;

    r->limbs[0] = (diff[0] & mask) | (temp[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (temp[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (temp[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (temp[3] & ~mask);
}

// =============================================================================
// Field Addition: r = (a + b) mod p
// =============================================================================

static inline void field_add_impl(thread FieldElement* r, thread const FieldElement* a, thread const FieldElement* b) {
    uint64_t carry = 0;
    uint64_t sum[4];

    // Add with carry chain
    sum[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    sum[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    sum[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    sum[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);

    // Reduce: if carry or sum >= p, subtract p
    uint64_t borrow = 0;
    uint64_t diff[4];

    diff[0] = sub_with_borrow(sum[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(sum[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(sum[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(sum[3], SECP256K1_P3, borrow, &borrow);

    // If carry from addition or no borrow from subtraction, use diff
    uint64_t use_diff = (carry != 0) | (borrow == 0);
    uint64_t mask = use_diff ? ~(uint64_t)0 : (uint64_t)0;

    r->limbs[0] = (diff[0] & mask) | (sum[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (sum[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (sum[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (sum[3] & ~mask);
}

// =============================================================================
// Field Subtraction: r = (a - b) mod p
// =============================================================================

static inline void field_sub_impl(thread FieldElement* r, thread const FieldElement* a, thread const FieldElement* b) {
    uint64_t borrow = 0;
    uint64_t diff[4];

    // Subtract with borrow chain
    diff[0] = sub_with_borrow(a->limbs[0], b->limbs[0], 0, &borrow);
    diff[1] = sub_with_borrow(a->limbs[1], b->limbs[1], borrow, &borrow);
    diff[2] = sub_with_borrow(a->limbs[2], b->limbs[2], borrow, &borrow);
    diff[3] = sub_with_borrow(a->limbs[3], b->limbs[3], borrow, &borrow);

    // If borrow, add p (result was negative)
    uint64_t mask = borrow ? ~(uint64_t)0 : (uint64_t)0;

    uint64_t carry = 0;
    uint64_t adj[4];
    adj[0] = add_with_carry(diff[0], SECP256K1_P0 & mask, 0, &carry);
    adj[1] = add_with_carry(diff[1], SECP256K1_P1 & mask, carry, &carry);
    adj[2] = add_with_carry(diff[2], SECP256K1_P2 & mask, carry, &carry);
    adj[3] = add_with_carry(diff[3], SECP256K1_P3 & mask, carry, &carry);

    r->limbs[0] = adj[0];
    r->limbs[1] = adj[1];
    r->limbs[2] = adj[2];
    r->limbs[3] = adj[3];
}

// =============================================================================
// Field Multiplication: r = (a * b) mod p
// =============================================================================

static inline void field_mul_impl(thread FieldElement* r, thread const FieldElement* a, thread const FieldElement* b) {
    // Fully unrolled 4x4 schoolbook multiplication
    uint64_t a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    uint64_t b0 = b->limbs[0], b1 = b->limbs[1], b2 = b->limbs[2], b3 = b->limbs[3];
    uint64_t product[8];
    uint64_t carry;

    // Row 0: a0 * b[0..3]
    uint64_t m_lo, m_hi;
    mul64_full(a0, b0, m_lo, m_hi);
    product[0] = m_lo; carry = m_hi;

    mul64_full(a0, b1, m_lo, m_hi);
    product[1] = m_lo + carry;
    carry = m_hi + (product[1] < m_lo ? (uint64_t)1 : (uint64_t)0);

    mul64_full(a0, b2, m_lo, m_hi);
    product[2] = m_lo + carry;
    carry = m_hi + (product[2] < m_lo ? (uint64_t)1 : (uint64_t)0);

    mul64_full(a0, b3, m_lo, m_hi);
    product[3] = m_lo + carry;
    carry = m_hi + (product[3] < m_lo ? (uint64_t)1 : (uint64_t)0);
    product[4] = carry;

    // Row 1: a1 * b[0..3]
    uint64_t t;
    mul64_full(a1, b0, m_lo, m_hi);
    t = product[1] + m_lo;
    carry = m_hi + (t < product[1] ? (uint64_t)1 : (uint64_t)0);
    product[1] = t;

    mul64_full(a1, b1, m_lo, m_hi);
    { uint64_t s1 = product[2] + m_lo;
      uint64_t c1 = (s1 < product[2]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t c2 = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = m_hi + c1 + c2; }
    product[2] = t;

    mul64_full(a1, b2, m_lo, m_hi);
    { uint64_t s1 = product[3] + m_lo;
      uint64_t c1 = (s1 < product[3]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t c2 = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = m_hi + c1 + c2; }
    product[3] = t;

    mul64_full(a1, b3, m_lo, m_hi);
    { uint64_t s1 = product[4] + m_lo;
      uint64_t c1 = (s1 < product[4]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t c2 = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = m_hi + c1 + c2; }
    product[4] = t;
    product[5] = carry;

    // Row 2: a2 * b[0..3]
    mul64_full(a2, b0, m_lo, m_hi);
    t = product[2] + m_lo;
    carry = m_hi + (t < product[2] ? (uint64_t)1 : (uint64_t)0);
    product[2] = t;

    mul64_full(a2, b1, m_lo, m_hi);
    { uint64_t s1 = product[3] + m_lo;
      uint64_t c1 = (s1 < product[3]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t c2 = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = m_hi + c1 + c2; }
    product[3] = t;

    mul64_full(a2, b2, m_lo, m_hi);
    { uint64_t s1 = product[4] + m_lo;
      uint64_t c1 = (s1 < product[4]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t c2 = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = m_hi + c1 + c2; }
    product[4] = t;

    mul64_full(a2, b3, m_lo, m_hi);
    { uint64_t s1 = product[5] + m_lo;
      uint64_t c1 = (s1 < product[5]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t c2 = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = m_hi + c1 + c2; }
    product[5] = t;
    product[6] = carry;

    // Row 3: a3 * b[0..3]
    mul64_full(a3, b0, m_lo, m_hi);
    t = product[3] + m_lo;
    carry = m_hi + (t < product[3] ? (uint64_t)1 : (uint64_t)0);
    product[3] = t;

    mul64_full(a3, b1, m_lo, m_hi);
    { uint64_t s1 = product[4] + m_lo;
      uint64_t c1 = (s1 < product[4]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t c2 = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = m_hi + c1 + c2; }
    product[4] = t;

    mul64_full(a3, b2, m_lo, m_hi);
    { uint64_t s1 = product[5] + m_lo;
      uint64_t c1 = (s1 < product[5]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t c2 = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = m_hi + c1 + c2; }
    product[5] = t;

    mul64_full(a3, b3, m_lo, m_hi);
    { uint64_t s1 = product[6] + m_lo;
      uint64_t c1 = (s1 < product[6]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t c2 = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = m_hi + c1 + c2; }
    product[6] = t;
    product[7] = carry;

    field_reduce(r, product);
}

// =============================================================================
// Field Squaring: r = a^2 mod p
// Optimized: only need upper triangle of multiplication
// =============================================================================

// Forward declaration for field_sqr_n_impl
static inline void field_sqr_impl(thread FieldElement* r, thread const FieldElement* a);

// Repeated squaring helper: r = r^(2^n) — in-place
static inline void field_sqr_n_impl(thread FieldElement* r, int n) {
    for (int i = 0; i < n; i++) {
        FieldElement tmp = *r;
        field_sqr_impl(r, &tmp);
    }
}

static inline void field_sqr_impl(thread FieldElement* r, thread const FieldElement* a) {
    // Fully unrolled squaring: exploits a[i]*a[j] == a[j]*a[i]
    uint64_t a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    uint64_t product[8];
    uint64_t carry;
    uint64_t t, c1, c2, c3;

    // -- Off-diagonal products (each appears twice) --
    uint64_t od01_lo, od01_hi;
    mul64_full(a0, a1, od01_lo, od01_hi);
    uint64_t od02_lo, od02_hi;
    mul64_full(a0, a2, od02_lo, od02_hi);
    uint64_t od03_lo, od03_hi;
    mul64_full(a0, a3, od03_lo, od03_hi);
    uint64_t od12_lo, od12_hi;
    mul64_full(a1, a2, od12_lo, od12_hi);
    uint64_t od13_lo, od13_hi;
    mul64_full(a1, a3, od13_lo, od13_hi);
    uint64_t od23_lo, od23_hi;
    mul64_full(a2, a3, od23_lo, od23_hi);

    // Accumulate off-diagonal into product[1..6]
    product[1] = od01_lo;

    product[2] = od02_lo + od01_hi;
    carry = (product[2] < od02_lo) ? (uint64_t)1 : (uint64_t)0;

    t = od03_lo + od02_hi;
    c1 = (t < od03_lo) ? (uint64_t)1 : (uint64_t)0;
    t += od12_lo;
    c2 = (t < od12_lo) ? (uint64_t)1 : (uint64_t)0;
    t += carry;
    c3 = (t < carry) ? (uint64_t)1 : (uint64_t)0;
    product[3] = t;
    carry = c1 + c2 + c3;

    t = od03_hi + od12_hi;
    c1 = (t < od03_hi) ? (uint64_t)1 : (uint64_t)0;
    t += od13_lo;
    c2 = (t < od13_lo) ? (uint64_t)1 : (uint64_t)0;
    t += carry;
    c3 = (t < carry) ? (uint64_t)1 : (uint64_t)0;
    product[4] = t;
    carry = c1 + c2 + c3;

    t = od13_hi + od23_lo;
    c1 = (t < od13_hi) ? (uint64_t)1 : (uint64_t)0;
    t += carry;
    c2 = (t < carry) ? (uint64_t)1 : (uint64_t)0;
    product[5] = t;
    carry = c1 + c2;

    product[6] = od23_hi + carry;

    // Double off-diagonal terms
    product[7] = product[6] >> 63;
    product[6] = (product[6] << 1) | (product[5] >> 63);
    product[5] = (product[5] << 1) | (product[4] >> 63);
    product[4] = (product[4] << 1) | (product[3] >> 63);
    product[3] = (product[3] << 1) | (product[2] >> 63);
    product[2] = (product[2] << 1) | (product[1] >> 63);
    product[1] = (product[1] << 1);
    product[0] = 0;

    // Add diagonal terms (a[i]^2)
    uint64_t m_lo, m_hi;
    mul64_full(a0, a0, m_lo, m_hi);
    product[0] = m_lo;
    t = product[1] + m_hi;
    carry = (t < product[1]) ? (uint64_t)1 : (uint64_t)0;
    product[1] = t;

    mul64_full(a1, a1, m_lo, m_hi);
    { uint64_t s1 = product[2] + m_lo;
      uint64_t ca = (s1 < product[2]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t cb = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = ca + cb; }
    product[2] = t;
    { uint64_t s1 = product[3] + m_hi;
      uint64_t ca = (s1 < product[3]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t cb = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = ca + cb; }
    product[3] = t;

    mul64_full(a2, a2, m_lo, m_hi);
    { uint64_t s1 = product[4] + m_lo;
      uint64_t ca = (s1 < product[4]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t cb = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = ca + cb; }
    product[4] = t;
    { uint64_t s1 = product[5] + m_hi;
      uint64_t ca = (s1 < product[5]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t cb = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = ca + cb; }
    product[5] = t;

    mul64_full(a3, a3, m_lo, m_hi);
    { uint64_t s1 = product[6] + m_lo;
      uint64_t ca = (s1 < product[6]) ? (uint64_t)1 : (uint64_t)0;
      t = s1 + carry;
      uint64_t cb = (t < s1) ? (uint64_t)1 : (uint64_t)0;
      carry = ca + cb; }
    product[6] = t;
    product[7] += m_hi + carry;

    field_reduce(r, product);
}

// =============================================================================
// Field Negation: r = -a mod p = p - a
// =============================================================================

static inline void field_neg_impl(thread FieldElement* r, thread const FieldElement* a) {
    // Check if a is zero
    uint64_t is_zero = ((a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3]) == 0) ? (uint64_t)1 : (uint64_t)0;

    uint64_t borrow = 0;
    r->limbs[0] = sub_with_borrow(SECP256K1_P0, a->limbs[0], 0, &borrow);
    r->limbs[1] = sub_with_borrow(SECP256K1_P1, a->limbs[1], borrow, &borrow);
    r->limbs[2] = sub_with_borrow(SECP256K1_P2, a->limbs[2], borrow, &borrow);
    r->limbs[3] = sub_with_borrow(SECP256K1_P3, a->limbs[3], borrow, &borrow);

    // If a was zero, result should be zero
    uint64_t mask = is_zero ? (uint64_t)0 : ~(uint64_t)0;
    r->limbs[0] &= mask;
    r->limbs[1] &= mask;
    r->limbs[2] &= mask;
    r->limbs[3] &= mask;
}

// =============================================================================
// Field Inversion: r = a^(-1) mod p
// Using Fermat's little theorem with optimized addition chain
// p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// =============================================================================

static inline void field_inv_impl(thread FieldElement* r, thread const FieldElement* a) {
    FieldElement x2, x3, x6, x12, x24, x48, x96, x192, x7, x31, x223;
    FieldElement x5, x11, x22;
    FieldElement t;

    // 1. x2 = a^2 * a  (2 consecutive ones)
    field_sqr_impl(&x2, a);
    field_mul_impl(&x2, &x2, a);

    // 2. x3 = x2^2 * a  (3 consecutive ones)
    field_sqr_impl(&x3, &x2);
    field_mul_impl(&x3, &x3, a);

    // 3. x6 = x3^(2^3) * x3  (6 consecutive ones)
    field_sqr_impl(&x6, &x3);
    field_sqr_n_impl(&x6, 2);
    field_mul_impl(&x6, &x6, &x3);

    // 4. x12 = x6^(2^6) * x6  (12 consecutive ones)
    t = x6;
    field_sqr_n_impl(&t, 6);
    field_mul_impl(&x12, &t, &x6);

    // 5. x24 = x12^(2^12) * x12  (24 consecutive ones)
    t = x12;
    field_sqr_n_impl(&t, 12);
    field_mul_impl(&x24, &t, &x12);

    // 6. x48 = x24^(2^24) * x24  (48 consecutive ones)
    t = x24;
    field_sqr_n_impl(&t, 24);
    field_mul_impl(&x48, &t, &x24);

    // 7. x96 = x48^(2^48) * x48  (96 consecutive ones)
    t = x48;
    field_sqr_n_impl(&t, 48);
    field_mul_impl(&x96, &t, &x48);

    // 8. x192 = x96^(2^96) * x96  (192 consecutive ones)
    t = x96;
    field_sqr_n_impl(&t, 96);
    field_mul_impl(&x192, &t, &x96);

    // 9. x7 = x6^2 * a  (7 consecutive ones)
    field_sqr_impl(&x7, &x6);
    field_mul_impl(&x7, &x7, a);

    // 10. x31 = x24^(2^7) * x7  (31 consecutive ones)
    t = x24;
    field_sqr_n_impl(&t, 7);
    field_mul_impl(&x31, &t, &x7);

    // 11. x223 = x192^(2^31) * x31  (223 consecutive ones)
    t = x192;
    field_sqr_n_impl(&t, 31);
    field_mul_impl(&x223, &t, &x31);

    // 12. x5 = x3^(2^2) * x2  (5 consecutive ones)
    t = x3;
    field_sqr_n_impl(&t, 2);
    field_mul_impl(&x5, &t, &x2);

    // 13. x11 = x6^(2^5) * x5  (11 consecutive ones)
    t = x6;
    field_sqr_n_impl(&t, 5);
    field_mul_impl(&x11, &t, &x5);

    // 14. x22 = x11^(2^11) * x11  (22 consecutive ones)
    t = x11;
    field_sqr_n_impl(&t, 11);
    field_mul_impl(&x22, &t, &x11);

    // 15. t = x223^2  (bit 32 is 0)
    field_sqr_impl(&t, &x223);

    // 16. t = t^(2^22) * x22  (append 22 ones)
    field_sqr_n_impl(&t, 22);
    field_mul_impl(&t, &t, &x22);

    // 17. t = t^(2^4)  (bits 9,8,7,6 are 0)
    field_sqr_n_impl(&t, 4);

    // 18. Process remaining 6 bits: 101101
    // bit 5: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);
    // bit 4: 0
    field_sqr_impl(&t, &t);
    // bit 3: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);
    // bit 2: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);
    // bit 1: 0
    field_sqr_impl(&t, &t);
    // bit 0: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(r, &t, a);
}

#endif // SECP256K1_FIELD_METAL_H
