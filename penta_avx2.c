/*
 * penta_avx2.c — PentaNet AVX2 Zero-Multiplier CPU Kernel
 * =========================================================
 *
 * Branchless pentanary matmul using AVX2 blend/select instructions.
 * No _mm256_mul_ps in the inner loop — only FADD, FSUB, and BLENDV.
 *
 * Weight encoding: {-2,-1,0,+1,+2} stored as int8 after 3-bit unpack.
 *
 * Inner loop logic (8 floats at a time):
 *   x2      = x + x                              (FADD  — no FMUL)
 *   contrib = blendv(x, x2,    mask_mag2)        (select ±2 → x+x)
 *   contrib = blendv(contrib, -contrib, mask_neg) (apply sign)
 *   contrib = blendv(0, contrib, mask_nonzero)    (zero out w=0)
 *   acc    += contrib                             (FADD)
 *
 * The only multiply is the final scale: out *= scale (one _mm256_mul_ps
 * per 8 output elements, amortized over K).
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o penta_avx2.so penta_avx2.c
 */

#include <immintrin.h>   /* AVX2 intrinsics */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif


/* ── 3-bit unpack ────────────────────────────────────────────────────────── */

/*
 * Unpack 3-bit packed int32 tensor → int8 tensor.
 * packed : (N, K_packs) int32,  values stored as actual+2 in bits [29:0]
 * out    : (N, K_orig)  int8,   values in {-2,-1,0,+1,+2}
 */
void penta_unpack_3bit(
    const int32_t* packed,
    int8_t*        out,
    int            N,
    int            K_packs,
    int            K_orig
) {
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < K_packs; g++) {
            int32_t p = packed[n * K_packs + g];
            for (int i = 0; i < 10; i++) {
                int k = g * 10 + i;
                if (k < K_orig) {
                    out[n * K_orig + k] = (int8_t)(((p >> (3 * i)) & 7) - 2);
                }
            }
        }
    }
}


/* ── Horizontal sum of __m256 ────────────────────────────────────────────── */

static inline float hsum256(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);          /* [1,1,3,3] */
    sum = _mm_add_ps(sum, shuf);                  /* [0+1, _, 2+3, _] */
    shuf = _mm_movehl_ps(shuf, sum);              /* [2+3, ...] */
    sum  = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}


/* ── AVX2 zero-multiplier matmul ─────────────────────────────────────────── */

/*
 * out[m, n] = (sum_k x[m,k] * w[n,k]) * scale
 *
 * x   : (M, K) float32
 * w   : (N, K) int8, values in {-2,-1,0,+1,+2}
 * out : (M, N) float32
 *
 * No floating-point multiply between x and w in the inner loop.
 */
void penta_matmul_avx2(
    const float*   x,
    const int8_t*  w,
    float*         out,
    int            M,
    int            N,
    int            K,
    float          scale
) {
    const __m256i zero_i = _mm256_setzero_si256();
    const __m256i two_i  = _mm256_set1_epi32(2);
    const __m256  zero_f = _mm256_setzero_ps();
    const __m256  scale_v = _mm256_set1_ps(scale);

    #pragma omp parallel for schedule(static) collapse(2)
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {

            __m256 acc = _mm256_setzero_ps();
            const float*  xrow = x + m * K;
            const int8_t* wrow = w + n * K;

            int k = 0;

            /* ── Main loop: 8 floats at a time ─────────────────────────── */
            for (; k <= K - 8; k += 8) {

                /* Load 8 float activations */
                __m256 xv = _mm256_loadu_ps(xrow + k);

                /* Load 8 int8 weights → sign-extend to int32 */
                __m128i w8  = _mm_loadl_epi64((const __m128i*)(wrow + k));
                __m256i w32 = _mm256_cvtepi8_epi32(w8);

                /* ── Masks (int32: all-1s or all-0s per lane) ─────────── */
                __m256i abs_w    = _mm256_abs_epi32(w32);
                /* nonzero: |w| > 0 */
                __m256i mask_nz  = _mm256_cmpgt_epi32(abs_w, zero_i);
                /* neg: w < 0  ↔  0 > w */
                __m256i mask_neg = _mm256_cmpgt_epi32(zero_i, w32);
                /* mag2: |w| == 2 */
                __m256i mask_m2  = _mm256_cmpeq_epi32(abs_w, two_i);

                /* Cast int masks to float masks (MSB is what blendv uses) */
                __m256 fmask_nz  = _mm256_castsi256_ps(mask_nz);
                __m256 fmask_neg = _mm256_castsi256_ps(mask_neg);
                __m256 fmask_m2  = _mm256_castsi256_ps(mask_m2);

                /* ── Zero-multiplier contribution ──────────────────────── */
                /* ×2 via FADD — no FMUL */
                __m256 xv2 = _mm256_add_ps(xv, xv);

                /* Select magnitude: x+x if |w|==2, else x */
                __m256 contrib = _mm256_blendv_ps(xv, xv2, fmask_m2);

                /* Apply sign: negate if w < 0 */
                __m256 neg_c = _mm256_sub_ps(zero_f, contrib);
                contrib = _mm256_blendv_ps(contrib, neg_c, fmask_neg);

                /* Zero out if w == 0 */
                contrib = _mm256_blendv_ps(zero_f, contrib, fmask_nz);

                /* Accumulate */
                acc = _mm256_add_ps(acc, contrib);
            }

            /* ── Horizontal reduce ─────────────────────────────────────── */
            float acc_s = hsum256(acc);

            /* ── Tail: remaining elements (K not multiple of 8) ────────── */
            for (; k < K; k++) {
                int8_t wk = wrow[k];
                float  xk = xrow[k];
                if      (wk ==  1) acc_s += xk;
                else if (wk == -1) acc_s -= xk;
                else if (wk ==  2) acc_s += xk + xk;   /* FADD, no FMUL */
                else if (wk == -2) acc_s -= xk + xk;
                /* wk == 0: skip */
            }

            /* Single scale multiply per output element (unavoidable) */
            out[m * N + n] = acc_s * scale;
        }
    }
}
