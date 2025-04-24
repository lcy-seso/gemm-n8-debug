#include "tl_templates/cuda/copy.h"
#include "tl_templates/cuda/debug.h"
#include "tl_templates/cuda/gemm_sm89.h"
#include "tl_templates/cuda/ldsm.h"
#include "tl_templates/cuda/reduce.h"
#include "tl_templates/cuda/threadblock_swizzle.h"

extern "C" __global__ void main_kernel(half_t* __restrict__ A,
                                       half_t* __restrict__ B,
                                       half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(64, 1)
    main_kernel(half_t* __restrict__ A, half_t* __restrict__ B,
                half_t* __restrict__ C) {
    float C_local[4];
    __shared__ half_t A_shared[1024];
    __shared__ half_t B_shared[1024];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        *(float2*)(C_local + (i * 2)) =
            make_float2(0.000000e+00f, 0.000000e+00f);
    }
#pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
        *(uint4*)(A_shared +
                  (((((i_1 * 512) + ((((int)threadIdx.x) >> 3) * 64)) +
                     ((((((int)threadIdx.x) >> 5) +
                        ((((int)threadIdx.x) & 7) >> 2)) &
                       1) *
                      32)) +
                    (((((((int)threadIdx.x) & 31) >> 4) +
                       ((((int)threadIdx.x) & 3) >> 1)) &
                      1) *
                     16)) +
                   (((((((int)threadIdx.x) & 15) >> 3) +
                      (((int)threadIdx.x) & 1)) &
                     1) *
                    8))) =
            *(uint4*)(A + ((i_1 * 512) + (((int)threadIdx.x) * 8)));
    }
#pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
        *(uint4*)(B_shared +
                  (((((i_2 * 512) + ((((int)threadIdx.x) >> 3) * 64)) +
                     ((((((int)threadIdx.x) >> 5) +
                        ((((int)threadIdx.x) & 7) >> 2)) &
                       1) *
                      32)) +
                    (((((((int)threadIdx.x) & 31) >> 4) +
                       ((((int)threadIdx.x) & 3) >> 1)) &
                      1) *
                     16)) +
                   (((((((int)threadIdx.x) & 15) >> 3) +
                      (((int)threadIdx.x) & 1)) &
                     1) *
                    8))) =
            *(uint4*)(B + ((i_2 * 512) + (((int)threadIdx.x) * 8)));
    }
    __syncthreads();

    tl::gemm_ss<16, 16, 64, 1, 2, 0, 1, 0>((&(A_shared[0])), (&(B_shared[0])),
                                           (&(C_local[0])));
#pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
        uint1 __1;
        float2 v_ = *(float2*)(C_local + (i_3 * 2));
        ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
        ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
        *(uint1*)(C +
                  ((((i_3 * 128) + (((((int)threadIdx.x) & 31) >> 2) * 16)) +
                    ((((int)threadIdx.x) >> 5) * 8)) +
                   ((((int)threadIdx.x) & 3) * 2))) = __1;
    }
}

#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() { return error_buf; }

extern "C" int init() {
    error_buf[0] = '\0';

    return 0;
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B,
                    half_t* __restrict__ C,
                    cudaStream_t stream = cudaStreamDefault) {
    main_kernel<<<dim3(1, 1, 1), dim3(64, 1, 1), 0, stream>>>(A, B, C);

    return 0;
}
