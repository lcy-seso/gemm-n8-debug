#include "gemm_sm89.h"
#include "utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

#if 0
    if (threadIdx.x == 0) {
        printf("A_shared:\n");
        const __half* A_shared_ptr = reinterpret_cast<const __half*>(A_shared);
        for (int i = 0; i < 1024; ++i) {
            printf("%.0f, ", __half2float(A_shared_ptr[i]));

            if ((i + 1) % 16 == 0) printf("\n");
        }
        printf("\n");

        printf("B_shared:\n");
        const __half* B_shared_ptr = reinterpret_cast<const __half*>(B_shared);
        for (int i = 0; i < 1024; ++i) {
            printf("%.0f, ", __half2float(B_shared_ptr[i]));

            if ((i + 1) % 16 == 0) printf("\n");
        }
        printf("\n");
    }
#endif

    tl::gemm_ss<16, 16, 64, 1, 2, 0, 1, 0, half_t, half_t, float>(
        (&(A_shared[0])), (&(B_shared[0])), (&(C_local[0])));

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

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B,
                    half_t* __restrict__ C,
                    cudaStream_t stream = cudaStreamDefault) {
    main_kernel<<<dim3(1, 1, 1), dim3(64, 1, 1), 0, stream>>>(A, B, C);

    return 0;
}

int main() {
    // static constexpr int M = 16;
    // static constexpr int N = 16;
    // static constexpr int K = 64;

    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 64;

    using DType = half_t;

    using MMA = cute::GemmTensorOp<M, N, K, 1, 2, 0, 1, 0, DType, DType, float>;

    using TiledMma = MMA::TileMma;
    print(TiledMma{});

    printf("Shared A layout:\n");
    print(typename MMA::SmemLayoutA{});

    printf("\nShared B layout:\n");
    print(typename MMA::SmemLayoutB{});

    printf("\nthreads: %d\n", int(size(TiledMma{})));

    return 0;

    // initialize data
    thrust::host_vector<DType> h_a(M * K);  // 64 * 16
    for (int i = 0; i < h_a.size(); ++i) {
        // h_a[i] = static_cast<DType>(rand_normal(0.05f, 1e-2f));
        h_a[i] = static_cast<DType>(i % 2048);
    }

    thrust::host_vector<DType> h_b(K * N);  // 16 * 64
    for (int i = 0; i < h_b.size(); ++i) {
        // h_b[i] = static_cast<DType>(rand_normal(0.03f, 5e-2f));
        h_b[i] = static_cast<DType>(i % 2048);
    }

#if 0
    // debug print
    print_matrix(
        reinterpret_cast<const __half*>(thrust::raw_pointer_cast(h_a.data())),
        h_a.size());
#endif

    thrust::host_vector<DType> h_c(M * N);  // 64 * 64
    thrust::fill(h_c.begin(), h_c.end(), static_cast<DType>(0.));

    thrust::device_vector<DType> d_a = h_a;
    thrust::device_vector<DType> d_b = h_b;
    thrust::device_vector<DType> d_c = h_c;

    call(thrust::raw_pointer_cast(d_a.data()),
         thrust::raw_pointer_cast(d_b.data()),
         thrust::raw_pointer_cast(d_c.data()));
    h_c = d_c;
    cudaDeviceSynchronize();

#if 1
    // debug print
    print_matrix(
        reinterpret_cast<const __half*>(thrust::raw_pointer_cast(h_c.data())),
        h_c.size());
#endif

    return 0;
}
