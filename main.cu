#include "gemm_sm89.h"

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

float rand_normal(float mean = 0.0f, float stddev = 1.0f) {
    // Box-Muller transform to generate random numbers with Normal distribution
    float u1 = ((float)rand()) / (float)RAND_MAX;
    float u2 = ((float)rand()) / (float)RAND_MAX;

    // Avoid log(0) by ensuring u1 is not zero
    if (u1 < 1e-10f) u1 = 1e-10f;

    // Compute Gaussian random value
    float r = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);

    // Scale and shift to get desired mean and standard deviation
    return mean + stddev * r;
}

int main() {
    // static constexpr int M = 16;
    // static constexpr int N = 16;
    // static constexpr int K = 64;

    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 32;

    using DType = half_t;

    // initialize data
    thrust::host_vector<DType> h_a(M * K);  // 64 * 16
    for (int i = 0; i < h_a.size(); ++i) {
        h_a[i] = static_cast<DType>(rand_normal(0.05f, 1e-2f));
        // h_a[i] = static_cast<DType>(i % 2048);
    }

    thrust::host_vector<DType> h_b(K * N);  // 16 * 64
    for (int i = 0; i < h_b.size(); ++i) {
        h_b[i] = static_cast<DType>(rand_normal(0.03f, 5e-2f));
        // h_b[i] = static_cast<DType>(i % 2048);
    }

    thrust::host_vector<DType> h_c(M * N);  // 64 * 64
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<DType> d_a = h_a;
    thrust::device_vector<DType> d_b = h_b;
    thrust::device_vector<DType> d_c = h_c;

    call(thrust::raw_pointer_cast(d_a.data()),
         thrust::raw_pointer_cast(d_b.data()),
         thrust::raw_pointer_cast(d_c.data()));
    cudaDeviceSynchronize();

    h_c = d_c;

    const __half* h_c_ptr =
        reinterpret_cast<const __half*>(thrust::raw_pointer_cast(h_c.data()));

    for (int i = 0; i < h_c.size(); ++i) {
        printf("%.2f, ", __half2float(h_c_ptr[i]));

        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }

    return 0;
}
