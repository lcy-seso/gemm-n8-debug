# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang
import tilelang.language as T

tilelang.disable_cache()


def matmul(
    M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"
):
    # add decorator @tilelang.jit if you want to return a torch function
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=64
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_N, block_K), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


M = 16  # M = T.symbolic("m") if you want to use dynamic shape
N = 16
K = 64
block_M = 16
block_N = 16
block_K = K

# 1. Define the kernel (matmul) and compile/lower it into an executable module
func = matmul(M, N, K, block_M, block_N, block_K)

# 2. Compile the kernel into a torch function
# out_idx specifies the index of the output buffer in the argument list
# if out_idx is specified, the tensor will be created during runtime
# target currently can be "cuda" or "hip" or "cpu".
jit_kernel = tilelang.compile(
    func, out_idx=[2], target="cuda", execution_backend="cython"
)
print(jit_kernel.get_kernel_source())

# 3. Test the kernel in Python with PyTorch data
import torch

# Create random input tensors on the GPU
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(N, K, device="cuda", dtype=torch.float16)

# Run the kernel through the Profiler
c = jit_kernel(a, b)

print(c)
# Reference multiplication using PyTorch
ref_c = a @ b.T

# Validate correctness
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")
