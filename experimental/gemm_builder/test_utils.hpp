#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "run_utils.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include <cassert>
#include <cstddef>

namespace ck_tile::test {

// Simple kernel for checking matrix multiplication.
//
// Computes C = A * B where:
//  - A is a matrix of size M x K in Row-Major format,
//  - B is a matrix of size K x N in Column-Major format,
//  - C is a matrix of size M x N in Row-Major format.
template <unsigned int BLOCK_SIZE>
__global__ void simple_gemm_kernel(const void* a,
                                   const void* b,
                                   void* c,
                                   const unsigned int M,
                                   const unsigned int N,
                                   const unsigned int K,
                                   const unsigned int LDA,
                                   const unsigned int LDB,
                                   const unsigned int LDC)
{
    const __hip_bfloat16* a_global = static_cast<const __hip_bfloat16*>(a);
    const __hip_bfloat16* b_global = static_cast<const __hip_bfloat16*>(b);
    __hip_bfloat16* c_global       = static_cast<__hip_bfloat16*>(c);

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;

    const unsigned int m_global = by * BLOCK_SIZE + ty;
    const unsigned int n_global = bx * BLOCK_SIZE + tx;

    float acc                   = 0.0F;
    const unsigned int k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for(unsigned int k_block = 0; k_block < k_blocks; k_block++)
    {
        // Pad shared memory layout by one to reduce bank conflicts.
        __shared__ __hip_bfloat16 a_shared[BLOCK_SIZE][BLOCK_SIZE + 1];
        __shared__ __hip_bfloat16 b_shared[BLOCK_SIZE][BLOCK_SIZE + 1];

        const unsigned int k_col_global = k_block * BLOCK_SIZE + tx;
        const unsigned int k_row_global = k_block * BLOCK_SIZE + ty;

        // Load elements from global memory to shared memory with padding.
        if(m_global < M && k_col_global < K)
        {
            a_shared[ty][tx] = a_global[m_global * LDA + k_col_global];
        }
        else
        {
            a_shared[ty][tx] = static_cast<__hip_bfloat16>(0.0F);
        }
        if(k_row_global < K && n_global < N)
        {
            b_shared[ty][tx] = b_global[k_row_global * LDB + n_global];
        }
        else
        {
            b_shared[ty][tx] = static_cast<__hip_bfloat16>(0.0F);
        }

        // Wait for all threads to load data before calculating.
        __syncthreads();
        for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        {
            acc += static_cast<float>(a_shared[ty][i]) * static_cast<float>(b_shared[i][tx]);
        }
    }
    // Write the accumulated value to global memory with boundary checks.
    if(m_global < M && n_global < N)
    {
        c_global[m_global * LDC + n_global] = static_cast<__hip_bfloat16>(acc);
    }
}

inline void RunReferenceGemm(const void* a_global,
                             const void* b_global,
                             void* c_global,
                             int M,
                             int N,
                             int K,
                             int LDA,
                             int LDB,
                             int LDC,
                             hipStream_t stream = hipStreamDefault)
{
    constexpr unsigned int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    simple_gemm_kernel<BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(a_global, b_global, c_global, M, N, K, LDA, LDB, LDC);
}

// Kernel to fill a bfloat16 tensor with random numbers in [-1, 1)
__global__ void FillUniformRandomKernel(__hip_bfloat16* data, int size, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        // Simple LCG for demonstration (not high quality randomness)
        unsigned int lcg = seed ^ idx;
        lcg              = lcg * 1664525u + 1013904223u;
        float val        = static_cast<float>(lcg & 0xFFFF) / 65536.0f;
        data[idx]        = static_cast<__hip_bfloat16>(val * 2.0f - 1.0f);
    }
}

inline void FillUniformRandomBf16(void* data,
                                  int size,
                                  unsigned int seed  = 1234,
                                  hipStream_t stream = hipStreamDefault)
{
    int block = 256;
    int grid  = (size + block - 1) / block;
    FillUniformRandomKernel<<<grid, block, 0, stream>>>(
        static_cast<__hip_bfloat16*>(data), size, seed);
}

// Kernel to calculate the difference between two bf16 device vectors and store in f32 device vector
__global__ void
CalculateDiffKernel(const __hip_bfloat16* a, const __hip_bfloat16* b, float* diff, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        diff[idx] = static_cast<float>(a[idx]) - static_cast<float>(b[idx]);
    }
}

class TensorDiff
{
    public:
    TensorDiff(const void* a, const void* b, int M, int N) : a_(a), b_(b), M_(M), N_(N) {}

    // Get the vector of differences.
    const std::vector<float>& Diff() const { return diff_; }

    // Returns the largest absolute difference.
    float LargestDiff() const
    {
        return *std::max_element(
            diff_.begin(), diff_.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
    }

    // Report information about the differences.
    void Report() const
    {
        std::cout << "Largest difference: " << LargestDiff() << std::endl;
        for(int m = 0; m < M_; ++m)
        {
            for(int n = 0; n < N_; ++n)
            {
                if(std::abs(diff_[m * N_ + n]) > 1e-3f)
                {
                    std::cout << "Difference at (" << m << ", " << n << "): " << diff_[m * N_ + n]
                              << std::endl;
                }
            }
        }
    }

    private:
    std::vector<float> CalculateDiff()
    {
        auto diff_dev      = ck_tile::runtime::AllocDevMem<float>(M_ * N_);
        int block          = 256;
        int grid           = (M_ * N_ + block - 1) / block;
        hipStream_t stream = hipStreamDefault;
        CalculateDiffKernel<<<grid, block, 0, stream>>>(static_cast<const __hip_bfloat16*>(a_),
                                                        static_cast<const __hip_bfloat16*>(b_),
                                                        diff_dev.get(),
                                                        M_ * N_);
        ck_tile::runtime::CheckHipError(hipDeviceSynchronize());
        std::vector<float> diff(M_ * N_);
        ck_tile::runtime::CheckHipError(
            hipMemcpy(diff.data(), diff_dev.get(), M_ * N_ * sizeof(float), hipMemcpyDeviceToHost));
        return diff;
    }
    const void* a_;
    const void* b_;
    int M_;
    int N_;
    std::vector<float> diff_ = CalculateDiff();
};

} // namespace ck_tile::test
