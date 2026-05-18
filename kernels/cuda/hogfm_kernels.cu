#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void hogfm_rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int rows,
    int cols,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float value = x[row * cols + col];
        sum += value * value;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shared[0] / cols + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        y[row * cols + col] = x[row * cols + col] * inv_rms * weight[col];
    }
}

extern "C" __global__ void hogfm_rotary_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int tokens,
    int heads,
    int dim,
    float base
) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * heads * dim;
    if (linear >= total) return;

    int col = linear % dim;
    int head_index = linear / dim;
    int token = head_index / heads;
    int half = dim / 2;
    if (col >= half * 2) {
        y[linear] = x[linear];
        return;
    }

    int pair_col = col % half;
    float inv_freq = powf(base, -static_cast<float>(pair_col) / half);
    float angle = token * inv_freq;
    float c = cosf(angle);
    float s = sinf(angle);
    int mate_col = col < half ? col + half : col - half;
    float a = x[head_index * dim + pair_col];
    float b = x[head_index * dim + pair_col + half];
    y[head_index * dim + pair_col] = a * c - b * s;
    y[head_index * dim + pair_col + half] = a * s + b * c;
}
