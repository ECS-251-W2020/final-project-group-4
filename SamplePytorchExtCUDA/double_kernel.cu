#include <cuda_runtime.h>

__global__ void double_kernel(const float* data, float* output, size_t n) {  //CUDA function
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = data[idx] * 2;
    }
}

void launch_double_ext_cuda_kernel(const float* data, float* output, size_t n){ //CPP function to call CUDA function above
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    double_kernel<<<blocks, threads>>>(data, output, n);
}
