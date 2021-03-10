#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__
void kmultiply(const float* a, float* b, int n) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < n)
        b[i] *= a[i];
}

extern "C"
void launch_multiply(const float* a, float* b, int n) {
    float* dA;
    float* dB;
    int cerr;

    cerr = cudaMalloc((void**)&dA, n*sizeof(float));
    cerr = cudaMalloc((void**)&dB, n*sizeof(float));
    cerr = cudaMemcpy(dA, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cerr = cudaMemcpy(dB, b, n*sizeof(float), cudaMemcpyHostToDevice);

    kmultiply<<<ceil((float)n/256), 256>>>(dA, dB, n);

    cerr = cudaThreadSynchronize();

    cerr = cudaMemcpy(b, dB, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
}
