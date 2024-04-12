#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1000000 // Size of the vectors (adjust as needed)

__global__ void sequential_vector_add(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void parallel_vector_add(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = (int*)malloc(N * sizeof(int));
    h_b = (int*)malloc(N * sizeof(int));
    h_c = (int*)malloc(N * sizeof(int));

    // Initialize host arrays (example with random values)
    for (int i = 0; i < N; i++) {
        h_a[i] = rand();
        h_b[i] = rand();
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Grid and block sizes for parallel execution
    int threadsPerBlock = 256;  // Adjust as needed based on GPU architecture
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  float sequential_time, parallel_time;
    // Measure execution times using a single loop
    cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);

    // Sequential execution
    cudaEventRecord(start_event, 0);
    for (int i = 0; i < 100; ++i) {  // Run the kernel multiple times for better timing
        sequential_vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(end_event, 0);
    cudaEventElapsedTime(&sequential_time, start_event, end_event);
    cudaEventDestroy(start_event);

    // Parallel execution
    cudaEventRecord(start_event, 0);
    for (int i = 0; i < 100; ++i) {  // Run the kernel multiple times for better timing
        parallel_vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(end_event, 0);
    cudaEventElapsedTime(&parallel_time, start_event, end_event);
    cudaEventDestroy(end_event);
    sequential_time /= 100.0f;  // Average time over multiple runs
    parallel_time /= 100.0f;

    // Copy results back from device to host (optional for verification)
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate speedup
    float speedup = sequential_time / parallel_time;

    printf("Sequential execution time: %.6f ms\n", sequential_time);
    printf("Parallel execution time: %.6f ms\n", parallel_time);
        printf("Speedup: %.2f\n", speedup);


    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
    }

