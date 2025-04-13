#include <cuda_runtime.h>

#include <iostream>

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define WARP_MASK (WARP_SIZE - 1)
__device__ inline int lane_id(void) { return threadIdx.x & WARP_MASK; }
__device__ inline int warp_id(void) { return threadIdx.x >> LOG_WARP_SIZE; }
// Warp scan
__device__ __forceinline__ int warp_scan(int val) {
  int x = val;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int y = __shfl_up_sync(0xffffffff, x, offset);
    if (lane_id() >= offset) x += y;
  }
  return x - val;
}

template <int threadsPerBlock>
__device__ int block_scan(int in) {
  __shared__ int sdata[threadsPerBlock >> LOG_WARP_SIZE];
  // A. Exclusive scan within each warp
  int warpPrefix = warp_scan(in);
  // B. Store in shared memory
  if (lane_id() == WARP_SIZE - 1) sdata[warp_id()] = warpPrefix + in;
  __syncthreads();
  // C. One warp scans in shared memory
  if (threadIdx.x < WARP_SIZE)
    sdata[threadIdx.x] = warp_scan(sdata[threadIdx.x]);
  __syncthreads();
  // D. Each thread calculates its final value
  int thread_out_element = warpPrefix + sdata[warp_id()];
  return thread_out_element;
}

template <int threadsPerBlock, int numElements, int batchSize>
__global__ void kernel_4(int *input, int *output) {
  int reductions[batchSize];
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_sum = 0;
#pragma unroll
  for (int i = 0; i < batchSize; i++) {
    const int idx = gtid * batchSize + i;
    if (idx < numElements) {
      total_sum += input[idx];
      reductions[i] = total_sum;
    }
  }
  int reduced_total_sum = block_scan<threadsPerBlock>(total_sum);
#pragma unroll
  for (int i = 0; i < batchSize; i++) {
    const int idx = gtid * batchSize + i;
    if (idx < numElements) {
      output[idx] = reduced_total_sum + reductions[i];
    }
  }
}

template <int threadsPerBlock, int numElements, int batchSize>
void launch_kernel_4(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock * batchSize - 1) /
                        (threadsPerBlock * batchSize);
  kernel_4<threadsPerBlock, numElements, batchSize>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}

template <int threadsPerBlock, int numElements, int batchSize>
void cpu_scan(int *input, int *output) {
  output[0] = input[0];
  for (int i = 1; i < numElements; i++) {
    if (!((i % (threadsPerBlock * batchSize)) == 0)) {
      output[i] = input[i] + output[i - 1];
    } else {
      output[i] = input[i];
    }
  }
}

__global__ void warmupKernel() { extern __shared__ int tmp[]; }

int main() {
  warmupKernel<<<1024, 1024>>>();
  cudaDeviceSynchronize();

  const int numElements = 1 << 30;
  const int threadsPerBlock = 1 << 7;
  const int batchSize = 4;

  // Allocate memory on host
  int *h_input = new int[numElements];
  int *h_output = new int[numElements];

  // Initialize input
  srand(42);
  for (int i = 0; i < numElements; i++) {
    h_input[i] = rand() % 100;
  }

  // Allocate memory on device
  int *d_input, *d_output;
  cudaMalloc(&d_input, numElements * sizeof(int));
  cudaMalloc(&d_output, numElements * sizeof(int));

  // Copy input to device
  cudaMemcpy(d_input, h_input, numElements * sizeof(int),
             cudaMemcpyHostToDevice);

  // Launch kernel
  launch_kernel_4<threadsPerBlock, numElements, batchSize>(d_input, d_output);

  // Copy output to host
  cudaMemcpy(h_output, d_output, numElements * sizeof(int),
             cudaMemcpyDeviceToHost);

  int *h_ref = new int[numElements];
  cpu_scan<threadsPerBlock, numElements, batchSize>(h_input, h_ref);

  // Verify results
  for (int i = 0; i < numElements; i++) {
    if (h_output[i] != h_ref[i]) {
      std::cout << "Test failed at index " << i << std::endl;
      return 1;
    }
  }
  std::cout << "Test passed!" << std::endl;

  // Measure performance
  cudaEvent_t start, stop;
  const unsigned int num_runs = 10000;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (unsigned int i = 0; i < num_runs; i++) {
    launch_kernel_4<threadsPerBlock, numElements, batchSize>(d_input, d_output);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  milliseconds /= num_runs;
  // Read numElements * sizeof(int) bytes, write numElements * sizeof(int) bytes
  auto bandwidth = numElements * sizeof(int) * 2 / milliseconds / 1e6;
  auto max_bandwidth = 3.3 * 1e3;  // 3.3 TB/s on H100
  std::cout << "Time taken: " << milliseconds << " ms" << std::endl;
  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
  std::cout << "Efficiency: " << bandwidth / max_bandwidth << std::endl;

  // Free memory
  cudaFree(d_input);
  cudaFree(d_output);
  delete[] h_input;
  delete[] h_output;
  delete[] h_ref;

  return 0;
}