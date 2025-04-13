#include <cuda_runtime.h>

#include <iostream>

template <int threadsPerBlock, int numElements>
__global__ void kernel_0(int *input, int *output) {
  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * threadsPerBlock + tid;

  output[gtid] = input[gtid];
  __syncthreads();

#pragma unroll
  for (unsigned int offset = 1; offset <= threadsPerBlock / 2; offset <<= 1) {
    int tmp;
    if (tid >= offset) {
      tmp = output[gtid - offset];
    }
    __syncthreads();

    if (tid >= offset && gtid < numElements) {
      output[gtid] += tmp;
    }
    __syncthreads();
  }
}

template <int threadsPerBlock, int numElements>
void launch_kernel_0(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  kernel_0<threadsPerBlock, numElements>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}

template <int threadsPerBlock, int numElements>
void cpu_scan(int *input, int *output) {
  output[0] = input[0];
  for (int i = 1; i < numElements; i++) {
    if (!((i & (threadsPerBlock - 1)) == 0)) {
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
  const int threadsPerBlock = 1 << 9;

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
  launch_kernel_0<threadsPerBlock, numElements>(d_input, d_output);

  // Copy output to host
  cudaMemcpy(h_output, d_output, numElements * sizeof(int),
             cudaMemcpyDeviceToHost);

  int *h_ref = new int[numElements];
  cpu_scan<threadsPerBlock, numElements>(h_input, h_ref);

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
    launch_kernel_0<threadsPerBlock, numElements>(d_input, d_output);
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