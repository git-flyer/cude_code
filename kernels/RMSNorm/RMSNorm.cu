# include <iostream>
# include <cstdio>
# include <cmath>
# include <random>
# include <chrono>
# include <cuda_runtime.h>
using std::cin;
using std::cout;
//  cpu实现, in是输入矩阵，shape为[batch, size]，按行存储
//  weight是可学习的缩放权重，shape为[size]，与特征维度对应
//  out是输出矩阵，shape与输入相同，为[batch, size]
//  batch 是样本数量（行数）
//  size 是每行特征维度（列数）
//  eps是分母中防止除0的小常熟，通常为1e-5
//  1. 对每一行，计算 RMS = 1/sqrt( mean(x^2) + eps )
//  2. 对该行每个元素，执行 out = in * weight * rms
void row_rmsnorm_f32_dim_cpu(float *in, float *weight, float *out, int batch, int size, float eps){
    for(int i = 0; i < batch; ++i){
        float *in_ptr = in + i * size;
        float *out_ptr = out + i * size;

        float sum = 0.0f;
        for(int j = 0; j < size; ++j){
            float val = in_ptr[j];
            sum += val * val;
        }
        // 分母 (归一化系数rms)
        float rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);
        for(int j = 0; j < size; ++j){
            float x = in_ptr[j] * weight[j];
            out_ptr[j] = x * rms;
        }
    }
}

// 被核函数调用
__inline__  __device__ float block_reduce(float val){
    int tid = threadIdx.x;
    int warpSize = 32;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    // warp_level 的归约
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        // 第一轮的 offset = 16;
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ float warpSums[32];
    if(lane == 0){
        warpSums[warp_id] = val;
    }
    __syncthreads();  // 确保每个warp内的0号线程将其warp内的归约结果都写入到了共享内存里

    // 再对共享内存里的32个归约结果使用一个warp进行最后一次归约
    if(warp_id == 0){
        val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warpSums[tid] : 0.0f;
        for(int offset = warpSize / 2; offset > 0; offset /= 2){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    else{
        val = 0.0f;
    }
    return val;
}

__global__ void row_rmsnorm_f32_dim(float *in, float *weight, float *out, int batch, int size, float eps){
    int bx = blockIdx.x;
    if(bx >= batch)    // 多余的线程块就什么也不做
        return;
    float *block_in = in + bx * size;  // 某个 block 处理的输入数据的起始地址
    float *block_out = out + bx * size; // 某个 block 处理的输出数据的起始地址
    float sum = 0.0f;

    // 每个线程块处理一个 1 * size 的数据行
    for(int i = threadIdx.x; i < size; i += blockDim.x){
        float x = block_in[i];
        sum += x * x;
    }
    __shared__ float shared_val;
    sum = block_reduce(sum);   // 将一个线程块内所有线程记录的sum全部加起来
    if(threadIdx.x == 0){
        shared_val = sum;   // 0 号线程的计算结果被写入到shared_mem
    }
    __syncthreads();    // 确保0号线程的计算结果已经被写入到shared_mem，后续线程块内的每个线程都需要这个值
    sum = shared_val;    // 所有的线程都得到了该size大小的数据块的归约结果

    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    for(int i = threadIdx.x; i < size; i += blockDim.x){
        block_out[i] = block_in[i] * scale * weight[i];
    }
}

float compute_max_error(const std::vector<float>& cpu_out,
                        const std::vector<float>& cuda_out, int n) {
  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float err = std::abs(cpu_out[i] - cuda_out[i]);
    max_err = std::max(max_err, err);
    if (max_err > 1.f) {
      std::cout << "Error at index " << i << ": CPU = " << cpu_out[i]
                << ", CUDA = " << cuda_out[i] << ", Error = " << err << "\n";
      break;
    }
  }
  return max_err;
}


int main() {
  const int batch = 16;   // 16个样本，每个样本的维度是1024
  const int size = 1024;   
  const float eps = 1e-6f;
  const int total = batch * size;

  // Host memory
  std::vector<float> h_input(total);
  std::vector<float> h_weight(size);
  std::vector<float> h_output_cpu(total);
  std::vector<float> h_output_cuda(total);

  // Random init
  std::random_device rd;  // 非确定性随机数生成器
  std::mt19937 gen(rd());  // Mersenne Twister 伪随机数生成器
  std::normal_distribution<float> dis(0.0f, 1.0f);  // 正态分布

  for (int i = 0; i < total; ++i) {   // 生成输入数据
    h_input[i] = dis(gen);
  }
  for (int i = 0; i < size; ++i) {
    h_weight[i] = dis(gen);
  }

  // CPU version
  auto start = std::chrono::high_resolution_clock::now();
  row_rmsnorm_f32_dim_cpu(h_input.data(), h_weight.data(), h_output_cpu.data(),
                          batch, size, eps);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "CPU RMSNorm took " << duration.count() << " microseconds.\n";

  // CUDA setup
  float *d_input, *d_weight, *d_output;
  cudaMalloc(&d_input, total * sizeof(float));
  cudaMalloc(&d_weight, size * sizeof(float));
  cudaMalloc(&d_output, total * sizeof(float));

  cudaMemcpy(d_input, h_input.data(), total * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Kernel launch config
  const int block_size = 1024; // block 用于处理一个输入向量
  const int grid_size = batch;  // One block per batch row
  dim3 grid(grid_size);
  dim3 block(block_size);

  // CUDA timing with events
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  int warpup = 10;
  for (int i = 0; i < warpup; i++) {
    // Warm-up run
    row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch,
                                         size, eps);
  }
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != 0) {
    printf("cuda error:%d\n", err);
  }
  cudaEventRecord(start_event);
  // row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch,
  // size, eps);
  int test_iter = 10;
  for (int i = 0; i < test_iter; ++i) {
    row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch,
                                         size, eps);
  }
  cudaEventRecord(stop_event);

  // Wait and measure
  cudaEventSynchronize(stop_event);
  float cuda_time;
  cudaEventElapsedTime(&cuda_time, start_event, stop_event);  // ms

  // Copy result back
  cudaMemcpy(h_output_cuda.data(), d_output, total * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "CUDA RMSNorm took " << cuda_time * 1000 / test_iter
            << " microseconds.\n";

  // Compare results
  float max_error = compute_max_error(h_output_cpu, h_output_cuda, total);
  std::cout << "Max absolute error (CPU vs CUDA): " << max_error << "\n";

  // Optional: print first few values
  std::cout << "\nFirst 10 outputs (CPU vs CUDA):\n";
  for (int i = 0; i < 10; ++i) {
    std::cout << "CPU: " << h_output_cpu[i] << " | CUDA: " << h_output_cuda[i]
              << " | Diff: " << std::abs(h_output_cpu[i] - h_output_cuda[i])
              << "\n";
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  return 0;
}
