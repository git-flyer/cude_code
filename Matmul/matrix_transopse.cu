# include <cstdio>
# include <cstdlib>
# include <iostream>
# include <cuda_runtime.h>
# include <cmath>
using std::cout;
using std::endl;



__global__ void naiveGmem(float *d_output, float *d_input, int M, int N){
    //输入矩阵是M* N的，M是行，N是列
    // 目的是把M * N 的矩阵转置成N * M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    if(x < N && y < M){
        d_output[x * M + y] = d_input[y * N + x];
    }
}


void call_naiveGmem(float *d_output, float *d_input, int M, int N){
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    naiveGmem<<<gridDim, blockDim>>>(d_output, d_input, M, N);
}

int main(){
    int M = 5, N = 5;
    size_t size = M * N * sizeof(float);

    // 主机端的输入输出数组的内存分配
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // 随机初始化输入矩阵
    for(int i = 0; i < M * N; ++i){
        h_input[i] = (float)rand() / RAND_MAX;
    }

    // 设备内存分配
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // 将输入矩阵从主机端复制到设备端
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warm_up_iter = 5;

    //
    for(int i = 0; i < warm_up_iter; ++i){
        call_naiveGmem(d_output, d_input, M, N);
    }

    int bench_iter = 5;

    //开始计时
    cudaEventRecord(start);
    for(int i = 0; i < bench_iter; ++i){
        call_naiveGmem(d_output, d_input, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //



    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        cout << "CUDA error: " << cudaGetErrorString(err) << endl;
        return 0;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive transpose kernel execution time: "
            << milliseconds / float(bench_iter) << " ms" << std::endl;

    // 将输出矩阵从设备端复制到主机端
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);


    // 输出转置前的矩阵
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            cout << h_input[i * N + j] << " ";
        }
        cout << endl;
    }
    cout<<"-----------------------"<< endl;
    // 输出转置后的矩阵
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            cout << h_output[i * M + j] << " ";
        }
        cout << endl;
    }

    // 释放主机端内存
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    cout << "Matrix transposition completed successfully!" << endl;

    return 0;
}
