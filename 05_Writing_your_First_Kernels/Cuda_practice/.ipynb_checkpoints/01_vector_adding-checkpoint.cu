#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000
#define block_size_1D 1024
#define block_size_3D_x 16
#define block_size_3D_y 4
#define block_size_3D_z 4




void add_vector_cpu(float *h_a, float *h_b, float * h_c,int n){
    for(int i=0; i<n; i++){
    h_c[i]=h_a[i]+h_b[i];
    }
}

__global__ void add_vector_cuda_1d(float *h_a, float *h_b, float * h_c,int n){
    int i = blockIdx.x * blockDim.x  + threadIdx.x;
    if(i<n){
    h_c[i] =h_a[i] +h_b[i];

}
}

__global__ void add_vector_cuda_3d(float *h_a, float *h_b, float * h_c,int n){
    int i = blockIdx.x +blockIdx.y* blockDim.x + blockIdx.z * blockDim.x* blockDim.y + threadIdx.x;
    if(i<n){
    h_c[i] =h_a[i] +h_b[i];

}
}

void init_vec(float *vec,int n){
    for(int i=0;i<n;i++)
    vec[i] = (float)rand()/RAND_MAX;
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu,*h_c_gpu_3d;
    float *d_a, *d_b, *d_c , *d_c_3;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b =(float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vec(h_a, N);
    init_vec(h_b, N);


    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_c_3, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int nx =100 ,ny=100,nz=1000;
    int num_blocks = (N + block_size_1D - 1) / block_size_1D;
    dim3 block_size_3d(block_size_3D_x,block_size_3D_y,block_size_3D_z);
    dim3 num_blocks_3d((nx+block_size_3d.x-1)/block_size_3d.x,
    (ny+block_size_3d.y-1)/block_size_3d.y,
    (nz+block_size_3d.z-1)/block_size_3d.z
    );

    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 roundedy
    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        add_vector_cpu(h_a, h_b, h_c_cpu, N);
        add_vector_cuda_1d<<<num_blocks, block_size_1D>>>(d_a, d_b, d_c, N);
        add_vector_cuda_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        add_vector_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;


    printf("Benchmarking CUDA implementation...\n");
    double gpu_total_time =0.0;

    for(int i=0;i<20;i++){
    double start_time =get_time();
    add_vector_cuda_1d <<<num_blocks, block_size_1D>>> (d_a,d_b,d_c,N);
    double end_time =get_time();
    gpu_total_time += end_time-start_time;

    }
    double  gpu_avg_time = gpu_total_time/20;


    printf("Benchmarking CUDA_3D implementation...\n");
    double gpu_total_time_3d =0.0;

    for(int i=0;i<20;i++){
    double start_time =get_time();
    add_vector_cuda_3d <<<num_blocks_3d, block_size_3d>>> (d_a,d_b,d_c_3,N);
    double end_time =get_time();
    gpu_total_time_3d += end_time-start_time;

    }
    double  gpu_avg_time_3d = gpu_total_time_3d/20;

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("GPU 3D average time: %f milliseconds\n", gpu_avg_time_3d*1000);

    printf("Speedup_1d: %fx\n", cpu_avg_time / gpu_avg_time);
    printf("Speedup_3d: %fx\n", cpu_avg_time / gpu_avg_time_3d);

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct_1d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct_1d = false;
            break;
        }
    }
    printf("Results are %s\n", correct_1d ? "correct" : "incorrect");

    cudaMemcpy(h_c_gpu_3d, d_c_3, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i=0; i<N; i++){
        if(fabs(h_c_cpu[i]-h_c_gpu_3d[i])>1e-5){
        correct_3d =false;
        break;
        }
    }
    printf("Results are %s\n", correct_3d ? "correct" : "incorrect");


    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c_3);

    return 0;

}