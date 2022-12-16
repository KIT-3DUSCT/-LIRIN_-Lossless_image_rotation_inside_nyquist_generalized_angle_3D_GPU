#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <math_constants.h>
#include "cuda_kernel.h"

#define FLOAT_DT float
#define coor_2d(COOR1, COOR2, MAX_COOR1, MAX_COOR2) COOR1 + COOR2 * MAX_COOR1
#define coor_3d(COOR1, COOR2, COOR3, MAX_COOR1, MAX_COOR2, MAX_COOR3) COOR1 + COOR2 * MAX_COOR1 + COOR3 * MAX_COOR1 * MAX_COOR2
#define MAX(a, b) ((a > b) ? (a) : (b))

__global__ void calculate_pixel(const FLOAT_DT *delay1, const FLOAT_DT *delay2, const FLOAT_DT *wavel1, const FLOAT_DT *wavel2,
                     const FLOAT_DT *image_real, const FLOAT_DT *image_imag, FLOAT_DT *output, int max_x, int max_y) {
    // Coordinates
    long long threadID = threadIdx.x+blockIdx.x*blockDim.x;
    int x = threadID % max_y;
    int y = (threadID/max_y)%max_x;
    int z = threadID/(max_x*max_y);

    // Get value of ps1
    FLOAT_DT value = CUDART_PI_F * 2.0 * delay1[z] / wavel1[x];
    if (max_y % 2 != 1 && max_y / 2 == z && x == 0) {
        value = 0;
    }
    FLOAT_DT ps1_value_real, ps1_value_imaginary;
    sincosf(value, &ps1_value_imaginary, &ps1_value_real);

    // Get value of ps2
    value = CUDART_PI_F * 2.0 * delay2[z] / wavel2[y];
    if (max_x % 2 != 1 && max_x / 2 == z && y == 0) {
        value = 0;
    }
    FLOAT_DT ps2_value_real, ps2_value_imaginary;
    sincosf(value, &ps2_value_imaginary, &ps2_value_real);

    // Final multiplication
    FLOAT_DT total_real = ps1_value_real * ps2_value_real + ps1_value_imaginary * ps2_value_imaginary;
    FLOAT_DT total_imaginary = ps1_value_real * ps2_value_imaginary - ps1_value_imaginary * ps2_value_real;

    atomicAdd(&output[z], (image_real[x + y * max_y] * total_real - image_imag[x+y*max_y]*total_imaginary) * sqrtf(total_real * total_real + total_imaginary * total_imaginary));
}

// image: 2d, delay1_in: 1d, delay2_in: 1d
// delay1 und delay2 werden implizit transponiert
void AmplitudeExtractionFourier2D(FLOAT_DT *image_real, FLOAT_DT *image_imag, FLOAT_DT *delay1, FLOAT_DT *delay2, FLOAT_DT fs_in, FLOAT_DT *output,
                                  int max_x, int max_y) {
    int longest_dimension;
    FLOAT_DT fs;

    longest_dimension = MAX(max_x, max_y);
    fs = fs_in - fs_in / (FLOAT_DT) longest_dimension;

    FLOAT_DT *wavel1, *wavel2;
    wavel1 = (FLOAT_DT*)malloc(max_y*sizeof(FLOAT_DT));
    wavel2 = (FLOAT_DT*)malloc(max_x*sizeof(FLOAT_DT));

    // Create wavel1
    if (max_y % 2 == 1) {
        for (int i = 0; i < max_y; i++) {
            // circshift + fftshift
            int shifted_i = (i + ((int) max_y / 2)) % max_y;
            wavel1[i] = 1 / ((FLOAT_DT) shifted_i * (fs / (FLOAT_DT) (max_y - 1)) - fs / 2);
        }
    } else {
        for (int i = 0; i < max_y; i++) {
            // Shift (f1(end:-1:size(CE,1)/2+2)=-f1(2:size(CE,1)/2);)
            int backwards_i = max_y - i - 1;

            if (backwards_i < max_y / 2 - 1) {
                wavel1[i] = -wavel1[backwards_i + 1];
            } else {
                wavel1[i] = 1 / ((FLOAT_DT) i * (fs / (FLOAT_DT) (max_y - 1)));
            }
        }
    }

    // Create wavel2
    if (max_x % 2 == 1) {
        for (int i = 0; i < max_x; i++) {
            // circshift + fftshift
            int shifted_i = (i + ((int) max_x / 2)) % max_x;
            wavel2[i] = 1 / ((FLOAT_DT) shifted_i * (fs / (FLOAT_DT) (max_x - 1)) - fs / 2);
        }
    } else {
        for (int i = 0; i < max_x; i++) {
            // Shift (f2(end:-1:size(CE,2)/2+2)=-f2(2:size(CE,2)/2);)
            int backwards_i = max_x - i - 1;

            if (backwards_i < max_x / 2 - 1) {
                wavel2[i] = -wavel2[backwards_i + 1];
            } else {
                wavel2[i] = 1 / ((FLOAT_DT) i * (fs / (FLOAT_DT) (max_x - 1)));
            }
        }
    }

    for (int i = 0; i < max_x * max_y; i++) {
        output[i] = 0.0f;
    }

    // Allocate space on GPU
    FLOAT_DT *gpu_delay1, *gpu_delay2, *gpu_wavel1, *gpu_wavel2, *gpu_image_real, *gpu_image_imag, *gpu_output;
    cudaMalloc(&gpu_delay1, max_x*max_y*sizeof(FLOAT_DT));
    cudaMalloc(&gpu_delay2, max_x*max_y*sizeof(FLOAT_DT));
    cudaMalloc(&gpu_wavel1, max_y*sizeof(FLOAT_DT));
    cudaMalloc(&gpu_wavel2, max_x*sizeof(FLOAT_DT));
    cudaMalloc(&gpu_image_real, max_x*max_y*sizeof(FLOAT_DT));
    cudaMalloc(&gpu_image_imag, max_x*max_y*sizeof(FLOAT_DT));
    cudaMalloc(&gpu_output, max_x*max_y*sizeof(FLOAT_DT));

    // Copy data to GPU
    cudaMemcpy(gpu_delay1, delay1, max_x*max_y*sizeof(FLOAT_DT), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_delay2, delay2, max_x*max_y*sizeof(FLOAT_DT), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_wavel1, wavel1, max_y*sizeof(FLOAT_DT), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_wavel2, wavel2, max_x*sizeof(FLOAT_DT), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_image_real, image_real, max_x*max_y*sizeof(FLOAT_DT), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_image_imag, image_imag, max_x*max_y*sizeof(FLOAT_DT), cudaMemcpyHostToDevice);

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
        printf("Sync kernel error1: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error1: %s\n", cudaGetErrorString(errAsync));

    // 1024: max -> 1024
    int count_threads = 1024;
    long long needed_blocks = ((long long) max_x*max_y*max_x*max_y/count_threads)+1;
    calculate_pixel<<<needed_blocks, count_threads>>>(gpu_delay1,gpu_delay2,gpu_wavel1,gpu_wavel2,gpu_image_real, gpu_image_imag, gpu_output,max_x,max_y);


    errSync  = cudaGetLastError();
    errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
        printf("Sync kernel error2: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error2: %s\n", cudaGetErrorString(errAsync));

    // Copy result from GPU
    cudaMemcpy(output, gpu_output, max_x*max_y*sizeof(FLOAT_DT), cudaMemcpyDeviceToHost);

    errSync  = cudaGetLastError();
    errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
        printf("Sync kernel error3: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error3: %s\n", cudaGetErrorString(errAsync));

    cudaFree(gpu_delay1);
    cudaFree(gpu_delay2);
    cudaFree(gpu_wavel1);
    cudaFree(gpu_wavel2);
    cudaFree(gpu_image_real);
    cudaFree(gpu_image_imag);
    cudaFree(gpu_output);
    free(wavel1);
    free(wavel2);
}
