#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H
#define FLOAT_DT float

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

void __declspec(dllexport) AmplitudeExtractionFourier2D(FLOAT_DT *image_real, FLOAT_DT *image_imag, FLOAT_DT *delay1, FLOAT_DT *delay2, FLOAT_DT fs_in, FLOAT_DT *output,
                                  int max_x, int max_y);

#ifdef __cplusplus
}
#endif

#endif