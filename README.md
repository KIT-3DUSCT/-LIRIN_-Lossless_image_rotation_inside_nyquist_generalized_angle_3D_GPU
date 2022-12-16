Rotates lossless in the Fourier domain! Abitrary angles, abitrary
rotation point. Limitation: Only z-axis rotation.
No padding, but wrap around.


# Usage

Call main file `LIRIN3DGPU.m` with parameters:
A: image 2d/3d
angle: rotation angle [degrees]
image_origin: x,y,z [meter]
rotation_point: x,y [meter]
resolution: resolution of pixels [meter]
out: rotated image 2d/3d

## Build GPU Code

Generate a DLL from the CUDA code:
```
build_dll.bat
```

Generate a MEX that includes the DLL. Run in Matlab:
```matlab
build_mex.m
```

## General structure

`cuda_kernel.cu` implements the function AmplitudeExtractionFourier2D on the GPU, with which lossless image rotation can be performed. The image is required inverse Fourier transformed. Therefore, the `GPUAmplitudeExtractionFourier2D.m` wrapper is used, which inversely Fourier-transforms the image and then calls the DLL generated from the kernel.
There is also the `mex_interface.c` file that implements the interface to enable data passing.
