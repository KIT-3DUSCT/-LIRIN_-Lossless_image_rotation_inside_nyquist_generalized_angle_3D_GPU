function [out] = GPUAmplitudeExtractionFourier2D(CE_in,delay1_in,delay2_in,fs_in)
CE_in = ifft2(CE_in);
image_real = real(CE_in);
image_imag = imag(CE_in);

out = mex_interface(image_real, delay1_in, delay2_in, fs_in, image_imag);
end

