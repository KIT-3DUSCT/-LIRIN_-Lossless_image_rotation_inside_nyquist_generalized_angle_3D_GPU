#include "mex.h"
#include "matrix.h"
#include "cuda_kernel.h"
#include <string.h>
#define FLOAT_DT float

void mexFunction(int output_n, mxArray *output[], int input_n, const mxArray *input[]) {
    // Input validation
    if (input_n != 5){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","Four inputs required.");
    }
    if (mxGetNumberOfDimensions(input[0]) > 2){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","First input is not a 2-dimensional-array");
    }
    if (!mxIsSingle(input[0])){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","First input should be singles");
    }
    if (mxGetNumberOfDimensions(input[1]) > 2){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","Second input is not a 2-dimensional-array");
    }
    if (!mxIsSingle(input[1])){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","Second input should be singles");
    }
    if (mxGetNumberOfDimensions(input[2]) > 2){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","Third input is not 2-dimensional-array");
    }
    if (!mxIsSingle(input[2])){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","Third input should be singles");
    }
    // Read dimension
    int* dimensions1 = (int*) mxGetDimensions(input[0]);
    int* dimensions2 = (int*) mxGetDimensions(input[1]);
    int* dimensions3 = (int*) mxGetDimensions(input[2]);
    int* dimensions4 = (int*) mxGetDimensions(input[3]);
    bool different_dimensions = false;

    if (dimensions1[0] != dimensions2[0] || dimensions1[2] != dimensions2[2]){
        different_dimensions = true; 
    }
    if (dimensions2[0] != dimensions3[0] || dimensions2[2] != dimensions3[2]){
        different_dimensions = true; 
    }
    if (different_dimensions){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","The dimensions of the input arrays are different");
    }
    if (mxGetNumberOfDimensions(input[3]) > 2 || dimensions4[0] != 1 || dimensions4[2] != 1){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","The fourth input should be a scalar");
    }
    if (!mxIsSingle(input[3])){
        mexErrMsgIdAndTxt("AmplitudeExtractionFourier2D:nrhs","The fourth input should be a single");
    }
    int max_y = dimensions1[0];
    int max_x = dimensions1[2];
    int total_length = max_x * max_y;

    // Read inputs
    FLOAT_DT *input1, *input2, *input3, *input5;
    FLOAT_DT input4;
    
    FLOAT_DT *output_local;
    double *output_double;
    output_local = (FLOAT_DT*)malloc(total_length*sizeof(FLOAT_DT));
    output_double = (double*)malloc(total_length*sizeof(double));

    input1 = (FLOAT_DT *)mxGetData(input[0]);
    input2 = (FLOAT_DT *)mxGetData(input[1]);
    input3 = (FLOAT_DT *)mxGetData(input[2]);
    input4 = (FLOAT_DT) mxGetScalar(input[3]);
    input5 = (FLOAT_DT *)mxGetData(input[4]);

    AmplitudeExtractionFourier2D(input1, input5, input2, input3, input4, output_local, max_x, max_y);

    for (int i = 0; i < total_length; i++) {
        output_double[i] = (double) output_local[i];
    }

    output[0] = mxCreateDoubleMatrix(total_length, 1, mxREAL);
    memcpy(mxGetPr(output[0]), output_double, total_length * sizeof(double));
    free(output_local);
    free(output_double);

    return;
}