#ifndef CONV2D_H
#define CONV2D_H

// =============================================================================
// Non-hierarchical — full convolution at once
// =============================================================================
void conv2d(
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    int depthInput,       int depthOutput,
    float* input_0, float* input_1, float* output_0
);

void conv2d_bias(
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    int depthInput,       int depthOutput,
    float* input_0, float* input_1, float* input_2, float* output_0
);

// =============================================================================
// Hierarchical — one output channel per firing
// input_0 is the full input (broadcast by PREESM)
// weight_0 is one filter slice (depthInput * kH * kW elements)
// output_0 is one output channel (outputHeight * outputWidth elements)
// =============================================================================
void conv2d_neuron(
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    int depthInput,	  int depthOutput,
    float* input_0, float* input_1, float* output_0
);

void conv2d_bias_neuron(
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    int depthInput,	  int depthOutput,
    float* input_0, float* input_1, float* input_2, float* output_0
);

#endif
