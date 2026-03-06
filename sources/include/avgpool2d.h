#ifndef AVGPOOL2D_H
#define AVGPOOL2D_H

// =============================================================================
// Non-hierarchical — full pooling at once
// =============================================================================
void avgpool2d(
    int depthInput,
    int inputHeight,  int inputWidth,
    int poolHeight,   int poolWidth,
    int strideHeight, int strideWidth,
    int padTop,       int padLeft,
    int padBottom,    int padRight,
    int outputHeight, int outputWidth,
    int countIncludePad,
    float* input_0, float* output_0
);

// =============================================================================
// Hierarchical — one channel per firing
// =============================================================================
void avgpool2d_neuron(
    int depthInput,
    int inputHeight,  int inputWidth,
    int poolHeight,   int poolWidth,
    int strideHeight, int strideWidth,
    int padTop,       int padLeft,
    int padBottom,    int padRight,
    int outputHeight, int outputWidth,
    int countIncludePad,
    float* input_0, float* output_0
);

// =============================================================================
// Global average pool — averages entire spatial extent per channel
// =============================================================================
void global_avgpool(int depth, int spatialSize, float* input_0, float* output_0);

// Hierarchical — one channel per firing
void global_avgpool_neuron(int spatialSize, float* input_0, float* output_0);

#endif
