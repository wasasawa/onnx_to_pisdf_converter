#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

// =============================================================================
// Non-hierarchical — full pooling at once
// =============================================================================
void maxpool2d(
    int depthInput,
    int inputHeight,  int inputWidth,
    int poolHeight,   int poolWidth,
    int strideHeight, int strideWidth,
    int padTop,       int padLeft,
    int padBottom,    int padRight,
    int outputHeight, int outputWidth,
    float* input_0, float* output_0
);

// =============================================================================
// Hierarchical — one channel per firing
// =============================================================================
void maxpool2d_neuron(
    int inputHeight,  int inputWidth,
    int poolHeight,   int poolWidth,
    int strideHeight, int strideWidth,
    int padTop,       int padLeft,
    int padBottom,    int padRight,
    int outputHeight, int outputWidth,
    float* input_0, float* output_0
);

#endif
