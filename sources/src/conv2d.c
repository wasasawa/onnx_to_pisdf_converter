#include "conv2d.h"
#include <string.h>

// =============================================================================
// Non-hierarchical
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
) {
    for (int oc = 0; oc < depthOutput; oc++) {
        for (int oh = 0; oh < outputHeight; oh++) {
            for (int ow = 0; ow < outputWidth; ow++) {
                float sum = 0.0f;
                for (int ic = 0; ic < depthInput; ic++) {
                    for (int kh = 0; kh < sizeKernelHeight; kh++) {
                        for (int kw = 0; kw < sizeKernelWidth; kw++) {
                            int ih = oh * strideHeight - padTop  + kh;
                            int iw = ow * strideWidth  - padLeft + kw;
                            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                int in_idx  = ic * inputHeight * inputWidth + ih * inputWidth + iw;
                                int w_idx   = oc * depthInput * sizeKernelHeight * sizeKernelWidth
                                            + ic * sizeKernelHeight * sizeKernelWidth
                                            + kh * sizeKernelWidth + kw;
                                sum += input_0[in_idx] * input_1[w_idx];
                            }
                        }
                    }
                }
                output_0[oc * outputHeight * outputWidth + oh * outputWidth + ow] = sum;
            }
        }
    }
}

void conv2d_bias(
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    int depthInput,       int depthOutput,
    float* input_0, float* input_1, float* input_2, float* output_0
) {
    conv2d(
        sizeKernelHeight, sizeKernelWidth,
        strideHeight, strideWidth,
        inputHeight, inputWidth,
        padTop, padLeft, padBottom, padRight,
        outputHeight, outputWidth,
        depthInput, depthOutput,
        input_0, input_1, output_0
    );
    // add bias per output channel
    for (int oc = 0; oc < depthOutput; oc++)
        for (int i = 0; i < outputHeight * outputWidth; i++)
            output_0[oc * outputHeight * outputWidth + i] += input_2[oc];
}

// =============================================================================
// Hierarchical — one output channel per firing
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
) {
    // weight_0 is one filter: [depthInput, kH, kW]
    for (int oh = 0; oh < outputHeight; oh++) {
        for (int ow = 0; ow < outputWidth; ow++) {
            float sum = 0.0f;
            for (int ic = 0; ic < depthInput; ic++) {
                for (int kh = 0; kh < sizeKernelHeight; kh++) {
                    for (int kw = 0; kw < sizeKernelWidth; kw++) {
                        int ih = oh * strideHeight - padTop  + kh;
                        int iw = ow * strideWidth  - padLeft + kw;
                        if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                            int in_idx = ic * inputHeight * inputWidth + ih * inputWidth + iw;
                            int w_idx  = ic * sizeKernelHeight * sizeKernelWidth
                                       + kh * sizeKernelWidth + kw;
                            sum += input_0[in_idx] * input_1[w_idx];
                        }
                    }
                }
            }
            output_0[oh * outputWidth + ow] = sum;
        }
    }
}

void conv2d_bias_neuron(
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    int depthInput,	  int depthOutput,
    float* input_0, float* input_1, float* input_2, float* output_0
) {
    conv2d_neuron(
        sizeKernelHeight, sizeKernelWidth,
        strideHeight, strideWidth,
        inputHeight, inputWidth,
        padTop, padLeft, padBottom, padRight,
        outputHeight, outputWidth,
        depthInput,
        input_0, input_1, output_0
    );
    // weight_1 aka input_2 is one bias scalar for this channel
    for (int i = 0; i < outputHeight * outputWidth; i++)
        output_0[i] += input_2[0];
}
