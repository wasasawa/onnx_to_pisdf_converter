#include "conv2d.h"
#include <string.h>

/* ----------------------------------------------------------------------------
 * Inner kernel shared by all variants.
 * Computes one output spatial map for ONE output channel.
 * Caller passes:
 *   in       – pointer to the full input tensor  (CHW layout)
 *   w        – pointer to one filter [depthInput × kH × kW]
 *   out      – pointer to one output channel     [outH × outW]
 * --------------------------------------------------------------------------*/
static void _conv2d_channel(
    int depthInput,
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int dilationHeight,   int dilationWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int outputHeight,     int outputWidth,
    const float* in, const float* w, float* out
) {
    for (int oh = 0; oh < outputHeight; oh++) {
        for (int ow = 0; ow < outputWidth; ow++) {
            float sum = 0.0f;
            for (int ic = 0; ic < depthInput; ic++) {
                const float* in_ch = in + ic * inputHeight * inputWidth;
                const float* w_ch  = w  + ic * sizeKernelHeight * sizeKernelWidth;
                for (int kh = 0; kh < sizeKernelHeight; kh++) {
                    for (int kw = 0; kw < sizeKernelWidth; kw++) {
                        int ih = oh * strideHeight - padTop  + kh * dilationHeight;
                        int iw = ow * strideWidth  - padLeft + kw * dilationWidth;
                        if ((unsigned)ih < (unsigned)inputHeight &&
                            (unsigned)iw < (unsigned)inputWidth) {
                            sum += in_ch[ih * inputWidth + iw]
                                 * w_ch [kh * sizeKernelWidth + kw];
                        }
                    }
                }
            }
            out[oh * outputWidth + ow] = sum;
        }
    }
}

// =============================================================================
// Non-hierarchical
// =============================================================================

void conv2d(
    int depthOutput,      int depthInput,
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int dilationHeight,   int dilationWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    float* input_0, float* input_1, float* output_0
) {
    int filter_size = depthInput * sizeKernelHeight * sizeKernelWidth;
    int out_spatial = outputHeight * outputWidth;

    for (int oc = 0; oc < depthOutput; oc++) {
        _conv2d_channel(
            depthInput,
            sizeKernelHeight, sizeKernelWidth,
            strideHeight, strideWidth,
            dilationHeight, dilationWidth,
            inputHeight, inputWidth,
            padTop, padLeft,
            outputHeight, outputWidth,
            input_0,
            input_1 + oc * filter_size,
            output_0 + oc * out_spatial
        );
    }
}

void conv2d_bias(
    int depthOutput,      int depthInput,
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int dilationHeight,   int dilationWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    float* input_0, float* input_1, float* input_2, float* output_0
) {
    conv2d(
        depthOutput, depthInput,
        sizeKernelHeight, sizeKernelWidth,
        strideHeight, strideWidth,
        dilationHeight, dilationWidth,
        inputHeight, inputWidth,
        padTop, padLeft, padBottom, padRight,
        outputHeight, outputWidth,
        input_0, input_1, output_0
    );
    /* Add bias: one scalar per output channel */
    int out_spatial = outputHeight * outputWidth;
    for (int oc = 0; oc < depthOutput; oc++) {
        float bias = input_2[oc];
        float* out_ch = output_0 + oc * out_spatial;
        for (int i = 0; i < out_spatial; i++)
            out_ch[i] += bias;
    }
}

// =============================================================================
// Hierarchical — one output channel per firing
// =============================================================================

void conv2d_neuron(
    int depthOutput,      int depthInput,
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int dilationHeight,   int dilationWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    float* input_0, float* input_1, float* output_0
) {
    /* input_1 is already the single filter slice for this channel */
    _conv2d_channel(
        depthInput,
        sizeKernelHeight, sizeKernelWidth,
        strideHeight, strideWidth,
        dilationHeight, dilationWidth,
        inputHeight, inputWidth,
        padTop, padLeft,
        outputHeight, outputWidth,
        input_0, input_1, output_0
    );
}

void conv2d_bias_neuron(
    int depthOutput,      int depthInput,
    int sizeKernelHeight, int sizeKernelWidth,
    int strideHeight,     int strideWidth,
    int dilationHeight,   int dilationWidth,
    int inputHeight,      int inputWidth,
    int padTop,           int padLeft,
    int padBottom,        int padRight,
    int outputHeight,     int outputWidth,
    float* input_0, float* input_1, float* input_2, float* output_0
) {
    conv2d_neuron(
        depthOutput, depthInput,
        sizeKernelHeight, sizeKernelWidth,
        strideHeight, strideWidth,
        dilationHeight, dilationWidth,
        inputHeight, inputWidth,
        padTop, padLeft, padBottom, padRight,
        outputHeight, outputWidth,
        input_0, input_1, output_0
    );
    /* input_2[0] is the bias scalar for this output channel */
    float bias      = input_2[0];
    int   out_spatial = outputHeight * outputWidth;
    for (int i = 0; i < out_spatial; i++)
        output_0[i] += bias;
}
