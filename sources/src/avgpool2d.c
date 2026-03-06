#include "avgpool2d.h"

/* ----------------------------------------------------------------------------
 * Inner kernel — one channel.
 * countIncludePad = 0 → divide by number of valid (non-padded) pixels
 * countIncludePad = 1 → always divide by poolHeight * poolWidth
 * --------------------------------------------------------------------------*/
static void _avgpool2d_channel(
    int inputHeight,  int inputWidth,
    int poolHeight,   int poolWidth,
    int strideHeight, int strideWidth,
    int padTop,       int padLeft,
    int outputHeight, int outputWidth,
    int countIncludePad,
    const float* in_ch, float* out_ch
) {
    int pool_area = poolHeight * poolWidth;

    for (int oh = 0; oh < outputHeight; oh++) {
        for (int ow = 0; ow < outputWidth; ow++) {
            float sum   = 0.0f;
            int   count = 0;
            for (int kh = 0; kh < poolHeight; kh++) {
                for (int kw = 0; kw < poolWidth; kw++) {
                    int ih = oh * strideHeight - padTop  + kh;
                    int iw = ow * strideWidth  - padLeft + kw;
                    if ((unsigned)ih < (unsigned)inputHeight &&
                        (unsigned)iw < (unsigned)inputWidth) {
                        sum += in_ch[ih * inputWidth + iw];
                        count++;
                    }
                }
            }
            int divisor = countIncludePad ? pool_area : count;
            out_ch[oh * outputWidth + ow] = divisor > 0 ? sum / divisor : 0.0f;
        }
    }
}

// =============================================================================
// Non-hierarchical
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
) {
    int in_spatial  = inputHeight  * inputWidth;
    int out_spatial = outputHeight * outputWidth;

    for (int c = 0; c < depthInput; c++) {
        _avgpool2d_channel(
            inputHeight, inputWidth,
            poolHeight, poolWidth,
            strideHeight, strideWidth,
            padTop, padLeft,
            outputHeight, outputWidth,
            countIncludePad,
            input_0  + c * in_spatial,
            output_0 + c * out_spatial
        );
    }
}

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
) {
    _avgpool2d_channel(
        inputHeight, inputWidth,
        poolHeight, poolWidth,
        strideHeight, strideWidth,
        padTop, padLeft,
        outputHeight, outputWidth,
        countIncludePad,
        input_0, output_0
    );
}

// =============================================================================
// Global average pool
// =============================================================================

void global_avgpool(int depth, int spatialSize, float* input_0, float* output_0) {
    float inv = 1.0f / spatialSize;
    for (int c = 0; c < depth; c++) {
        const float* in_ch = input_0 + c * spatialSize;
        float sum = 0.0f;
        for (int i = 0; i < spatialSize; i++)
            sum += in_ch[i];
        output_0[c] = sum * inv;
    }
}

/* Hierarchical — one channel per firing */
void global_avgpool_neuron(int spatialSize, float* input_0, float* output_0) {
    float sum = 0.0f;
    float inv = 1.0f / spatialSize;
    for (int i = 0; i < spatialSize; i++)
        sum += input_0[i];
    output_0[0] = sum * inv;
}
