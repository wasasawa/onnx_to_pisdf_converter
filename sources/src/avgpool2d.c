#include "avgpool2d.h"

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
    float* input_0,   float* output_0
) {
    for (int c = 0; c < depthInput; c++) {
        float* in_ch  = input_0  + c * inputHeight  * inputWidth;
        float* out_ch = output_0 + c * outputHeight * outputWidth;
        for (int oh = 0; oh < outputHeight; oh++) {
            for (int ow = 0; ow < outputWidth; ow++) {
                float sum   = 0.0f;
                int   count = 0;
                for (int kh = 0; kh < poolHeight; kh++) {
                    for (int kw = 0; kw < poolWidth; kw++) {
                        int ih = oh * strideHeight - padTop  + kh;
                        int iw = ow * strideWidth  - padLeft + kw;
                        if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                            sum += in_ch[ih * inputWidth + iw];
                            count++;
                        }
                    }
                }
                out_ch[oh * outputWidth + ow] = count > 0 ? sum / count : 0.0f;
            }
        }
    }
}

// =============================================================================
// Hierarchical — one channel per firing
// =============================================================================
void avgpool2d_neuron(
    int inputHeight,  int inputWidth,
    int poolHeight,   int poolWidth,
    int strideHeight, int strideWidth,
    int padTop,       int padLeft,
    int padBottom,    int padRight,
    int outputHeight, int outputWidth,
    float* input_0,   float* output_0
) {
    for (int oh = 0; oh < outputHeight; oh++) {
        for (int ow = 0; ow < outputWidth; ow++) {
            float sum   = 0.0f;
            int   count = 0;
            for (int kh = 0; kh < poolHeight; kh++) {
                for (int kw = 0; kw < poolWidth; kw++) {
                    int ih = oh * strideHeight - padTop  + kh;
                    int iw = ow * strideWidth  - padLeft + kw;
                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                        sum += input_0[ih * inputWidth + iw];
                        count++;
                    }
                }
            }
            output_0[oh * outputWidth + ow] = count > 0 ? sum / count : 0.0f;
        }
    }
}

// =============================================================================
// Global average pool — average all spatial positions per channel
// =============================================================================
void global_avgpool(int depth, int spatialSize, float* input_0, float* output_0) {
    for (int c = 0; c < depth; c++) {
        float sum = 0.0f;
        for (int i = 0; i < spatialSize; i++)
            sum += input_0[c * spatialSize + i];
        output_0[c] = sum / spatialSize;
    }
}

// Hierarchical — one channel per firing
// input_0 is spatialSize elements, output_0 is 1 element
void global_avgpool_neuron(int spatialSize, float* input_0, float* output_0) {
    float sum = 0.0f;
    for (int i = 0; i < spatialSize; i++)
        sum += input_0[i];
    output_0[0] = sum / spatialSize;
}
