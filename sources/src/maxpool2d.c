#include "maxpool2d.h"
#include <float.h>

// =============================================================================
// Non-hierarchical
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
) {
    for (int c = 0; c < depthInput; c++) {
        float* in_ch  = input_0  + c * inputHeight  * inputWidth;
        float* out_ch = output_0 + c * outputHeight * outputWidth;
        for (int oh = 0; oh < outputHeight; oh++) {
            for (int ow = 0; ow < outputWidth; ow++) {
                float max_val = -FLT_MAX;
                for (int kh = 0; kh < poolHeight; kh++) {
                    for (int kw = 0; kw < poolWidth; kw++) {
                        int ih = oh * strideHeight - padTop  + kh;
                        int iw = ow * strideWidth  - padLeft + kw;
                        if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                            float val = in_ch[ih * inputWidth + iw];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                out_ch[oh * outputWidth + ow] = max_val;
            }
        }
    }
}

// =============================================================================
// Hierarchical — one channel per firing
// input_0  is one channel: [inputHeight  * inputWidth]
// output_0 is one channel: [outputHeight * outputWidth]
// =============================================================================

void maxpool2d_neuron(
    int inputHeight,  int inputWidth,
    int poolHeight,   int poolWidth,
    int strideHeight, int strideWidth,
    int padTop,       int padLeft,
    int padBottom,    int padRight,
    int outputHeight, int outputWidth,
    float* input_0, float* output_0
) {
    for (int oh = 0; oh < outputHeight; oh++) {
        for (int ow = 0; ow < outputWidth; ow++) {
            float max_val = -FLT_MAX;
            for (int kh = 0; kh < poolHeight; kh++) {
                for (int kw = 0; kw < poolWidth; kw++) {
                    int ih = oh * strideHeight - padTop  + kh;
                    int iw = ow * strideWidth  - padLeft + kw;
                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                        float val = input_0[ih * inputWidth + iw];
                        if (val > max_val) max_val = val;
                    }
                }
            }
            output_0[oh * outputWidth + ow] = max_val;
        }
    }
}
