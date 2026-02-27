#include "load_weights.h"


#define WEIGHTS_PATH "weights.bin"

static void load_section(int offset, int size, void* output) {
    FILE* f = fopen(WEIGHTS_PATH, "rb");
    if (!f) {
        fprintf(stderr, "[load_weights] Cannot open " WEIGHTS_PATH "\n");
        return;
    }
    fseek(f, offset, SEEK_SET);
    fread(output, 1, size, f);
    fclose(f);
}

void load_weights_float(int offset, int size, float* output) {
    load_section(offset, size, output);
}

void load_weights_int64(int offset, int size, long long* output) {
    load_section(offset, size, output);
}

void load_weights_int32(int offset, int size, int* output) {
    load_section(offset, size, output);
}

void load_weights_uint8(int offset, int size, uint8_t* output) {
    load_section(offset, size, output);
}
