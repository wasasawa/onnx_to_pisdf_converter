#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

void load_weights_float (int offset, int size, float*     output);
void load_weights_int64 (int offset, int size, long long* output);
void load_weights_int32 (int offset, int size, int*       output);
void load_weights_uint8 (int offset, int size, uint8_t*   output);


#endif
