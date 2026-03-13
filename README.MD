# ONNX to PiSDF Converter

Converts ONNX neural network models to PiSDF (Parameterized and Interfaced Synchronous DataFlow) format for the [PREESM](https://preesm.github.io/) framework.


## Structure
```
.
├── bin/
│   └── weights.bin                # Generated binary weights file (created at runtime)
│
├── models/                        # Place your ONNX models here
│
├── output_graphs/                 # Generated PiSDF .pi output files
│
├── scripts/
│   ├── main.py                    # Entry point
│   ├── parser.py                  # Parses ONNX model, writes weights to binary
│   ├── converter.py               # Builds the IR graph, generates XML
│   ├── structure.py               # IR data structure definitions
│   ├── pi_generator.py            # Generates actor .pi files (hierarchical mode)
│   ├── actor_codegen.py           # Per-op C++ code generator classes (oneDNN)
│   └── model_codegen.py           # Orchestrates actor code generation for a full model
│
└── sources/
    ├── include/                   # Hand-written reference headers
    │   ├── add.h, conv2d.h, ...   # One header per operator
    │   └── runtime/               # Generated-kernel support headers
    │       ├── dnnl_utils.h       # oneDNN engine/stream/memory helpers
    │       └── blas_utils.h       # BLAS helpers (GEMM)
    │
    ├── src/                       # Hand-written C implementations
    │   ├── add.c, conv2d.c, ...
    │   └── loadWeights.c
    │
    ├── generated/                 # Output of model_codegen.py (created at runtime)
    │   ├── <model>_actors.h       # Function declarations for all actors in the model
    │   └── <model>_actors.cpp     # oneDNN-backed implementations
    │
    └── pi/                        # Generated .pi actor files (hierarchical mode)
```

## Usage
```bash
python scripts/main.py <model_path> [hierarchical] [isGeneratedKernels] [output_path] [weights_path]
```

| Argument | Description | Default |
|---|---|---|
| `model_path` | Path to the ONNX model | required |
| `hierarchical` | `1` for hierarchical `.pi` subgraphs, `0` for flat | `0` |
| `isGeneratedKernels` | `1` to generate oneDNN actor code, `0` for hand-written | `1` |
| `output_path` | Path for the generated `.pi` graph | `output_graphs/output.pi` |
| `weights_path` | Path for the generated `weights.bin` | `bin/weights.bin` |

### Examples
```bash
# Flat graph + generated oneDNN kernels (recommended)
python main.py ../models/cifar10_resnet32.onnx 0 1 ../output_graphs/resnet32.pi ../bin/weights.bin

# Flat graph + hand-written kernels (no code generation)
python main.py ../models/cifar10_resnet32.onnx 0 0 ../output_graphs/resnet32.pi ../bin/weights.bin


Both commands will:
1. Parse the ONNX model
2. Run shape inference
3. Build the IR graph
4. Write `weights.bin`
5. *(isGeneratedKernels=1)* Generate `sources/generated/<model>_actors.h/.cpp`
6. Generate the PiSDF graph at `output_path`
7. *(hierarchical=1)* Generate actor `.pi` subgraphs in `sources/pi/`

## Runtime Kernel Generator

When `isGeneratedKernels=1` (the default), the converter generates model-specific
C++ actor implementations backed by [oneDNN](https://oneapi-src.github.io/oneDNN/)
(formerly DNNL) primitives, replacing the generic hand-written C kernels.

5. Generate the PiSDF graph in `output_path/`
6. *(Hierarchical only)* Generate actor `.pi` files in `sources/pi/`

### Deploying to the PREESM runtime project

After running the converter:

1. Copy `sources/generated/<model>_actors.h` → `Code/include/runtime/`
2. Copy `sources/generated/<model>_actors.cpp` → `Code/src/runtime/`
3. Copy `sources/include/runtime/dnnl_utils.h` → `Code/include/runtime/`


### System requirements for generated kernels

```bash
# Ubuntu / Debian
sudo apt install libdnnl-dev libopenblas-dev
```


## Requirements
```bash
pip install onnx numpy
```

## Supported Operators

| Category | Operators |
|---|---|
| Convolution | `Conv` |
| Pooling | `MaxPool`, `AveragePool`, `GlobalAveragePool` |
| Activation | `Relu`, `Sigmoid`, `Tanh`, `Softmax`, `Dropout` |
| Linear | `MatMul`, `Gemm` |
| Reshape | `Reshape`, `Flatten` |
| Other | `Add`, `Concat`, `BatchNormalization`, `Transpose`, `Pad`, `Slice` |

> Add variants (`add_same` / `add_bias` / `add_scalar` / `add_generic`) are auto-detected from tensor shapes at conversion time.

## Extending with New Operators

To add support for an unsupported operator:

1. Add it to `ONNX_TO_OPTYPE` in `converter.py`
2. Add its `OpType` entry to `OPTYPE_TO_H` and `OPTYPE_TO_PI` in `converter.py`
3. Handle parameter extraction in `extract_parameters_from_actor()` in `ir_builder.py`
4. *(Hierarchical only)* Add its `RATE_EXPRESSIONS` entry in `pi_generator.py`
5. Implement the corresponding C function in `sources/src/` and `sources/include/`

Unknown operators are skipped with a warning during conversion.

## Notes


- Weights loader actor contains mlultiple versions for when the type is different, uses offset_map from create_weight_file_sanctionned() that gives info about what type is in each section + it's size
- The refinement of the I/O actors + weights_loader should be modified for each specific use case (and for each type concerning load_weights)
- The weights binary is sectioned by dtype (`FLOAT`, `INT64`, etc.) — `load_weights.h` and `load_weights.c` contain specific versions for each type, for now the converter doesn't assign the right function automatically it has to be done manually
- The IR graph is built in a single pass with runtime refinement for operators that have multiple variants (e.g. `Conv` with/without bias, `Add` broadcast pattern detection)

