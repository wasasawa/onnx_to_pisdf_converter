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
│   ├── pi_generator.py            # Generates actor .pi files (hierarchical mode)
│   └── structure.py               # IR data structure definitions
│
└── sources/
    ├── include/                   # Reference headers — copy into your PREESM Code/include/
    │   ├── add.h
    │   ├── avgpool2d.h
    │   ├── conv2d.h
    │   ├── load_weights.h         # Weights loader, contains mlultiple versions for when the type is different
    │   ├── maxpool2d.h
    │   ├── ops.h
    │   └── transpose.h
    │
    ├── src/                       # Reference C implementations — copy into your PREESM Code/src/
    │   ├── add.c
    │   ├── avgpool2d.c
    │   ├── conv2d.c
    │   ├── load_weights.c         
    │   ├── maxpool2d.c
    │   ├── ops.c
    │   └── transpose.c
    │
    └── pi/                        # Generated .pi actor files (hierarchical mode)
```

## Usage
```bash
python scripts/main.py <model_path> [hierarchical] [output_path] [weights_path]
```

| Argument | Description | Default |
|---|---|---|
| `model_path` | Path to the ONNX model | required |
| `hierarchical` | `1` for hierarchical mode, `0` for flat | `0` |
| `output_path` | Directory for the generated `.pi` graph | `output_graphs/` |
| `weights_path` | Directory for the generated `weights.bin` | `bin/` |

### Examples
```bash
# Flat mode — converts mnist-12, outputs to default locations
python scripts/main.py models/mnist-12.onnx 0

# Hierarchical mode — converts squeezenet, outputs to custom locations
python scripts/main.py models/squeezenet1.1.onnx 1 output/ weights/
```

Both commands will:
1. Parse the ONNX model
2. Run shape inference
3. Build the IR graph
4. Write `weights.bin`
5. Generate the PiSDF graph in `output_path/`
6. *(Hierarchical only)* Generate actor `.pi` files in `sources/pi/`

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

