# ONNX to PiSDF Converter

Converts ONNX neural network models to PiSDF format. Hierarchical version not ready yet.

## Structure
```
.
├── scripts/
│   ├── main.py          # Main entry point
│   ├── parser.py        # Parses the information from the onnx format
│   ├── converter.py     # Fills the IR Graph with info from parser
│   └── structure.py     # Classes and structures used
├── sources/
│   ├── include/
│   │   ├── add.h
│   │   └── loadWeights.h
│   └── src/
│       ├── add.c
│       └── loadWeights.c
├── models/              # Put your .onnx models here
├── bin/                 # Generated weights.bin
├── output_graphs/       # Generated .pi output
└── README.md
```

## Usage
```bash
python scripts/main.py models/your_model.onnx
```

This will:
1. Parse the ONNX model
2. Run shape inference
3. Build the IR graph
4. Write `bin/weights.bin`
5. Generate output graph in `output_graphs/`

## Requirements
```bash
pip install onnx numpy
```

## Supported Operators

`Conv`, `Relu`, `Sigmoid`, `Tanh`, `MaxPool`, `AveragePool`, `GlobalAveragePool`,
`MatMul`, `Softmax`, `Reshape`, `Flatten`, `Concat`, `Add`, `Dropout`

## Notes

- Weight binary is sectioned by dtype (FLOAT, INT64, etc.)
- Add variants (same/bias/scalar/generic) are auto-detected from tensor shapes
- Unknown operators are skipped with a warning — add them to `ONNX_TO_OPTYPE` in `converter.py` and handle parameter extraction in `converter.py/extract_parameters_from_actor()`
