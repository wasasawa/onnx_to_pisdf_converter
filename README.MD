# ONNX to PiSDF Converter

Converts ONNX neural network models to PiSDF format. Hierarchial version not ready yet.

## Structure
```
.
├── hierarchy/              # Version with hierarchical actors (.pi files)
│   ├── scripts/
│   │   ├── main.py	    # Main entry point
│   │   ├── parser.py	    # Parses the information from the onnx format
│   │   ├── converter.py    # Fills the IR Graph with ifo from parser
│   │   └── structure.py    # Classes and structures used
│   ├── sources/
│   │   ├── include/
│   │   │   └── loadWeights.h
│   │   └── src/
│   │       └── loadWeights.c
│   ├── models/             # Put your .onnx models here
│   ├── bin/                # Generated weights.bin
│   └── output_graphs/      # Generated .pi output
│
├── no_hierarchy/           # Version with flat actors (.h files)
│   ├── scripts/
│   │   ├── main.py
│   │   ├── parser.py
│   │   ├── converter.py
│   │   └── structure.py
│   ├── sources/
│   │   ├── include/
│   │   │   └── loadWeights.h
│   │   └── src/
│   │       └── loadWeights.c
│   ├── models/             # Put your .onnx models here
│   ├── bin/                # Generated weights.bin
│   └── output_graphs/      # Generated XML output
│
└── README.md
```

## Usage
```bash
cd no_hierarchy/scripts
python main.py ../models/your_model.onnx

# or for hierarchy version
cd hierarchy/scripts
python main.py ../models/your_model.onnx
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
- Unknown operators are skipped with a warning — add them to `ONNX_TO_OPTYPE` in `converter.py` and handle parameter extraction in converter.py/extract_parameters_from_actor()

