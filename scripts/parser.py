from collections import defaultdict
import onnx
import struct
import math
from onnx import TensorProto

def resolve_auto_pad(input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, 
                     auto_pad, pads, dilation_h=1, dilation_w=1):
    """
    Convert ONNX auto_pad to explicit [padTop, padLeft, padBottom, padRight].
    
    auto_pad values:
        - "NOTSET": Use explicit pads
        - "SAME_UPPER": Pad so output = ceil(input / stride), extra pad goes bottom/right
        - "SAME_LOWER": Same but extra pad goes top/left
        - "VALID": No padding
    """
    if auto_pad == "VALID":
        return 0, 0, 0, 0
    
    if auto_pad == "NOTSET" or not auto_pad:
        # Use explicit pads [top, left, bottom, right]
        pad_top = pads[0] if len(pads) > 0 else 0
        pad_left = pads[1] if len(pads) > 1 else 0
        pad_bottom = pads[2] if len(pads) > 2 else pad_top
        pad_right = pads[3] if len(pads) > 3 else pad_left
        return pad_top, pad_left, pad_bottom, pad_right
    
    # SAME_UPPER or SAME_LOWER: compute padding for "same" output size
    eff_kernel_h = (kernel_h - 1) * dilation_h + 1
    eff_kernel_w = (kernel_w - 1) * dilation_w + 1
    
    out_h = math.ceil(input_h / stride_h)
    out_w = math.ceil(input_w / stride_w)
    
    pad_h_total = max(0, (out_h - 1) * stride_h + eff_kernel_h - input_h)
    pad_w_total = max(0, (out_w - 1) * stride_w + eff_kernel_w - input_w)
    
    if auto_pad == "SAME_UPPER":
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
    else:  # SAME_LOWER
        pad_bottom = pad_h_total // 2
        pad_top = pad_h_total - pad_bottom
        pad_right = pad_w_total // 2
        pad_left = pad_w_total - pad_right
    
    return pad_top, pad_left, pad_bottom, pad_right


def parse_onnx_model(model_path):
    """
    Load ONNX model and extract all relevant information.
    
    Separates:
        - Real inputs: data tensors fed into the graph
        - Initializers: weights/biases stored in the model
        - Nodes: computation operators
        - Outputs: final results
    
    Returns:
        Dictionary with all parsed model components
    """
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    # Initializers are weights/biases embedded in the model
    initializer_names = {init.name for init in graph.initializer}

    # Real inputs are graph inputs that aren't initializers

    real_inputs = [
        {
            "name": inp.name,
            "type": inp.type.tensor_type.elem_type,
            "shape": [dim.dim_value for dim in inp.type.tensor_type.shape.dim],
        }
        for inp in graph.input
        if inp.name not in initializer_names
    ]


    # Extract initializer metadata and data

    initializers = [
        {
            "name": init.name,
            "data_type": init.data_type,
            "dims": list(init.dims),
            "data": init,
        }
        for init in graph.initializer
    ]

    # Extract output info
    outputs = [
        {
            "name": out.name,
            "type": out.type.tensor_type.elem_type,
            "shape": [
                dim.dim_value if dim.dim_value > 0 else dim.dim_param
                for dim in out.type.tensor_type.shape.dim
            ],
        }
        for out in graph.output
    ]

    # Parse nodes with their attributes
    nodes = []
    for node in graph.node:
        attr_dict = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                attr_dict[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                attr_dict[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attr_dict[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                attr_dict[attr.name] = (
                    attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                )

        nodes.append({
            "op_type": node.op_type,
            "name": node.name or f"{node.op_type}_{len(nodes)}",
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": attr_dict,
        })

    return {
        "name" : graph.name,
        "inputs": real_inputs,
        "outputs": outputs,
        "initializers": initializers,
        "initializer_names": initializer_names,
        # Maps each initializer tensor name to its ONNX data-type integer
        # (e.g. TensorProto.FLOAT=1, TensorProto.INT64=7).  Used in fill_IRGraph
        # to stamp the correct dtype on every IRTensor instead of defaulting all to "float".
        "initializer_dtype_map": {init.name: init.data_type for init in graph.initializer},
        "nodes": nodes,
        "model": model,
    }

def infer_tensor_shapes(model, initializers):
    """
    Run ONNX shape inference to get dimensions of all tensors.
    
    Used for:
        - Determining port rates in PiSDF
        - Computing parallelism factors
        - Detecting Add broadcasting patterns
    
    Returns:
        Dictionary mapping tensor names to their shapes
    """
    inferred_model = onnx.shape_inference.infer_shapes(model)
    shapes = {}

    # Intermediate tensors (between nodes)
    for vi in inferred_model.graph.value_info:
        dims = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in vi.type.tensor_type.shape.dim
        ]
        shapes[vi.name] = dims

    # Weight tensors
    for init in initializers:
        shapes[init["name"]] = init["dims"]

    # Graph inputs
    for inp in inferred_model.graph.input:
        dims = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in inp.type.tensor_type.shape.dim
        ]
        shapes[inp.name] = dims

    # Graph outputs
    for out in inferred_model.graph.output:
        dims = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in out.type.tensor_type.shape.dim
        ]
        shapes[out.name] = dims

    return shapes



def create_weights_file(initializers, output_path="weights.bin"):
    """
    Write all weights and biases to a binary file.
    
    Weights are written in order of appearance in the ONNX model.
    The fork node in the generated graph distributes them to actors.
    """
    print(f"Creating weights file: {output_path}")

    total_size = 0
    with open(output_path, "wb") as f:
        for init in initializers:
            data = init["data"]
            if data.float_data:
                for val in data.float_data:
                    f.write(struct.pack("f", val))
                total_size += len(data.float_data) * 4
            elif data.raw_data:
                f.write(data.raw_data)
                total_size += len(data.raw_data)

    print(f"  Total size: {total_size} bytes")
    print(f"  Weights/biases: {len(initializers)}")
    return total_size


def create_weights_file_sectioned(initializers, output_path="bin/weights.bin"):
    """
    Write all initializer tensors to a binary file grouped by dtype.

    Returns
    -------
    offset_map : dict
        {
          "FLOAT": {
              "section_start": 0,
              "section_size":  1024,   # bytes
              "tensors": {
                  "conv1.weight": {
                      "offset_in_section": 0,
                      "offset_in_file":    0,
                      "size":              256,  
                      "n_elems":           64,
                  }, ...
              }
          }, ...
        }
    """
    

    DTYPE_CONFIG = {
        TensorProto.FLOAT:    ("f", 4),
        TensorProto.DOUBLE:   ("d", 8),
        TensorProto.INT8:     ("b", 1),
        TensorProto.INT16:    ("h", 2),
        TensorProto.INT32:    ("i", 4),
        TensorProto.INT64:    ("q", 8),
        TensorProto.UINT8:    ("B", 1),
        TensorProto.UINT16:   ("H", 2),
        TensorProto.UINT32:   ("I", 4),
        TensorProto.UINT64:   ("Q", 8),
        TensorProto.BOOL:     ("B", 1),
        TensorProto.FLOAT16:  (None, 2),
        TensorProto.BFLOAT16: (None, 2),
    }

    PROTO_FIELD = {
        TensorProto.FLOAT:   "float_data",
        TensorProto.DOUBLE:  "double_data",
        TensorProto.INT8:    "int32_data",
        TensorProto.INT16:   "int32_data",
        TensorProto.INT32:   "int32_data",
        TensorProto.INT64:   "int64_data",
        TensorProto.UINT8:   "int32_data",
        TensorProto.UINT16:  "int32_data",
        TensorProto.UINT32:  "uint64_data",
        TensorProto.UINT64:  "uint64_data",
        TensorProto.BOOL:    "int32_data",
    }

    def extract_raw_bytes(tensor_data, dtype):
        fmt_char, itemsize = DTYPE_CONFIG[dtype]
        if tensor_data.raw_data:
            return tensor_data.raw_data
        field = PROTO_FIELD.get(dtype)
        if field:
            values = getattr(tensor_data, field)
            if values:
                return struct.pack(f"{len(values)}{fmt_char}", *values)
        return b""

    # Group initializers by dtype
    groups = defaultdict(list)
    for init in initializers:
        data  = init["data"]
        dtype = data.data_type
        if dtype not in DTYPE_CONFIG:
            print(f"  [SKIP] '{init['name']}': unsupported dtype {dtype}")
            continue
        raw_bytes = extract_raw_bytes(data, dtype)
        if not raw_bytes:
            print(f"  [WARN] '{init['name']}': empty, skipping")
            continue
        dtype_name = TensorProto.DataType.Name(dtype)
        groups[dtype_name].append((init, raw_bytes))

    # Write file section by section
    offset_map = {}
    current_offset = 0

    print(f"Creating weights file: {output_path}")
    with open(output_path, "wb") as f:
        for dtype_name, entries in groups.items():
            _, itemsize   = DTYPE_CONFIG[next(
                k for k in DTYPE_CONFIG
                if TensorProto.DataType.Name(k) == dtype_name
            )]
            section_start = current_offset
            section_size  = sum(len(b) for _, b in entries)

            print(f"  [{dtype_name}] {len(entries)} tensors, {section_size} bytes at offset {section_start}")

            offset_map[dtype_name] = {
                "section_start": section_start,
                "section_size":  section_size,
                "tensors":       {},
            }

            offset_in_section = 0
            for init, raw_bytes in entries:
                n_elems = len(raw_bytes) // itemsize
                f.write(raw_bytes)

                offset_map[dtype_name]["tensors"][init["name"]] = {
                    "offset_in_section": offset_in_section,
                    "offset_in_file":    section_start + offset_in_section,
                    "size":              len(raw_bytes),
                    "n_elems":           n_elems,
                }

                offset_in_section  += len(raw_bytes)
                current_offset     += len(raw_bytes)

    print(f"  Total: {current_offset} bytes")
    return offset_map

