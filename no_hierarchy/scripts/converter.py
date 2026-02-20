from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from structure import *
from parser import *

def get_pi_file_for_actor(actor: IRActor) -> str:
    """
    Select the appropriate .pi file for an actor.
    For most ops this is straightforward from OPTYPE_TO_PI.
    Special cases:
        - CONV: depends on whether bias input is present
        - ADD: depends on the size pattern of the two inputs
    """

    # -------------------------------------------------------------------------
    # CONV: check if bias input is present (3rd input)
    # -------------------------------------------------------------------------
    if actor.op_type in (OpType.CONV2D, OpType.CONV2D_BIAS):
        has_bias = len(actor.inputs) >= 3
        return "Algo/conv2d_bias.pi" if has_bias else "Algo/conv2d.pi"

    # -------------------------------------------------------------------------
    # ADD: detect pattern from linked input tensor sizes
    # -------------------------------------------------------------------------
    elif actor.op_type in (OpType.ADD_SAME, OpType.ADD_BIAS, OpType.ADD_SCALAR, OpType.ADD_GENERIC):
        t1 = actor.inputs[0][1] if len(actor.inputs) > 0 else None
        t2 = actor.inputs[1][1] if len(actor.inputs) > 1 else None

        size1 = t1.size if t1 else 0
        size2 = t2.size if t2 else 0

        if size1 == size2:
            return "Algo/add_same.pi"
        elif size2 == 1:
            return "Algo/add_scalar.pi"
        elif size1 > size2 and size2 > 0 and size1 % size2 == 0:
            return "Algo/add_bias.pi"
        else:
            return "Algo/add_generic.pi"

    # -------------------------------------------------------------------------
    # Everything else: straight lookup from the mapping table
    # -------------------------------------------------------------------------
    return OPTYPE_TO_PI.get(actor.op_type, "")

def extract_parameters_from_actor(actor: IRActor, initializer_names: set) -> dict:
    """
    Extract parameters from an actor using its already-linked IRTensor objects.
    Returns { param_name: int_value }
    """
    params = {}
    op    = actor.op_type
    attrs = actor.attributes

    def input_tensor(idx) -> Optional[IRTensor]:
        return actor.inputs[idx][1] if idx < len(actor.inputs) else None

    def output_tensor(idx) -> Optional[IRTensor]:
        return actor.outputs[idx][1] if idx < len(actor.outputs) else None

    # =========================================================================
    # RELU, SIGMOID, TANH, DROPOUT
    # Params: size
    # =========================================================================
    if op in (OpType.RELU, OpType.SIGMOID, OpType.TANH, OpType.DROPOUT):
        t = input_tensor(0)
        if t:
            params["size"] = t.size

    # =========================================================================
    # ADD variants
    # Params: size1, size2
    # =========================================================================
    elif op in (OpType.ADD_SAME, OpType.ADD_BIAS, OpType.ADD_SCALAR, OpType.ADD_GENERIC):
        t1 = input_tensor(0)
        t2 = input_tensor(1)
        if t1:
            params["size1"] = t1.size
        if t2:
            params["size2"] = t2.size

    # =========================================================================
    # CONV2D / CONV2D_BIAS
    # input_0: [N, C_in, H, W]
    # input_1: [C_out, C_in, kH, kW]  ← weight tensor
    # input_2: [C_out]                 ← bias (CONV2D_BIAS only)
    # =========================================================================
    elif op in (OpType.CONV2D, OpType.CONV2D_BIAS):
        input_t  = input_tensor(0)
        weight_t = input_tensor(1)
        output_t = output_tensor(0)

        if weight_t and len(weight_t.shape) >= 4:
            params["depthOutput"] = weight_t.shape[0]
            params["depthInput"]  = weight_t.shape[1]
            kernel_h              = weight_t.shape[-2]
            kernel_w              = weight_t.shape[-1]
        else:
            ks       = attrs.get("kernel_shape", [3, 3])
            kernel_h = ks[0]
            kernel_w = ks[1]

        params["sizeKernelHeight"] = kernel_h
        params["sizeKernelWidth"]  = kernel_w

        strides = attrs.get("strides", [1, 1])
        params["strideHeight"] = strides[0]
        params["strideWidth"]  = strides[1]

        dilations = attrs.get("dilations", [1, 1])
        pads      = attrs.get("pads", [0, 0, 0, 0])
        auto_pad  = attrs.get("auto_pad", "NOTSET")

        if input_t:
            in_h = input_t.height
            in_w = input_t.width
            params["inputHeight"] = in_h
            params["inputWidth"]  = in_w

            pad_top, pad_left, pad_bottom, pad_right = resolve_auto_pad(
                in_h, in_w, kernel_h, kernel_w,
                strides[0], strides[1], auto_pad, pads,
                dilations[0], dilations[1]
            )
            params["padTop"]    = pad_top
            params["padLeft"]   = pad_left
            params["padBottom"] = pad_bottom
            params["padRight"]  = pad_right

        if output_t:
            params["outputHeight"] = output_t.height
            params["outputWidth"]  = output_t.width

    # =========================================================================
    # MAXPOOL2D / AVGPOOL2D
    # input_0: [N, C, H, W]
    # =========================================================================
    elif op in (OpType.MAXPOOL2D, OpType.AVGPOOL2D):
        input_t  = input_tensor(0)
        output_t = output_tensor(0)

        if input_t:
            params["depthInput"] = input_t.channels
            in_h = input_t.height
            in_w = input_t.width
            params["inputHeight"] = in_h
            params["inputWidth"]  = in_w

        ks       = attrs.get("kernel_shape", [2, 2])
        kernel_h = ks[0]
        kernel_w = ks[1]
        params["poolHeight"] = kernel_h
        params["poolWidth"]  = kernel_w

        strides = attrs.get("strides", ks)
        params["strideHeight"] = strides[0]
        params["strideWidth"]  = strides[1]

        dilations = attrs.get("dilations", [1, 1])
        pads      = attrs.get("pads", [0, 0, 0, 0])
        auto_pad  = attrs.get("auto_pad", "NOTSET")
        ceil_mode = attrs.get("ceil_mode", 0) == 1

        if input_t:
            pad_top, pad_left, pad_bottom, pad_right = resolve_auto_pad(
                in_h, in_w, kernel_h, kernel_w,
                strides[0], strides[1], auto_pad, pads,
                dilations[0], dilations[1]
            )
            params["padTop"]    = pad_top
            params["padLeft"]   = pad_left
            params["padBottom"] = pad_bottom
            params["padRight"]  = pad_right

        if output_t:
            params["outputHeight"] = output_t.height
            params["outputWidth"]  = output_t.width

    # =========================================================================
    # GLOBAL AVERAGE POOL
    # input_0: [N, C, H, W]
    # =========================================================================
    elif op == OpType.GLOBAL_AVGPOOL:
        t = input_tensor(0)
        if t:
            params["depth"]       = t.channels
            params["spatialSize"] = t.height * t.width

    # =========================================================================
    # MATMUL
    # input_0: [M, K]
    # input_1: [K, N]
    # =========================================================================
    elif op == OpType.MATMUL:
        a = input_tensor(0)
        b = input_tensor(1)
        if a:
            params["M"] = a.shape[-2] if len(a.shape) >= 2 else 1
            params["K"] = a.shape[-1]
        if b:
            params["N"] = b.shape[-1]

    # =========================================================================
    # SOFTMAX
    # Params: size
    # =========================================================================
    elif op == OpType.SOFTMAX:
        t = input_tensor(0)
        if t:
            params["size"] = t.size

    # =========================================================================
    # RESHAPE
    # Params: inputSize, outputSize, shapeSize
    # shapeSize = number of dimensions in the target shape (e.g. [1, 512] -> 2)
    # =========================================================================
    elif op == OpType.RESHAPE:
        input_t  = input_tensor(0)
        output_t = output_tensor(0)
        shape_t  = input_tensor(1)   # the shape tensor (e.g. [1, 512])
        if input_t:
            params["inputSize"]  = input_t.size
        if output_t:
            params["outputSize"] = output_t.size
        if shape_t:
            params["shapeSize"]  = shape_t.size  # number of dims

    # =========================================================================
    # FLATTEN
    # Params: inputSize, outputSize
    # =========================================================================
    elif op == OpType.FLATTEN:
        input_t  = input_tensor(0)
        output_t = output_tensor(0)
        if input_t:
            params["inputSize"]  = input_t.size
        if output_t:
            params["outputSize"] = output_t.size

    # =========================================================================
    # CONCAT
    # Params: size1, size2 (first two inputs)
    # =========================================================================
    elif op == OpType.CONCAT:
        for idx in range(min(2, len(actor.inputs))):
            t = input_tensor(idx)
            if t:
                params[f"size{idx + 1}"] = t.size

    return params


def fill_IRGraph(model_data, shapes, offset_map) -> IRGraph:
    graph = IRGraph(model_data["name"])

    # Create all the tensors from shape inference
    for name, dims in shapes.items():
        graph.create_tensor(name, dims)

    registry = {} # Dic to deduplicate params, (param_name, value) -> unique_id

    # Create all the Actors
    for node in model_data["nodes"]:
        
        # =========================================================================
        # Resolve op type to the enum and create the actor 
        # ========================================================================= 
        op_key = node["op_type"].lower()
        op_type = ONNX_TO_OPTYPE.get(op_key) 
        if op_type is None:       # If op_type doesn't exist in enum, skip actor cuz can't be handled, you gotta add the actor urself :)
            print(f"[SKIP] Unknown op '{node['op_type']}' (node: '{node['name']}')")
            continue
        actor = graph.create_actor(op_type, node["name"])
        actor.attributes = node["attributes"] # Copying attributes dict (from onnx) to the actor to extract params
      

        # =========================================================================
        # Create the (Port, Tensor) for each actor, enumerate to have an idx for input_0 input_1 etc...
        # =========================================================================        

        # Adding inputs, printing a warning if we can't find the name in the graph (Should never happen tho since infer_shapes and parse_model should give trhe same tensor names)
        input_idx = 0
        weight_idx = 0
        
        for tensor_name in node["inputs"]:
            if tensor_name not in model_data["initializer_names"]:
                input_tensor = graph.get_tensor(tensor_name)
                if input_tensor is None:
                    print(f"  [WARN] Actor '{actor.unique_name}': input tensor '{tensor_name}' not found in graph")
                    continue
                port_name = f"input_{input_idx}"
                actor.add_input(port_name, input_tensor)
                graph.link(input_tensor, None, actor.get_port(port_name))
                input_idx += 1
        
            else:
                weight_tensor = graph.get_tensor(tensor_name)
                weight_tensor.declare_weight()
                if weight_tensor is None:
                    print(f"  [WARN] Actor '{actor.unique_name}': weight tensor '{tensor_name}' not found in graph")
                    continue
                port_name = f"weight_{weight_idx}"
                actor.add_weight(port_name, weight_tensor)
                graph.link(weight_tensor, None, actor.get_port(port_name))
                weight_idx += 1
        
        # Same but for outputs
        for idx, tensor_name in enumerate(node["outputs"]):
            output_tensor = graph.get_tensor(tensor_name)
            if output_tensor is None:
                print(f"  [WARN] Actor '{actor.unique_name}': output tensor '{tensor_name}' not found in graph")
                continue
            port_name = f"output_{idx}"
            actor.add_output(port_name, output_tensor)
            graph.link(output_tensor, actor.get_port(port_name), None)
        
    # =========================================================================
    # Handle params, first extract from attributes, then add to actor class
    # =========================================================================            
    for actor in graph.actors:
        raw_params = extract_parameters_from_actor(actor, model_data["initializer_names"])
        for param_name, value in raw_params.items():
            ir_param = graph.get_or_create_param(param_name, value)
            actor.add_param(param_name, ir_param)
        #actor.source = get_pi_file_for_actor(actor) 
        # actor.source = "Code/include/test.h"
    # ===============================================================================================
    # I/O Actors | load_weights and split_weights & fork actors for weights (one per dtype section)
    # ===============================================================================================
    add_io_actors(graph, model_data)

    add_weight_fork_actors(graph, model_data, offset_map)

    add_load_weights_actors(graph, offset_map)
    # =========================================================================
    # Detect and add BROADCAST actors for tensors with multiple consumers
    # =========================================================================     
    add_broadcast_actors(graph)
    return graph

def add_io_actors(graph: IRGraph, model_data: dict):

    for idx, inp in enumerate(model_data["inputs"]):
        tensor = graph.get_tensor(inp["name"])
        input_actor = graph.create_actor(OpType.LOAD_INPUT, f"load_input_{idx}")
        input_actor.source = "Code/include/load_input.h"

        size_param = graph.get_or_create_param("InputSize", tensor.size)
        input_actor.add_param("InputSize", size_param)

        input_actor.add_output("output", tensor)
        graph.link(tensor, input_actor.get_port("output"), None)

        for consumer_port in tensor.consumers:
            graph.link(tensor, input_actor.get_port("output"), consumer_port)

    for idx, out in enumerate(model_data["outputs"]):
        tensor = graph.get_tensor(out["name"])
        output_actor = graph.create_actor(OpType.OUTPUT, f"output_{idx}")
        output_actor.source = "Code/include/output.h"

        size_param = graph.get_or_create_param("OutputSize", tensor.size)
        output_actor.add_param("OutputSize", size_param)

        output_actor.add_input("input", tensor)
        graph.link(tensor, tensor.producer, output_actor.get_port("input"))  

def add_weight_fork_actors(graph: IRGraph, model_data: dict, offset_map: dict):
    """
    Create one SPLIT_WEIGHTS fork actor per dtype section and wire all weight edges.
    
    """ 

    ITEMSIZE = {
        "FLOAT": 4, "DOUBLE": 8,
        "INT8": 1, "INT16": 2, "INT32": 4, "INT64": 8,
        "UINT8": 1, "UINT16": 2, "UINT32": 4, "UINT64": 8,
        "BOOL": 1, "FLOAT16": 2, "BFLOAT16": 2,
    }

    # Build weight_consumers map: tensor_name -> list of (actor, port)
    weight_consumers = defaultdict(list)
    for actor in graph.actors:
        for port, tensor in actor.weights:
            weight_consumers[tensor.name].append((actor, port))

    global_weight_idx = 0

    for dtype_name, section in offset_map.items(): 

        itemsize       = ITEMSIZE.get(dtype_name, 4)
        total_elements = sum(info["n_elems"] for info in section["tensors"].values())
        dtype_lower    = dtype_name.lower()

        # Size param for this section
        size_param = graph.get_or_create_param(f"sizeWeights_{dtype_name}", total_elements)

        # Fork actor
        fork = graph.create_actor(OpType.SPLIT_WEIGHTS, f"split_weights_{dtype_name}") 
        fork.add_param("sizeWeights", size_param)

        # tensor representing the whole section as a fifo input
        section_tensor = graph.get_or_create_tensor(
            f"__weights_section_{dtype_name}__",
            [total_elements],
            dtype=dtype_lower
        )
        fork.add_input("input", section_tensor)

        # One output port per weight tensor
        for weight_name, info in section["tensors"].items():
            tensor = graph.get_tensor(weight_name)
            if tensor is None:
                print(f"  [WARN] '{weight_name}' not found in graph, skipping")
                global_weight_idx += 1
                continue

            consumers = weight_consumers.get(weight_name, [])

            # Port name encodes destination actor(s)
            if consumers:
                suffix = "_and_".join(a.unique_name for a, _ in consumers)
            else:
                suffix = "unused"

            port_name = f"weight_{global_weight_idx}_to_{suffix}"

            fork.add_output(port_name, tensor)
            graph.link(tensor, fork.get_port(port_name), None)

            # Wire to each consuming actor's weight port
            for actor, consumer_port in consumers:
                graph.link(tensor, fork.get_port(port_name), consumer_port)

            global_weight_idx += 1

def add_load_weights_actors(graph: IRGraph, offset_map: dict):
    for dtype_name, section in offset_map.items():
        total_elements = sum(info["n_elems"] for info in section["tensors"].values())
        dtype_lower    = dtype_name.lower()

        section_tensor = graph.get_or_create_tensor(
            f"__weights_section_{dtype_name}__",
            [total_elements],
            dtype=dtype_lower
        )

        # offset and size as PREESM config params
        offset_param = graph.get_or_create_param("offset", section["section_start"])
        size_param   = graph.get_or_create_param("size",   section["section_size"])

        load = graph.create_actor(OpType.LOAD_WEIGHTS, f"load_weights_{dtype_name}")
        load.source = "Code/include/load_weights.h"
        load.add_param("offset", offset_param)
        load.add_param("size",   size_param)
        load.add_output("output", section_tensor)
        graph.link(section_tensor, load.get_port("output"), None)

        # Wire to the fork that consumes this section
        for actor in graph.actors:
            if actor.op_type == OpType.SPLIT_WEIGHTS:
                for port, tensor in actor.inputs:
                    if tensor.name == section_tensor.name:
                        graph.link(section_tensor, load.get_port("output"), port)

def add_broadcast_actors(graph: IRGraph):
    """
    For each tensor that has multiple consumers,
    insert a BROADCAST actor between producer and consumers.
    """

    # We copy the list because we will modify graph.tensors during iteration
    original_tensors = graph.tensors

    for tensor in original_tensors:

        # Only data tensors (skip weights or internal)
        if tensor.isWeight:
            continue

        if tensor.producer is None:
            continue

        if len(tensor.consumers) <= 1:
            continue

        # -----------------------------------------------------
        # Create BROADCAST actor
        # -----------------------------------------------------
        bcast_actor = graph.create_actor(OpType.BROADCAST, f"broadcast")

        # -----------------------------------------------------
        # Create input tensor for broadcast (same shape/dtype)
        # -----------------------------------------------------
        in_tensor = graph.get_or_create_tensor(
            f"{tensor.name}_to_broadcast",
            tensor.shape,
            tensor.dtype
        )

        # Input port
        bcast_in_port = bcast_actor.add_input("input", in_tensor)

        # Rewire producer → broadcast
        graph.link(in_tensor, tensor.producer, bcast_in_port)

        # -----------------------------------------------------
        # For each consumer, create one output
        # -----------------------------------------------------
        old_consumers = list(tensor.consumers)
        tensor.consumers.clear()

        for idx, consumer_port in enumerate(old_consumers):

            # Create output tensor per consumer
            out_tensor = graph.get_or_create_tensor(
                f"{tensor.name}_bcast_{idx}",
                tensor.shape,
                tensor.dtype
            )

            out_port = bcast_actor.add_output(f"output_{idx}", out_tensor)

            # Link broadcast → consumer
            graph.link(out_tensor, out_port, consumer_port)

        # --------------------------------------------
        # Remove old tensor completely
        # --------------------------------------------
        del graph._tensors[tensor.name]

def _generate_actor_node(graph_el, actor: IRActor):

    if actor.op_type == OpType.SPLIT_WEIGHTS:
        kind = "fork"
    elif actor.op_type == OpType.BROADCAST:
        kind = "broadcast"
    else:
        kind = "actor"
    actor_el = SubElement(graph_el, "node", attrib={
        "id":     actor.unique_name,
        "kind":   kind,
        "period": "0",
    })

    # --- <data key="graph_desc"> ---
    SubElement(actor_el, "data", attrib={"key": "graph_desc"}).text = actor.source

    # Loop function name derived from the header filename
    # e.g. "Code/include/conv2d.h" -> "conv2d"
    loop_fn = actor.unique_name

    # --- <loop> ---
    loop_el = SubElement(actor_el, "loop", attrib={"name": loop_fn})

    # config params first
    for port, param in actor.params:
        SubElement(loop_el, "param", attrib={
            "direction": "IN",
            "isConfig":  "true",
            "name":      port.name,
            "type":      "int",
        })

    # data inputs
    for port, tensor in actor.inputs:
        SubElement(loop_el, "param", attrib={
            "direction": "IN",
            "isConfig":  "false",
            "name":      port.name,
            "type":      tensor.dtype,
        })

    # weight inputs
    for port, tensor in actor.weights:
        SubElement(loop_el, "param", attrib={
            "direction": "IN",
            "isConfig":  "false",
            "name":      port.name,
            "type":      tensor.dtype,
        })

    # outputs
    for port, tensor in actor.outputs:
        SubElement(loop_el, "param", attrib={
            "direction": "OUT",
            "isConfig":  "false",
            "name":      port.name,
            "type":      tensor.dtype,
        })

    # --- <port> elements ---
    for port, param in actor.params:
        SubElement(actor_el, "port", attrib={
            "kind": "cfg_input",
            "name": port.name,
        })

    for port, tensor in actor.inputs:
        SubElement(actor_el, "port", attrib={
            "kind":       "input",
            "name":       port.name,
            "expr":       str(port.rate),
            "annotation": "NONE",
        })

    for port, tensor in actor.weights:
        SubElement(actor_el, "port", attrib={
            "kind":       "input",
            "name":       port.name,
            "expr":       str(port.rate),
            "annotation": "NONE",
        })

    for port, tensor in actor.outputs:
        SubElement(actor_el, "port", attrib={
            "kind":       "output",
            "name":       port.name,
            "expr":       str(port.rate),
            "annotation": "NONE",
        })

def generate_xml(graph: IRGraph, model_data) -> str:


    # -------------------------------------------------------------------------
    # Root
    # -------------------------------------------------------------------------
    root = Element("graphml", attrib={
        "xmlns": "http://graphml.graphdrawing.org/xmlns",
    })

    # -------------------------------------------------------------------------
    # Graph node
    # -------------------------------------------------------------------------
    graph_el = SubElement(root, "graph", attrib={
        "edgedefault": "directed", 
    })
    
    graph_el0 = SubElement(graph_el, "data", attrib={
        "key":"name"
    })
    # -------------------------------------------------------------------------
    # Parameters (config nodes)
    # -------------------------------------------------------------------------
    for param in graph.params:
        SubElement(graph_el, "node", attrib={
            "id": param.unique_id,
            "kind": "param",
            "expr": str(param.value)
        })

    # -------------------------------------------------------------------------
    # I/O Actors
    # -------------------------------------------------------------------------
   
    # add_io_actors(graph, model_data) 
    # -------------------------------------------------------------------------
    # Actors (computation nodes)
    # -------------------------------------------------------------------------
    for actor in graph.actors:
        _generate_actor_node(graph_el, actor)
    # -------------------------------------------------------------------------
    # FIFO edges (data flow)
    # -------------------------------------------------------------------------
    for src_port, dst_port, dtype in graph.get_fifo_edges():
        src_actor = src_port.actor.unique_name if src_port.actor else "?"
        dst_actor = dst_port.actor.unique_name if dst_port.actor else "?"
        SubElement(graph_el, "edge", attrib={
            "kind":         "fifo",
            "source":       src_actor,
            "sourceport":   src_port.name,
            "target":       dst_actor,
            "targetport":   dst_port.name,
            "type":         dtype,
        })

    # -------------------------------------------------------------------------
    # Dependency edges (params -> actors)
    # -------------------------------------------------------------------------
    for param, port in graph.get_dependency_edges():
        dst_actor = port.actor.unique_name if port.actor else "?"
        SubElement(graph_el, "edge", attrib={
            "kind":         "dependency",
            "source":       param.unique_id,
            "target":       dst_actor,
            "targetport":   port.name,
        })

    # -------------------------------------------------------------------------
    # Pretty print
    # -------------------------------------------------------------------------
    raw = tostring(root, encoding="unicode")
    return parseString(raw).toprettyxml(indent="  ")


def write_xml(graph: IRGraph, model_data, output_path: str = "../output_graphs/output.pi"):
    xml_str = generate_xml(graph, model_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"XML written to {output_path}")

