from xml.etree.ElementTree import Element, SubElement, ElementTree, indent
from converter import OpType, IRActor, IRGraph, OPTYPE_TO_PI, OPTYPE_TO_H
import os

RATE_EXPRESSIONS = {
    # -------------------------------------------------------------------------
    # Conv — one firing per output channel
    # input is reused for each channel- induces broadcast (hopefully ?) 
    # weights are split one filter per firing 
    # -------------------------------------------------------------------------
    "conv2d": {
        "inputs": {
            "input_0":  "depthInput * inputHeight * inputWidth",
            "weight_0": "depthInput * sizeKernelHeight * sizeKernelWidth",
        },
        "outputs": {
            "output_0": "outputHeight * outputWidth",
        },
    },
    "conv2d_bias": {
        "inputs": {
            "input_0":  "depthInput * inputHeight * inputWidth",
            "weight_0": "depthInput * sizeKernelHeight * sizeKernelWidth",
            "weight_1": "1",  # one bias value per output channel per firing
        },
        "outputs": {
            "output_0": "outputHeight * outputWidth",
        },
    },

    # -------------------------------------------------------------------------
    # Pool — one firing per channel
    # each firing takes one full channel and outputs the pooled result
    # -------------------------------------------------------------------------
    "maxpool2d": {
        "inputs":  {"input_0": "inputHeight * inputWidth"},
        "outputs": {"output_0": "outputHeight * outputWidth"},
    },
    "avgpool2d": {
        "inputs":  {"input_0": "inputHeight * inputWidth"},
        "outputs": {"output_0": "outputHeight * outputWidth"},
    },
    "global_avgpool": {
        "inputs":  {"input_0": "spatialSize"},  # full spatial size per channel
        "outputs": {"output_0": "1"},           # one value out per channel
    },

    # -------------------------------------------------------------------------
    # Element-wise ops — one element in, one element out per firing
    # -------------------------------------------------------------------------
    "relu": {
        "inputs":  {"input_0": "1"},
        "outputs": {"output_0": "1"},
    },
    "sigmoid": {
        "inputs":  {"input_0": "1"},
        "outputs": {"output_0": "1"},
    },
    "tanh": {
        "inputs":  {"input_0": "1"},
        "outputs": {"output_0": "1"},
    },
    "softmax": {
        "inputs":  {"input_0": "1"},
        "outputs": {"output_0": "1"},
    },
    "dropout": {
        "inputs":  {"input_0": "1"},
        "outputs": {"output_0": "1"},
    },

    # -------------------------------------------------------------------------
    # Add — one element per input per firing
    # -------------------------------------------------------------------------
    "add_same": {
        "inputs":  {"input_0": "1", "input_1": "1"},
        "outputs": {"output_0": "1"},
    },
    "add_bias": {
        "inputs":  {"input_0": "1", "input_1": "1"},
        "outputs": {"output_0": "1"},
    },
    "add_scalar": {
        "inputs":  {"input_0": "1", "input_1": "1"},
        "outputs": {"output_0": "1"},
    },
    "add_generic": {
        "inputs":  {"input_0": "1", "input_1": "1"},
        "outputs": {"output_0": "1"},
    },

    # -------------------------------------------------------------------------
    # MatMul — one firing per output value
    # each firing takes one row from input and one column from weights
    # -------------------------------------------------------------------------
    "matmul": {
        "inputs":  {"input_0": "K", "weight_0": "K"},
        "outputs": {"output_0": "1"},
    },

    # -------------------------------------------------------------------------
    # Reshape/Flatten/Transpose — one element per firing, just reorders data - Maybe better to put as glogbal later ? cuz
    # -------------------------------------------------------------------------
    "reshape": {
        "inputs":  {"input_0": "1", "weight_0": "1"},
        "outputs": {"output_0": "1"},
    },
    "flatten": {
        "inputs":  {"input_0": "1"},
        "outputs": {"output_0": "1"},
    },
    "transpose": {
        "inputs":  {"input_0": "1"},
        "outputs": {"output_0": "1"},
    },

    # -------------------------------------------------------------------------
    # Concat — one element from each input per firing
    # -------------------------------------------------------------------------
    "concat": {
        "inputs":  {"input_0": "1", "input_1": "1"},
        "outputs": {"output_0": "1"},
    },
}

# Ops that keep .h and never get a .pi
SKIP_OPS = {
    OpType.LOAD_WEIGHTS,
    OpType.LOAD_INPUT,
    OpType.OUTPUT,
    OpType.SPLIT_WEIGHTS,
    OpType.CONSTANT_FILL,
}


def _get_op_name(actor: IRActor) -> str:
    return OPTYPE_TO_PI[actor.op_type].split("/")[-1].replace(".pi", "")


def _get_expr(rate_exprs: dict, port_name: str, section: str, fallback: int) -> str:
    return rate_exprs.get(section, {}).get(port_name, str(fallback))


def generate_pi_file(actor: IRActor, output_dir: str) -> str:
    """
    Generate a .pi file for one actor op type from its IR representation.
    Returns the path to the generated file.
    """
    op_name    = _get_op_name(actor)
    pi_path    = os.path.join(output_dir, f"{op_name}.pi")
    rate_exprs = RATE_EXPRESSIONS.get(op_name, {})
    inner_name = f"{op_name}_neuron"
    h_source   = OPTYPE_TO_H[actor.op_type]

    root     = Element("graphml", attrib={"xmlns": "http://graphml.graphdrawing.org/graphml"})
    graph_el = SubElement(root, "graph", attrib={"edgedefault": "directed"})
    SubElement(graph_el, "data", attrib={"key": "name"}).text = op_name

    # -------------------------------------------------------------------------
    # cfg_in_iface nodes
    # -------------------------------------------------------------------------
    for port, param in actor.params:
        SubElement(graph_el, "node", attrib={
            "id":   port.name,
            "kind": "cfg_in_iface",
        })

    # -------------------------------------------------------------------------
    # src nodes
    # -------------------------------------------------------------------------
    for port, tensor in actor.inputs:
        expr   = _get_expr(rate_exprs, port.name, "inputs", port.rate)
        src_el = SubElement(graph_el, "node", attrib={"id": port.name, "kind": "src"})
        SubElement(src_el, "port", attrib={
            "annotation": "NONE",
            "expr":       expr,
            "kind":       "output",
            "name":       port.name,
        })

    for port, tensor in actor.weights:
        expr   = _get_expr(rate_exprs, port.name, "inputs", port.rate)
        src_el = SubElement(graph_el, "node", attrib={"id": port.name, "kind": "src"})
        SubElement(src_el, "port", attrib={
            "annotation": "NONE",
            "expr":       expr,
            "kind":       "output",
            "name":       port.name,
        })

    # -------------------------------------------------------------------------
    # snk nodes
    # -------------------------------------------------------------------------
    for port, tensor in actor.outputs:
        expr   = _get_expr(rate_exprs, port.name, "outputs", port.rate)
        snk_el = SubElement(graph_el, "node", attrib={"id": port.name, "kind": "snk"})
        SubElement(snk_el, "port", attrib={
            "annotation": "NONE",
            "expr":       expr,
            "kind":       "input",
            "name":       port.name,
        })

    # -------------------------------------------------------------------------
    # Inner actor node
    # -------------------------------------------------------------------------
    inner_el = SubElement(graph_el, "node", attrib={"id": inner_name, "kind": "actor"})
    SubElement(inner_el, "data", attrib={"key": "graph_desc"}).text = h_source

    loop_el = SubElement(inner_el, "loop", attrib={"name": op_name})

    for port, param in actor.params:
        SubElement(loop_el, "param", attrib={
            "direction": "IN", "isConfig": "true",
            "name": port.name, "type": "int",
        })
    for port, tensor in actor.inputs:
        SubElement(loop_el, "param", attrib={
            "direction": "IN", "isConfig": "false",
            "name": port.name, "type": tensor.dtype,
        })
    for port, tensor in actor.weights:
        SubElement(loop_el, "param", attrib={
            "direction": "IN", "isConfig": "false",
            "name": port.name, "type": tensor.dtype,
        })
    for port, tensor in actor.outputs:
        SubElement(loop_el, "param", attrib={
            "direction": "OUT", "isConfig": "false",
            "name": port.name, "type": tensor.dtype,
        })

    # ports on inner actor
    for port, param in actor.params:
        SubElement(inner_el, "port", attrib={
            "kind": "cfg_input",
            "name": port.name,
        })
    for port, tensor in actor.inputs:
        expr = _get_expr(rate_exprs, port.name, "inputs", port.rate)
        SubElement(inner_el, "port", attrib={
            "annotation": "NONE", "expr": expr,
            "kind": "input", "name": port.name,
        })
    for port, tensor in actor.weights:
        expr = _get_expr(rate_exprs, port.name, "inputs", port.rate)
        SubElement(inner_el, "port", attrib={
            "annotation": "NONE", "expr": expr,
            "kind": "input", "name": port.name,
        })
    for port, tensor in actor.outputs:
        expr = _get_expr(rate_exprs, port.name, "outputs", port.rate)
        SubElement(inner_el, "port", attrib={
            "annotation": "NONE", "expr": expr,
            "kind": "output", "name": port.name,
        })

    # -------------------------------------------------------------------------
    # fifo edges
    # -------------------------------------------------------------------------
    for port, tensor in actor.inputs:
        SubElement(graph_el, "edge", attrib={
            "kind": "fifo", "type": tensor.dtype,
            "source": port.name, "sourceport": port.name,
            "target": inner_name, "targetport": port.name,
        })
    for port, tensor in actor.weights:
        SubElement(graph_el, "edge", attrib={
            "kind": "fifo", "type": tensor.dtype,
            "source": port.name, "sourceport": port.name,
            "target": inner_name, "targetport": port.name,
        })
    for port, tensor in actor.outputs:
        SubElement(graph_el, "edge", attrib={
            "kind": "fifo", "type": tensor.dtype,
            "source": inner_name, "sourceport": port.name,
            "target": port.name, "targetport": port.name,
        })

    # -------------------------------------------------------------------------
    # dependency edges
    # -------------------------------------------------------------------------
    for port, param in actor.params:
        # param -> inner actor
        SubElement(graph_el, "edge", attrib={
            "kind":       "dependency",
            "source":     port.name,
            "target":     inner_name,
            "targetport": port.name,
        })
        # param -> src nodes whose expression references this param
        for in_port, tensor in [*actor.inputs, *actor.weights]:
            expr = _get_expr(rate_exprs, in_port.name, "inputs", in_port.rate)
            if port.name in expr:
                SubElement(graph_el, "edge", attrib={
                    "kind":   "dependency",
                    "source": port.name,
                    "target": in_port.name,
                })
        # param -> snk nodes whose expression references this param
        for out_port, tensor in actor.outputs:
            expr = _get_expr(rate_exprs, out_port.name, "outputs", out_port.rate)
            if port.name in expr:
                SubElement(graph_el, "edge", attrib={
                    "kind":   "dependency",
                    "source": port.name,
                    "target": out_port.name,
                })

    # -------------------------------------------------------------------------
    # Write file
    # -------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    tree = ElementTree(root)
    indent(tree, space="    ")
    tree.write(pi_path, encoding="UTF-8", xml_declaration=True)
    return pi_path


def generate_all_pi_files(graph: IRGraph, output_dir: str = "../Algo"):
    """
    Generate one .pi per unique op type, skipping special actors.
    """
    seen = set()
    for actor in graph.actors:
        if actor.op_type in SKIP_OPS:
            continue
        op_name = _get_op_name(actor)
        if op_name not in seen:
            path = generate_pi_file(actor, output_dir)
            print(f"  Generated {path}")
            seen.add(op_name)
