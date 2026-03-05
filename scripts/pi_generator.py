from xml.etree.ElementTree import Element, SubElement, ElementTree, indent
from converter import OpType, IRActor, IRGraph, OPTYPE_TO_PI, OPTYPE_TO_H, OPTYPE_TO_LOOP_FN
import os

RATE_EXPRESSIONS = {
    # -------------------------------------------------------------------------
    # Conv — fires depthOutput times (one per output channel)
    # src produces the full input depthOutput times → PREESM broadcasts
    # weights split one filter per firing
    # snk accumulates all output channels
    # -------------------------------------------------------------------------
    "conv2d": {
        "src_rates": {
            "input_0":  "depthInput * inputHeight * inputWidth",
            "input_1": "depthOutput * depthInput * sizeKernelHeight * sizeKernelWidth",
        },
        "inputs": {
            "input_0":  "depthInput * inputHeight * inputWidth",
            "input_1": "depthInput * sizeKernelHeight * sizeKernelWidth",
        },
        "outputs": {
            "output_0": "outputHeight * outputWidth",
        },
        "snk_rates": {
            "output_0": "depthOutput * outputHeight * outputWidth",
        },
    },
    "conv2d_bias": {
        "src_rates": {
            "input_0":  "depthInput * inputHeight * inputWidth",
            "input_1": "depthOutput * depthInput * sizeKernelHeight * sizeKernelWidth",
            "input_2": "depthOutput",
        },
        "inputs": {
            "input_0":  "depthInput * inputHeight * inputWidth",
            "input_1": "depthInput * sizeKernelHeight * sizeKernelWidth",
            "input_2": "1",
        },
        "outputs": {
            "output_0": "outputHeight * outputWidth",
        },
        "snk_rates": {
            "output_0": "depthOutput * outputHeight * outputWidth",
        },
    },

    # -------------------------------------------------------------------------
    # Pool — fires depthInput times (one per channel)
    # -------------------------------------------------------------------------
    "maxpool2d": {
        "src_rates": {
            "input_0": "depthInput * inputHeight * inputWidth",
        },
        "inputs":  {"input_0": "inputHeight * inputWidth"},
        "outputs": {"output_0": "outputHeight * outputWidth"},
        "snk_rates": {
            "output_0": "depthInput * outputHeight * outputWidth",
        },
    },
    "avgpool2d": {
        "src_rates": {
            "input_0": "depthInput * inputHeight * inputWidth",
        },
        "inputs":  {"input_0": "inputHeight * inputWidth"},
        "outputs": {"output_0": "outputHeight * outputWidth"},
        "snk_rates": {
            "output_0": "depthInput * outputHeight * outputWidth",
        },
    },

    # -------------------------------------------------------------------------
    # Global avgpool — fires depth times (one per channel)
    # -------------------------------------------------------------------------
    "global_avgpool": {
        "src_rates": {
            "input_0": "depth * spatialSize",
        },
        "inputs":  {"input_0": "spatialSize"},
        "outputs": {"output_0": "1"},
        "snk_rates": {
            "output_0": "depth",
        },
    },

    # -------------------------------------------------------------------------
    # Element-wise — fires size times (one element per firing)
    # src produces size tokens, snk consumes size tokens
    # -------------------------------------------------------------------------
    "relu": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "size"},
    },
    "sigmoid": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "size"},
    },
    "tanh": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "size"},
    },
    "dropout": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "size"},
    },

    # -------------------------------------------------------------------------
    # Softmax — needs full vector, fires once
    # -------------------------------------------------------------------------
    "softmax": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "size"},
        "outputs":    {"output_0": "size"},
        "snk_rates":  {"output_0": "size"},
    },

    # -------------------------------------------------------------------------
    # Add variants — fires size1 times (one element per firing)
    # input_0 produces size1 tokens (one per firing)
    # input_1:
    #   add_same:   size1 tokens (one per firing, same size)
    #   add_bias:   size2 tokens (cycles through bias, PREESM wraps)
    #   add_scalar: 1 token (broadcast by PREESM to all firings)
    # -------------------------------------------------------------------------
    "add_same": {
        "src_rates":  {"input_0": "size1", "input_1": "size1"},
        "inputs":     {"input_0": "1",     "input_1": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "size1"},
    },
    "add_bias": {
        "src_rates":  {"input_0": "size1", "input_1": "size2"},
        "inputs":     {"input_0": "1",     "input_1": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "size1"},
    },
    "add_scalar": {
        "src_rates":  {"input_0": "size1", "input_1": "1"},
        "inputs":     {"input_0": "1",     "input_1": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "size1"},
    },
    "add_generic": {
        "src_rates":  {"input_0": "size1", "input_1": "size2"},
        "inputs":     {"input_0": "1",     "input_1": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "size1"},
    },

    # -------------------------------------------------------------------------
    # MatMul — fires M*N times (one output element per firing)
    # input_0: one row of A per firing (K elements), total M*N rows consumed
    #   but each row is reused N times → src produces M*K, actor fires M*N
    #   PREESM handles the reuse via broadcast
    # weight_0: one col of B per firing (K elements), total M*N cols consumed
    #   each col reused M times → src produces N*K
    # -------------------------------------------------------------------------
    "matmul": {
        "src_rates":  {"input_0": "M * K", "input_1": "N * K"},
        "inputs":     {"input_0": "K",     "input_1": "K*N"},
        "outputs":    {"output_0": "N"},
        "snk_rates":  {"output_0": "M * N"},
    },

    # -------------------------------------------------------------------------
    # Reshape/Flatten — element-wise passthrough
    # -------------------------------------------------------------------------
    "reshape": {
        "src_rates":  {"input_0": "outputSize", "input_1": "shapeSize"},
        "inputs":     {"input_0": "1",         "input_1": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "outputSize"},
    },
    "reshape_w": {
        "src_rates":  {"input_0": "outputSize", "input_1": "shapeSize"},
        "inputs":     {"input_0": "1",          "input_1": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "outputSize"},
    },
    "flatten": {
        "src_rates":  {"input_0": "inputSize"},
        "inputs":     {"input_0": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "outputSize"},
    },

    # -------------------------------------------------------------------------
    # Transpose — full tensor at once (cannot be element-wise)
    # -------------------------------------------------------------------------
    "transpose": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "size"},
        "outputs":    {"output_0": "size"},
        "snk_rates":  {"output_0": "size"},
    },

    # -------------------------------------------------------------------------
    # Concat — fires size1+size2 times (one element per firing from each stream)
    # -------------------------------------------------------------------------
    "concat": {
        "src_rates":  {"input_0": "size1", "input_1": "size2"},
        "inputs":     {"input_0": "1",     "input_1": "1"},
        "outputs":    {"output_0": "1"},
        "snk_rates":  {"output_0": "size1 + size2"},
    },
}

# Ops that keep .h and never get a .pi
SKIP_OPS = {
    OpType.LOAD_WEIGHTS,
    OpType.LOAD_INPUT,
    OpType.OUTPUT,
    OpType.SPLIT_WEIGHTS,
    OpType.CONSTANT_FILL,
    OpType.BROADCAST,
    OpType.FORK,
    OpType.JOIN,
    OpType.ZERO,
    OpType.SINK,
    OpType.CONSTANT_FILL,
    OpType.RANGE_FILL,
    OpType.POLICYNET,
}


def _get_op_name(actor: IRActor) -> str:
    return OPTYPE_TO_PI[actor.op_type].split("/")[-1].replace(".pi", "")


def _get_src_expr(rate_exprs: dict, port_name: str, fallback: int) -> str:
    # src node rate — use src_rates if present, fall back to inputs rate, then fallback
    return (
        rate_exprs.get("src_rates", {}).get(port_name)
        or rate_exprs.get("inputs",    {}).get(port_name)
        or str(fallback)
    )

def _get_snk_expr(rate_exprs: dict, port_name: str, fallback: int) -> str:
    # snk node rate — use snk_rates if present, fall back to outputs rate, then fallback
    return (
        rate_exprs.get("snk_rates",  {}).get(port_name)
        or rate_exprs.get("outputs", {}).get(port_name)
        or str(fallback)
    )

def _get_actor_in_expr(rate_exprs: dict, port_name: str, fallback: int) -> str:
    return rate_exprs.get("inputs", {}).get(port_name, str(fallback))

def _get_actor_out_expr(rate_exprs: dict, port_name: str, fallback: int) -> str:
    return rate_exprs.get("outputs", {}).get(port_name, str(fallback))


def generate_pi_file(actor: IRActor, output_dir: str) -> str:
    op_name    = _get_op_name(actor) 
    if any(p.name == "keep" for p, _ in actor.params):
        op_name = f"dy_{op_name}"
    pi_path    = os.path.join(output_dir, f"{op_name}.pi") # use PI dict?
    rate_exprs = RATE_EXPRESSIONS.get(op_name, {})
    loop_name  = OPTYPE_TO_LOOP_FN[actor.op_type]
    inner_name = f"{loop_name}_neuron"
    h_source   = OPTYPE_TO_H[actor.op_type]

    root     = Element("graphml", attrib={"xmlns": "http://graphml.graphdrawing.org/xmlns"})
    graph_el = SubElement(root, "graph", attrib={"edgedefault": "directed"})
    SubElement(graph_el, "data", attrib={"key": "name"}).text = op_name

    # -------------------------------------------------------------------------
    # cfg_in_iface nodes
    # -------------------------------------------------------------------------
    for port, param in actor.params:
        SubElement(graph_el, "node", attrib={"id": port.name, "kind": "cfg_in_iface"})

    # -------------------------------------------------------------------------
    # src nodes — use src_rates
    # -------------------------------------------------------------------------
    for port, tensor in actor.inputs:
        expr   = _get_src_expr(rate_exprs, port.name, port.rate)
        src_el = SubElement(graph_el, "node", attrib={"id": port.name, "kind": "src"})
        SubElement(src_el, "port", attrib={
            "annotation": "NONE", "expr": expr,
            "kind": "output", "name": port.name,
        })

    for port, tensor in actor.weights:
        expr   = _get_src_expr(rate_exprs, port.name, port.rate)
        src_el = SubElement(graph_el, "node", attrib={"id": port.name, "kind": "src"})
        SubElement(src_el, "port", attrib={
            "annotation": "NONE", "expr": expr,
            "kind": "output", "name": port.name,
        })

    # -------------------------------------------------------------------------
    # snk nodes — use snk_rates
    # -------------------------------------------------------------------------
    for port, tensor in actor.outputs:
        expr   = _get_snk_expr(rate_exprs, port.name, port.rate)
        snk_el = SubElement(graph_el, "node", attrib={"id": port.name, "kind": "snk"})
        SubElement(snk_el, "port", attrib={
            "annotation": "NONE", "expr": expr,
            "kind": "input", "name": port.name,
        })

    # -------------------------------------------------------------------------
    # Inner actor node — uses inputs/outputs rates (per firing)
    # -------------------------------------------------------------------------
    inner_el = SubElement(graph_el, "node", attrib={"id": inner_name, "kind": "actor"})
    SubElement(inner_el, "data", attrib={"key": "graph_desc"}).text = h_source

    loop_el = SubElement(inner_el, "loop", attrib={"name": inner_name})
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

    for port, param in actor.params:
        SubElement(inner_el, "port", attrib={"kind": "cfg_input", "name": port.name})
    for port, tensor in actor.inputs:
        expr = _get_actor_in_expr(rate_exprs, port.name, port.rate)
        SubElement(inner_el, "port", attrib={
            "annotation": "NONE", "expr": expr,
            "kind": "input", "name": port.name,
        })
    for port, tensor in actor.weights:
        expr = _get_actor_in_expr(rate_exprs, port.name, port.rate)
        SubElement(inner_el, "port", attrib={
            "annotation": "NONE", "expr": expr,
            "kind": "input", "name": port.name,
        })
    for port, tensor in actor.outputs:
        expr = _get_actor_out_expr(rate_exprs, port.name, port.rate)
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
    # dependency edges — check both src_rates and inputs for param references
    # -------------------------------------------------------------------------
    for port, param in actor.params:
        SubElement(graph_el, "edge", attrib={
            "kind": "dependency", "source": port.name,
            "target": inner_name, "targetport": port.name,
        })
        for in_port, tensor in [*actor.inputs, *actor.weights]:
            src_expr   = _get_src_expr(rate_exprs, in_port.name, in_port.rate)
            actor_expr = _get_actor_in_expr(rate_exprs, in_port.name, in_port.rate)
            if port.name in src_expr or port.name in actor_expr:
                SubElement(graph_el, "edge", attrib={
                    "kind": "dependency", "source": port.name,
                    "target": in_port.name,
                })
        for out_port, tensor in actor.outputs:
            snk_expr   = _get_snk_expr(rate_exprs, out_port.name, out_port.rate)
            actor_expr = _get_actor_out_expr(rate_exprs, out_port.name, out_port.rate)
            if port.name in snk_expr or port.name in actor_expr:
                SubElement(graph_el, "edge", attrib={
                    "kind": "dependency", "source": port.name,
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


def generate_all_pi_files(graph: IRGraph, output_dir: str = "../sources.pi"):
    seen = set()
    for actor in graph.actors:
        if actor.op_type in SKIP_OPS :
            continue
        op_name = _get_op_name(actor)
        is_dynamic = any(p.name == "keep" for p, _ in actor.params)
        dynamic_key = f"dy_{op_name}" if is_dynamic else op_name
        if dynamic_key not in seen:
            path = generate_pi_file(actor, output_dir)
            print(f"  Generated {path}")
            seen.add(dynamic_key)
