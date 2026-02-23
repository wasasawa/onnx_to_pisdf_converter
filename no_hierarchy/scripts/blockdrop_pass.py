from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

from structure import (
    IRGraph, IRActor, IRPort, IRTensor, IRParam,
    OpType, PortDir, OPTYPE_TO_H,
)


# ===========================================================================
# 1.  RATE-EXPRESSION HELPERS
# ===========================================================================

def _mul(base_rate, param_name: str) -> str:
    """e.g. (16384, 'keep_0') → '16384*keep_0'"""
    return f"{base_rate}*{param_name}"


def _inv(base_rate, param_name: str) -> str:
    """e.g. (16384, 'keep_0') → '16384*(1-keep_0)'"""
    return f"{base_rate}*(1-{param_name})"


# ===========================================================================
# 2.  RESIDUAL-BLOCK DESCRIPTOR
# ===========================================================================

@dataclass
class ResidualBlock:
    """
    Complete description of one detected residual block.

    add_actor             : The ADD_SAME / ADD_BIAS merge actor.
    fork_actor            : The BROADCAST actor that branches both paths.
    fork_compute_port     : fork's output port heading to compute path.
    fork_compute_tensor   : tensor from fork to first compute actor.
    compute_actors        : Ordered compute-path actors (fork→ADD, exclusive).
    compute_result_tensor : Tensor currently flowing into ADD's compute port
                            (the actual live tensor, not the stale actor.inputs ref).
    add_compute_port      : ADD's input port for the compute result.
    block_output_size     : Scalar token size of the compute result tensor.
    """
    add_actor             : IRActor
    fork_actor            : IRActor
    fork_compute_port     : IRPort
    fork_compute_tensor   : IRTensor
    compute_actors        : list[IRActor]
    compute_result_tensor : IRTensor
    add_compute_port      : IRPort
    block_output_size     : int


# ===========================================================================
# 3.  TENSOR-CENTRIC GRAPH MAPS
#     All traversal is done through these maps, never through actor.inputs /
#     actor.outputs, which may reference stale (deleted) tensors.
# ===========================================================================

def _build_producer_map(graph: IRGraph) -> dict[str, IRActor]:
    """tensor_name → producing actor  (built from live graph.tensors)."""
    return {
        t.name: t.producer.actor
        for t in graph.tensors
        if t.producer is not None and t.producer.actor is not None
    }


def _build_actor_output_tensors(graph: IRGraph) -> dict[str, list[IRTensor]]:
    """
    actor.unique_name → list of tensors it currently produces.
    Built from graph.tensors, NOT from actor.outputs (may be stale).
    """
    result: dict[str, list[IRTensor]] = defaultdict(list)
    for t in graph.tensors:
        if t.producer is not None and t.producer.actor is not None:
            result[t.producer.actor.unique_name].append(t)
    return result


def _build_actor_input_tensors(graph: IRGraph) -> dict[str, list[IRTensor]]:
    """
    actor.unique_name → list of tensors currently flowing *into* it.
    Built from graph.tensors, NOT from actor.inputs (may be stale).
    """
    result: dict[str, list[IRTensor]] = defaultdict(list)
    for t in graph.tensors:
        for port in t.consumers:
            if port.actor is not None:
                result[port.actor.unique_name].append(t)
    return result


def _build_consumer_to_tensor(graph: IRGraph) -> dict[int, IRTensor]:
    """
    id(port) → tensor  for every consumer port in the live graph.
    Allows look-up of the ACTUAL tensor associated with an actor's input port,
    bypassing the stale actor.inputs[i][1] reference.
    """
    result: dict[int, IRTensor] = {}
    for t in graph.tensors:
        for port in t.consumers:
            result[id(port)] = t
    return result


# ===========================================================================
# 4.  TOPOLOGICAL DEPTH (tensor-centric)
# ===========================================================================

def _compute_actor_depths(graph: IRGraph,
                           actor_input_tensors: dict[str, list[IRTensor]],
                           producer_map: dict[str, IRActor]) -> dict[str, int]:
    """
    Kahn's algorithm: assign topological depth to every actor.
    Uses actor_input_tensors (tensor-centric) instead of actor.inputs.
    """
    preds: dict[str, set[str]] = {a.unique_name: set() for a in graph.actors}

    for a in graph.actors:
        for t in actor_input_tensors.get(a.unique_name, []):
            p = producer_map.get(t.name)
            if p:
                preds[a.unique_name].add(p.unique_name)

    depth: dict[str, int] = {}
    in_deg = {k: len(v) for k, v in preds.items()}
    queue  = deque(n for n, d in in_deg.items() if d == 0)
    for n in queue:
        depth[n] = 0

    while queue:
        n = queue.popleft()
        a = graph.get_actor(n)
        if not a:
            continue
        for t in _actor_out_tensors_live(a, graph):
            for cp in t.consumers:
                if cp.actor is None:
                    continue
                c = cp.actor.unique_name
                depth[c] = max(depth.get(c, 0), depth[n] + 1)
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return depth


def _actor_out_tensors_live(actor: IRActor, graph: IRGraph) -> list[IRTensor]:
    """Return live output tensors of actor (via graph.tensors, not actor.outputs)."""
    return [t for t in graph.tensors
            if t.producer is not None and t.producer.actor is actor]


# ===========================================================================
# 5.  BACKWARD / FORWARD BFS (tensor-centric)
# ===========================================================================

def _ancestors_bfs(start: IRActor,
                   actor_input_tensors: dict[str, list[IRTensor]],
                   producer_map: dict[str, IRActor]) -> set[str]:
    """
    Backward BFS through data edges (tensor-centric).
    Returns unique_names of all upstream actors, including start.
    """
    visited: set[str] = set()
    queue: deque[IRActor] = deque([start])
    while queue:
        a = queue.popleft()
        if a.unique_name in visited:
            continue
        visited.add(a.unique_name)
        for t in actor_input_tensors.get(a.unique_name, []):
            p = producer_map.get(t.name)
            if p and p.unique_name not in visited:
                queue.append(p)
    return visited


def _can_reach_actor(from_tensor: IRTensor,
                     target: IRActor,
                     actor_output_tensors: dict[str, list[IRTensor]]) -> bool:
    """
    Forward BFS from from_tensor: can we reach target actor?
    Uses live actor_output_tensors instead of actor.outputs.
    """
    visited_t: set[str] = set()
    queue: deque[IRTensor] = deque([from_tensor])
    while queue:
        t = queue.popleft()
        if t.name in visited_t:
            continue
        visited_t.add(t.name)
        for cp in t.consumers:
            if cp.actor is None:
                continue
            if cp.actor is target:
                return True
            for ot in actor_output_tensors.get(cp.actor.unique_name, []):
                if ot.name not in visited_t:
                    queue.append(ot)
    return False


def _collect_compute_actors(fork_compute_tensor: IRTensor,
                             add_actor: IRActor,
                             actor_output_tensors: dict[str, list[IRTensor]],
                             depth_map: dict[str, int]) -> list[IRActor]:
    """
    Forward BFS from fork_compute_tensor: collect all actors strictly between
    the fork and ADD (both exclusive), sorted by topological depth.
    Uses live actor_output_tensors.
    """
    visited_a: set[str] = set()
    visited_t: set[str] = set()
    result: list[IRActor] = []
    queue: deque[IRTensor] = deque([fork_compute_tensor])

    while queue:
        t = queue.popleft()
        if t.name in visited_t:
            continue
        visited_t.add(t.name)
        for cp in t.consumers:
            a = cp.actor
            if a is None or a is add_actor or a.unique_name in visited_a:
                continue
            visited_a.add(a.unique_name)
            result.append(a)
            for ot in actor_output_tensors.get(a.unique_name, []):
                queue.append(ot)

    result.sort(key=lambda a: depth_map.get(a.unique_name, 0))
    return result


# ===========================================================================
# 6.  BLOCK DETECTION
# ===========================================================================

_CONV_OPS = frozenset({OpType.CONV2D, OpType.CONV2D_BIAS})


def _is_residual_add(actor: IRActor) -> bool:
    return (
        actor.op_type in (OpType.ADD_SAME, OpType.ADD_BIAS, OpType.ADD_GENERIC)
        and len(actor.inputs) == 2
    )


def detect_residual_blocks(graph: IRGraph) -> list[ResidualBlock]:
    """
    Detect all residual blocks in the graph, returning ResidualBlock descriptors.

    Robustly handles the stale-tensor issue from add_broadcast_actors by
    using consumer_to_tensor map to find the ACTUAL tensors currently wired
    to ADD's input ports.

    A block requires:
      ① ADD actor with 2 inputs, both with live producer actors.
      ② At least one path contains a CONV2D actor (compute path).
      ③ A common BROADCAST ancestor (the fork point) exists.
    """
    # Build all tensor-centric maps once
    producer_map         = _build_producer_map(graph)
    actor_out_tensors    = _build_actor_output_tensors(graph)
    actor_in_tensors     = _build_actor_input_tensors(graph)
    consumer_to_tensor   = _build_consumer_to_tensor(graph)
    depth_map            = _compute_actor_depths(graph, actor_in_tensors, producer_map)

    blocks: list[ResidualBlock] = []

    for add_actor in graph.actors:
        if not _is_residual_add(add_actor):
            continue

        # ── Resolve the ACTUAL current tensors for ADD's two inputs ─────────
        # actor.inputs[i][1] may be stale (deleted tensor). Use consumer_to_tensor.
        p0, _ = add_actor.inputs[0]
        p1, _ = add_actor.inputs[1]
        t0 = consumer_to_tensor.get(id(p0))
        t1 = consumer_to_tensor.get(id(p1))

        if t0 is None or t1 is None:
            # One port has no live tensor flowing in — not a standard residual ADD
            continue

        prod0 = producer_map.get(t0.name)
        prod1 = producer_map.get(t1.name)

        if prod0 is None or prod1 is None:
            continue

        # ── Classify compute path vs skip path ──────────────────────────────
        anc0 = _ancestors_bfs(prod0, actor_in_tensors, producer_map)
        anc1 = _ancestors_bfs(prod1, actor_in_tensors, producer_map)

        def _has_conv(aset: set[str]) -> bool:
            return any(
                (a := graph.get_actor(n)) is not None and a.op_type in _CONV_OPS
                for n in aset
            )

        conv0, conv1 = _has_conv(anc0), _has_conv(anc1)
        if not conv0 and not conv1:
            continue

        if conv0 and conv1:
            # Both paths have conv → deeper producer is the compute path
            d0 = depth_map.get(prod0.unique_name, 0)
            d1 = depth_map.get(prod1.unique_name, 0)
            if d0 >= d1:
                compute_prod, compute_port, compute_tensor = prod0, p0, t0
                compute_anc = anc0
            else:
                compute_prod, compute_port, compute_tensor = prod1, p1, t1
                compute_anc = anc1
        elif conv0:
            compute_prod, compute_port, compute_tensor = prod0, p0, t0
            compute_anc = anc0
        else:
            compute_prod, compute_port, compute_tensor = prod1, p1, t1
            compute_anc = anc1

        skip_anc = anc1 if compute_anc is anc0 else anc0

        # ── Find the BROADCAST fork in the common ancestor set ─────────────
        common_bcasts = [
            n for n in (compute_anc & skip_anc)
            if (a := graph.get_actor(n)) is not None
            and a.op_type == OpType.BROADCAST
        ]
        if not common_bcasts:
            continue

        # Deepest broadcast = the branch point closest to the ADD
        fork_name  = max(common_bcasts, key=lambda n: depth_map.get(n, 0))
        fork_actor = graph.get_actor(fork_name)

        # ── Identify which fork output leads to the compute path ───────────
        fork_compute_port   = None
        fork_compute_tensor = None
        for out_port, out_tensor in fork_actor.outputs:
            if _can_reach_actor(out_tensor, compute_prod, actor_out_tensors):
                fork_compute_port   = out_port
                fork_compute_tensor = out_tensor
                break

        if fork_compute_port is None:
            continue

        # ── Gather ordered compute-path actors ─────────────────────────────
        compute_actors = _collect_compute_actors(
            fork_compute_tensor, add_actor, actor_out_tensors, depth_map
        )
        if not compute_actors:
            continue

        blocks.append(ResidualBlock(
            add_actor             = add_actor,
            fork_actor            = fork_actor,
            fork_compute_port     = fork_compute_port,
            fork_compute_tensor   = fork_compute_tensor,
            compute_actors        = compute_actors,
            compute_result_tensor = compute_tensor,
            add_compute_port      = compute_port,
            block_output_size     = compute_tensor.size,
        ))

    return blocks


# ===========================================================================
# 7.  PARAMETER HELPERS
# ===========================================================================

def _create_keep_param(graph: IRGraph, block_idx: int) -> IRParam:
    uid = f"keep_{block_idx}"
    if uid not in graph._params:
        graph._params[uid] = IRParam(name=uid, value=1, unique_id=uid)
    return graph._params[uid]


def _add_cfg_port(actor: IRActor, keep_param: IRParam) -> None:
    port_name = keep_param.unique_id
    if any(p.name == port_name for p, _ in actor.params):
        return
    actor.add_param(port_name, keep_param)


# ===========================================================================
# 8.  STEP 1 — MODIFY THE BROADCAST FORK ACTOR
# ===========================================================================

def _modify_fork_broadcast(block: ResidualBlock, keep_param: IRParam) -> None:
    _add_cfg_port(block.fork_actor, keep_param)
    block.fork_compute_port.rate = _mul(block.fork_compute_port.rate,
                                         keep_param.unique_id)


# ===========================================================================
# 9.  STEP 2 — MODIFY COMPUTE-ACTOR DATA RATES
# ===========================================================================

def _modify_compute_actor_data_rates(actor: IRActor,
                                      keep_param: IRParam) -> None:
    """
    Add keep cfg port; multiply all data input/output port rates by keep.
    Uses actor.inputs and actor.outputs — the PORTS are always valid even if
    their tensor references may be stale.  We only need port.rate here.
    """
    _add_cfg_port(actor, keep_param)
    pname = keep_param.unique_id
    for port, _ in actor.inputs:
        port.rate = _mul(port.rate, pname)
    for port, _ in actor.outputs:
        port.rate = _mul(port.rate, pname)


# ===========================================================================
# 10.  STEP 3 — WEIGHT GATE INSERTION
# ===========================================================================

def _insert_weight_gate(graph: IRGraph,
                         compute_actor: IRActor,
                         weight_idx: int,
                         keep_param: IRParam,
                         label: str) -> None:
    """
    Insert Gate_Weights fork + Sink for one weight FIFO.

    Before:
        source ──[weight_tensor]──▶ compute_actor.weight_port  (rate W)
    After:
        source ──[weight_tensor]──▶ Gate.weight_in  (rate W, static)
            Gate.to_conv   (W*keep)     ──▶ compute_actor.weight_port
            Gate.discarded (W*(1-keep)) ──▶ Sink.input
    """
    weight_port, weight_tensor = compute_actor.weights[weight_idx]
    W     = weight_tensor.size   # always the original static size
    pname = keep_param.unique_id
    dtype = weight_tensor.dtype

    # ─── Gate_Weights fork ────────────────────────────────────────────────
    gate_name  = f"Gate_{label}"
    gate_actor = IRActor(
        name=gate_name, op_type=OpType.BROADCAST,
        unique_name=gate_name, source="",
        attributes={"_kind": "gate"},
    )
    graph._actors[gate_name] = gate_actor
    _add_cfg_port(gate_actor, keep_param)

    gate_in_port = IRPort(name="weight_in", direction=PortDir.IN,
                          rate=W, actor=gate_actor)
    gate_actor.inputs.append((gate_in_port, weight_tensor))

    gated_tensor = graph.create_tensor(
        name=f"{weight_tensor.name}_gated_{label}",
        shape=weight_tensor.shape, dtype=dtype,
    )
    to_conv_port = IRPort(name="to_conv", direction=PortDir.OUT,
                          rate=_mul(W, pname), actor=gate_actor)
    gate_actor.outputs.append((to_conv_port, gated_tensor))
    graph.link(gated_tensor, to_conv_port, None)

    discard_tensor = graph.create_tensor(
        name=f"{weight_tensor.name}_discard_{label}",
        shape=weight_tensor.shape, dtype=dtype,
    )
    discard_port = IRPort(name="discarded", direction=PortDir.OUT,
                          rate=_inv(W, pname), actor=gate_actor)
    gate_actor.outputs.append((discard_port, discard_tensor))
    graph.link(discard_tensor, discard_port, None)

    # ─── Rewire weight_tensor → gate instead of directly to compute actor ──
    if weight_port in weight_tensor.consumers:
        weight_tensor.consumers.remove(weight_port)
    graph.link(weight_tensor, None, gate_in_port)

    compute_actor.weights[weight_idx] = (weight_port, gated_tensor)
    weight_port.rate = _mul(W, pname)
    graph.link(gated_tensor, None, weight_port)

    # ─── Sink actor ───────────────────────────────────────────────────────
    sink_name  = f"Sink_{label}"
    sink_actor = IRActor(
        name=sink_name, op_type=OpType.BROADCAST,
        unique_name=sink_name, source="Code/include/utilities.h",
        attributes={"_kind": "sink", "_loop_fn": "sink"},
    )
    graph._actors[sink_name] = sink_actor
    _add_cfg_port(sink_actor, keep_param)

    sink_in_port = IRPort(name="input", direction=PortDir.IN,
                          rate=_inv(W, pname), actor=sink_actor)
    sink_actor.inputs.append((sink_in_port, discard_tensor))
    graph.link(discard_tensor, None, sink_in_port)


# ===========================================================================
# 11.  STEP 4 — ZERO ACTOR + SELECT (JOIN) ACTOR
# ===========================================================================

def _insert_zero_and_select(graph: IRGraph,
                              block: ResidualBlock,
                              keep_param: IRParam,
                              block_idx: int) -> None:
    """
    Insert Zero + Select actors before ADD's compute input.

    Zero.output   (S*(1-keep)) → Select.zero
    compute_result (S*keep)    → Select.b_output
    Select.output  (S, static) → ADD.compute_input   [ADD unchanged]
    """
    pname  = keep_param.unique_id
    S      = block.block_output_size
    dtype  = block.compute_result_tensor.dtype

    # ─── Zero actor ───────────────────────────────────────────────────────
    zero_name  = f"zero_block{block_idx}"
    zero_actor = IRActor(
        name=zero_name, op_type=OpType.BROADCAST,
        unique_name=zero_name, source="Code/include/utilities.h",
        attributes={"_kind": "zero", "_loop_fn": "Zero"},
    )
    graph._actors[zero_name] = zero_actor
    _add_cfg_port(zero_actor, keep_param)

    zero_tensor = graph.create_tensor(
        name=f"zero_tensor_block{block_idx}",
        shape=block.compute_result_tensor.shape, dtype=dtype,
    )
    zero_out = IRPort(name="output", direction=PortDir.OUT,
                      rate=_inv(S, pname), actor=zero_actor)
    zero_actor.outputs.append((zero_out, zero_tensor))
    graph.link(zero_tensor, zero_out, None)

    # ─── Select (join) actor ──────────────────────────────────────────────
    select_name  = f"select_block{block_idx}"
    select_actor = IRActor(
        name=select_name, op_type=OpType.BROADCAST,
        unique_name=select_name, source="",
        attributes={"_kind": "select"},   # emitted as kind="join"
    )
    graph._actors[select_name] = select_actor
    _add_cfg_port(select_actor, keep_param)

    # Input "zero" ← Zero actor
    sel_zero_port = IRPort(name="zero", direction=PortDir.IN,
                           rate=_inv(S, pname), actor=select_actor)
    select_actor.inputs.append((sel_zero_port, zero_tensor))
    graph.link(zero_tensor, None, sel_zero_port)

    # Input "b_output" ← compute result tensor (currently feeding ADD's compute port)
    compute_result   = block.compute_result_tensor
    add_compute_port = block.add_compute_port

    sel_b_port = IRPort(name="b_output", direction=PortDir.IN,
                        rate=_mul(S, pname), actor=select_actor)
    select_actor.inputs.append((sel_b_port, compute_result))

    # Remove ADD's compute port from compute_result's consumers; add select's port
    if add_compute_port in compute_result.consumers:
        compute_result.consumers.remove(add_compute_port)
    graph.link(compute_result, None, sel_b_port)

    # Output of select → new tensor → ADD's compute input (static rate S)
    select_out_tensor = graph.create_tensor(
        name=f"select_out_block{block_idx}",
        shape=block.compute_result_tensor.shape, dtype=dtype,
    )
    sel_out_port = IRPort(name="output", direction=PortDir.OUT,
                          rate=S, actor=select_actor)
    select_actor.outputs.append((sel_out_port, select_out_tensor))
    graph.link(select_out_tensor, sel_out_port, None)

    # Redirect ADD's compute input port to consume select_out_tensor.
    # Match by PORT IDENTITY — tensor refs in actor.inputs may be stale.
    add_actor = block.add_actor
    for i, (p, _) in enumerate(add_actor.inputs):
        if p is add_compute_port:
            add_actor.inputs[i] = (p, select_out_tensor)
            break

    add_compute_port.rate = S   # remains static
    graph.link(select_out_tensor, None, add_compute_port)


# ===========================================================================
# 12.  PER-BLOCK TRANSFORMATION ORCHESTRATOR
# ===========================================================================

def _transform_block(graph: IRGraph,
                     block: ResidualBlock,
                     block_idx: int) -> IRParam:
    keep_param = _create_keep_param(graph, block_idx)

    # Step 1 — Modify the fork BROADCAST
    _modify_fork_broadcast(block, keep_param)

    # Steps 2 & 3 — Compute actors: data rates + weight gates
    gate_counter = 0
    for actor in block.compute_actors:
        _modify_compute_actor_data_rates(actor, keep_param)
        for w_idx in range(len(actor.weights)):
            _, wt = actor.weights[w_idx]
            if wt.producer is not None:   # only gate weights with a source actor
                label = f"{actor.unique_name}_w{w_idx}_{gate_counter}"
                _insert_weight_gate(graph, actor, w_idx, keep_param, label)
                gate_counter += 1

    # Step 4 — Zero + Select before ADD
    _insert_zero_and_select(graph, block, keep_param, block_idx)

    return keep_param


# ===========================================================================
# 13.  PUBLIC ENTRY POINT
# ===========================================================================

def apply_blockdrop_pass(graph: IRGraph) -> int:
    """
    Detect all residual blocks and apply the BlockDrop transformation in-place.
    Returns the number of blocks transformed.
    """
    blocks = detect_residual_blocks(graph)
    if not blocks:
        print("[BlockDrop] No residual blocks detected. Graph unchanged.")
        return 0

    print(f"[BlockDrop] Detected {len(blocks)} residual block(s).")
    for idx, block in enumerate(blocks):
        keep = _transform_block(graph, block, idx)
        compute_names = [a.unique_name for a in block.compute_actors]
        print(f"  Block {idx}: ADD='{block.add_actor.unique_name}'  "
              f"fork='{block.fork_actor.unique_name}'  "
              f"param='{keep.unique_id}'  compute={compute_names}")
    return len(blocks)


# ===========================================================================
# 14.  XML GENERATION
# ===========================================================================

_SPECIAL_KIND_TO_XML_KIND = {
    "gate"  : "fork",
    "sink"  : "actor",
    "zero"  : "actor",
    "select": "join",
}


def _emit_node(graph_el: Element, actor: IRActor,
               xml_kind: str, loop_fn: str) -> None:
    node = SubElement(graph_el, "node", attrib={
        "id": actor.unique_name, "kind": xml_kind, "period": "0",
    })
    SubElement(node, "data", attrib={"key": "graph_desc"}).text = actor.source
    loop_el = SubElement(node, "loop", attrib={"name": loop_fn})

    for port, _ in actor.params:
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

    for port, _ in actor.params:
        SubElement(node, "port", attrib={"kind": "cfg_input", "name": port.name})
    for port, _ in actor.inputs:
        SubElement(node, "port", attrib={
            "kind": "input", "name": port.name,
            "expr": str(port.rate), "annotation": "NONE",
        })
    for port, _ in actor.weights:
        SubElement(node, "port", attrib={
            "kind": "input", "name": port.name,
            "expr": str(port.rate), "annotation": "NONE",
        })
    for port, _ in actor.outputs:
        SubElement(node, "port", attrib={
            "kind": "output", "name": port.name,
            "expr": str(port.rate), "annotation": "NONE",
        })


def _generate_actor_node_bd(graph_el: Element, actor: IRActor) -> None:
    special = actor.attributes.get("_kind")

    if special in _SPECIAL_KIND_TO_XML_KIND:
        xml_kind = _SPECIAL_KIND_TO_XML_KIND[special]
        loop_fn  = actor.attributes.get("_loop_fn", actor.unique_name)
        _emit_node(graph_el, actor, xml_kind, loop_fn)
        return

    # Standard converter actor kinds (inlined to avoid importing converter.py)
    if actor.op_type == OpType.BROADCAST:
        kind = "broadcast"
    elif actor.op_type == OpType.SPLIT_WEIGHTS:
        kind = "fork"
    else:
        kind = "actor"
    _emit_node(graph_el, actor, kind, actor.unique_name)


def generate_blockdrop_xml(graph: IRGraph, model_data: dict) -> str:
    """Generate PiSDF-compatible GraphML XML for a BlockDrop-transformed graph."""
    root     = Element("graphml", attrib={"xmlns": "http://graphml.graphdrawing.org/xmlns"})
    graph_el = SubElement(root, "graph", attrib={"edgedefault": "directed"})
    SubElement(graph_el, "data", attrib={"key": "name"})

    for param in graph.params:
        SubElement(graph_el, "node", attrib={
            "id": param.unique_id, "kind": "param", "expr": str(param.value),
        })

    for actor in graph.actors:
        _generate_actor_node_bd(graph_el, actor)

    for src_port, dst_port, dtype in graph.get_fifo_edges():
        src_actor = src_port.actor.unique_name if src_port.actor else "?"
        dst_actor = dst_port.actor.unique_name if dst_port.actor else "?"
        SubElement(graph_el, "edge", attrib={
            "kind": "fifo",
            "source": src_actor,  "sourceport": src_port.name,
            "target": dst_actor,  "targetport": dst_port.name,
            "type":   dtype,
        })

    for param, port in graph.get_dependency_edges():
        dst_actor = port.actor.unique_name if port.actor else "?"
        SubElement(graph_el, "edge", attrib={
            "kind": "dependency",
            "source": param.unique_id,
            "target": dst_actor, "targetport": port.name,
        })

    raw = tostring(root, encoding="unicode")
    return parseString(raw).toprettyxml(indent="  ")


def write_blockdrop_xml(graph: IRGraph, model_data: dict,
                         output_path: str = "output_blockdrop.pi") -> None:
    xml_str = generate_blockdrop_xml(graph, model_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"[BlockDrop] XML written to  {output_path}")
