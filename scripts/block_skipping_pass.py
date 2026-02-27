from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

from structure import (
    IRGraph, IRActor, IRPort, IRTensor, IRParam,
    OpType, PortDir, OPTYPE_TO_H, OPTYPE_TO_LOOP_FN
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
    Complete structural description of one detected residual block.

    Fields
    ------
    add_actor
        The ADD_SAME / ADD_BIAS merge actor.
    broadcast_actor
        The BROADCAST actor whose two outputs branch into the compute path
        and the skip path respectively.
    broadcast_compute_port
        The output port of broadcast_actor that leads to the compute path.
    broadcast_compute_tensor
        The tensor produced by broadcast_compute_port (the first token of
        the compute path).
    compute_actors
        Ordered list of actors on the compute path, from broadcast_actor to
        add_actor (both exclusive), sorted by topological depth.
    compute_result_tensor
        The live tensor currently flowing into ADD's compute input port.
    add_compute_port
        The input port of add_actor that receives the compute result.
    block_output_size
        Scalar element count of the compute result tensor.
    """
    add_actor               : IRActor
    broadcast_actor         : IRActor
    broadcast_compute_port  : IRPort
    broadcast_compute_tensor: IRTensor
    compute_actors          : list[IRActor]
    compute_result_tensor   : IRTensor
    add_compute_port        : IRPort
    block_output_size       : int


# ===========================================================================
# 3.  TENSOR-CENTRIC GRAPH MAPS
#
#  All traversal uses these maps, never actor.inputs / actor.outputs, which
#  may reference stale (deleted) tensors after add_broadcast_actors().
# ===========================================================================

def _build_producer_map(graph: IRGraph) -> dict[str, IRActor]:
    """tensor_name → producing actor, built from live graph.tensors."""
    return {
        t.name: t.producer.actor
        for t in graph.tensors
        if t.producer is not None and t.producer.actor is not None
    }


def _build_actor_output_tensors(graph: IRGraph) -> dict[str, list[IRTensor]]:
    """
    actor.unique_name → list of tensors it currently produces.
    Built from graph.tensors, not from actor.outputs (which may be stale).
    """
    result: dict[str, list[IRTensor]] = defaultdict(list)
    for t in graph.tensors:
        if t.producer is not None and t.producer.actor is not None:
            result[t.producer.actor.unique_name].append(t)
    return result


def _build_actor_input_tensors(graph: IRGraph) -> dict[str, list[IRTensor]]:
    """
    actor.unique_name → list of tensors currently flowing into it.
    Built from graph.tensors, not from actor.inputs (which may be stale).
    """
    result: dict[str, list[IRTensor]] = defaultdict(list)
    for t in graph.tensors:
        for port in t.consumers:
            if port.actor is not None:
                result[port.actor.unique_name].append(t)
    return result


def _build_consumer_to_tensor(graph: IRGraph) -> dict[int, IRTensor]:
    """
    id(port) → live tensor for every consumer port in the graph.
    Allows look-up of the actual tensor wired to an actor's input port,
    bypassing the stale actor.inputs[i][1] reference.
    """
    result: dict[int, IRTensor] = {}
    for t in graph.tensors:
        for port in t.consumers:
            result[id(port)] = t
    return result


# ===========================================================================
# 4.  TOPOLOGICAL DEPTH  (tensor-centric Kahn's algorithm)
# ===========================================================================

def _compute_actor_depths(
    graph               : IRGraph,
    actor_input_tensors : dict[str, list[IRTensor]],
    producer_map        : dict[str, IRActor],
) -> dict[str, int]:
    """
    Assign a topological depth to every actor using Kahn's algorithm.
    Uses actor_input_tensors (tensor-centric) instead of actor.inputs.

    CRITICAL — in_deg must be counted per incoming tensor edge, not per
    unique predecessor actor.

    A SPLIT_WEIGHTS actor typically has two weight tensors (kernel + bias/BN)
    going to each compute actor.  If in_deg were counted as the number of
    *unique* predecessor actors (a set), it would be 1 for that split-weights
    contribution, but the Kahn loop decrements in_deg once per tensor, i.e.
    twice.  That drives in_deg to zero before the data-path predecessor fires,
    causing every compute actor to be enqueued prematurely with depth ≈ 2
    instead of its true depth.

    Concretely for ResNet-8: BROADCAST_2 (correct depth 4) and CONV2D_2
    (premature depth 4 instead of 7) both appeared at the same depth, triggering
    the equal-depth warning and causing Block 0 to be silently skipped.  Block 1
    had its compute and skip paths swapped (CONV2D_5 the projection appeared
    deeper than CONV2D_3→CONV2D_4 the true compute path), and Block 2 picked
    BROADCAST_0 as the deepest common ancestor instead of BROADCAST_1, pulling
    the entire second block into its spurious compute actor list.
    """
    # Count incoming edges per tensor, matching the Kahn loop's decrement rate.
    in_deg: dict[str, int] = defaultdict(int)
    for a in graph.actors:
        for t in actor_input_tensors.get(a.unique_name, []):
            if producer_map.get(t.name) is not None:
                in_deg[a.unique_name] += 1

    depth: dict[str, int] = {}
    queue: deque[str] = deque()
    for a in graph.actors:
        if in_deg.get(a.unique_name, 0) == 0:
            depth[a.unique_name] = 0
            queue.append(a.unique_name)

    while queue:
        n = queue.popleft()
        actor = graph.get_actor(n)
        if not actor:
            continue
        for t in _live_output_tensors(actor, graph):
            for cp in t.consumers:
                if cp.actor is None:
                    continue
                c = cp.actor.unique_name
                depth[c] = max(depth.get(c, 0), depth[n] + 1)
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return depth


def _live_output_tensors(actor: IRActor, graph: IRGraph) -> list[IRTensor]:
    """Return the live output tensors of actor, via graph.tensors."""
    return [t for t in graph.tensors
            if t.producer is not None and t.producer.actor is actor]


# ===========================================================================
# 5.  BACKWARD / FORWARD BFS  (tensor-centric)
# ===========================================================================

def _ancestors_bfs(
    start               : IRActor,
    actor_input_tensors : dict[str, list[IRTensor]],
    producer_map        : dict[str, IRActor],
) -> set[str]:
    """
    Backward BFS through data edges.
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


def _can_reach_actor(
    from_tensor         : IRTensor,
    target              : IRActor,
    actor_output_tensors: dict[str, list[IRTensor]],
) -> bool:
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


def _collect_compute_actors(
    broadcast_compute_tensor: IRTensor,
    add_actor               : IRActor,
    actor_output_tensors    : dict[str, list[IRTensor]],
    depth_map               : dict[str, int],
) -> list[IRActor]:
    """
    Forward BFS from broadcast_compute_tensor: collect all actors strictly
    between the broadcast and ADD (both exclusive), sorted by depth.
    """
    visited_a: set[str] = set()
    visited_t: set[str] = set()
    result: list[IRActor] = []
    queue: deque[IRTensor] = deque([broadcast_compute_tensor])

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

# Op-types whose output tensors are "weights" — consumed by compute actors
# via weight ports.  Any actor that receives at least one tensor from these
# sources is a weight-bearing actor.
_WEIGHT_SOURCE_OPS: frozenset[OpType] = frozenset({
    OpType.SPLIT_WEIGHTS,
    OpType.LOAD_WEIGHTS,
})


def _count_weight_actors_on_path(
    from_tensor         : IRTensor,
    add_actor           : IRActor,
    actor_out_tensors   : dict[str, list[IRTensor]],
    actor_in_tensors    : dict[str, list[IRTensor]],
    producer_map        : dict[str, IRActor],
) -> int:
    """
    Forward BFS from from_tensor to add_actor (exclusive).

    Returns the number of distinct weight-bearing actors encountered —
    i.e. actors that consume at least one tensor produced by a
    SPLIT_WEIGHTS or LOAD_WEIGHTS actor.
    """
    visited_a: set[str] = set()
    visited_t: set[str] = set()
    count = 0
    queue: deque[IRTensor] = deque([from_tensor])

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
            # Check if any input to this actor comes from a weight-source
            for in_t in actor_in_tensors.get(a.unique_name, []):
                prod = producer_map.get(in_t.name)
                if prod is not None and prod.op_type in _WEIGHT_SOURCE_OPS:
                    count += 1
                    break   # Count the actor once regardless of # weight ports
            for ot in actor_out_tensors.get(a.unique_name, []):
                if ot.name not in visited_t:
                    queue.append(ot)

    return count

def _is_residual_add(actor: IRActor) -> bool:
    return (
        actor.op_type in (OpType.ADD_SAME, OpType.ADD_BIAS, OpType.ADD_GENERIC)
        and len(actor.inputs) == 2
    )


def detect_residual_blocks(graph: IRGraph) -> list[ResidualBlock]:
    """
    Detect all residual blocks in the graph.

    Algorithm
    ---------
    For each ADD actor with two live data inputs:

    1.  Find the deepest common BROADCAST ancestor of both input producers.
        "Deepest" (max topological depth) avoids picking a preceding block's
        broadcast when multiple common ancestors exist.

    2.  For each of the broadcast's two output tensors, do a forward BFS to
        the ADD and count weight-bearing actors (actors fed by SPLIT_WEIGHTS
        or LOAD_WEIGHTS).  The output with the higher count is the compute
        path; the other is the skip path.

    3.  If both outputs have equal weight-actor counts the block is ambiguous
        and skipped with a warning.  This should not occur in any standard
        ResNet topology.

    Robustness against stale tensor references
    ------------------------------------------
    actor.inputs[i][1] may reference a deleted tensor after the converter's
    add_broadcast_actors() pass.  All tensor look-ups use consumer_to_tensor
    (id(port) → live tensor), which is always current.
    """
    producer_map       = _build_producer_map(graph)
    actor_out_tensors  = _build_actor_output_tensors(graph)
    actor_in_tensors   = _build_actor_input_tensors(graph)
    consumer_to_tensor = _build_consumer_to_tensor(graph)
    depth_map          = _compute_actor_depths(graph, actor_in_tensors, producer_map)

    blocks: list[ResidualBlock] = []

    for add_actor in graph.actors:
        if not _is_residual_add(add_actor):
            continue

        # ── Resolve live tensors for ADD's two input ports ───────────────────
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

        # ── Find deepest common BROADCAST ancestor ───────────────────────────
        # Depth is only used here to pick the innermost (deepest) BROADCAST
        # when multiple common ancestors exist.  It is NOT used to classify
        # compute vs skip (weight-actor count does that).
        anc0 = _ancestors_bfs(prod0, actor_in_tensors, producer_map)
        anc1 = _ancestors_bfs(prod1, actor_in_tensors, producer_map)

        common_broadcasts = [
            name for name in (anc0 & anc1)
            if (a := graph.get_actor(name)) is not None
            and a.op_type == OpType.BROADCAST
        ]
        if not common_broadcasts:
            continue

        broadcast_name  = max(common_broadcasts, key=lambda n: depth_map.get(n, 0))
        broadcast_actor = graph.get_actor(broadcast_name)

        # ── Classify paths by weight-actor count ─────────────────────────────
        # For each output of the broadcast, count how many actors on the path
        # to ADD consume at least one weight tensor (from SPLIT_WEIGHTS /
        # LOAD_WEIGHTS).  The compute path always has strictly more such actors
        # than the skip path, regardless of the skip variant:
        #   identity    → skip=0, compute≥2
        #   slice+pad   → skip=0, compute≥2  (depth would falsely report equal)
        #   1×1 proj    → skip=1, compute≥2
        w_counts: list[tuple[int, IRPort, IRTensor]] = []
        for out_port, out_tensor in broadcast_actor.outputs:
            w = _count_weight_actors_on_path(
                out_tensor, add_actor,
                actor_out_tensors, actor_in_tensors, producer_map,
            )
            w_counts.append((w, out_port, out_tensor))

        w_counts.sort(key=lambda x: x[0], reverse=True)   # highest count first
        max_w, broadcast_compute_port, broadcast_compute_tensor = w_counts[0]
        min_w = w_counts[-1][0]   # lowest count (skip path)

        if max_w <= min_w:
            # Both paths have identical weight-actor counts — genuinely
            # ambiguous (not expected for any standard ResNet topology).
            print(f"[BlockSkipping] WARNING: ADD '{add_actor.unique_name}' "
                  f"has equal weight-actor counts on both paths ({max_w}), "
                  f"cannot determine compute vs skip path — skipping.")
            continue

        # ── Find compute result tensor and ADD compute port ──────────────────
        # The compute path ends at the ADD input whose live tensor is produced
        # by an actor reachable from broadcast_compute_tensor.
        if _can_reach_actor(broadcast_compute_tensor, prod0, actor_out_tensors):
            compute_prod, compute_port, compute_tensor = prod0, p0, t0
        else:
            compute_prod, compute_port, compute_tensor = prod1, p1, t1

        # ── Gather ordered compute-path actors ───────────────────────────────
        compute_actors = _collect_compute_actors(
            broadcast_compute_tensor, add_actor, actor_out_tensors, depth_map
        )
        if not compute_actors:
            continue

        blocks.append(ResidualBlock(
            add_actor                = add_actor,
            broadcast_actor          = broadcast_actor,
            broadcast_compute_port   = broadcast_compute_port,
            broadcast_compute_tensor = broadcast_compute_tensor,
            compute_actors           = compute_actors,
            compute_result_tensor    = compute_tensor,
            add_compute_port         = compute_port,
            block_output_size        = compute_tensor.size,
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
    """Idempotently add keep_<N> as a cfg_input port on actor."""
    port_name = keep_param.unique_id
    if any(p.name == port_name for p, _ in actor.params):
        return
    actor.add_param(port_name, keep_param)


# ===========================================================================
# 8.  STEP 1 — MODIFY THE BROADCAST ACTOR
# ===========================================================================

def _modify_broadcast(block: ResidualBlock, keep_param: IRParam) -> None:
    """
    Add keep_<N> as cfg_input to the broadcast actor and scale the
    compute-path output port rate by keep_<N>.  The skip-path output rate
    is left unchanged.
    """
    _add_cfg_port(block.broadcast_actor, keep_param)
    block.broadcast_compute_port.rate = _mul(
        block.broadcast_compute_port.rate, keep_param.unique_id
    )


# ===========================================================================
# 9.  STEP 2 — MODIFY COMPUTE-ACTOR DATA RATES
# ===========================================================================

def _modify_compute_actor_data_rates(actor: IRActor, keep_param: IRParam) -> None:
    """
    Add keep_<N> as cfg_input and multiply all data input / output port
    rates by keep_<N>.

    Note: operates directly on IRPort.rate via actor.inputs / actor.outputs.
    The port objects are always valid even when their tensor references are
    stale — we only need port.rate here.
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

def _insert_weight_gate(
    graph        : IRGraph,
    compute_actor: IRActor,
    weight_idx   : int,
    keep_param   : IRParam,
    label        : str,
) -> None:
    """
    Insert a Gate_Weights fork and a Sink actor for one weight FIFO.

    Gate_Weights is a Preesm *fork* (one input, two outputs at different
    rates summing to the input rate), not a broadcast.

    Before:
        source ──[weight_tensor]──► compute_actor.weight_port  (rate W)

    After:
        source ──[weight_tensor]──► Gate_Weights.weight_in     (rate W)
            Gate_Weights.to_conv      (W * keep_<N>)  ──► compute_actor.weight_port
            Gate_Weights.discarded (W*(1-keep_<N>))   ──► Sink.input
    """
    weight_port, weight_tensor = compute_actor.weights[weight_idx]
    W     = weight_tensor.size   # always the original static size
    pname = keep_param.unique_id
    dtype = weight_tensor.dtype

    # ── Gate_Weights fork ─────────────────────────────────────────────────
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

    # ── Rewire: weight_tensor → gate_actor, not directly to compute_actor ─
    if weight_port in weight_tensor.consumers:
        weight_tensor.consumers.remove(weight_port)
    graph.link(weight_tensor, None, gate_in_port)

    compute_actor.weights[weight_idx] = (weight_port, gated_tensor)
    weight_port.rate = _mul(W, pname)
    graph.link(gated_tensor, None, weight_port)

    # ── Sink actor ────────────────────────────────────────────────────────
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
# 11.  STEP 4 — ZERO + SELECT (JOIN) ACTORS
# ===========================================================================

def _insert_zero_and_select(
    graph      : IRGraph,
    block      : ResidualBlock,
    keep_param : IRParam,
    block_idx  : int,
) -> None:
    """
    Insert a Zero actor and a Select (join) actor before ADD's compute input.

        Zero.output   (S*(1-keep_<N>)) ──► Select.zero
        compute_last  (S*keep_<N>)     ──► Select.b_output
        Select.output (S, static)      ──► ADD.compute_input  [ADD unchanged]

    When keep=1: Zero produces 0 tokens; Select forwards S compute tokens.
    When keep=0: Zero produces S tokens; Select forwards S zeros; ADD
                 effectively passes the skip path straight through.
    """
    pname = keep_param.unique_id
    S     = block.block_output_size
    dtype = block.compute_result_tensor.dtype

    # ── Zero actor ────────────────────────────────────────────────────────
    zero_name  = f"zero_block{block_idx}"
    zero_actor = IRActor(
        name=zero_name, op_type=OpType.BROADCAST,
        unique_name=zero_name, source="Code/include/utilities.h",
        attributes={"_kind": "zero", "_loop_fn": "zero"},
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

    # ── Select (join) actor ───────────────────────────────────────────────
    select_name  = f"select_block{block_idx}"
    select_actor = IRActor(
        name=select_name, op_type=OpType.BROADCAST,
        unique_name=select_name, source="",
        attributes={"_kind": "select"},   # emitted as kind="join" in XML
    )
    graph._actors[select_name] = select_actor
    _add_cfg_port(select_actor, keep_param)

    # Input "zero" ← Zero actor
    sel_zero_port = IRPort(name="zero", direction=PortDir.IN,
                           rate=_inv(S, pname), actor=select_actor)
    select_actor.inputs.append((sel_zero_port, zero_tensor))
    graph.link(zero_tensor, None, sel_zero_port)

    # Input "b_output" ← compute result tensor (currently feeding ADD)
    compute_result   = block.compute_result_tensor
    add_compute_port = block.add_compute_port

    sel_b_port = IRPort(name="b_output", direction=PortDir.IN,
                        rate=_mul(S, pname), actor=select_actor)
    select_actor.inputs.append((sel_b_port, compute_result))

    # Remove ADD's compute port from compute_result's consumers; replace
    # with select's input port.
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
    # Match by port identity — tensor references in actor.inputs may be stale.
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

def _transform_block(
    graph     : IRGraph,
    block     : ResidualBlock,
    block_idx : int,
) -> IRParam:
    """Apply all four transformation steps for one residual block."""
    keep_param = _create_keep_param(graph, block_idx)

    # Step 1 — Broadcast: scale compute-path output rate
    _modify_broadcast(block, keep_param)

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
# 12.5  POLICY NETWORK ACTOR
# ===========================================================================

def _insert_policy_network(graph: IRGraph, keep_params: list[IRParam]) -> IRActor:
    """
    Insert the policyNetwork actor into the graph.

    The policyNetwork is a configuration-output actor: its loop function
    ``setCombination`` dynamically writes each keep_<N> parameter at
    runtime.  In PiSDF terms it has one *cfg_output* port per keep param,
    each wired via a dependency edge to the corresponding param node.

    The actor carries no data FIFOs — only the cfg_output ports.  Its
    ``_cfg_outputs`` attribute records the ordered list of keep params so
    the XML emitter can generate the correct dependency edges.

    Parameters
    ----------
    graph       : IRGraph
        The graph being transformed (modified in-place).
    keep_params : list[IRParam]
        Ordered list of keep_<N> params produced by the block transforms,
        one per residual block.

    Returns
    -------
    IRActor
        The newly created policyNetwork actor.
    """
    actor = IRActor(
        name="policyNetwork",
        op_type=OpType.BROADCAST,          # closest available op; overridden by _kind
        unique_name="policyNetwork",
        source="Code/include/utilities.h",
        attributes={
            "_kind"       : "policy",
            "_loop_fn"    : "setCombination",
            "_cfg_outputs": keep_params,   # ordered list of IRParam
        },
    )
    graph._actors["policyNetwork"] = actor
    return actor


# ===========================================================================
# 13.  PUBLIC ENTRY POINT
# ===========================================================================

def apply_block_skipping_pass(graph: IRGraph) -> int:
    """
    Detect all residual blocks and apply the block-skipping transformation
    in-place.  Returns the number of blocks transformed.

    After all blocks are transformed a single ``policyNetwork`` actor is
    inserted.  It owns a cfg_output port for every keep_<N> parameter and
    its dependency edges replace the static param values as the runtime
    source of the block-skip decisions.
    """
    blocks = detect_residual_blocks(graph)
    if not blocks:
        print("[BlockSkipping] No residual blocks detected. Graph unchanged.")
        return 0

    print(f"[BlockSkipping] Detected {len(blocks)} residual block(s).")
    keep_params: list[IRParam] = []
    for idx, block in enumerate(blocks):
        keep = _transform_block(graph, block, idx)
        keep_params.append(keep)
        compute_names = [a.unique_name for a in block.compute_actors]
        print(
            f"  Block {idx}: ADD='{block.add_actor.unique_name}'  "
            f"broadcast='{block.broadcast_actor.unique_name}'  "
            f"param='{keep.unique_id}'  compute={compute_names}"
        )

    _insert_policy_network(graph, keep_params)
    print(f"[BlockSkipping] Inserted policyNetwork actor "
          f"(controls: {[p.unique_id for p in keep_params]}).")

    return len(blocks)


# ===========================================================================
# 14.  XML GENERATION
# ===========================================================================

# Mapping from the internal _kind attribute to the Preesm XML kind string.
# Gate_Weights actors are Preesm forks (split data).
# Zero and Sink actors are plain Preesm actors.
# Select actors are Preesm joins.
# policyNetwork is a plain Preesm actor with cfg_output ports.
_SPECIAL_KIND_TO_XML_KIND: dict[str, str] = {
    "gate"  : "fork",
    "sink"  : "actor",
    "zero"  : "actor",
    "select": "join",
    "policy": "actor",
}


def _emit_node(graph_el: Element, actor: IRActor,
               xml_kind: str, loop_fn: str) -> None:
    """Emit one <node> element with its <loop> and <port> children."""
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


def _emit_policy_node(graph_el: Element, actor: IRActor) -> None:
    """
    Emit the <node> element for the policyNetwork actor.

    The policy actor has only cfg_output ports — one per keep_<N> param.
    Its loop params are ``direction="OUT" isConfig="true"`` with type
    ``int64_t``, and its ports carry ``kind="cfg_output"``.  It has no
    data FIFO ports at all.
    """
    keep_params: list[IRParam] = actor.attributes["_cfg_outputs"]
    loop_fn: str               = actor.attributes["_loop_fn"]

    node = SubElement(graph_el, "node", attrib={
        "id": actor.unique_name, "kind": "actor", "period": "0",
    })
    SubElement(node, "data", attrib={"key": "graph_desc"}).text = actor.source
    loop_el = SubElement(node, "loop", attrib={"name": loop_fn})

    for param in keep_params:
        SubElement(loop_el, "param", attrib={
            "direction": "OUT", "isConfig": "true",
            "name": param.unique_id, "type": "int64_t",
        })

    for param in keep_params:
        SubElement(node, "port", attrib={
            "kind": "cfg_output", "name": param.unique_id, "annotation": "NONE",
        })


def _generate_actor_node(graph_el: Element, actor: IRActor) -> None:
    """Emit the correct <node> for any actor, including block-skipping actors."""
    special = actor.attributes.get("_kind")

    if special == "policy":
        _emit_policy_node(graph_el, actor)
        return

    if special in _SPECIAL_KIND_TO_XML_KIND:
        xml_kind = _SPECIAL_KIND_TO_XML_KIND[special]
        loop_fn  = actor.attributes.get("_loop_fn", actor.unique_name)
        _emit_node(graph_el, actor, xml_kind, loop_fn)
        return

    # Standard converter actor kinds
    if actor.op_type == OpType.BROADCAST:
        kind = "broadcast"
    elif actor.op_type == OpType.SPLIT_WEIGHTS:
        kind = "fork"
    else:
        kind = "actor"
    _emit_node(graph_el, actor, kind, OPTYPE_TO_LOOP_FN.get(actor.op_type, ""))


def _emit_policy_dependency_edges(graph_el: Element, graph: IRGraph) -> None:
    """
    Emit dependency edges from the policyNetwork actor to each keep_<N>
    param node.

        <edge kind="dependency"
              source="policyNetwork" sourceport="keep_N"
              target="keep_N"/>

    These replace the static param values as the authoritative source of
    the block-skip decisions at runtime.
    """
    policy_actor = graph.get_actor("policyNetwork")
    if policy_actor is None:
        return

    keep_params: list[IRParam] = policy_actor.attributes.get("_cfg_outputs", [])
    for param in keep_params:
        SubElement(graph_el, "edge", attrib={
            "kind"      : "dependency",
            "source"    : "policyNetwork",
            "sourceport": param.unique_id,
            "target"    : param.unique_id,
        })


def generate_block_skipping_xml(graph: IRGraph, model_data: dict) -> str:
    """Generate PiSDF-compatible GraphML XML for a block-skipping-transformed graph."""
    root     = Element("graphml", attrib={"xmlns": "http://graphml.graphdrawing.org/xmlns"})
    graph_el = SubElement(root, "graph", attrib={"edgedefault": "directed"})
    SubElement(graph_el, "data", attrib={"key": "name"})

    for param in graph.params:
        SubElement(graph_el, "node", attrib={
            "id": param.unique_id, "kind": "param", "expr": str(param.value),
        })

    for actor in graph.actors:
        _generate_actor_node(graph_el, actor)

    for src_port, dst_port, dtype in graph.get_fifo_edges():
        src_actor = src_port.actor.unique_name if src_port.actor else "?"
        dst_actor = dst_port.actor.unique_name if dst_port.actor else "?"
        SubElement(graph_el, "edge", attrib={
            "kind":       "fifo",
            "source":     src_actor, "sourceport": src_port.name,
            "target":     dst_actor, "targetport":  dst_port.name,
            "type":       dtype,
        })

    for param, port in graph.get_dependency_edges():
        dst_actor = port.actor.unique_name if port.actor else "?"
        SubElement(graph_el, "edge", attrib={
            "kind":       "dependency",
            "source":     param.unique_id,
            "target":     dst_actor, "targetport": port.name,
        })

    # Emit policyNetwork → keep_<N> dependency edges after all standard edges.
    _emit_policy_dependency_edges(graph_el, graph)

    raw = tostring(root, encoding="unicode")
    return parseString(raw).toprettyxml(indent="  ")


def write_block_skipping_xml(
    graph       : IRGraph,
    model_data  : dict,
    output_path : str = "output_block_skipping.pi",
) -> None:
    """Write the block-skipping-transformed graph to a .pi file."""
    xml_str = generate_block_skipping_xml(graph, model_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"[BlockSkipping] XML written to  {output_path}")

