"""
block_skipping_pass.py
======================
Block-Skipping Dynamic Layer-Skipping Transformation Pass for PiSDF/PiMM Graphs.

References
----------
- Wu et al., "BlockDrop: Dynamic Inference Paths in Residual Networks", CVPR 2018.
- Bhattacharya et al., "PiMM: Parameterized and Interfaced dataflow MetaModel
  for MPSoCs runtime reconfiguration", SAMOS 2013.


DETECTION STRATEGY  (architecture-agnostic)
============================================
A residual block is any subgraph of the form:

        BROADCAST ──► [compute path, ≥1 actor] ──► ADD
                  └──► [skip path, ≥0 actors]  ──► ADD

Necessary and sufficient conditions:

  ①  ADD actor with exactly 2 data inputs, both with live producer actors.
  ②  A common BROADCAST ancestor exists for both inputs.
      (The converter inserts a BROADCAST actor whenever a tensor has more
       than one consumer, so this is always the structural fork point.)
  ③  The two paths have different topological depths.  The deeper path is
      the compute path; the shallower is the skip path.

This is intentionally free of any constraint on *which operators* appear on
either path.  The following skip-connection variants are all handled:

  ▸ Identity shortcut — no operators between broadcast and ADD.
  ▸ Projection (1×1 Conv + BN) — as in ResNet-50/101/152 stage-first blocks.
  ▸ Any other projection type (BN only, Slice+Pad, etc.).

Crucially, the old approach of using CONV-presence heuristics on backward-BFS
ancestor sets was fragile: the ancestor BFS always traces back far enough to
include convolutions from *preceding* blocks, making both paths appear to
contain convolutions regardless of the actual skip-connection type.  Pure
depth comparison is the correct and robust discriminator.


TRANSFORMATION STRATEGY  (matches dynamic_resnet8 reference exactly)
======================================================================
For each detected residual block the pass performs four steps, all
parameterised by a new Boolean PiSDF parameter  keep_<N>  (default 1 =
execute block, 0 = skip block).

  Step 1 — Modify the BROADCAST actor that branches both paths:
    ▸ Add  keep_<N>  as cfg_input.
    ▸ Multiply the compute-path output port rate by  keep_<N>.
    ▸ Leave the skip-path output port rate unchanged.

  Step 2 — For every actor on the compute path (broadcast → ADD, exclusive):
    ▸ Add  keep_<N>  as cfg_input.
    ▸ Multiply all data input and output port rates by  keep_<N>.

  Step 3 — Weight gating for compute-path actors.
    For each weight port whose tensor has a live producer actor:

        Before:
            SPLIT_WEIGHTS ──[W]──► compute_actor.weight_port  (rate W)

        After:
            SPLIT_WEIGHTS ──[W]──► Gate_Weights.weight_in     (rate W, static)
                Gate_Weights.to_conv       (W * keep_<N>)  ──► compute_actor.weight_port
                Gate_Weights.discarded  (W*(1-keep_<N>))  ──► Sink.input

        Both Gate_Weights (a Preesm fork) and Sink receive  keep_<N>  as
        cfg_input.

  Step 4 — Insert Zero + Select actors before ADD's compute input:
        Zero.output   (S*(1-keep_<N>))  ──► Select.zero
        compute_last  (S*keep_<N>)      ──► Select.b_output
        Select.output (S, static)       ──► ADD.compute_input

    The ADD actor itself is never modified.  The skip path is never modified.


TERMINOLOGY NOTE — Broadcast vs Fork in Preesm
================================================
In Preesm, "broadcast" and "fork" are two distinct actor kinds:

  ▸ broadcast  (kind="broadcast") — one input, N outputs, all at the same
    rate.  Used to replicate a token to multiple consumers.  This is what
    the converter creates via add_broadcast_actors().

  ▸ fork       (kind="fork") — one input, N outputs at *different* rates
    that sum to the input rate.  Used to split a token stream.  The
    SPLIT_WEIGHTS actor and the Gate_Weights actors introduced by this
    pass are forks.

All variables and comments in this module use "broadcast" when referring to
the BROADCAST actors that branch the compute and skip paths, and "gate" /
"fork" when referring to the weight-gating fork actors.


STALE-TENSOR HANDLING
=====================
The converter's add_broadcast_actors() replaces every multi-consumer tensor T
with a BROADCAST actor and per-consumer output tensors, then deletes T from
graph._tensors — but does NOT update actor.inputs / actor.outputs / actor.weights.
Those lists still hold references to deleted objects.

All graph traversal here is therefore **tensor-centric**: built from
graph.tensors, not from actor port lists.  See _build_* helpers below.


Usage
-----
    from block_skipping_pass import apply_block_skipping_pass, write_block_skipping_xml

    graph = main(model_path, output_xml)         # existing pipeline unchanged
    n     = apply_block_skipping_pass(graph)     # in-place; returns block count
    write_block_skipping_xml(graph, model_data, "output_bs.pi")
"""

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
# 4.  TOPOLOGICAL DEPTH  (tensor-centric Kahn's algorithm)
# ===========================================================================

def _compute_actor_depths(
    graph: IRGraph,
    actor_input_tensors : dict[str, list[IRTensor]],
    producer_map        : dict[str, IRActor],
) -> dict[str, int]:
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

_CONV_OPS = frozenset({OpType.CONV2D, OpType.CONV2D_BIAS})


def _is_residual_add(actor: IRActor) -> bool:
    return (
        actor.op_type in (OpType.ADD_SAME, OpType.ADD_BIAS, OpType.ADD_GENERIC)
        and len(actor.inputs) == 2
    )


def detect_residual_blocks(graph: IRGraph) -> list[ResidualBlock]:
    """
    Detect all residual blocks in the graph.

    Detection is fully generic with respect to skip-connection type.
    The only structural requirements are:

      ①  ADD actor with 2 live-wired data inputs.
      ②  A common BROADCAST ancestor for both inputs
         (always inserted by the converter for any multi-consumer tensor).
      ③  Strictly different topological depths for the two input producers.
         The deeper producer's path is the compute path; the shallower is
         the skip path.  No constraint is placed on what operators appear
         on either path — identity, 1×1 Conv+BN, BN-only, Slice+Pad, or
         any future variant are all handled transparently.

    Robustness against stale tensor references
    ------------------------------------------
    actor.inputs[i][1] may reference a deleted tensor object after the
    converter's add_broadcast_actors() pass.  All tensor look-ups here use
    consumer_to_tensor (id(port) → live tensor), which is always current.
    """
    # Build all tensor-centric maps once
    producer_map          = _build_producer_map(graph)
    actor_out_tensors     = _build_actor_output_tensors(graph)
    actor_in_tensors      = _build_actor_input_tensors(graph)
    consumer_to_tensor    = _build_consumer_to_tensor(graph)
    depth_map             = _compute_actor_depths(graph, actor_in_tensors, producer_map)

    blocks: list[ResidualBlock] = []

    for add_actor in graph.actors:
        if not _is_residual_add(add_actor):
            continue

        # ── Resolve the actual live tensors for ADD's two input ports ────────
        # actor.inputs[i][1] may be stale; use consumer_to_tensor instead.
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

        # ── Classify compute path vs skip path by topological depth ──────────
        #
        # The compute path is always deeper than the skip path in every
        # ResNet variant:
        #   • Identity skip    — skip producer IS the broadcast actor itself
        #                        (0 intermediate actors), compute is much deeper.
        #   • Projection skip  — skip projection chain (conv+bn, or bn, or
        #                        slice+pad, etc.) is always shorter than the
        #                        compute chain.
        #
        # Using depth directly avoids the fragility of op-type heuristics
        # (e.g. CONV-presence on ancestor sets), which fail because backward
        # BFS always traverses back to convolutions in preceding blocks.
        d0 = depth_map.get(prod0.unique_name, 0)
        d1 = depth_map.get(prod1.unique_name, 0)

        if d0 == d1:
            # Equal depth — cannot determine compute vs skip.  This should
            # not occur in any standard ResNet topology; skip with a warning.
            print(f"[BlockSkipping] WARNING: ADD '{add_actor.unique_name}' "
                  f"has equal-depth inputs (d={d0}), skipping.")
            continue

        if d0 > d1:
            compute_prod, compute_port, compute_tensor = prod0, p0, t0
        else:
            compute_prod, compute_port, compute_tensor = prod1, p1, t1

        # ── Find the deepest common BROADCAST ancestor ───────────────────────
        #
        # Both paths diverge from a single BROADCAST actor inserted by the
        # converter (add_broadcast_actors).  We need the DEEPEST one to avoid
        # picking an outer broadcast from a preceding block that happens to be
        # a common ancestor of both paths.
        anc_compute = _ancestors_bfs(compute_prod, actor_in_tensors, producer_map)

        # Determine skip producer: the other ADD input
        skip_prod = prod1 if compute_prod is prod0 else prod0
        anc_skip  = _ancestors_bfs(skip_prod,  actor_in_tensors, producer_map)

        common_broadcasts = [
            name for name in (anc_compute & anc_skip)
            if (a := graph.get_actor(name)) is not None
            and a.op_type == OpType.BROADCAST
        ]
        if not common_broadcasts:
            continue

        broadcast_name  = max(common_broadcasts, key=lambda n: depth_map.get(n, 0))
        broadcast_actor = graph.get_actor(broadcast_name)

        # ── Identify which broadcast output leads to the compute path ────────
        broadcast_compute_port   = None
        broadcast_compute_tensor = None
        for out_port, out_tensor in broadcast_actor.outputs:
            if _can_reach_actor(out_tensor, compute_prod, actor_out_tensors):
                broadcast_compute_port   = out_port
                broadcast_compute_tensor = out_tensor
                break

        if broadcast_compute_port is None:
            continue

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
# 13.  PUBLIC ENTRY POINT
# ===========================================================================

def apply_block_skipping_pass(graph: IRGraph) -> int:
    """
    Detect all residual blocks and apply the block-skipping transformation
    in-place.  Returns the number of blocks transformed.
    """
    blocks = detect_residual_blocks(graph)
    if not blocks:
        print("[BlockSkipping] No residual blocks detected. Graph unchanged.")
        return 0

    print(f"[BlockSkipping] Detected {len(blocks)} residual block(s).")
    for idx, block in enumerate(blocks):
        keep = _transform_block(graph, block, idx)
        compute_names = [a.unique_name for a in block.compute_actors]
        print(
            f"  Block {idx}: ADD='{block.add_actor.unique_name}'  "
            f"broadcast='{block.broadcast_actor.unique_name}'  "
            f"param='{keep.unique_id}'  compute={compute_names}"
        )
    return len(blocks)


# ===========================================================================
# 14.  XML GENERATION
# ===========================================================================

# Mapping from the internal _kind attribute to the Preesm XML kind string.
# Gate_Weights actors are Preesm forks (split data).
# Zero and Sink actors are plain Preesm actors.
# Select actors are Preesm joins.
_SPECIAL_KIND_TO_XML_KIND: dict[str, str] = {
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


# ===========================================================================
# 15.  SELF-TEST
#
#  Tests THREE block topologies to validate the generic detection:
#
#  Block A — identity skip (no ops between broadcast and ADD)
#     CONV0 → RELU0 → BROADCAST_A
#                       ├─ output_0 → CONV1 → RELU1 → CONV2 → ADD_A
#                       └─ output_1 ─────────────────────────→ ADD_A
#
#  Block B — projection skip (Conv on skip path)
#     ADD_A → RELU2 → BROADCAST_B
#                       ├─ output_0 → CONV3 → RELU3 → CONV4 → ADD_B
#                       └─ output_1 → CONV5 (1×1 proj) ──────→ ADD_B
#
#  Both blocks have SPLIT_WEIGHTS providing learnable weights to compute
#  convolutions, to exercise the weight-gating path.
#
#  The stale-tensor condition from add_broadcast_actors() is reproduced
#  faithfully: original multi-consumer tensors are deleted from
#  graph._tensors after BROADCAST actors are inserted.
# ===========================================================================

def _make_test_graph() -> IRGraph:
    """
    Build a two-block toy graph that reproduces the stale-tensor condition
    and covers both identity-skip and projection-skip topologies.
    """
    g     = IRGraph(name="toy_two_block")
    SHAPE = [1, 16, 8, 8]
    S     = 16 * 8 * 8      # 1024 elements
    W     = 16 * 16 * 3 * 3 # 2304 weight elements

    def _actor(op: OpType, uname: str) -> IRActor:
        a = IRActor(name=uname, op_type=op, unique_name=uname,
                    source=OPTYPE_TO_H.get(op, ""), attributes={})
        g._actors[uname] = a
        return a

    def _tensor(name: str, shape=None) -> IRTensor:
        return g.create_tensor(name, shape or SHAPE)

    def _wire_data(producer_actor: IRActor, t: IRTensor,
                   consumer_actor: IRActor, port_name: str) -> IRPort:
        """Wire producer → tensor → consumer, returning the consumer port."""
        out_port = IRPort(name=f"output_0", direction=PortDir.OUT, rate=S, actor=producer_actor)
        producer_actor.outputs.append((out_port, t))
        g.link(t, out_port, None)
        in_port = IRPort(name=port_name, direction=PortDir.IN, rate=S, actor=consumer_actor)
        consumer_actor.inputs.append((in_port, t))
        g.link(t, None, in_port)
        return in_port

    # ── SPLIT_WEIGHTS mock (weights for all four compute convs) ───────────
    split = _actor(OpType.SPLIT_WEIGHTS, "SPLIT_WEIGHTS_0")
    weights = []
    for i in range(4):
        wt = g.create_tensor(f"w{i}", [16, 16, 3, 3])
        wp = IRPort(name=f"w{i}_out", direction=PortDir.OUT, rate=W, actor=split)
        split.outputs.append((wp, wt))
        g.link(wt, wp, None)
        weights.append(wt)

    # ── CONV0 → RELU0 ─────────────────────────────────────────────────────
    x = _tensor("x")
    conv0 = _actor(OpType.CONV2D, "CONV2D_0")
    c0_out = _tensor("c0_out")
    _wire_data(conv0, c0_out, _actor(OpType.RELU, "RELU_0"), "input_0")
    relu0 = g.get_actor("RELU_0")
    r0_out = _tensor("r0_out")

    # Make RELU_0 produce r0_out with TWO consumers (add + conv_block_A)
    r0_port = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=relu0)
    relu0.outputs.append((r0_port, r0_out))
    g.link(r0_out, r0_port, None)

    # ── Block A: identity skip ─────────────────────────────────────────────
    # Actors: CONV1, RELU1, CONV2, ADD_A
    conv1 = _actor(OpType.CONV2D, "CONV2D_1")
    relu1 = _actor(OpType.RELU,   "RELU_1")
    conv2 = _actor(OpType.CONV2D, "CONV2D_2")
    add_a = _actor(OpType.ADD_SAME, "ADD_SAME_0")

    # Temporarily wire r0_out to conv1 and add_a (two consumers → will be broadcast)
    p_c1_in = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=conv1)
    p_c1_w  = IRPort(name="weight_0", direction=PortDir.IN, rate=W, actor=conv1)
    conv1.inputs  = [(p_c1_in, r0_out)]
    conv1.weights = [(p_c1_w, weights[0])]
    g.link(r0_out, None, p_c1_in)
    g.link(weights[0], None, p_c1_w)

    c1_out = _tensor("c1_out")
    c1_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=conv1)
    conv1.outputs = [(c1_op, c1_out)]; g.link(c1_out, c1_op, None)

    p_r1 = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=relu1)
    relu1.inputs = [(p_r1, c1_out)]; g.link(c1_out, None, p_r1)
    r1_out = _tensor("r1_out")
    r1_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=relu1)
    relu1.outputs = [(r1_op, r1_out)]; g.link(r1_out, r1_op, None)

    p_c2_in = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=conv2)
    p_c2_w  = IRPort(name="weight_0", direction=PortDir.IN, rate=W, actor=conv2)
    conv2.inputs  = [(p_c2_in, r1_out)]; g.link(r1_out, None, p_c2_in)
    conv2.weights = [(p_c2_w, weights[1])]; g.link(weights[1], None, p_c2_w)
    c2_out = _tensor("c2_out")
    c2_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=conv2)
    conv2.outputs = [(c2_op, c2_out)]; g.link(c2_out, c2_op, None)

    p_a0_0 = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=add_a)
    p_a0_1 = IRPort(name="input_1", direction=PortDir.IN, rate=S, actor=add_a)
    add_a.inputs = [(p_a0_0, c2_out), (p_a0_1, r0_out)]  # p_a0_1 → r0_out (will be stale)
    g.link(c2_out, None, p_a0_0)
    g.link(r0_out, None, p_a0_1)
    a0_out = _tensor("a0_out")
    a0_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=add_a)
    add_a.outputs = [(a0_op, a0_out)]; g.link(a0_out, a0_op, None)

    # ── Simulate add_broadcast_actors on r0_out (identity skip) ───────────
    bcast_a = _actor(OpType.BROADCAST, "BROADCAST_0")
    bca_in_t = g.create_tensor("r0_out_to_bcast", SHAPE)
    bca_in_p = IRPort(name="input", direction=PortDir.IN, rate=S, actor=bcast_a)
    bcast_a.inputs = [(bca_in_p, bca_in_t)]
    g.link(bca_in_t, r0_port, bca_in_p)   # RELU_0.output_0 → bcast

    r0_out.consumers.clear()   # simulate del
    bca_t0 = g.create_tensor("r0_out_bcast_0", SHAPE)
    bca_p0 = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=bcast_a)
    bcast_a.outputs.append((bca_p0, bca_t0))
    g.link(bca_t0, bca_p0, p_c1_in)   # → CONV1 compute input

    bca_t1 = g.create_tensor("r0_out_bcast_1", SHAPE)
    bca_p1 = IRPort(name="output_1", direction=PortDir.OUT, rate=S, actor=bcast_a)
    bcast_a.outputs.append((bca_p1, bca_t1))
    g.link(bca_t1, bca_p1, p_a0_1)    # → ADD_A skip input (identity)

    del g._tensors["r0_out"]   # stale; actor.inputs still references it

    # ── RELU_A (post-add, feeds Block B) ─────────────────────────────────
    relu_a = _actor(OpType.RELU, "RELU_A")
    p_ra = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=relu_a)
    relu_a.inputs = [(p_ra, a0_out)]; g.link(a0_out, None, p_ra)
    ra_out = _tensor("ra_out")
    ra_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=relu_a)
    relu_a.outputs = [(ra_op, ra_out)]; g.link(ra_out, ra_op, None)

    # ── Block B: projection skip (CONV5 on skip path) ─────────────────────
    conv3 = _actor(OpType.CONV2D, "CONV2D_3")
    relu3 = _actor(OpType.RELU,   "RELU_3")
    conv4 = _actor(OpType.CONV2D, "CONV2D_4")
    conv5 = _actor(OpType.CONV2D, "CONV2D_5")   # 1×1 projection
    add_b = _actor(OpType.ADD_SAME, "ADD_SAME_1")

    # ra_out has two consumers (conv3, conv5) → will be broadcast
    p_c3_in = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=conv3)
    p_c3_w  = IRPort(name="weight_0", direction=PortDir.IN, rate=W, actor=conv3)
    conv3.inputs  = [(p_c3_in, ra_out)]; g.link(ra_out, None, p_c3_in)
    conv3.weights = [(p_c3_w, weights[2])]; g.link(weights[2], None, p_c3_w)
    c3_out = _tensor("c3_out")
    c3_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=conv3)
    conv3.outputs = [(c3_op, c3_out)]; g.link(c3_out, c3_op, None)

    p_r3 = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=relu3)
    relu3.inputs = [(p_r3, c3_out)]; g.link(c3_out, None, p_r3)
    r3_out = _tensor("r3_out")
    r3_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=relu3)
    relu3.outputs = [(r3_op, r3_out)]; g.link(r3_out, r3_op, None)

    p_c4_in = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=conv4)
    p_c4_w  = IRPort(name="weight_0", direction=PortDir.IN, rate=W, actor=conv4)
    conv4.inputs  = [(p_c4_in, r3_out)]; g.link(r3_out, None, p_c4_in)
    conv4.weights = [(p_c4_w, weights[3])]; g.link(weights[3], None, p_c4_w)
    c4_out = _tensor("c4_out")
    c4_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=conv4)
    conv4.outputs = [(c4_op, c4_out)]; g.link(c4_out, c4_op, None)

    # CONV5 = projection (1×1 conv on skip path, no weight gate needed here
    # since weights[*] only cover compute convs in this toy graph)
    p_c5_in = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=conv5)
    conv5.inputs = [(p_c5_in, ra_out)]   # stale after broadcast
    g.link(ra_out, None, p_c5_in)
    c5_out = _tensor("c5_out")
    c5_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=conv5)
    conv5.outputs = [(c5_op, c5_out)]; g.link(c5_out, c5_op, None)

    p_ab_0 = IRPort(name="input_0", direction=PortDir.IN, rate=S, actor=add_b)
    p_ab_1 = IRPort(name="input_1", direction=PortDir.IN, rate=S, actor=add_b)
    add_b.inputs = [(p_ab_0, c4_out), (p_ab_1, ra_out)]  # p_ab_1 → ra_out (will be stale)
    g.link(c4_out, None, p_ab_0)
    g.link(ra_out, None, p_ab_1)
    ab_out = _tensor("ab_out")
    ab_op = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=add_b)
    add_b.outputs = [(ab_op, ab_out)]; g.link(ab_out, ab_op, None)

    # ── Simulate add_broadcast_actors on ra_out (projection skip) ─────────
    bcast_b = _actor(OpType.BROADCAST, "BROADCAST_1")
    bcb_in_t = g.create_tensor("ra_out_to_bcast", SHAPE)
    bcb_in_p = IRPort(name="input", direction=PortDir.IN, rate=S, actor=bcast_b)
    bcast_b.inputs = [(bcb_in_p, bcb_in_t)]
    g.link(bcb_in_t, ra_op, bcb_in_p)   # RELU_A.output_0 → bcast

    ra_out.consumers.clear()
    bcb_t0 = g.create_tensor("ra_out_bcast_0", SHAPE)
    bcb_p0 = IRPort(name="output_0", direction=PortDir.OUT, rate=S, actor=bcast_b)
    bcast_b.outputs.append((bcb_p0, bcb_t0))
    g.link(bcb_t0, bcb_p0, p_c3_in)   # → CONV3 compute input

    bcb_t1 = g.create_tensor("ra_out_bcast_1", SHAPE)
    bcb_p1 = IRPort(name="output_1", direction=PortDir.OUT, rate=S, actor=bcast_b)
    bcast_b.outputs.append((bcb_p1, bcb_t1))
    g.link(bcb_t1, bcb_p1, p_c5_in)   # → CONV5 projection (skip path)
    g.link(bcb_t1, bcb_p1, p_ab_1)    # also direct to ADD_B skip port

    del g._tensors["ra_out"]

    return g


if __name__ == "__main__":
    print("=" * 65)
    print("BlockSkipping Self-Test — Two-Block Graph (identity + projection)")
    print("=" * 65)

    g = _make_test_graph()

    print("\n[Before]")
    g.print_summary()

    n = apply_block_skipping_pass(g)

    print(f"\n[After]  {n} block(s) transformed.")
    g.print_summary()

    print("\n[Block-skipping FIFO edges]")
    bd_kw = ("Gate_", "Sink_", "zero_", "select_", "_gated_",
             "_discard_", "zero_tensor", "select_out")
    for src, dst, _ in g.get_fifo_edges():
        line = (f"  {src.actor.unique_name}.{src.name}[{src.rate}]"
                f" → {dst.actor.unique_name}.{dst.name}[{dst.rate}]")
        if any(k in line for k in bd_kw):
            print(line)

    print("\n[keep_ dependency edges]")
    for param, port in g.get_dependency_edges():
        if param.unique_id.startswith("keep_"):
            print(f"  {param.unique_id} → {port.actor.unique_name}.{port.name}")

    print("\n[Broadcast output rates after transform]")
    for bname in ("BROADCAST_0", "BROADCAST_1"):
        b = g.get_actor(bname)
        if b:
            for p, _ in b.outputs:
                print(f"  {bname}.{p.name}: rate={p.rate}")

    print("\n[ADD inputs after transform]")
    for aname in ("ADD_SAME_0", "ADD_SAME_1"):
        add = g.get_actor(aname)
        if add:
            for p, t in add.inputs:
                live = g.get_tensor(t.name) is not None
                print(f"  {aname}.{p.name}: tensor='{t.name}' rate={p.rate} live={live}")

    print()
    expected_blocks = 2
    assert n == expected_blocks, f"Expected {expected_blocks} blocks, got {n}"
    print(f"✓ Detected and transformed all {expected_blocks} block types correctly.")
