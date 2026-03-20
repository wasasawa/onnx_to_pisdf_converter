from __future__ import annotations
from structure import IRGraph, IRActor, OpType, OPTYPE_TO_PI
from pi_generator import RATE_EXPRESSIONS, SKIP_OPS


# ═════════════════════════════════════════════════════════════════════════
# 1.  COARSE RATE EXPRESSIONS  (symbolic "grain" / "grain_ch")
#
#     Only the "inputs" and "outputs" dicts change (the inner actor rates).
#     "src_rates" and "snk_rates" stay identical to the originals.
# ═════════════════════════════════════════════════════════════════════════

COARSE_RATE_EXPRESSIONS = {

    # ── element-wise: inner processes 'grain' elements per firing ────────
    "relu": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "grain"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "size"},
    },
    "sigmoid": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "grain"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "size"},
    },
    "tanh": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "grain"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "size"},
    },
    "dropout": {
        "src_rates":  {"input_0": "size"},
        "inputs":     {"input_0": "grain"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "size"},
    },

    # ── add variants ─────────────────────────────────────────────────────
    "add_same": {
        "src_rates":  {"input_0": "size1", "input_1": "size1"},
        "inputs":     {"input_0": "grain", "input_1": "grain"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "size1"},
    },
    "add_bias": {
        "src_rates":  {"input_0": "size1", "input_1": "size2"},
        "inputs":     {"input_0": "grain", "input_1": "grain"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "size1"},
    },
    "add_scalar": {
        "src_rates":  {"input_0": "size1", "input_1": "1"},
        "inputs":     {"input_0": "grain", "input_1": "1"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "size1"},
    },
    "add_generic": {
        "src_rates":  {"input_0": "size1", "input_1": "size2"},
        "inputs":     {"input_0": "grain", "input_1": "grain"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "size1"},
    },

    # ── reshape / flatten ─────────────────────────────────────────────────
    "reshape": {
        "src_rates":  {"input_0": "outputSize", "input_1": "shapeSize"},
        "inputs":     {"input_0": "grain",      "input_1": "grain"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "outputSize"},
    },
    "flatten": {
        "src_rates":  {"input_0": "inputSize"},
        "inputs":     {"input_0": "grain"},
        "outputs":    {"output_0": "grain"},
        "snk_rates":  {"output_0": "outputSize"},
    },

    # ── matmul / gemm: inner processes 'grain' rows per firing ───────────
    "matmul": {
        "src_rates":  {"input_0": "M * K",     "input_1": "K * N"},
        "inputs":     {"input_0": "grain * K", "input_1": "K * N"},
        "outputs":    {"output_0": "grain * N"},
        "snk_rates":  {"output_0": "M * N"},
    },
    "gemm": {
        "src_rates":  {"input_0": "M * K",     "input_1": "K * N", "input_2": "sizeC"},
        "inputs":     {"input_0": "grain * K", "input_1": "K * N", "input_2": "sizeC"},
        "outputs":    {"output_0": "grain * N"},
        "snk_rates":  {"output_0": "M * N"},
    },

    # ── conv: grain_ch = number of output channels per firing ────────────
    "conv2d": {
        "src_rates": {
            "input_0": "depthInput * inputHeight * inputWidth",
            "input_1": "depthOutput * depthInput * sizeKernelHeight * sizeKernelWidth",
        },
        "inputs": {
            "input_0": "depthInput * inputHeight * inputWidth",
            "input_1": "grain_ch * depthInput * sizeKernelHeight * sizeKernelWidth",
        },
        "outputs": {
            "output_0": "grain_ch * outputHeight * outputWidth",
        },
        "snk_rates": {
            "output_0": "depthOutput * outputHeight * outputWidth",
        },
    },
    "conv2d_bias": {
        "src_rates": {
            "input_0": "depthInput * inputHeight * inputWidth",
            "input_1": "depthOutput * depthInput * sizeKernelHeight * sizeKernelWidth",
            "input_2": "depthOutput",
        },
        "inputs": {
            "input_0": "depthInput * inputHeight * inputWidth",
            "input_1": "grain_ch * depthInput * sizeKernelHeight * sizeKernelWidth",
            "input_2": "grain_ch",
        },
        "outputs": {
            "output_0": "grain_ch * outputHeight * outputWidth",
        },
        "snk_rates": {
            "output_0": "depthOutput * outputHeight * outputWidth",
        },
    },

    # ── pool: grain_ch = number of channels per firing ───────────────────
    "maxpool2d": {
        "src_rates": {"input_0": "depthInput * inputHeight * inputWidth"},
        "inputs":    {"input_0": "grain_ch * inputHeight * inputWidth"},
        "outputs":   {"output_0": "grain_ch * outputHeight * outputWidth"},
        "snk_rates": {"output_0": "depthInput * outputHeight * outputWidth"},
    },
    "avgpool2d": {
        "src_rates": {"input_0": "depthInput * inputHeight * inputWidth"},
        "inputs":    {"input_0": "grain_ch * inputHeight * inputWidth"},
        "outputs":   {"output_0": "grain_ch * outputHeight * outputWidth"},
        "snk_rates": {"output_0": "depthInput * outputHeight * outputWidth"},
    },
    "global_avgpool": {
        "src_rates": {"input_0": "depth * spatialSize"},
        "inputs":    {"input_0": "grain_ch * spatialSize"},
        "outputs":   {"output_0": "grain_ch"},
        "snk_rates": {"output_0": "depth"},
    },
}

# ── which existing param to read and which grain name to attach ──────────

_OP_CONFIG = {
    # element-wise
    "relu":          ("size",        "grain"),
    "sigmoid":       ("size",        "grain"),
    "tanh":          ("size",        "grain"),
    "dropout":       ("size",        "grain"),
    # add variants
    "add_same":      ("size1",       "grain"),
    "add_bias":      ("size1",       "grain"),
    "add_scalar":    ("size1",       "grain"),
    "add_generic":   ("size1",       "grain"),
    # shape
    "reshape":       ("outputSize",  "grain"),
    "flatten":       ("outputSize",  "grain"),
    # linear (split by rows)
    "matmul":        ("M",           "grain"),
    "gemm":          ("M",           "grain"),
    # conv
    "conv2d":        ("depthOutput", "grain_ch"),
    "conv2d_bias":   ("depthOutput", "grain_ch"),
    # pool
    "maxpool2d":     ("depthInput",  "grain_ch"),
    "avgpool2d":     ("depthInput",  "grain_ch"),
    "global_avgpool":("depth",       "grain_ch"),
}


# ═════════════════════════════════════════════════════════════════════════
# 2.  GRAIN COMPUTATION
# ═════════════════════════════════════════════════════════════════════════

def _best_grain(total: int, target: int) -> int:
    """
    Find G such that total / G ≈ target and total % G == 0.

    Searches outward from the ideal repetition count until a divisor
    of total is found. Always returns a valid divisor.
    """
    if total <= 0 or target <= 0:
        return max(total, 1)
    target = min(target, total)

    if total % target == 0:
        return total // target

    for delta in range(1, total):
        for rep in (target + delta, target - delta):
            if rep < 1 or rep > total:
                continue
            if total % rep == 0:
                return total // rep

    return total


def _get_param_int(actor: IRActor, name: str) -> int:
    """Read an integer param value from actor, or 0 if missing."""
    for port, param in actor.params:
        if port.name == name:
            try:
                return int(param.value)
            except (ValueError, TypeError):
                return 0
    return 0


# ═════════════════════════════════════════════════════════════════════════
# 3.  PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

def coarsen_graph(graph: IRGraph, target_parallelism: int = 8) -> None:
    """
    Prepare the graph for coarse-grained hierarchical .pi generation.

    Does two things:
      1. Patches RATE_EXPRESSIONS in-place with coarse versions
      2. Attaches a grain / grain_ch IRParam to each applicable actor

    Call this BEFORE generate_all_pi_files().
    """

    # Step 1: patch the global RATE_EXPRESSIONS dict
    RATE_EXPRESSIONS.update(COARSE_RATE_EXPRESSIONS)

    # Step 2: attach per-instance grain params
    P = target_parallelism

    for actor in graph.actors:
        if actor.op_type in SKIP_OPS:
            continue
        if actor.op_type not in OPTYPE_TO_PI:
            continue

        op_name = OPTYPE_TO_PI[actor.op_type].split("/")[-1].replace(".pi", "")
        config = _OP_CONFIG.get(op_name)
        if config is None:
            continue

        param_name, grain_name = config
        if any(p.name == grain_name for p, _ in actor.params):
            continue  # already coarsened
        
        total = _get_param_int(actor, param_name)
        if total <= 0:
            continue

        grain_value = _best_grain(total, P)
        grain_param = graph.get_or_create_param(grain_name, grain_value)
        actor.add_param(grain_name, grain_param)

        rep = total // grain_value
        print(f"  [Coarse] {actor.unique_name}: {param_name}={total}, "
              f"{grain_name}={grain_value}, rep={rep}")