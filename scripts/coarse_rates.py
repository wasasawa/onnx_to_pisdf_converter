from __future__ import annotations
from structure import IRGraph, IRActor, OpType, OPTYPE_TO_PI, IRParam
from pi_generator import RATE_EXPRESSIONS, SKIP_OPS

# Name (and node id) of the single global parallelism knob in the top-level .pi.
TARGET_PARAM = "TargetParallelism"


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

def _divisors(n: int) -> list[int]:
    """All positive divisors of n, sorted ascending (includes 1 and n)."""
    ds = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.add(i)
            ds.add(n // i)
        i += 1
    return sorted(ds)


def _grain_expr(total: int, knob: str = TARGET_PARAM,
                max_parallelism: int | None = None) -> str:
    """
    Divisor-floor rule as a PREESM parameter expression.

        rep   = largest divisor of `total` that is <= knob   ( >= 1 always )
              = max over divisors d of ( d * floor(min(knob,d)/d) )
        grain = total / rep

    The d==1 term is always 1 (for knob >= 1), so it seeds the max and
    guarantees rep >= 1. Only divisors <= max_parallelism are offered as
    candidates (None = all), which bounds the expression length for split
    dimensions with many divisors.
    """
    divs = [d for d in _divisors(total)
            if max_parallelism is None or d <= max_parallelism]
    rep = "1"                                   # the d==1 term
    for d in divs:
        if d == 1:
            continue
        term = f"{d}*floor(min({knob},{d})/{d})"
        rep = f"max({rep}, {term})"
    return f"{total} / ({rep})"


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

def coarsen_graph(graph: IRGraph, target_parallelism: int = 8,
                  max_parallelism: int | None = None) -> None:
    """
    Prepare the graph for coarse-grained hierarchical .pi generation.

    Does two things:
      1. Patches RATE_EXPRESSIONS in-place with coarse versions.
      2. Adds the single global `TargetParallelism` knob and attaches a derived
         grain / grain_ch IRParam (grain = divisor_floor(total, TargetParallelism))
         to each applicable actor, recording the TargetParallelism -> grain
         param dependency so the top-level .pi can wire it.

    Call this BEFORE generate_all_pi_files().
    """

    # Step 1: patch the global RATE_EXPRESSIONS dict
    RATE_EXPRESSIONS.update(COARSE_RATE_EXPRESSIONS)

    # Step 2a: the single global knob (clean node id == TARGET_PARAM).
    knob = IRParam(name=TARGET_PARAM, value=str(target_parallelism),
                   unique_id=TARGET_PARAM)
    graph._params[TARGET_PARAM] = knob

    # param -> param dependency edges (source_uid, target_uid); read by the
    # top-level XML writers via getattr(graph, "_param_deps", []).
    param_deps = getattr(graph, "_param_deps", None)
    if param_deps is None:
        param_deps = []
        graph._param_deps = param_deps

    # Step 2b: attach a derived grain param to each applicable actor.
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

        # Derived grain param, shared by all actors with the same `total`.
        uid = f"{grain_name}_{total}"
        if uid not in graph._params:
            expr = _grain_expr(total, TARGET_PARAM, max_parallelism)
            graph._params[uid] = IRParam(name=grain_name, value=expr, unique_id=uid)
            param_deps.append((TARGET_PARAM, uid))
        actor.add_param(grain_name, graph._params[uid])

        print(f"  [Coarse] {actor.unique_name}: {param_name}={total} -> "
              f"{uid} = {graph._params[uid].value}")