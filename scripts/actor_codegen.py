from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from structure import IRActor, OpType


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ActorCodegen(ABC):
    """Generates one generic C++ function for an operator type."""

    @abstractmethod
    def decl(self, actor: IRActor, params: dict) -> str:
        """Function declaration with semicolon (for the .h file)."""

    @abstractmethod
    def defn(self, actor: IRActor, params: dict, parallel: bool = False) -> str:
        """Full function definition (for the .cpp file)."""

    def loop_fn(self, actor: IRActor) -> str:
        """C function name PREESM calls — same for every actor of this type."""
        return self._base_name()

    def _base_name(self) -> str:
        return type(self).__name__.replace("Codegen", "").lower()

    # -- Block-skip variant -------------------------------------------------
    # Residual ADD actors are rewritten by block_skipping_pass.py to gate their
    # compute-path input by a `keep` cfg param.  When keep == 0 the compute FIFO
    # carries 0 tokens, so the kernel must not read it.  Codegens that support
    # this emit a second `<base>_skip` function alongside the normal one.
    # Default: no variant.

    def skip_loop_fn(self, actor: IRActor) -> str:
        return f"{self.loop_fn(actor)}_skip"

    def skip_decl(self, actor: IRActor, params: dict) -> Optional[str]:
        return None

    def skip_defn(self, actor: IRActor, params: dict) -> Optional[str]:
        return None

    @staticmethod
    def _skip_sig(actor: IRActor) -> str:
        """Like _full_sig but with a trailing `int keep` cfg param.

        block_skipping_pass.py appends the `keep` cfg port after the actor's
        existing size params, and cfg params are always emitted before data
        ports, so `keep` lands right after the size params in the loop param
        order — matched here.
        """
        parts = [f"int {port.name}" for port, _ in actor.params]
        parts.append("int keep")
        for port, tensor in actor.inputs + actor.weights:
            parts.append(f"{tensor.dtype}* {port.name}")
        for port, tensor in actor.outputs:
            parts.append(f"{tensor.dtype}* {port.name}")
        return ", ".join(parts)

    @staticmethod
    def _call_args(actor: IRActor) -> str:
        """Argument list to forward to the normal kernel (`_full_sig` minus keep)."""
        names = [port.name for port, _ in actor.params]
        for port, _ in actor.inputs + actor.weights:
            names.append(port.name)
        for port, _ in actor.outputs:
            names.append(port.name)
        return ", ".join(names)

    @staticmethod
    def _full_sig(actor: IRActor) -> str:
        """
        Complete C signature:
          int <param>, ...       cfg_input ports — passed by PREESM as ints
          <dtype>* <port>, ...   data / weight inputs
          <dtype>* <port>, ...   outputs
        """
        parts = [f"int {port.name}" for port, _ in actor.params]
        for port, tensor in actor.inputs + actor.weights:
            parts.append(f"{tensor.dtype}* {port.name}")
        for port, tensor in actor.outputs:
            parts.append(f"{tensor.dtype}* {port.name}")
        return ", ".join(parts)

    @staticmethod
    def _key_names(actor: IRActor) -> str:
        """Comma-separated param names used as the std::map cache key."""
        return ", ".join(port.name for port, _ in actor.params)

    @staticmethod
    def _cache_qualifier(parallel: bool) -> str:
        """Storage qualifier for the static cache map."""
        return "thread_local static" if parallel else "thread_local static"

    @staticmethod
    def _omp_preamble(parallel: bool) -> str:
        """omp_set_num_threads(1) is emitted once at static init in model_codegen.py,
        not inside each actor invocation.  Return empty string."""
        return ""

    @staticmethod
    def _grain_name(actor: IRActor) -> str:
        """Return 'grain_ch', 'grain', or '' depending on coarse-rate params.

        coarsen_graph() attaches one of these to applicable actors before
        code generation.  When present the generated parallel body uses the
        param as the per-firing element/channel count instead of the
        hard-coded scalar '1'.
        """
        for port, _ in actor.params:
            if port.name in ("grain_ch", "grain"):
                return port.name
        return ""


# ===========================================================================
# Convolution
# ===========================================================================

class Conv2DCodegen(ActorCodegen):
    """Conv2D without bias."""

    def _base_name(self): return "conv2d"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        kn  = self._key_names(actor)
        cq  = self._cache_qualifier(parallel)
        omp = self._omp_preamble(parallel)
        # Coarse parallel: grain_ch output channels per firing.
        # Fine parallel (no grain param): 1 output channel per firing.
        g     = self._grain_name(actor) if parallel else ""
        w_oc  = g or ("1" if parallel else "depthOutput")
        d_oc  = w_oc

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    // Cache key = shape only.  Spider2 DELAYED mode reallocates FIFOs each
    // iteration, so the weight pointer changes; we cache only the primitive
    // and redo the cheap reorder when the pointer changes.
    struct _Entry {{
        convolution_forward prim;
        memory              weights_mem;
        memory::desc        weights_desc_optimal;
        memory::desc        src_md, dst_md;          // cached — never changes
        bool                ready           = false;
        const void*         last_weight_ptr = nullptr;
    }};
    {cq} std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[std::vector<int>{{{kn}}}];

    if (!_e.ready) {{
        auto& eng = rt::cpu_engine();
        memory::dims src_dims = {{1, depthInput,  inputHeight,       inputWidth}};
        memory::dims w_dims   = {{{w_oc}, depthInput, sizeKernelHeight, sizeKernelWidth}};
        memory::dims dst_dims = {{1, {d_oc}, outputHeight,      outputWidth}};
        memory::dims strides  = {{strideHeight, strideWidth}};
        memory::dims dils     = {{dilationHeight - 1, dilationWidth - 1}};
        memory::dims pad_l    = {{padTop, padLeft}};
        memory::dims pad_r    = {{padBottom, padRight}};

        _e.src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        _e.dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
        auto w_any = memory::desc(w_dims,  memory::data_type::f32, memory::format_tag::any);

        auto pd = convolution_forward::primitive_desc(eng,
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            _e.src_md, w_any, _e.dst_md,
            strides, dils, pad_l, pad_r);

        _e.weights_desc_optimal = pd.weights_desc();
        auto w_plain            = memory::desc(w_dims, memory::data_type::f32, memory::format_tag::oihw);
        _e.weights_mem          = rt::reorder_to_optimal(input_1, w_plain, _e.weights_desc_optimal);
        _e.prim                 = convolution_forward(pd);
        _e.last_weight_ptr      = input_1;
        _e.ready                = true;
    }} else if (_e.last_weight_ptr != input_1) {{
        // Primitive reused — only redo the weight reorder (FIFO ptr changed).
        memory::dims w_dims = {{{w_oc}, depthInput, sizeKernelHeight, sizeKernelWidth}};
        auto w_plain        = memory::desc(w_dims, memory::data_type::f32, memory::format_tag::oihw);
        _e.weights_mem      = rt::reorder_to_optimal(input_1, w_plain, _e.weights_desc_optimal);
        _e.last_weight_ptr  = input_1;
    }}

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{
        {{DNNL_ARG_SRC,     rt::wrap(input_0,  _e.src_md)}},
        {{DNNL_ARG_WEIGHTS, _e.weights_mem}},
        {{DNNL_ARG_DST,     rt::wrap(output_0, _e.dst_md)}},
    }});
    stm.wait();
}}
"""


class Conv2DBiasCodegen(ActorCodegen):
    """Conv2D with bias fused."""

    def _base_name(self): return "conv2d_bias"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        kn  = self._key_names(actor)
        cq  = self._cache_qualifier(parallel)
        omp = self._omp_preamble(parallel)
        # Coarse parallel: grain_ch output channels per firing.
        # Fine parallel (no grain param): 1 output channel per firing.
        g    = self._grain_name(actor) if parallel else ""
        w_oc = g or ("1" if parallel else "depthOutput")
        d_oc = w_oc
        b_oc = w_oc

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    // Cache key = shape only.  Spider2 DELAYED mode reallocates FIFOs each
    // iteration, so the weight pointer changes; we cache only the primitive
    // and redo the cheap reorder when the pointer changes.
    struct _Entry {{
        convolution_forward prim;
        memory              weights_mem;
        memory::desc        weights_desc_optimal;
        memory::desc        src_md, bias_md, dst_md;  // cached — never changes
        bool                ready           = false;
        const void*         last_weight_ptr = nullptr;
    }};
    {cq} std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[std::vector<int>{{{kn}}}];

    if (!_e.ready) {{
        auto& eng = rt::cpu_engine();
        memory::dims src_dims  = {{1, depthInput,  inputHeight,       inputWidth}};
        memory::dims w_dims    = {{{w_oc}, depthInput, sizeKernelHeight, sizeKernelWidth}};
        memory::dims bias_dims = {{{b_oc}}};
        memory::dims dst_dims  = {{1, {d_oc}, outputHeight,      outputWidth}};
        memory::dims strides   = {{strideHeight, strideWidth}};
        memory::dims dils      = {{dilationHeight - 1, dilationWidth - 1}};
        memory::dims pad_l     = {{padTop, padLeft}};
        memory::dims pad_r     = {{padBottom, padRight}};

        _e.src_md  = memory::desc(src_dims,  memory::data_type::f32, memory::format_tag::nchw);
        _e.bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::a);
        _e.dst_md  = memory::desc(dst_dims,  memory::data_type::f32, memory::format_tag::nchw);
        auto w_any = memory::desc(w_dims,    memory::data_type::f32, memory::format_tag::any);

        auto pd = convolution_forward::primitive_desc(eng,
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            _e.src_md, w_any, _e.bias_md, _e.dst_md,
            strides, dils, pad_l, pad_r);

        _e.weights_desc_optimal = pd.weights_desc();
        auto w_plain            = memory::desc(w_dims, memory::data_type::f32, memory::format_tag::oihw);
        _e.weights_mem          = rt::reorder_to_optimal(input_1, w_plain, _e.weights_desc_optimal);
        _e.prim                 = convolution_forward(pd);
        _e.last_weight_ptr      = input_1;
        _e.ready                = true;
    }} else if (_e.last_weight_ptr != input_1) {{
        // Primitive reused — only redo the weight reorder (FIFO ptr changed).
        memory::dims w_dims = {{{w_oc}, depthInput, sizeKernelHeight, sizeKernelWidth}};
        auto w_plain        = memory::desc(w_dims, memory::data_type::f32, memory::format_tag::oihw);
        _e.weights_mem      = rt::reorder_to_optimal(input_1, w_plain, _e.weights_desc_optimal);
        _e.last_weight_ptr  = input_1;
    }}

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{
        {{DNNL_ARG_SRC,     rt::wrap(input_0,  _e.src_md)}},
        {{DNNL_ARG_WEIGHTS, _e.weights_mem}},
        {{DNNL_ARG_BIAS,    rt::wrap(input_2,  _e.bias_md)}},
        {{DNNL_ARG_DST,     rt::wrap(output_0, _e.dst_md)}},
    }});
    stm.wait();
}}
"""


# ===========================================================================
# Pooling
# ===========================================================================

class MaxPool2DCodegen(ActorCodegen):

    def _base_name(self): return "maxpool2d"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        kn  = self._key_names(actor)
        cq  = self._cache_qualifier(parallel)
        omp = self._omp_preamble(parallel)
        # Coarse parallel: grain_ch channels per firing. Fine: 1 channel.
        g  = self._grain_name(actor) if parallel else ""
        ch = g or ("1" if parallel else "depthInput")

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    struct _Entry {{ pooling_forward prim; memory::desc src_md, dst_md; bool ready = false; }};
    {cq} std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[{{{kn}}}];

    if (!_e.ready) {{
        _e.src_md = memory::desc({{1, {ch},  inputHeight,  inputWidth}},
                                 memory::data_type::f32, memory::format_tag::nchw);
        _e.dst_md = memory::desc({{1, {ch},  outputHeight, outputWidth}},
                                 memory::data_type::f32, memory::format_tag::nchw);
        auto pd = pooling_forward::primitive_desc(rt::cpu_engine(),
            prop_kind::forward_inference, algorithm::pooling_max,
            _e.src_md, _e.dst_md,
            {{strideHeight, strideWidth}}, {{poolHeight, poolWidth}},
            {{0, 0}},
            {{padTop, padLeft}}, {{padBottom, padRight}});
        _e.prim  = pooling_forward(pd);
        _e.ready = true;
    }}

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{{{DNNL_ARG_SRC, rt::wrap(input_0, _e.src_md)}},
                           {{DNNL_ARG_DST, rt::wrap(output_0, _e.dst_md)}}}});
    stm.wait();
}}
"""


class AvgPool2DCodegen(ActorCodegen):

    def _base_name(self): return "avgpool2d"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        kn  = self._key_names(actor)
        cq  = self._cache_qualifier(parallel)
        omp = self._omp_preamble(parallel)
        # Coarse parallel: grain_ch channels per firing. Fine: 1 channel.
        g  = self._grain_name(actor) if parallel else ""
        ch = g or ("1" if parallel else "depthInput")

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    struct _Entry {{ pooling_forward prim; memory::desc src_md, dst_md; bool ready = false; }};
    {cq} std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[{{{kn}}}];

    if (!_e.ready) {{
        auto alg = (countIncludePad != 0)
            ? algorithm::pooling_avg_include_padding
            : algorithm::pooling_avg_exclude_padding;
        _e.src_md = memory::desc({{1, {ch},  inputHeight,  inputWidth}},
                                 memory::data_type::f32, memory::format_tag::nchw);
        _e.dst_md = memory::desc({{1, {ch},  outputHeight, outputWidth}},
                                 memory::data_type::f32, memory::format_tag::nchw);
        auto pd = pooling_forward::primitive_desc(rt::cpu_engine(),
            prop_kind::forward_inference, alg,
            _e.src_md, _e.dst_md,
            {{strideHeight, strideWidth}}, {{poolHeight, poolWidth}},
            {{0, 0}},
            {{padTop, padLeft}}, {{padBottom, padRight}});
        _e.prim  = pooling_forward(pd);
        _e.ready = true;
    }}

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{{{DNNL_ARG_SRC, rt::wrap(input_0, _e.src_md)}},
                           {{DNNL_ARG_DST, rt::wrap(output_0, _e.dst_md)}}}});
    stm.wait();
}}
"""


class GlobalAvgPoolCodegen(ActorCodegen):
    """Global average pool: mean over spatialSize for each of depth channels."""

    def _base_name(self): return "global_avgpool"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                # Coarse: grain_ch channels per firing
                return f"""\
void {fn}({sig})
{{
    for (int c = 0; c < {g}; ++c) {{
        float sum = 0.0f;
        const float* src = input_0 + c * spatialSize;
        for (int i = 0; i < spatialSize; ++i) sum += src[i];
        output_0[c] = sum / static_cast<float>(spatialSize);
    }}
}}
"""
            # Fine: 1 channel per firing
            return f"""\
void {fn}({sig})
{{
    float sum = 0.0f;
    for (int i = 0; i < spatialSize; ++i) sum += input_0[i];
    output_0[0] = sum / static_cast<float>(spatialSize);
}}
"""

        return f"""\
void {fn}({sig})
{{
    for (int c = 0; c < depth; ++c) {{
        float sum = 0.0f;
        const float* ch = input_0 + c * spatialSize;
        for (int i = 0; i < spatialSize; ++i) sum += ch[i];
        output_0[c] = sum / static_cast<float>(spatialSize);
    }}
}}
"""


# ===========================================================================
# Element-wise activation (Relu, Sigmoid, Tanh, Dropout)
# ===========================================================================

class ReluCodegen(ActorCodegen):

    def _base_name(self): return "relu"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        kn  = self._key_names(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                return f"""\
void {fn}({sig})
{{
    for (int i = 0; i < {g}; ++i)
        output_0[i] = input_0[i] > 0.f ? input_0[i] : 0.f;
}}
"""
            # Fine: 1 element per firing
            return f"""\
void {fn}({sig})
{{
    output_0[0] = input_0[0] > 0.f ? input_0[0] : 0.f;
}}
"""

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    struct _Entry {{ eltwise_forward prim; memory::desc md; bool ready = false; }};
    thread_local static std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[{{{kn}}}];

    if (!_e.ready) {{
        _e.md    = memory::desc({{size}}, memory::data_type::f32, memory::format_tag::a);
        auto pd  = eltwise_forward::primitive_desc(rt::cpu_engine(),
            prop_kind::forward_inference,
            algorithm::eltwise_relu, _e.md, _e.md, 0.f, 0.f);
        _e.prim  = eltwise_forward(pd);
        _e.ready = true;
    }}

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{{{DNNL_ARG_SRC, rt::wrap(input_0, _e.md)}},
                           {{DNNL_ARG_DST, rt::wrap(output_0, _e.md)}}}});
    stm.wait();
}}
"""


class SigmoidCodegen(ActorCodegen):

    def _base_name(self): return "sigmoid"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        kn  = self._key_names(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                return f"""\
void {fn}({sig})
{{
    for (int i = 0; i < {g}; ++i)
        output_0[i] = 1.0f / (1.0f + std::exp(-input_0[i]));
}}
"""
            return f"""\
void {fn}({sig})
{{
    output_0[0] = 1.0f / (1.0f + std::exp(-input_0[0]));
}}
"""

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    struct _Entry {{ eltwise_forward prim; memory::desc md; bool ready = false; }};
    thread_local static std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[{{{kn}}}];

    if (!_e.ready) {{
        _e.md    = memory::desc({{size}}, memory::data_type::f32, memory::format_tag::a);
        auto pd  = eltwise_forward::primitive_desc(rt::cpu_engine(),
            prop_kind::forward_inference,
            algorithm::eltwise_logistic, _e.md, _e.md, 0.f, 0.f);
        _e.prim  = eltwise_forward(pd);
        _e.ready = true;
    }}

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{{{DNNL_ARG_SRC, rt::wrap(input_0, _e.md)}},
                           {{DNNL_ARG_DST, rt::wrap(output_0, _e.md)}}}});
    stm.wait();
}}
"""


class TanhCodegen(ActorCodegen):
    # NOTE: named "tanh_op" to avoid collision with <cmath>'s ::tanh

    def _base_name(self): return "tanh_op"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        kn  = self._key_names(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                return f"""\
void {fn}({sig})
{{
    for (int i = 0; i < {g}; ++i)
        output_0[i] = std::tanh(input_0[i]);
}}
"""
            return f"""\
void {fn}({sig})
{{
    output_0[0] = std::tanh(input_0[0]);
}}
"""

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    struct _Entry {{ eltwise_forward prim; memory::desc md; bool ready = false; }};
    thread_local static std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[{{{kn}}}];

    if (!_e.ready) {{
        _e.md    = memory::desc({{size}}, memory::data_type::f32, memory::format_tag::a);
        auto pd  = eltwise_forward::primitive_desc(rt::cpu_engine(),
            prop_kind::forward_inference,
            algorithm::eltwise_tanh, _e.md, _e.md, 0.f, 0.f);
        _e.prim  = eltwise_forward(pd);
        _e.ready = true;
    }}

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{{{DNNL_ARG_SRC, rt::wrap(input_0, _e.md)}},
                           {{DNNL_ARG_DST, rt::wrap(output_0, _e.md)}}}});
    stm.wait();
}}
"""


class DropoutCodegen(ActorCodegen):
    """Dropout is a passthrough at inference time."""

    def _base_name(self): return "dropout"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                return f"""\
void {fn}({sig})
{{
    std::memcpy(output_0, input_0, {g} * sizeof(float));
}}
"""
            return f"""\
void {fn}({sig})
{{
    output_0[0] = input_0[0];
}}
"""

        return f"""\
void {fn}({sig})
{{
    // Dropout is identity at inference time.
    std::memcpy(output_0, input_0, size * sizeof(float));
}}
"""


# ===========================================================================
# Add variants
# ===========================================================================

class AddSameCodegen(ActorCodegen):
    """Element-wise add of two tensors of equal size."""

    def _base_name(self): return "add_same"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        kn  = self._key_names(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                return f"""\
void {fn}({sig})
{{
    for (int i = 0; i < {g}; ++i)
        output_0[i] = input_0[i] + input_1[i];
}}
"""
            return f"""\
void {fn}({sig})
{{
    output_0[0] = input_0[0] + input_1[0];
}}
"""

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    struct _Entry {{ dnnl::binary prim; memory::desc md; bool ready = false; }};
    thread_local static std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[{{{kn}}}];

    if (!_e.ready) {{
        _e.md    = memory::desc({{size1}}, memory::data_type::f32, memory::format_tag::a);
        auto pd  = dnnl::binary::primitive_desc(rt::cpu_engine(),
            algorithm::binary_add, _e.md, _e.md, _e.md);
        _e.prim  = dnnl::binary(pd);
        _e.ready = true;
    }}

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{
        {{DNNL_ARG_SRC_0, rt::wrap(input_0, _e.md)}},
        {{DNNL_ARG_SRC_1, rt::wrap(input_1, _e.md)}},
        {{DNNL_ARG_DST,   rt::wrap(output_0, _e.md)}},
    }});
    stm.wait();
}}
"""

    def skip_decl(self, actor, params):
        return f"void {self.skip_loop_fn(actor)}({self._skip_sig(actor)});"

    def skip_defn(self, actor, params):
        fn   = self.skip_loop_fn(actor)
        sig  = self._skip_sig(actor)
        base = self.loop_fn(actor)
        args = self._call_args(actor)
        return f"""\
void {fn}({sig})
{{
    // Residual block-skip: input_0 = skip path, input_1 = gated compute path.
    if (keep) {{
        // Block kept: reuse the oneDNN-backed {base} kernel (skip + compute).
        {base}({args});
    }} else {{
        // Block dropped: the compute FIFO carries 0 tokens — forward skip only.
        std::memcpy(output_0, input_0, size1 * sizeof(float));
    }}
}}
"""


class AddBiasCodegen(ActorCodegen):
    """Add where input_1 (bias) is smaller and broadcast cyclically."""

    def _base_name(self): return "add_bias"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                # COARSE_RATE_EXPRESSIONS gives grain tokens of input_1 per firing,
                # so bias cycles within the grain slice — no modulo needed.
                return f"""\
void {fn}({sig})
{{
    for (int i = 0; i < {g}; ++i)
        output_0[i] = input_0[i] + input_1[i];
}}
"""
            return f"""\
void {fn}({sig})
{{
    output_0[0] = input_0[0] + input_1[0];
}}
"""

        return f"""\
void {fn}({sig})
{{
    // output[i] = input_0[i] + input_1[i % size2]  (cyclic broadcast)
    for (int i = 0; i < size1; ++i)
        output_0[i] = input_0[i] + input_1[i % size2];
}}
"""


class AddScalarCodegen(ActorCodegen):
    """Add where input_1 is a single scalar broadcast to all elements."""

    def _base_name(self): return "add_scalar"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                # input_1 rate is "1" in coarse rates — scalar broadcast.
                return f"""\
void {fn}({sig})
{{
    const float s = input_1[0];
    for (int i = 0; i < {g}; ++i)
        output_0[i] = input_0[i] + s;
}}
"""
            return f"""\
void {fn}({sig})
{{
    output_0[0] = input_0[0] + input_1[0];
}}
"""

        return f"""\
void {fn}({sig})
{{
    const float s = input_1[0];
    for (int i = 0; i < size1; ++i)
        output_0[i] = input_0[i] + s;
}}
"""


class AddGenericCodegen(ActorCodegen):
    """Runtime-dispatch add: same-size, scalar, or cyclic broadcast."""

    def _base_name(self): return "add_generic"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                # Both inputs have grain tokens in coarse mode.
                return f"""\
void {fn}({sig})
{{
    for (int i = 0; i < {g}; ++i)
        output_0[i] = input_0[i] + input_1[i];
}}
"""
            return f"""\
void {fn}({sig})
{{
    output_0[0] = input_0[0] + input_1[0];
}}
"""

        return f"""\
void {fn}({sig})
{{
    if (size1 == size2) {{
        for (int i = 0; i < size1; ++i) output_0[i] = input_0[i] + input_1[i];
    }} else if (size2 == 1) {{
        const float s = input_1[0];
        for (int i = 0; i < size1; ++i) output_0[i] = input_0[i] + s;
    }} else {{
        for (int i = 0; i < size1; ++i) output_0[i] = input_0[i] + input_1[i % size2];
    }}
}}
"""

    def skip_decl(self, actor, params):
        return f"void {self.skip_loop_fn(actor)}({self._skip_sig(actor)});"

    def skip_defn(self, actor, params):
        fn   = self.skip_loop_fn(actor)
        sig  = self._skip_sig(actor)
        base = self.loop_fn(actor)
        args = self._call_args(actor)
        return f"""\
void {fn}({sig})
{{
    // Residual block-skip: input_0 = skip path, input_1 = gated compute path.
    if (keep) {{
        // Block kept: reuse the {base} kernel (residual operands are equal-size).
        {base}({args});
    }} else {{
        // Block dropped: the compute FIFO carries 0 tokens — forward skip only.
        std::memcpy(output_0, input_0, size1 * sizeof(float));
    }}
}}
"""


# ===========================================================================
# Linear / matrix ops
# ===========================================================================

class MatMulCodegen(ActorCodegen):
    """Matrix multiplication A[M,K] x B[K,N] → C[M,N] via CBLAS."""

    def _base_name(self): return "matmul"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            g = self._grain_name(actor)
            rows = g if g else "1"
            comment = f"C[{rows} x N] = A[{rows} x K] * B[K x N]"
            return f"""\
void {fn}({sig})
{{
    // {comment}
    rt::sgemm(false, false, {rows}, N, K,
              1.0f, input_0, K,
                    input_1, N,
              0.0f, output_0, N);
}}
"""

        return f"""\
void {fn}({sig})
{{
    // C[M x N] = A[M x K] * B[K x N]
    rt::sgemm(false, false, M, N, K,
              1.0f, input_0, K,
                    input_1, N,
              0.0f, output_0, N);
}}
"""


class GemmCodegen(ActorCodegen):
    """ONNX Gemm: Y = alpha * op(A) * op(B) + beta * C via CBLAS."""

    def _base_name(self): return "gemm"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            g    = self._grain_name(actor)
            rows = g if g else "1"
            # Tile bias across grain rows so sgemm beta-accumulates correctly.
            bias_init = (
                f"for (int r = 0; r < {rows}; ++r)\n"
                f"        std::memcpy(output_0 + r * N, input_2, sizeC * sizeof(float));"
                if g else
                "std::memcpy(output_0, input_2, sizeC * sizeof(float));"
            )
            return f"""\
void {fn}({sig})
{{
    // Y[{rows} x N] = alpha * A[{rows} x K] * B[K x N] + beta * C
    {bias_init}
    rt::sgemm(
        transA != 0, transB != 0,
        {rows}, N, K,
        static_cast<float>(alpha),
        input_0, transA ? {rows} : K,
        input_1, transB ? K : N,
        static_cast<float>(beta),
        output_0, N);
}}
"""

        return f"""\
void {fn}({sig})
{{
    // Y = alpha * op(A)[M x K] * op(B)[K x N] + beta * C[sizeC]
    // alpha and beta are passed as int (always 1 in standard ONNX models).
    std::memcpy(output_0, input_2, sizeC * sizeof(float));
    rt::sgemm(
        transA != 0, transB != 0,
        M, N, K,
        static_cast<float>(alpha),
        input_0, transA ? M : K,
        input_1, transB ? K : N,
        static_cast<float>(beta),
        output_0, N);
}}
"""


class SoftmaxCodegen(ActorCodegen):

    def _base_name(self): return "softmax"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        kn  = self._key_names(actor)
        # Softmax needs the full vector — fires once in hierarchical mode.
        # Just add thread safety.
        cq  = self._cache_qualifier(parallel)
        omp = self._omp_preamble(parallel)

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    struct _Entry {{ softmax_forward prim; memory::desc md; bool ready = false; }};
    {cq} std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[{{{kn}}}];

    if (!_e.ready) {{
        _e.md    = memory::desc({{outerSize, size / outerSize}},
                               memory::data_type::f32, memory::format_tag::ab);
        auto pd  = softmax_forward::primitive_desc(rt::cpu_engine(),
            prop_kind::forward_inference, algorithm::softmax_accurate,
            _e.md, _e.md, /*axis=*/1);
        _e.prim  = softmax_forward(pd);
        _e.ready = true;
    }}

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{{{DNNL_ARG_SRC, rt::wrap(input_0, _e.md)}},
                           {{DNNL_ARG_DST, rt::wrap(output_0, _e.md)}}}});
    stm.wait();
}}
"""


# ===========================================================================
# Shape / memory ops
# ===========================================================================

class ReshapeCodegen(ActorCodegen):
    """Reshape is a shape-only operation — data is memcopied unchanged."""

    def _base_name(self): return "reshape"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                return f"""\
void {fn}({sig})
{{
    // input_1 carries shape tensor — ignored at inference.
    std::memcpy(output_0, input_0, {g} * sizeof(float));
}}
"""
            return f"""\
void {fn}({sig})
{{
    output_0[0] = input_0[0];
}}
"""

        return f"""\
void {fn}({sig})
{{
    // input_1 carries the target shape tensor (int64_t) — ignored at inference.
    std::memcpy(output_0, input_0, outputSize * sizeof(float));
}}
"""


class FlattenCodegen(ActorCodegen):
    """Flatten is a contiguous reshape — pure memcpy."""

    def _base_name(self): return "flatten"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            g = self._grain_name(actor)
            if g:
                return f"""\
void {fn}({sig})
{{
    std::memcpy(output_0, input_0, {g} * sizeof(float));
}}
"""
            return f"""\
void {fn}({sig})
{{
    output_0[0] = input_0[0];
}}
"""

        return f"""\
void {fn}({sig})
{{
    std::memcpy(output_0, input_0, outputSize * sizeof(float));
}}
"""


class ConcatCodegen(ActorCodegen):
    """
    Concatenation along the channel / last axis.
    Generates one std::memcpy per input, so the body adapts to the
    actual number of inputs of the representative actor.
    """

    def _base_name(self): return "concat"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)

        if parallel:
            # Each firing: 1 element from each input → output
            all_inputs = actor.inputs + actor.weights
            copies = []
            for i, (port, _) in enumerate(all_inputs):
                copies.append(f"    output_0[{i}] = {port.name}[0];")
            body = "\n".join(copies)
            return f"""\
void {fn}({sig})
{{
{body}
}}
"""

        all_inputs = actor.inputs + actor.weights
        copies = []
        offset = "0"
        for i, (port, _) in enumerate(all_inputs):
            size_var = f"size{i + 1}"
            dst = f"output_0 + {offset}" if offset != "0" else "output_0"
            copies.append(f"    std::memcpy({dst}, {port.name}, {size_var} * sizeof(float));")
            offset = f"{offset} + {size_var}" if offset != "0" else size_var

        body = "\n".join(copies)
        return f"""\
void {fn}({sig})
{{
{body}
}}
"""


# ===========================================================================
# Normalisation
# ===========================================================================

class BatchNormCodegen(ActorCodegen):
    """Batch normalisation, inference mode (uses precomputed mean/variance)."""

    def _base_name(self): return "batchnorm_spatial"

    def decl(self, actor, params):
        return f"void {self.loop_fn(actor)}({self._full_sig(actor)});"

    def defn(self, actor, params, parallel=False):
        fn  = self.loop_fn(actor)
        sig = self._full_sig(actor)
        # BatchNorm fires once in hierarchical mode (no rate expression).
        # Just add thread safety.
        cq  = self._cache_qualifier(parallel)
        omp = self._omp_preamble(parallel)
        # epsilon is stored as a mangled string in IRParam (float→int precision
        # loss), so we read it directly from the ONNX attribute and bake it in.
        # The 'epsilon' cfg_input port is kept in the signature to satisfy
        # PREESM's port matching; its runtime value is ignored.
        shape_params = ", ".join(
            port.name for port, _ in actor.params if port.name != "epsilon"
        )

        return f"""\
void {fn}({sig})
{{
    using namespace dnnl;
    // 'epsilon' is accepted as int to match the cfg_input port but its value
    // is unusable (float→int truncation). A fixed standard value is used.
    static constexpr float kEps = 1e-5f;
    struct _Entry {{
        batch_normalization_forward prim;
        memory                      scale_shift_mem;
        memory::desc                md, md1;         // cached — never changes
        bool                        ready = false;
    }};
    {cq} std::map<std::vector<int>, _Entry> _cache;
    auto& _e = _cache[{{{shape_params}}}];

    if (!_e.ready) {{
        auto& eng = rt::cpu_engine();
        _e.md  = memory::desc({{1, channels, spatialSize}},
                              memory::data_type::f32, memory::format_tag::abc);
        _e.md1 = memory::desc({{channels}},
                              memory::data_type::f32, memory::format_tag::a);
        auto pd = batch_normalization_forward::primitive_desc(eng,
            prop_kind::forward_inference, _e.md, _e.md, kEps,
            normalization_flags::use_scale |
            normalization_flags::use_shift |
            normalization_flags::use_global_stats);
        _e.prim            = batch_normalization_forward(pd);
        _e.scale_shift_mem = memory(pd.weights_desc(), eng);
        _e.ready = true;
    }}

    // oneDNN v2: scale and shift interleaved [gamma_0..gamma_C, beta_0..beta_C]
    float* ss = static_cast<float*>(_e.scale_shift_mem.get_data_handle());
    std::memcpy(ss,             input_1, channels * sizeof(float));
    std::memcpy(ss + channels,  input_2, channels * sizeof(float));

    auto& stm = rt::cpu_stream();
    _e.prim.execute(stm, {{
        {{DNNL_ARG_SRC,         rt::wrap(input_0, _e.md)}},
        {{DNNL_ARG_MEAN,        rt::wrap(input_3, _e.md1)}},
        {{DNNL_ARG_VARIANCE,    rt::wrap(input_4, _e.md1)}},
        {{DNNL_ARG_SCALE_SHIFT, _e.scale_shift_mem}},
        {{DNNL_ARG_DST,         rt::wrap(output_0, _e.md)}},
    }});
    stm.wait();
}}
"""


# ===========================================================================
# Operator Registry
# ===========================================================================

CODEGEN_REGISTRY: dict[OpType, ActorCodegen] = {
    # Convolution
    OpType.CONV2D:        Conv2DCodegen(),
    OpType.CONV2D_BIAS:   Conv2DBiasCodegen(),
    # Pooling
    OpType.MAXPOOL2D:     MaxPool2DCodegen(),
    OpType.AVGPOOL2D:     AvgPool2DCodegen(),
    OpType.GLOBAL_AVGPOOL: GlobalAvgPoolCodegen(),
    # Activations
    OpType.RELU:          ReluCodegen(),
    OpType.SIGMOID:       SigmoidCodegen(),
    OpType.TANH:          TanhCodegen(),
    OpType.DROPOUT:       DropoutCodegen(),
    # Add variants
    OpType.ADD_SAME:      AddSameCodegen(),
    OpType.ADD_BIAS:      AddBiasCodegen(),
    OpType.ADD_SCALAR:    AddScalarCodegen(),
    OpType.ADD_GENERIC:   AddGenericCodegen(),
    # Linear
    OpType.MATMUL:        MatMulCodegen(),
    OpType.GEMM:          GemmCodegen(),
    OpType.SOFTMAX:       SoftmaxCodegen(),
    # Shape / memory
    OpType.RESHAPE:       ReshapeCodegen(),
    OpType.FLATTEN:       FlattenCodegen(),
    OpType.CONCAT:        ConcatCodegen(),
    # Normalisation
    OpType.BATCHNORM:     BatchNormCodegen(),
    # TRANSPOSE, SLICE, PAD have variable param counts (rank/ndim-dependent)
    # and cannot be expressed as a single generic signature — they fall back
    # to the handwritten kernels via OPTYPE_TO_LOOP_FN.
}


def get_codegen(op_type: OpType) -> Optional[ActorCodegen]:
    return CODEGEN_REGISTRY.get(op_type)
