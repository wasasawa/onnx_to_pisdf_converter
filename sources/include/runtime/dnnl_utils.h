#pragma once
#include <dnnl.hpp>
#include <omp.h>

// ---------------------------------------------------------------------------
// Global engine / stream — created once per process.
// All generated actors share these.
// ---------------------------------------------------------------------------
namespace rt {

inline dnnl::engine& cpu_engine() {
    static dnnl::engine eng = []() {
        omp_set_num_threads(1);
        return dnnl::engine(dnnl::engine::kind::cpu, 0);
    }();
    return eng;
}

inline dnnl::stream& cpu_stream() {
    thread_local static dnnl::stream stm(cpu_engine());
    return stm;
}

// ---------------------------------------------------------------------------
// Reorder helper.
// Used at actor init time to convert weights from plain nchw/oihw
// into the optimal blocked format that oneDNN chose for the primitive.
// The returned memory object owns the reordered buffer.
// ---------------------------------------------------------------------------
inline dnnl::memory reorder_to_optimal(
    const void*               src_data,
    const dnnl::memory::desc& src_md,   // plain descriptor, e.g. oihw
    const dnnl::memory::desc& dst_md)   // descriptor returned by pd.weights_desc()
{
    auto& eng = cpu_engine();
    auto& stm = cpu_stream();

    auto src_mem = dnnl::memory(src_md, eng, const_cast<void*>(src_data));
    auto dst_mem = dnnl::memory(dst_md, eng);   // allocates internal buffer

    dnnl::reorder(src_mem, dst_mem).execute(stm, src_mem, dst_mem);
    stm.wait();

    return dst_mem;   // caller stores this; it owns its buffer
}

// ---------------------------------------------------------------------------
// Wrap a raw pointer in a dnnl::memory without copying (zero-copy view).
// ---------------------------------------------------------------------------
inline dnnl::memory wrap(void* ptr, const dnnl::memory::desc& md) {
    return dnnl::memory(md, cpu_engine(), ptr);
}

} // namespace rt
