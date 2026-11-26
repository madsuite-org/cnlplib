"""
CUDA for libraries that export the cnlp device tier (`P_device`, `P_*_d`).

`CNLPModel(lib, args...; backend = CUDABackend())` returns a `CNLPModel{CuVector{Float64}}`: the
same instance, with its metadata and buffers on the device, evaluated through the device
callbacks on device pointers. This extension only supplies what differs from the host:
the pointer conversion, the stream handoff, and the product kernels.
"""
module CNLPLibCUDAExt

using CNLPLib, NLPModels, CUDACore
using CUDACore: CuArray, CuVector, StridedCuVector, CUDABackend
using CNLPLib: CNLPModel
import Libdl

const CuModel = CNLPModel{<:CuVector{Float64}}

# A contiguous CuArray (or contiguous view of one) as the raw device pointer the ABI takes.
CNLPLib._ptr(a::StridedCuVector{Float64}) = reinterpret(Ptr{Cdouble}, UInt(pointer(a)))

# The library launches on the legacy default stream and synchronizes its context before
# returning; CUDA.jl works on task-local streams. Finishing this task's stream first is
# what orders the library's reads after our writes.
CNLPLib._sync(::CuModel) = CUDACore.synchronize()

function CNLPLib._device_model(::CUDABackend, m::CNLPModel{Vector{Float64}})
    sym(s) = Symbol(m.prefix, "_", s)
    dp = Libdl.dlsym(m.lib.handle, sym(:device); throw_error = false)
    dp === nothing && throw(ArgumentError(
        "`$(m.prefix)` has no device tier: the library does not export `$(sym(:device))` " *
        "(compile it with `compile_library(...; device = :cuda)`)"))
    space = ccall(dp, Int64, (Int64,), m.id)
    space == 1 || throw(ArgumentError("`$(sym(:device))` reports memory space $space, not CUDA (1)"))
    hm = m.meta
    meta = NLPModelMeta(
        hm.nvar; ncon = hm.ncon, nnzj = hm.nnzj, nnzh = hm.nnzh,
        x0 = CuVector(hm.x0), lvar = CuVector(hm.lvar), uvar = CuVector(hm.uvar),
        y0 = CuVector(hm.y0), lcon = CuVector(hm.lcon), ucon = CuVector(hm.ucon),
        minimize = hm.minimize, name = hm.name,
    )
    fp(s) = Libdl.dlsym(m.lib.handle, sym(s))
    VT = typeof(meta.x0)
    return CNLPModel(
        meta, Counters(), m.lib, m.id,
        fp(:obj_d), fp(:grad_d), fp(:cons_d), fp(:jac_d), fp(:hess_d),
        m.jrows, m.jcols, m.hrows, m.hcols,
        CuVector(m.jrows), CuVector(m.jcols), CuVector(m.hrows), CuVector(m.hcols),
        VT(undef, hm.nnzj), VT(undef, hm.nnzh), CUDACore.zeros(Float64, hm.ncon),
        m.prefix, m.vars, m.cons, m.pars,
    )
end

# ── the precompilation model on the device (src/precompile.jl), for CNLPLibcuMadNLPExt ──
# Its callbacks take device pointers, as a library's device tier does; the toy copies to the
# host and back, which is all precompilation needs.
_dw(p, n) = unsafe_wrap(CuArray, reinterpret(CUDACore.CuPtr{Float64}, UInt(p)), n)
_put(p, v) = (copyto!(_dw(p, length(v)), v); Int64(0))
_d_obj(::Int64, x::Ptr{Cdouble}, f::Ptr{Cdouble}) =
    (v = Array(_dw(x, 2)); GC.@preserve v CNLPLib._toy_obj(Int64(1), pointer(v), f))
_d_grad(::Int64, x::Ptr{Cdouble}, g::Ptr{Cdouble}) = (v = Array(_dw(x, 2)); _put(g, [2(v[1] - 1), 2(v[2] - 2)]))
_d_cons(::Int64, x::Ptr{Cdouble}, c::Ptr{Cdouble}) = _put(c, [sum(Array(_dw(x, 2)))])
_d_jac(::Int64, ::Ptr{Cdouble}, v::Ptr{Cdouble}) = _put(v, [1.0, 1.0])
_d_hess(::Int64, ::Ptr{Cdouble}, ::Ptr{Cdouble}, w::Cdouble, v::Ptr{Cdouble}) = _put(v, [2w, 2w])

function _toy_device_model(; fixed::Bool = false)
    m = CNLPLib._toy_model(; fixed)
    hm = m.meta
    meta = NLPModelMeta(
        hm.nvar; ncon = hm.ncon, nnzj = hm.nnzj, nnzh = hm.nnzh,
        x0 = CuVector(hm.x0), lvar = CuVector(hm.lvar), uvar = CuVector(hm.uvar),
        y0 = CuVector(hm.y0), lcon = CuVector(hm.lcon), ucon = CuVector(hm.ucon), name = hm.name,
    )
    return CNLPModel(
        meta, Counters(), m.lib, m.id,
        @cfunction(_d_obj, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_d_grad, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_d_cons, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_d_jac, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_d_hess, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Ptr{Cdouble})),
        m.jrows, m.jcols, m.hrows, m.hcols,
        CuVector(m.jrows), CuVector(m.jcols), CuVector(m.hrows), CuVector(m.hcols),
        CUDACore.zeros(Float64, hm.nnzj), CUDACore.zeros(Float64, hm.nnzh), CUDACore.zeros(Float64, hm.ncon),
        m.prefix, m.vars, m.cons, m.pars,
    )
end

# ── the solver side: a problem's device callbacks as a ProblemModel{CuVector} ────────

const CuProblem = CNLPLib.ProblemModel{<:CuVector{Float64}}
CNLPLib._psync(::CuProblem) = CUDACore.synchronize()

function CNLPLib._device_problem_model(::Val{:cuda}, m::CNLPLib.ProblemModel{Vector{Float64}},
                                       d::CNLPLib.CNLPDeviceProblem)
    hm = m.meta
    meta = NLPModelMeta(
        hm.nvar; ncon = hm.ncon, nnzj = hm.nnzj, nnzh = hm.nnzh,
        x0 = CuVector(hm.x0), lvar = CuVector(hm.lvar), uvar = CuVector(hm.uvar),
        y0 = CuVector(hm.y0), lcon = CuVector(hm.lcon), ucon = CuVector(hm.ucon),
        minimize = hm.minimize,
    )
    return CNLPLib.ProblemModel(
        meta, Counters(), m.p, d.obj_d, d.grad_d, d.cons_d, d.jac_d, d.hess_d,
        m.jrows, m.jcols, m.hrows, m.hcols,
        CuVector(m.jrows), CuVector(m.jcols), CuVector(m.hrows), CuVector(m.hcols),
        CUDACore.zeros(Float64, hm.nnzj), CUDACore.zeros(Float64, hm.nnzh), CUDACore.zeros(Float64, hm.ncon),
    )
end

# ── the products' COO matvecs, on the device ──────────────────────────────────────────

function _coo_kernel!(y, rows, cols, vals, x, n)
    k = (CUDACore.blockIdx().x - Int32(1)) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if k <= n
        @inbounds CUDACore.@atomic y[rows[k]] += vals[k] * x[cols[k]]
    end
    return
end
# the Hessian is stored as its lower triangle; off-diagonal entries act both ways
function _sym_kernel!(y, rows, cols, vals, x, n)
    k = (CUDACore.blockIdx().x - Int32(1)) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if k <= n
        @inbounds begin
            i, j, h = rows[k], cols[k], vals[k]
            CUDACore.@atomic y[i] += h * x[j]
            i == j || (CUDACore.@atomic y[j] += h * x[i])
        end
    end
    return
end
function _launch!(f, y, rows, cols, vals, x)
    fill!(y, 0.0)
    n = length(vals)
    n == 0 && return y
    CUDACore.@cuda threads = 256 blocks = cld(n, 256) f(y, rows, cols, vals, x, n)
    return y
end

CNLPLib._coo_mul!(y::StridedCuVector{Float64}, rows::CuVector{Int}, cols::CuVector{Int},
                     vals::CuVector{Float64}, v::StridedCuVector{Float64}) =
    _launch!(_coo_kernel!, y, rows, cols, vals, v)
CNLPLib._sym_mul!(y::StridedCuVector{Float64}, rows::CuVector{Int}, cols::CuVector{Int},
                     vals::CuVector{Float64}, v::StridedCuVector{Float64}) =
    _launch!(_sym_kernel!, y, rows, cols, vals, v)

end # module
