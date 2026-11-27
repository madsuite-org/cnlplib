# Precompilation. A model from a library evaluates through C callbacks; a toy model whose
# callbacks are Julia functions (`@cfunction`) runs the same code with no library and no C
# compiler, so the workloads here and in the solver extensions (MadNLP, NLPModelsIpopt,
# cuMadNLP) compile what a first load, evaluation and solve would.
#
#   min (x1 - 1)^2 + (x2 - 2)^2   s.t.   x1 + x2 = 1,   0 <= x <= 5
#
# and, with `fixed = true`, the same with x2 fixed at 0.5 and 1 <= x1 + x2 <= 2: MadNLP builds
# a different solver for a model with fixed variables, so both are compiled.

_w(p, n) = unsafe_wrap(Array, p, n)
_toy_obj(::Int64, x::Ptr{Cdouble}, f::Ptr{Cdouble}) =
    (v = _w(x, 2); unsafe_store!(f, (v[1] - 1)^2 + (v[2] - 2)^2); Int64(0))
_toy_grad(::Int64, x::Ptr{Cdouble}, g::Ptr{Cdouble}) =
    (v = _w(x, 2); w = _w(g, 2); w[1] = 2(v[1] - 1); w[2] = 2(v[2] - 2); Int64(0))
_toy_cons(::Int64, x::Ptr{Cdouble}, c::Ptr{Cdouble}) = (v = _w(x, 2); unsafe_store!(c, v[1] + v[2]); Int64(0))
_toy_jac(::Int64, ::Ptr{Cdouble}, v::Ptr{Cdouble}) = (fill!(_w(v, 2), 1.0); Int64(0))
_toy_hess(::Int64, ::Ptr{Cdouble}, ::Ptr{Cdouble}, w::Cdouble, v::Ptr{Cdouble}) = (fill!(_w(v, 2), 2w); Int64(0))

"""
    _toy_model(; fixed = false) -> CNLPModel{Vector{Float64}}

The precompilation model: a `CNLPModel` over Julia callbacks, valid in this process only.
"""
_toy_bounds(fixed) = fixed ? ([0.0, 0.5], [5.0, 0.5], [1.0], [2.0]) : (zeros(2), fill(5.0, 2), [1.0], [1.0])

function _toy_model(; fixed::Bool = false)
    lv, uv, lc, uc = _toy_bounds(fixed)
    meta = NLPModelMeta(2; ncon = 1, nnzj = 2, nnzh = 2, x0 = [0.5, 0.5], lvar = lv,
                        uvar = uv, lcon = lc, ucon = uc, name = "toy")
    return CNLPModel(
        meta, Counters(), CLib("", C_NULL, String[]), Int64(1),
        @cfunction(_toy_obj, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_toy_grad, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_toy_cons, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_toy_jac, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_toy_hess, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Ptr{Cdouble})),
        [1, 1], [1, 2], [1, 2], [1, 2], [1, 1], [1, 2], [1, 2], [1, 2],
        zeros(2), zeros(2), zeros(1), "toy", (;), (;), (;),
    )
end

# The same problem as a solver receives it (`cnlp_problem`: callbacks taking a context), read
# as a `ProblemModel`.
_tp_obj(::Ptr{Cvoid}, x::Ptr{Cdouble}, f::Ptr{Cdouble}) = _toy_obj(Int64(1), x, f)
_tp_grad(::Ptr{Cvoid}, x::Ptr{Cdouble}, g::Ptr{Cdouble}) = _toy_grad(Int64(1), x, g)
_tp_cons(::Ptr{Cvoid}, x::Ptr{Cdouble}, c::Ptr{Cdouble}) = _toy_cons(Int64(1), x, c)
_tp_js(::Ptr{Cvoid}, r::Ptr{Int64}, c::Ptr{Int64}) = (_w(r, 2) .= (1, 1); _w(c, 2) .= (1, 2); Int64(0))
_tp_jac(::Ptr{Cvoid}, x::Ptr{Cdouble}, v::Ptr{Cdouble}) = _toy_jac(Int64(1), x, v)
_tp_hs(::Ptr{Cvoid}, r::Ptr{Int64}, c::Ptr{Int64}) = (_w(r, 2) .= (1, 2); _w(c, 2) .= (1, 2); Int64(0))
_tp_hess(::Ptr{Cvoid}, x::Ptr{Cdouble}, y::Ptr{Cdouble}, w::Cdouble, v::Ptr{Cdouble}) = _toy_hess(Int64(1), x, y, w, v)
const _TOY_DATA = Dict(f => ([0.5, 0.5], _toy_bounds(f)...) for f in (false, true))

"""
    _toy_problem(; fixed = false) -> ProblemModel{Vector{Float64}}

The precompilation model as a cnlp solver receives it.
"""
function _toy_problem(; fixed::Bool = false)
    x0, lv, uv, lc, uc = _TOY_DATA[fixed]
    p = CNLPProblem(C_NULL, 2, 1, 2, 2, 1, pointer(x0), pointer(lv), pointer(uv), pointer(lc), pointer(uc),
        C_NULL, C_NULL, C_NULL,
        @cfunction(_tp_obj, Int64, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_tp_grad, Int64, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_tp_cons, Int64, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_tp_js, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
        @cfunction(_tp_jac, Int64, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble})),
        @cfunction(_tp_hs, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
        @cfunction(_tp_hess, Int64, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Ptr{Cdouble})))
    return ProblemModel(p)
end

# the model API, evaluated once on every entry point
function _exercise(m)
    x = copy(m.meta.x0)
    y = zeros(m.meta.ncon)
    v = ones(m.meta.nvar)
    NLPModels.obj(m, x); NLPModels.grad(m, x); NLPModels.cons(m, x)
    NLPModels.jac_structure(m); NLPModels.hess_structure(m)
    NLPModels.jac_coord(m, x); NLPModels.hess_coord(m, x, y); NLPModels.hess_coord(m, x)
    NLPModels.hess_coord(m, x, y; obj_weight = 0.5)
    NLPModels.jprod(m, x, v); NLPModels.jtprod(m, x, y); NLPModels.hprod(m, x, y, v)
    return nothing
end

@setup_workload begin
    @compile_workload begin
        _exercise(_toy_model())
        _exercise(_toy_problem())
    end
end
