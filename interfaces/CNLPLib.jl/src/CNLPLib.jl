"""
    CNLPLib

Consume an NLP exposed by a shared library through a plain C interface as an
`NLPModels.AbstractNLPModel`, so any NLPModels-compatible solver (MadNLP,
Ipopt, ...) can solve it. The main producer of such libraries is ExaModels'
tape recorder (`compile_library`), but any
library implementing the ABI below works, in any language.

Implements cnlp ABI 0.2 (normative spec: `cnlp.h` at
https://cnlplib.madsuite.org).

A model is a `CNLPModel{VT}`: `VT = Vector{Float64}` evaluates through the library's
host callbacks; with `device = :cuda` (CUDA extension) `VT` is a `CuVector` and the
same instance evaluates through the device tier (`P_*_d`) on device pointers. Any
number of instances may coexist per library, and a library may carry several models,
one prefix each: `CNLPModel("@grid", :acopf, ...)`.

# The libblastrampoline seam

A `juliac`-compiled library carries its own (privatized) Julia runtime, but
libblastrampoline is resolved once per process by soname, so it is *shared*
with the host. The library's runtime initializes lazily on its first call and
re-forwards the shared trampoline to its own BLAS, silently clearing the
host's forwarding (symptom: `Error: no BLAS/LAPACK library loaded for
dgemm_()` and solvers degrading to garbage). [`load`](@ref) snapshots the
host's BLAS configuration before `dlopen`, and [`CNLPModel`](@ref) restores it
after the library's first call; [`restore_blas!`](@ref) can be called again at
any time.

# Usage

```julia
using CNLPLib, MadNLP
lib = CNLPLib.load("librecorder.so")
m = CNLPModel(lib, 1000; prefix = "rec")   # rec_new(1000) → id, fixes BLAS
res = madnlp(m)
```

Instantiation arguments are positional, one per schema field — the same
spelling `ExaModel(core, arg1, arg2, ...)` uses on the producer side.
"""
module CNLPLib

using Libdl
using LinearAlgebra
import NLPModels: NLPModels, AbstractNLPModel, NLPModelMeta, Counters,
    increment!, @lencheck

export CNLPModel, argtype, available_models, restore_blas!, schema_json, set_path!, @cnlp_str,
    get_vars, get_cons, get_pars, get_value, set_value!,
    solution, multipliers, multipliers_L, multipliers_U, writers, write_solution, default_writer,
    readers, default_reader, read_start, read_parameters!

include("patchversion.jl")
include("registry.jl")
include("private_runtime.jl")

"""
    CLib

Handle to a loaded NLP shared library: the `dlopen` handle plus the snapshot
of the host's BLAS forwarding taken *before* the library was loaded. Create
with [`load`](@ref).
"""
struct CLib
    path::String
    handle::Ptr{Cvoid}
    blas_libs::Vector{String}
end

# ── library path registry ────────────────────────────────────────────────────

const _PATHS = String[]
const _LIBS = Dict{String, CLib}()

"""
    set_path!(dirs...)

Set the library search path used by name-based loading (`CNLPLib.lib("acopf")`,
`CNLPModel("@acopf", ...)`, `cnlp"acopf"`). Initialized from the colon-separated
`CNLPLIB_PATH` environment variable; calling `set_path!` replaces it.
"""
function set_path!(dirs::AbstractString...)
    empty!(_PATHS)
    append!(_PATHS, String.(dirs))
    return copy(_PATHS)
end

function _paths()
    if isempty(_PATHS)
        # CNLPMODELS_PATH: the name before the package was renamed
        env = get(ENV, "CNLPLIB_PATH", get(ENV, "CNLPMODELS_PATH", ""))
        isempty(env) || append!(_PATHS, split(env, ":"))
    end
    return _PATHS
end

"""
    lib(name::AbstractString) -> CLib

Resolve `lib<name>.<dlext>` against the search path ([`set_path!`](@ref) /
`CNLPLIB_PATH`) — also accepting `<dir>/lib/lib<name>.<dlext>`, the layout
`compile_library` produces — then among the libraries the `cnlp` command installed
(`\$CNLPLIB_HOME/lib`, by default `~/.cnlplib/lib`), and otherwise in the cnlplib registry,
installed there on first use ([`install`](@ref)); load it, and cache the handle by name.
`name@nightly` takes the registry's nightly build.
"""
function lib(name::AbstractString)
    get!(_LIBS, String(name)) do
        base = first(split(name, '@'))   # `emp@nightly`: the registry's channel
        fname = "lib" * base * "." * Libdl.dlext
        if base == name
            for d in _paths(), cand in (joinpath(d, fname), joinpath(d, name, "lib", fname), joinpath(d, "lib", fname))
                isfile(cand) && return load(cand)
            end
        end
        # installed in $CNLPLIB_HOME (by the `cnlp` command, or by this package), else
        # installed there now from the cnlplib registry
        return load(_registry_lib(name))
    end
end

"""
    cnlp"name"

The shortcut onto the library path: resolve `lib<name>.<dlext>` against
[`set_path!`](@ref) / `CNLPLIB_PATH`, check that it exists and loads as a
shared library, and return the (cached) handle — usable directly as the
library argument of [`CNLPModel`](@ref):

    m = CNLPModel(cnlp"acopf", bus, vmin, 100.0)

Resolution and validation happen at run time (never at parse time), so the
literal is safe inside precompiled code.
"""
macro cnlp_str(name)
    return :(lib($name))
end

"""
    load(path::AbstractString) -> CLib

Snapshot the host's BLAS configuration, then `dlopen` the library with
`RTLD_LOCAL | RTLD_DEEPBIND` (so the library prefers its own bundled
dependencies and leaks no symbols into the host).

An **unbundled** juliac library — one linked against the standard
`libjulia` rather than carrying a privatized copy — cannot be loaded
into a Julia process as-is: sharing the host's runtime, its first call
aborts the whole process. On Linux and macOS, `load` detects this and
gives the library a private copy of the *installed* runtime instead
(patched in scratch, one per library file, cached), after which it
behaves exactly like a bundled one. On Windows the unbundled form is
refused with an explanation — compile with `bundle = true` there. The
Julia running here must match the version the library was linked
against.
"""
function load(path::AbstractString)
    blas_libs = [l.libname for l in BLAS.get_config().loaded_libs]
    handle = _dlopen_model(String(path), Libdl.RTLD_LOCAL | Libdl.RTLD_DEEPBIND)
    return CLib(String(path), handle, blas_libs)
end

"""
    restore_blas!(lib::CLib)

Re-forward the host's libblastrampoline to the BLAS/LAPACK libraries that were
active before `lib` was loaded. Idempotent; called automatically by
[`CNLPModel`](@ref) after the library's first call.
"""
function restore_blas!(lib::CLib)
    for (i, l) in enumerate(lib.blas_libs)
        BLAS.lbt_forward(l; clear = (i == 1))
    end
    return lib
end

"""
    CNLPModel{VT}

A model instance of a cnlp library as an `AbstractNLPModel{Float64, VT}`. `VT` is where
the solver's vectors live: `Vector{Float64}` on the host, a GPU vector type for a model
built with `device = ...` (the evaluation pointers are then the library's device tier).
Structure, layout, parameters, readers and writers are instance state and work the same
for every `VT`.
"""
struct CNLPModel{VT <: AbstractVector{Float64}, IT <: AbstractVector{Int}} <: AbstractNLPModel{Float64, VT}
    meta::NLPModelMeta{Float64, VT}
    counters::Counters
    lib::CLib
    id::Int64
    # evaluation: the host callbacks, or the device tier (`P_*_d`)
    obj_p::Ptr{Cvoid}
    grad_p::Ptr{Cvoid}
    cons_p::Ptr{Cvoid}
    jac_p::Ptr{Cvoid}
    hess_p::Ptr{Cvoid}
    # the sparsity structures, read once on the host
    jrows::Vector{Int}
    jcols::Vector{Int}
    hrows::Vector{Int}
    hcols::Vector{Int}
    # the same structures where VT lives, value buffers for the products, and the
    # zero multipliers of the objective-only Hessian
    vjrows::IT
    vjcols::IT
    vhrows::IT
    vhcols::IT
    jbuf::VT
    hbuf::VT
    zeroy::VT
    # the model's named blocks and the prefix they are addressed under
    prefix::String
    vars::NamedTuple
    cons::NamedTuple
    pars::NamedTuple
end

"""
    BlockRef

A named block of a compiled model — a variable, constraint or parameter — with
the slice of the solution (or of the parameter vector) it occupies.

`offset` is 0-based, as the library reports it; `dims` is the block's shape, so
a slice can be reshaped the way `ExaModels` does rather than handed back flat.
`index` is the library's own numbering, which is how parameters are addressed
for [`get_value`](@ref) / [`set_value!`](@ref).
"""
struct BlockRef
    name::Symbol
    kind::Symbol          # :var, :con, :par
    offset::Int
    length::Int
    dims::Dims
    index::Int64
end

Base.length(b::BlockRef) = b.length
Base.size(b::BlockRef) = b.dims

Base.show(io::IO, b::BlockRef) = print(
    io, "BlockRef(:", b.name, ", ", b.kind, ", ", join(b.dims, "×"),
    " at ", b.offset, ")")

# The library's published layout, or empty tuples when it publishes none.
# Named blocks are an optional part of the ABI, like the schema and builder:
# a hand-written C library that implements only the evaluators has nothing to
# name, and is consumed exactly as before.
function _read_layout(lib::CLib, prefix::AbstractString, id::Int64)
    nb = Libdl.dlsym(lib.handle, Symbol(prefix, "_nblocks"); throw_error = false)
    nb === nothing && return ((;), (;), (;))
    try
        return _read_layout_strict(lib, prefix, id, nb)
    catch err
        # The layout is an OPTIONAL tier: a library whose block query fails
        # is nonconformant, but that must not make the MODEL unusable — the
        # evaluators are independent of it. Warn (once per construction, with
        # the real error) and carry on with no named blocks. Observed live:
        # a producer bug returned status 2 for six of seventeen models, and
        # every one of them was thereby unconstructable rather than merely
        # nameless.
        @warn "reading the named-block layout of `$prefix` failed; the model
               works, but get_vars/get_cons/get_pars will report no blocks" err
        return ((;), (;), (;))
    end
end

function _read_layout_strict(lib::CLib, prefix::AbstractString, id::Int64, nb::Ptr{Cvoid})
    n = ccall(nb, Int64, (Int64,), id)
    n < 0 && return ((;), (;), (;))
    bp = Libdl.dlsym(lib.handle, Symbol(prefix, "_block"))
    np = Libdl.dlsym(lib.handle, Symbol(prefix, "_block_name"))
    vars, cons, pars = Pair{Symbol,BlockRef}[], Pair{Symbol,BlockRef}[], Pair{Symbol,BlockRef}[]
    for k in 0:(Int(n) - 1)
        out = zeros(Int64, 4 + 8)
        _check(ccall(bp, Int64, (Int64, Int64, Ptr{Int64}), id, Int64(k), out),
               Symbol(prefix, "_block"))
        need = ccall(np, Int64, (Int64, Int64, Ptr{UInt8}, Int64), id, Int64(k), C_NULL, Int64(0))
        buf = Vector{UInt8}(undef, Int(need))
        ccall(np, Int64, (Int64, Int64, Ptr{UInt8}, Int64), id, Int64(k), buf, need)
        name = Symbol(String(buf))
        nd = Int(out[4])
        b = BlockRef(name,
                     out[1] == 0 ? :var : out[1] == 1 ? :con : :par,
                     Int(out[2]), Int(out[3]),
                     Dims(Int(out[4 + i]) for i in 1:nd),
                     Int64(k))
        push!(b.kind === :var ? vars : b.kind === :con ? cons : pars, name => b)
    end
    return (NamedTuple(vars), NamedTuple(cons), NamedTuple(pars))
end

@noinline _status_error(sym, st) =
    error("$sym returned nonzero status $st")

@inline function _check(st::Int64, sym::Symbol)
    st == 0 || _status_error(sym, st)
    return nothing
end

"""
    schema_json(lib::CLib; prefix = "rec") -> String
    schema_json(lib::CLib, model::Symbol) -> String

The library's data schema as published by `<prefix>_schema` (the schema/builder surface,
structured libraries only): a JSON description of the fields, kinds
(scalar/array/table) and column types. In a library carrying several models
the schema is per model — name it as a symbol, exactly as in [`CNLPModel`](@ref).
"""
function schema_json(lib::CLib; prefix::AbstractString = "rec")
    fp = Libdl.dlsym(lib.handle, Symbol(prefix, "_schema"); throw_error = false)
    # A model with no structured schema is not an error to ask about: it takes
    # one value or none, and `argtype` says which. Returning `nothing` lets a
    # caller ask about any model uniformly.
    fp === nothing && return nothing
    n = ccall(fp, Int64, (Ptr{UInt8}, Int64), C_NULL, Int64(0))
    buf = Vector{UInt8}(undef, Int(n))
    ccall(fp, Int64, (Ptr{UInt8}, Int64), buf, n)
    return String(buf)
end

schema_json(lib::CLib, model::Symbol) = schema_json(lib; prefix = String(model))

# Top-level field names from the schema JSON, in declaration order. The
# schema is this package's own contract (see the README), so a structural
# scan suffices: field objects sit at bracket depth 2 inside "fields"
# (table columns at depth 4 are thereby excluded).
function _schema_field_names(json::AbstractString)
    names = String[]
    r = findfirst("\"fields\"", json)
    r === nothing && error("library schema has no \"fields\" array")
    depth = 0
    i = last(r)
    while i < lastindex(json)
        i = nextind(json, i)
        ch = json[i]
        if ch == '[' || ch == '{'
            depth += 1
        elseif ch == ']' || ch == '}'
            depth -= 1
            depth <= 0 && break
        elseif ch == '"'
            j = findnext('"', json, nextind(json, i))
            tok = json[nextind(json, i):prevind(json, j)]
            if depth == 2 && tok == "name"
                k1 = findnext('"', json, nextind(json, j))
                k2 = findnext('"', json, nextind(json, k1))
                push!(names, json[nextind(json, k1):prevind(json, k2)])
                i = k2
            else
                i = j
            end
        end
    end
    return names
end

# ── Instantiation: positional arguments → model id ───────────────────────────
#
# Arguments are positional — one value per schema field, in the order the
# library publishes them. That is deliberately the same convention the producer
# side uses: an `ExaCore` built against placeholders is instantiated as
# `ExaModel(core, arg1, arg2, ...)` and compiled as
# `compile_library(out, core, arg1, ...)`, so a compiled model is consumed the
# way it was written. The schema a compiled recipe publishes names its fields
# `arg1`, `arg2`, ... for exactly that reason.
#
# A field's value is a scalar, an array, or a table (a vector of named tuples,
# sent columnar) — the same grammar the producer derives its schema from.
#
# A lone integer takes the one-knob `<prefix>_new(n)` when the library exports
# it, and falls through to the builder otherwise. The two surfaces are disjoint
# in practice: `compile_library` emits `P_new` and no builder precisely when the
# schema is a single integer scalar.

function _instantiate(lib::CLib, prefix::AbstractString, args::Tuple)
    # A lone STRING argument goes to `<prefix>_new_str` — the entry point for
    # models whose one argument is a string (a case-file path, say). There is
    # no ambiguity with model SELECTION: by the time `_instantiate` runs, a
    # leading selector string has already been consumed, so a string here is
    # an argument. `argtype` is the witness a caller consults; the symbol is
    # the witness this routing probes.
    if length(args) == 1 && args[1] isa AbstractString
        p = Libdl.dlsym(lib.handle, Symbol(prefix, "_new_str"); throw_error = false)
        p === nothing && error(
            "`$prefix` has no string entry point (`$(prefix)_new_str`); its " *
            "signature is `" * _argtype(lib, prefix) * "`")
        # a data-set file by name, as `cnlp solve` takes one; any other string as given
        id = ccall(p, Int64, (Cstring,), _string_arg(args[1]))
        id > 0 || _status_error(Symbol(prefix, "_new_str"), id)
        return id
    end
    if length(args) == 1 && args[1] isa Integer
        p = Libdl.dlsym(lib.handle, Symbol(prefix, "_new"); throw_error = false)
        if p !== nothing
            id = ccall(p, Int64, (Int64,), Int64(args[1]))
            id > 0 || _status_error(Symbol(prefix, "_new"), id)
            return id
        end
    end
    # A FIXED model — a library whose `<prefix>_nargs()` reports 0 — consumes
    # no instantiation data: `<prefix>_new` keeps its one-integer C signature
    # but ignores the value, so no arguments at all is the natural call. A
    # library that does not declare its arity keeps the old behaviour and
    # falls through to the builder surface (where a schema with no fields is
    # the other legitimate "no instance data" case).
    if isempty(args)
        q = Libdl.dlsym(lib.handle, Symbol(prefix, "_nargs"); throw_error = false)
        if q !== nothing && ccall(q, Int64, ()) == 0
            p = Libdl.dlsym(lib.handle, Symbol(prefix, "_new"))
            id = ccall(p, Int64, (Int64,), Int64(0))
            id > 0 || _status_error(Symbol(prefix, "_new"), id)
            return id
        end
    end
    return _fill_data(lib, prefix, _bind(lib, prefix, args))
end

# Positional arguments against the schema's field order. No arguments at all is
# the "no instance data" case, and is checked the same way — a schema declaring
# fields says so here, rather than the library reporting itself incomplete
# several calls later.
function _bind(lib::CLib, prefix::AbstractString, args::Tuple)
    Libdl.dlsym(lib.handle, Symbol(prefix, "_data_begin"); throw_error = false) === nothing &&
        error("this library has no builder surface: it instantiates from a " *
              "single integer, `CNLPModel(lib, n)`")
    names = _schema_field_names(schema_json(lib; prefix = prefix))
    length(names) == length(args) || error(
        "given $(length(args)) argument$(length(args) == 1 ? "" : "s") but the " *
        "library's schema declares $(length(names)) field" *
        "$(length(names) == 1 ? "" : "s")" *
        (isempty(names) ? "" : ": " * join(names, ", ")))
    return NamedTuple{Tuple(Symbol.(names))}(args)
end

# One builder field, by value dispatch: numbers are scalars, numeric vectors
# are arrays, vectors of named tuples are tables (sent column-by-column). The
# library itself validates names, kinds, completeness and column lengths.
_push_field(fp, sym, b, f, val::AbstractFloat) =
    _check(ccall(fp(:set_scalar_f64), Int64, (Int64, Cstring, Cdouble),
        b, f, Cdouble(val)), sym(:set_scalar_f64))
_push_field(fp, sym, b, f, val::Integer) =
    _check(ccall(fp(:set_scalar_i64), Int64, (Int64, Cstring, Clonglong),
        b, f, Clonglong(val)), sym(:set_scalar_i64))
_push_field(fp, sym, b, f, val::AbstractVector{<:AbstractFloat}) =
    _check(ccall(fp(:set_array_f64), Int64, (Int64, Cstring, Ptr{Cdouble}, Int64),
        b, f, convert(Vector{Float64}, val), Int64(length(val))), sym(:set_array_f64))
_push_field(fp, sym, b, f, val::AbstractVector{<:Integer}) =
    _check(ccall(fp(:set_array_i64), Int64, (Int64, Cstring, Ptr{Clonglong}, Int64),
        b, f, convert(Vector{Int64}, val), Int64(length(val))), sym(:set_array_i64))
function _push_field(fp, sym, b, f, val::AbstractVector{<:NamedTuple})
    for cn in fieldnames(eltype(val))
        _push_col(fp, sym, b, f, string(cn), [getfield(r, cn) for r in val])
    end
end
_push_field(fp, sym, b, f, val) =
    error("unsupported field $f::$(typeof(val)): fields must be numbers, " *
          "numeric vectors, or vectors of named tuples of numbers")

_push_col(fp, sym, b, f, c, col::AbstractVector{<:AbstractFloat}) =
    _check(ccall(fp(:set_col_f64), Int64, (Int64, Cstring, Cstring, Ptr{Cdouble}, Int64),
        b, f, c, convert(Vector{Float64}, col), Int64(length(col))), sym(:set_col_f64))
_push_col(fp, sym, b, f, c, col::AbstractVector{<:Integer}) =
    _check(ccall(fp(:set_col_i64), Int64, (Int64, Cstring, Cstring, Ptr{Clonglong}, Int64),
        b, f, c, convert(Vector{Int64}, col), Int64(length(col))), sym(:set_col_i64))
_push_col(fp, sym, b, f, c, col) =
    error("unsupported column type $(eltype(col)) for $f.$c")

function _fill_data(lib::CLib, prefix::AbstractString, data::NamedTuple)
    sym(s) = Symbol(prefix, "_", s)
    fp(s) = Libdl.dlsym(lib.handle, sym(s))
    b = ccall(fp(:data_begin), Int64, ())
    b > 0 || _status_error(sym(:data_begin), b)
    for (fname, val) in pairs(data)
        _push_field(fp, sym, b, string(fname), val)
    end
    ccall(fp(:data_ready), Int64, (Int64,), b) == 1 ||
        error("library reports data incomplete or inconsistent after all fields were set")
    id = ccall(fp(:new_from_data), Int64, (Int64,), b)
    id > 0 || _status_error(sym(:new_from_data), id)
    return id
end

"""
    CNLPModel(lib::CLib, arg1, arg2, ...; prefix = "rec", name = basename(lib.path))

Create a model instance from `lib` and wrap it as an `AbstractNLPModel`.

The arguments are the values the model is instantiated with — one per field of
the library's schema, positionally, in the order the library publishes them.
This is the same spelling the producer side uses (`ExaModel(core, arg1, ...)`,
`compile_library(out, core, arg1, ...)`), so a compiled model is consumed the
way it was written.

Each value is a **number**, a **numeric vector**, or a **table** (a vector of
named tuples, sent to the builder column by column and validated against
the library's schema). A lone integer uses `<prefix>_new(n)` when the library
exports it and the builder otherwise; with no arguments at all the model is
built from no instance data, which is valid when the schema declares no fields.

Any number of instances may coexist per library. The first library call lazily
finishes its runtime initialization, after which the host's BLAS forwarding is
restored — see [`restore_blas!`](@ref).

    m = CNLPModel(lib, 1000; prefix = "rosen")             # rosen_new(1000)
    m = CNLPModel(lib, [(i = 1, pd = 0.4)], [0.9], 100.0)  # table, array, scalar
"""
function CNLPModel(
    lib::CLib,
    args...;
    prefix::AbstractString = "rec",
    name::AbstractString = basename(lib.path),
    device::Symbol = :cpu,
    input::Union{Nothing, Pair} = nothing,
)
    sym(s) = Symbol(prefix, "_", s)
    fp(s) = Libdl.dlsym(lib.handle, sym(s))

    # `input = :name => path`: instantiate through one of the model's own
    # instance readers (the readers tier) instead of its entry point.
    id = if input === nothing
        _instantiate(lib, prefix, args)
    else
        isempty(args) || throw(ArgumentError("give either instantiation arguments or `input`, not both"))
        _instantiate_reader(lib, prefix, Symbol(first(input)), String(last(input)))
    end
    # The library's runtime is fully initialized by now: repair whatever its
    # lazy initialization did to the shared trampoline.
    restore_blas!(lib)

    nvar = Int(ccall(fp(:nvar), Int64, (Int64,), id))
    ncon = Int(ccall(fp(:ncon), Int64, (Int64,), id))
    nnzj = Int(ccall(fp(:nnzj), Int64, (Int64,), id))
    nnzh = Int(ccall(fp(:nnzh), Int64, (Int64,), id))

    x0 = zeros(nvar); lvar = zeros(nvar); uvar = zeros(nvar)
    lcon = zeros(ncon); ucon = zeros(ncon)
    st = ccall(fp(:meta), Int64,
        (Int64, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
        id, x0, lvar, uvar, lcon, ucon)
    _check(st, sym(:meta))

    # The objective sense is optional in the ABI: a library without P_minimize
    # minimizes.
    mp = Libdl.dlsym(lib.handle, sym(:minimize); throw_error = false)
    minimize = mp === nothing ? true : ccall(mp, Int64, (Int64,), id) != 0

    meta = NLPModelMeta(
        nvar; ncon = ncon, nnzj = nnzj, nnzh = nnzh,
        x0 = x0, lvar = lvar, uvar = uvar, lcon = lcon, ucon = ucon,
        minimize = minimize, name = String(name),
    )
    vars, cons, pars = _read_layout(lib, prefix, id)
    jrows, jcols = _structure(fp(:jac_structure), id, nnzj, sym(:jac_structure))
    hrows, hcols = _structure(fp(:hess_structure), id, nnzh, sym(:hess_structure))
    m = CNLPModel(
        meta, Counters(), lib, id,
        fp(:obj), fp(:grad), fp(:cons), fp(:jac), fp(:hess),
        jrows, jcols, hrows, hcols, jrows, jcols, hrows, hcols,
        Vector{Float64}(undef, nnzj), Vector{Float64}(undef, nnzh), zeros(ncon),
        String(prefix), vars, cons, pars,
    )
    # Restore once more on the way out: the structure calls above are the
    # first entries into THIS model's generated code, and in a multi-model
    # library a later model's first calls can lazily clobber the trampoline
    # again after the post-instantiation restore. Observed live: model two of
    # an OPF library printing `no BLAS/LAPACK library loaded for dcopy_()`
    # at solve time until an explicit restore.
    restore_blas!(lib)
    device === :cpu && return m
    # The device tier (`P_device`, `P_*_d`): a model on the same instance whose
    # metadata and buffers live on the device. Provided by a package extension.
    hasmethod(_device_model, Tuple{Val{device}, typeof(m)}) || throw(ArgumentError(
        "`device = :$device` needs its extension loaded (for :cuda, `using CUDA` or `using CUDACore`)"))
    return _device_model(Val(device), m)
end

"""
    _device_model(::Val{device}, m::CNLPModel)

The same instance as `m` with `VT` on `device`, evaluated through the device tier.
Methods live in package extensions.
"""
function _device_model end

# A sparsity structure (1-based COO), read from the library.
function _structure(fp::Ptr{Cvoid}, id::Int64, nnz::Int, name::Symbol)
    r = Vector{Int}(undef, nnz); c = Vector{Int}(undef, nnz)
    _check(ccall(fp, Int64, (Int64, Ptr{Int64}, Ptr{Int64}), id, r, c), name)
    return r, c
end

"""
    CNLPModel(spec::AbstractString, arg1, arg2, ...; kwargs...)

String-based construction:

  - `"@opf"` resolves the **name** `opf` via [`lib`](@ref) against
    [`set_path!`](@ref) / `CNLPLIB_PATH` (`\$dir/libopf.so`, or the
    bundled layouts `\$dir/opf/lib/libopf.so` and `\$dir/lib/libopf.so`);
    the prefix defaults to the name;
  - any other string is a filesystem **path**, relative to the current
    directory or absolute, exactly as written — either the shared library
    itself, or a bundle directory `compile_library` produced, in which case
    the library is found inside it. The prefix defaults to the resolved
    name stripped of `lib` and the extension (`.../librosen.so` → `"rosen"`,
    `/path/to/opf/` → `"opf"`).

Override either default with `prefix=`.

    m = CNLPModel("@acopf", bus, vmin, 100.0)     # search path
    m = CNLPModel("rosen", 1000)                  # ./rosen (file or bundle dir)
    m = CNLPModel("/opt/models/rosen", 1000)      # full path
"""
function CNLPModel(
    spec::AbstractString, args...;
    prefix::AbstractString = _default_prefix(spec), kwargs...,
)
    return CNLPModel(_resolve_spec(spec), args...; prefix = prefix, kwargs...)
end

"""
    CNLPModel(lib, model::Symbol, arg1, arg2, ...; name = String(model))

Select a model **by name** in a library carrying several.

One shared library may export any number of models, each under its own symbol
prefix, each with its own schema and its own instances — `:acopf` binds
`acopf_new`, `acopf_obj`, `acopf_meta`, ... So the symbol *is* the prefix, and
naming it is the whole of model selection:

    m = CNLPModel("@grid", :acopf, bus, 100.0)   # acopf_* inside libgrid.so
    d = CNLPModel("@grid", :dcopf, bus)          # dcopf_* in the same library

`lib` is anything the single-model constructors accept — a `CLib`, a `cnlp"..."`
literal, an `"@name"`, or a path. Models in one library share the library's
address space and, for a library carrying a language runtime, its single runtime
copy; they are otherwise independent, and instances of different models coexist
exactly as instances of one model do.

Omitting the symbol keeps the single-model spelling, where the prefix falls back
to the library name — so a one-model library is unaffected.
"""
function CNLPModel(
    lib::CLib, model::Symbol, args...; name::AbstractString = String(model), kwargs...,
)
    # naming the model and passing `prefix=` are the same knob: not both
    haskey(kwargs, :prefix) && throw(MethodError(CNLPModel, (lib, model, args...)))
    _require_model(lib, model)
    return CNLPModel(lib, args...; prefix = String(model), name = name, kwargs...)
end

# A name this library does not carry must be reported HERE. `_nvar` is the
# witness because the ABI requires it of every model whichever way the model is
# instantiated — unlike `_new` (absent from builder-only models) or
# `_data_begin` (absent from one-knob ones). Without this check a mistyped name
# surfaces several calls later as "this library has no builder surface", which
# describes a single-model library rather than a name that is not in this one.
function _require_model(lib::CLib, model::Symbol)
    Libdl.dlsym(lib.handle, Symbol(model, "_nvar"); throw_error = false) === nothing &&
        error("library $(lib.path) carries no model named `$model` " *
              "(it exports no `$(model)_nvar`)")
    return nothing
end

function CNLPModel(spec::AbstractString, model::Symbol, args...; kwargs...)
    return CNLPModel(_resolve_spec(spec), model, args...; kwargs...)
end

# `@name` resolves on the search path; any other string is a filesystem path,
# relative to the current directory or absolute, exactly as written.
_is_name(spec::AbstractString) = startswith(spec, "@")

_default_prefix(spec::AbstractString) =
    _is_name(spec) ? String(first(split(spec[2:end], '@'))) : _prefix_from_path(spec)

# `librosen.so` → `rosen`; a bundle directory or a file not following the
# `lib<name>` convention keeps its stem, and `prefix=` remains the override
# for libraries whose symbols are named independently of the file.
function _prefix_from_path(path::AbstractString)
    base = first(splitext(basename(rstrip(path, '/'))))
    return startswith(base, "lib") && length(base) > 3 ? base[4:end] : base
end

# A path names a shared library directly, or a bundle DIRECTORY — the layout
# `compile_library` produces — in which case the library is found inside it.
# Returned ABSOLUTE: `dlopen` treats a slash-free relative like `qp.so` as a
# soname to search the system path for, not as a file in the current
# directory — it resolved locally only by environmental accident and failed
# in CI.
function _resolve_path(spec::AbstractString)
    isfile(spec) && return abspath(spec)
    if isdir(spec)
        fname = "lib" * basename(rstrip(spec, '/')) * "." * Libdl.dlext
        for cand in (joinpath(spec, "lib", fname), joinpath(spec, fname))
            isfile(cand) && return abspath(cand)
        end
        error("no shared library in $spec (tried lib/$fname and $fname)")
    end
    error("no shared library at $spec")
end

# Cached like name-resolution: absolute-path keys cannot collide with bare
# names, so the one registry serves both.
_resolve_spec(spec::AbstractString) =
    _is_name(spec) ? lib(spec[2:end]) :
    get!(() -> load(_resolve_path(spec)), _LIBS, abspath(spec))

"""
    argtype(lib, model) -> String

What a model instantiates **from**: the types its entry point takes, in order,
each optionally followed by `|` and a description.

This is the question a consumer holding only a library cannot otherwise answer.
`""` is a model that takes nothing, `"int|size"` one that takes a size,
`"string|…"` one whose entry point takes a string, and a longer list is a
structured model taking one value per field:

    argtype(lib, :acopf)   # "int|arg1,Vector{f64}|v0,Table{i::int pd::f64}|bus"

Every model answers, including those with no builder schema — which is why
this exists rather than reading the schema, which only structured models have.
A library that publishes no signature returns `""`.
"""
argtype(lib::CLib, model::Union{Symbol, AbstractString}) = _argtype(lib, String(model))
argtype(spec::AbstractString, model::Union{Symbol, AbstractString}) =
    _argtype(_resolve_spec(spec), String(model))

function _argtype(lib::CLib, prefix::AbstractString)
    fp = Libdl.dlsym(lib.handle, Symbol(prefix, "_argtype"); throw_error = false)
    fp === nothing && return ""
    n = ccall(fp, Int64, (Ptr{UInt8}, Int64), C_NULL, Int64(0))
    n <= 0 && return ""
    buf = Vector{UInt8}(undef, Int(n))
    ccall(fp, Int64, (Ptr{UInt8}, Int64), buf, n)
    return String(buf)
end

"""
    available_models(lib) -> Vector{Symbol}

The models a library carries, by name.

Every other entry point needs a prefix to start from; this is the one question
a caller can ask having only a path. Pair it with [`CNLPModel`](@ref):

    for name in available_models("@grid")
        m = CNLPModel("@grid", name, data)
    end

A library that does not publish a catalogue — a hand-written C library, or one
compiled before libraries said what they carried — returns an empty vector, and
selecting a model by name still works if you know the name.
"""
available_models(lib::CLib) = _catalogue(lib)
available_models(spec::AbstractString) = _catalogue(_resolve_spec(spec))

function _catalogue(lib::CLib)
    n = Libdl.dlsym(lib.handle, :cnlp_nmodels; throw_error = false)
    n === nothing && return Symbol[]
    count = ccall(n, Int64, ())
    count <= 0 && return Symbol[]
    np = Libdl.dlsym(lib.handle, :cnlp_model_name)
    out = Vector{Symbol}(undef, Int(count))
    for k in 0:(Int(count) - 1)
        need = ccall(np, Int64, (Int64, Ptr{UInt8}, Int64), Int64(k), C_NULL, Int64(0))
        buf = Vector{UInt8}(undef, Int(need))
        ccall(np, Int64, (Int64, Ptr{UInt8}, Int64), Int64(k), buf, need)
        out[k + 1] = Symbol(String(buf))
    end
    return out
end

# ── Named blocks ─────────────────────────────────────────────────────────────

"""
    get_vars(model[, name])
    get_cons(model[, name])
    get_pars(model[, name])

The model's **named** blocks of that kind, as a `NamedTuple` — or one of them
by name.

A compiled library publishes the names its model was written with, so a caller
who never sees the Julia source can still address a slice of the solution by
name. Pair them with [`solution`](@ref), [`multipliers`](@ref),
[`multipliers_L`](@ref) / [`multipliers_U`](@ref), and [`get_value`](@ref) /
[`set_value!`](@ref) for parameters — the same spellings `ExaModels` uses for a
model built in Julia.

Named blocks are optional in the ABI: a library that publishes none reports
empty tuples, and everything else about it still works.

    m = CNLPModel("@grid", :acopf, bus)
    keys(get_vars(m))                 # (:pg, :qg, :vm, :va)
    solution(result, get_vars(m, :pg))

Modelling packages export accessors with these names too (ExaModels itself
does); with both loaded, Julia makes the bare name ambiguous — qualify as
`CNLPLib.get_vars`.
"""
get_vars(m::CNLPModel) = getfield(m, :vars)

@doc (@doc get_vars)
get_cons(m::CNLPModel) = getfield(m, :cons)

@doc (@doc get_vars)
get_pars(m::CNLPModel) = getfield(m, :pars)

get_vars(m::CNLPModel, name::Symbol) = _named_block(m, :var, name)
@doc (@doc get_vars)
get_cons(m::CNLPModel, name::Symbol) = _named_block(m, :con, name)
@doc (@doc get_vars)
get_pars(m::CNLPModel, name::Symbol) = _named_block(m, :par, name)

# The two ways of being wrong are separated here for the same reason as in
# ExaModels: named blocks share one namespace, so asking for a variable that is
# really a parameter is at least as likely as a typo, and wants a different fix.
function _named_block(m::CNLPModel, kind::Symbol, name::Symbol)
    nt = kind === :var ? get_vars(m) : kind === :con ? get_cons(m) : get_pars(m)
    haskey(nt, name) && return nt[name]
    for (k, other) in ((:var, get_vars(m)), (:con, get_cons(m)), (:par, get_pars(m)))
        if haskey(other, name)
            fn = k === :var ? "get_vars" : k === :con ? "get_cons" : "get_pars"
            throw(ArgumentError("`$name` is a $(k === :var ? "variable" :
                k === :con ? "constraint" : "parameter") ($fn), not a $kind"))
        end
    end
    what = kind === :var ? "variable" : kind === :con ? "constraint" : "parameter"
    throw(ArgumentError(
        "this model has no named $what `$name`; it has $(keys(nt))" *
        (isempty(keys(nt)) ?
         " (none — this library publishes no named blocks)" : "")))
end

"""
    solution(result, block)

The part of `result.solution` belonging to variable `block`, reshaped to the
block's own dimensions.
"""
solution(result, b::BlockRef) =
    reshape(view(result.solution, (b.offset + 1):(b.offset + b.length)), b.dims...)

# The same extraction from a bare vector — for callers holding `res.solution`
# rather than the result object; without this method that spelling failed
# with a `FieldError` pointing at Array, which explains nothing.
solution(x::AbstractVector, b::BlockRef) =
    reshape(view(x, (b.offset + 1):(b.offset + b.length)), b.dims...)

"""
    multipliers(result, block)

The dual variables for constraint `block`.
"""
multipliers(result, b::BlockRef) =
    reshape(view(result.multipliers, (b.offset + 1):(b.offset + b.length)), b.dims...)

@doc (@doc solution)
multipliers(y::AbstractVector, b::BlockRef) =
    reshape(view(y, (b.offset + 1):(b.offset + b.length)), b.dims...)

"""
    multipliers_L(result, block)
    multipliers_U(result, block)

The lower- and upper-bound dual variables for variable `block`.
"""
multipliers_L(result, b::BlockRef) =
    reshape(view(result.multipliers_L, (b.offset + 1):(b.offset + b.length)), b.dims...)

@doc (@doc solution)
multipliers_L(z::AbstractVector, b::BlockRef) =
    reshape(view(z, (b.offset + 1):(b.offset + b.length)), b.dims...)

@doc (@doc multipliers_L)
multipliers_U(result, b::BlockRef) =
    reshape(view(result.multipliers_U, (b.offset + 1):(b.offset + b.length)), b.dims...)

@doc (@doc solution)
multipliers_U(z::AbstractVector, b::BlockRef) =
    reshape(view(z, (b.offset + 1):(b.offset + b.length)), b.dims...)

"""
    get_value(model, param)
    set_value!(model, param, values)

Read or update a parameter block's values. Parameters are model state: a write
takes effect for every later evaluation of this instance, and other instances
of the same model keep their own.

Unlike `ExaModels.get_value`, this returns a copy rather than a view — the
values live in the library's address space, not the caller's.
"""
function get_value(m::CNLPModel, b::BlockRef)
    b.kind === :par || throw(ArgumentError("`$(b.name)` is a $(b.kind), not a parameter"))
    out = Vector{Float64}(undef, b.length)
    fp = Libdl.dlsym(m.lib.handle, Symbol(m.prefix, "_get_value"))
    _check(ccall(fp, Int64, (Int64, Int64, Ptr{Cdouble}, Int64),
                 m.id, b.index, out, Int64(b.length)), Symbol(m.prefix, "_get_value"))
    return reshape(out, b.dims...)
end

@doc (@doc get_value)
function set_value!(m::CNLPModel, b::BlockRef, values)
    b.kind === :par || throw(ArgumentError("`$(b.name)` is a $(b.kind), not a parameter"))
    v = convert(Vector{Float64}, vec(collect(values)))
    length(v) == b.length || throw(DimensionMismatch(
        "parameter `$(b.name)` has $(b.length) elements, got $(length(v))"))
    fp = Libdl.dlsym(m.lib.handle, Symbol(m.prefix, "_set_value"))
    _check(ccall(fp, Int64, (Int64, Int64, Ptr{Cdouble}, Int64),
                 m.id, b.index, v, Int64(b.length)), Symbol(m.prefix, "_set_value"))
    return m
end

# ── evaluation, for every VT ──────────────────────────────────────────────────────
# Two hooks are specialized per storage by the extensions: `_ptr`, an array as the raw
# pointer the ABI takes (a host array converts itself in `ccall`), and `_sync`, which
# orders the library's reads after our writes (a no-op on the host). The products
# assemble from the COO values the ABI provides through `_coo_mul!` and `_sym_mul!`.

_ptr(a::AbstractVector{Float64}) = a
_sync(::CNLPModel) = nothing

function NLPModels.obj(m::CNLPModel, x::AbstractVector{Float64})
    @lencheck m.meta.nvar x
    increment!(m, :neval_obj)
    _sync(m)
    out = Ref{Cdouble}(0.0)
    _check(ccall(m.obj_p, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble}), m.id, _ptr(x), out), :obj)
    return out[]
end

function NLPModels.grad!(m::CNLPModel, x::AbstractVector{Float64}, g::AbstractVector{Float64})
    @lencheck m.meta.nvar x g
    increment!(m, :neval_grad)
    _sync(m)
    _check(ccall(m.grad_p, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble}), m.id, _ptr(x), _ptr(g)), :grad)
    return g
end

# The ABI draws no linear/nonlinear distinction: all constraints are nonlinear.
for (fn, inc) in ((:cons!, :neval_cons), (:cons_nln!, :neval_cons_nln))
    @eval function NLPModels.$fn(m::CNLPModel, x::AbstractVector{Float64}, c::AbstractVector{Float64})
        @lencheck m.meta.nvar x
        @lencheck m.meta.ncon c
        increment!(m, $(QuoteNode(inc)))
        m.meta.ncon == 0 && return c
        _sync(m)
        _check(ccall(m.cons_p, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble}), m.id, _ptr(x), _ptr(c)), :cons)
        return c
    end
end

function NLPModels.jac_structure!(m::CNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    @lencheck m.meta.nnzj rows cols
    copyto!(rows, m.jrows); copyto!(cols, m.jcols)
    return rows, cols
end

function NLPModels.hess_structure!(m::CNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    @lencheck m.meta.nnzh rows cols
    copyto!(rows, m.hrows); copyto!(cols, m.hcols)
    return rows, cols
end

function NLPModels.jac_coord!(m::CNLPModel, x::AbstractVector{Float64}, vals::AbstractVector{Float64})
    @lencheck m.meta.nvar x
    @lencheck m.meta.nnzj vals
    increment!(m, :neval_jac)
    m.meta.nnzj == 0 && return vals
    _sync(m)
    _check(ccall(m.jac_p, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble}), m.id, _ptr(x), _ptr(vals)), :jac)
    return vals
end

function NLPModels.hess_coord!(
    m::CNLPModel, x::AbstractVector{Float64}, y::AbstractVector{Float64},
    vals::AbstractVector{Float64}; obj_weight::Real = 1.0,
)
    @lencheck m.meta.nvar x
    @lencheck m.meta.ncon y
    @lencheck m.meta.nnzh vals
    increment!(m, :neval_hess)
    m.meta.nnzh == 0 && return vals
    _sync(m)
    _check(ccall(m.hess_p, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Ptr{Cdouble}),
                 m.id, _ptr(x), _ptr(y), Cdouble(obj_weight), _ptr(vals)), :hess)
    return vals
end

NLPModels.hess_coord!(m::CNLPModel, x::AbstractVector{Float64}, vals::AbstractVector{Float64};
                      obj_weight::Real = 1.0) =
    NLPModels.hess_coord!(m, x, m.zeroy, vals; obj_weight)

# y = A v for COO (rows, cols, vals), and for a lower triangle stored as COO.
function _coo_mul!(y, rows, cols, vals, v)
    fill!(y, 0.0)
    @inbounds for k in eachindex(vals)
        y[rows[k]] += vals[k] * v[cols[k]]
    end
    return y
end
function _sym_mul!(y, rows, cols, vals, v)
    fill!(y, 0.0)
    @inbounds for k in eachindex(vals)
        i, j, h = rows[k], cols[k], vals[k]
        y[i] += h * v[j]
        i == j || (y[j] += h * v[i])
    end
    return y
end

for (fn, inc) in ((:jprod_nln!, :neval_jprod), (:jprod!, :neval_jprod))
    @eval function NLPModels.$fn(m::CNLPModel, x::AbstractVector{Float64}, v::AbstractVector{Float64},
                                 Jv::AbstractVector{Float64})
        @lencheck m.meta.nvar x v
        @lencheck m.meta.ncon Jv
        increment!(m, $(QuoteNode(inc)))
        NLPModels.jac_coord!(m, x, m.jbuf)
        return _coo_mul!(Jv, m.vjrows, m.vjcols, m.jbuf, v)
    end
end

for (fn, inc) in ((:jtprod_nln!, :neval_jtprod), (:jtprod!, :neval_jtprod))
    @eval function NLPModels.$fn(m::CNLPModel, x::AbstractVector{Float64}, v::AbstractVector{Float64},
                                 Jtv::AbstractVector{Float64})
        @lencheck m.meta.nvar x Jtv
        @lencheck m.meta.ncon v
        increment!(m, $(QuoteNode(inc)))
        NLPModels.jac_coord!(m, x, m.jbuf)
        return _coo_mul!(Jtv, m.vjcols, m.vjrows, m.jbuf, v)
    end
end

function NLPModels.hprod!(
    m::CNLPModel, x::AbstractVector{Float64}, y::AbstractVector{Float64},
    v::AbstractVector{Float64}, Hv::AbstractVector{Float64}; obj_weight::Real = 1.0,
)
    @lencheck m.meta.nvar x v Hv
    @lencheck m.meta.ncon y
    increment!(m, :neval_hprod)
    NLPModels.hess_coord!(m, x, y, m.hbuf; obj_weight)
    return _sym_mul!(Hv, m.vhrows, m.vhcols, m.hbuf, v)
end

NLPModels.hprod!(m::CNLPModel, x::AbstractVector{Float64}, v::AbstractVector{Float64},
                 Hv::AbstractVector{Float64}; obj_weight::Real = 1.0) =
    NLPModels.hprod!(m, x, m.zeroy, v, Hv; obj_weight)

# ── solution writers ─────────────────────────────────────────────────────────────
# A library may carry writers for its models (P_nwriters, P_writer_name,
# P_write_<name>): code that turns a solution into the model's own file format,
# such as a solved MATPOWER case. They run on the host.

"""
    writers(m) -> Vector{Symbol}

The solution writers the library carries for this model; empty if none.
"""
function writers(m::CNLPModel)
    np = Libdl.dlsym(m.lib.handle, Symbol(m.prefix, "_nwriters"); throw_error = false)
    np === nothing && return Symbol[]
    fp = Libdl.dlsym(m.lib.handle, Symbol(m.prefix, "_writer_name"))
    out = Symbol[]
    for k in 0:(Int(ccall(np, Int64, ())) - 1)
        n = ccall(fp, Int64, (Int64, Ptr{UInt8}, Int64), k, C_NULL, 0)
        buf = Vector{UInt8}(undef, n)
        ccall(fp, Int64, (Int64, Ptr{UInt8}, Int64), k, buf, n)
        push!(out, Symbol(String(buf)))
    end
    return out
end

"""
    write_solution(m, writer, path, result)
    write_solution(m, writer, path, x, y, zl, zu; objective, status = "")

Write a solution with one of the library's writers (see [`writers`](@ref)).
`result` is a solver result with `solution`, `multipliers`, `multipliers_L`,
`multipliers_U`, `objective` and `status` (MadNLP's, NLPModelsIpopt's), in the
NLPModels sign convention, which is converted to the cnlp one for a maximizing model.
The explicit form takes `y`, `zl`, `zu` already in the cnlp convention
(σ∇f + Jᵀy − zL + zU = 0, cnlp-solution 2.1). Arrays on a device are copied to the
host first.
"""
function write_solution(m::CNLPModel, writer::Symbol, path::AbstractString,
                        x::AbstractVector, y::AbstractVector, zl::AbstractVector, zu::AbstractVector;
                        objective::Real, status::AbstractString = "")
    writer in writers(m) || throw(ArgumentError("`$(m.prefix)` has no writer `$writer`; it has $(writers(m))"))
    @lencheck m.meta.nvar x zl zu
    @lencheck m.meta.ncon y
    fp = Libdl.dlsym(m.lib.handle, Symbol(m.prefix, "_write_", writer))
    hx, hy, hl, hu = (Vector{Float64}(Array(v)) for v in (x, y, zl, zu))
    st = ccall(fp, Int64, (Int64, Cstring, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cstring),
               m.id, path, hx, hy, hl, hu, Cdouble(objective), status)
    st == 0 || error("writer `$writer` of `$(m.prefix)` could not write $path (status $st)")
    return path
end

function write_solution(m::CNLPModel, writer::Symbol, path::AbstractString, result)
    y, zl, zu = _to_cnlp_signs(m, result.multipliers, result.multipliers_L, result.multipliers_U)
    return write_solution(m, writer, path, result.solution, y, zl, zu;
                          objective = result.objective, status = string(result.status))
end

# Multiplier signs. A cnlp solution satisfies σ∇f + Jᵀy − zL + zU = 0 (σ = −1 when the model
# maximizes; cnlp-solution 2.1). NLPModels/SolverCore results (MadNLP, NLPModelsIpopt) satisfy
# ∇f + Jᵀy − zL + zU = 0, so for a maximizing model all three are negated, each way.
_to_cnlp_signs(m::CNLPModel, y, zl, zu) =
    m.meta.minimize ? (y, zl, zu) : (-Array(y), -Array(zl), -Array(zu))
_from_cnlp_signs(m::CNLPModel, y, zl, zu) = _to_cnlp_signs(m, y, zl, zu)

# ── the standard solution format (cnlp-solution 0.2: formats/solution.md) ──────────
# HDF5, written by the HDF5 extension (`using HDF5`): root attributes, the `/solution`
# group (as ExaModelsIO stores a solution) and `/blocks/<kind>/<name>`. Other encodings
# are a library's writers.

# Solver statuses (MadNLP, SolverCore) mapped to the solver-neutral ones.
const _STATUS = Dict(
    "SOLVE_SUCCEEDED" => "solved", "SOLVED_TO_ACCEPTABLE_LEVEL" => "acceptable",
    "INFEASIBLE_PROBLEM_DETECTED" => "infeasible", "DIVERGING_ITERATES" => "unbounded",
    "MAXIMUM_ITERATIONS_EXCEEDED" => "iteration_limit", "MAXIMUM_WALLTIME_EXCEEDED" => "time_limit",
    "first_order" => "solved", "acceptable" => "acceptable", "infeasible" => "infeasible",
    "unbounded" => "unbounded", "max_iter" => "iteration_limit", "max_time" => "time_limit",
)
_status(s) = get(_STATUS, string(s), "error")

# c(x) through the host callback, whatever VT the model evaluates on.
function _host_cons(m::CNLPModel, x::Vector{Float64})
    c = zeros(m.meta.ncon)
    m.meta.ncon == 0 && return c
    fp = Libdl.dlsym(m.lib.handle, Symbol(m.prefix, "_cons"))
    _check(ccall(fp, Int64, (Int64, Ptr{Cdouble}, Ptr{Cdouble}), m.id, x, c), Symbol(m.prefix, "_cons"))
    return c
end

# The solution as plain host data, organized by the library's named blocks.
function _solution_record(m::CNLPModel, result)
    x = Vector{Float64}(Array(result.solution))
    y, zl, zu = (Vector{Float64}(Array(v)) for v in
                 _to_cnlp_signs(m, result.multipliers, result.multipliers_L, result.multipliers_U))
    c = _host_cons(m, x)
    sl(v, b) = reshape(v[(b.offset + 1):(b.offset + b.length)], b.dims...)
    blocks = Tuple{String,String,Vector{Pair{String,Array{Float64}}}}[]
    for (name, b) in pairs(get_vars(m))
        push!(blocks, ("variables", String(name), ["value" => sl(x, b),
            "multiplier_L" => sl(zl, b), "multiplier_U" => sl(zu, b)]))
    end
    for (name, b) in pairs(get_cons(m))
        push!(blocks, ("constraints", String(name), ["value" => sl(c, b), "multiplier" => sl(y, b)]))
    end
    for (name, b) in pairs(get_pars(m))
        push!(blocks, ("parameters", String(name), ["value" => reshape(Vector{Float64}(get_value(m, b)), b.dims...)]))
    end
    return (; library = basename(m.lib.path), model = m.prefix, nvar = m.meta.nvar, ncon = m.meta.ncon,
            minimize = m.meta.minimize, objective = Float64(result.objective),
            status = _status(result.status), solver_status = string(result.status),
            iterations = hasproperty(result, :iter) ? Int(result.iter) : -1, x, y, zl, zu, blocks)
end

# `_write_standard(path, record)` is defined by the HDF5 extension.
function _write_standard end

# The writer the library names as its default for this model, or `nothing`.
function default_writer(m::CNLPModel)
    fp = Libdl.dlsym(m.lib.handle, Symbol(m.prefix, "_default_writer"); throw_error = false)
    fp === nothing && return nothing
    n = ccall(fp, Int64, (Int64, Ptr{UInt8}, Int64), m.id, C_NULL, 0)
    n <= 0 && return nothing
    buf = Vector{UInt8}(undef, n)
    ccall(fp, Int64, (Int64, Ptr{UInt8}, Int64), m.id, buf, n)
    return Symbol(String(buf))
end

"""
    write_solution(m, path, result; format = nothing)

Write a solution in the model's default format: the library's default writer if it
names one (e.g. a MATPOWER case), otherwise the standard cnlp-solution format (HDF5,
needs `using HDF5`). `format` forces `:cnlp` (the standard format) or one of
[`writers`](@ref).
"""
function write_solution(m::CNLPModel, path::AbstractString, result; format::Union{Nothing,Symbol} = nothing)
    f = format !== nothing ? format : something(default_writer(m), :cnlp)
    f === :cnlp || return write_solution(m, f, path, result)
    isempty(methods(_write_standard)) && throw(ArgumentError("the standard solution format needs HDF5: `using HDF5`"))
    return _write_standard(path, _solution_record(m, result))
end

# ── readers: the model's own input formats ──────────────────────────────────────
# (cnlp.h, the readers tier). Kinds: :instance, :parameters, :start.

const _READER_KINDS = (:instance, :parameters, :start)

# The ABI string convention (call once to size, once to fill), for the two
# shapes readers use: (buf, cap) and (k, buf, cap).
function _copyout(fp)
    n = ccall(fp, Int64, (Ptr{UInt8}, Int64), C_NULL, 0)
    n <= 0 && return ""
    buf = Vector{UInt8}(undef, n)
    ccall(fp, Int64, (Ptr{UInt8}, Int64), buf, n)
    return String(buf)
end
function _copyout(fp, k::Int64)
    n = ccall(fp, Int64, (Int64, Ptr{UInt8}, Int64), k, C_NULL, 0)
    n <= 0 && return ""
    buf = Vector{UInt8}(undef, n)
    ccall(fp, Int64, (Int64, Ptr{UInt8}, Int64), k, buf, n)
    return String(buf)
end

"""
    readers(lib, model::Symbol) -> Vector{Pair{Symbol,Symbol}}
    readers(m) -> Vector{Pair{Symbol,Symbol}}

The model's own input readers, as `name => kind` with kind `:instance`,
`:parameters` or `:start`; empty if the library carries none.
"""
function readers(lib::CLib, model::Symbol)
    np = Libdl.dlsym(lib.handle, Symbol(model, "_nreaders"); throw_error = false)
    np === nothing && return Pair{Symbol,Symbol}[]
    fn = Libdl.dlsym(lib.handle, Symbol(model, "_reader_name"))
    fk = Libdl.dlsym(lib.handle, Symbol(model, "_reader_kind"))
    return [Symbol(_copyout(fn, Int64(k))) => _READER_KINDS[ccall(fk, Int64, (Int64,), k) + 1]
            for k in 0:(Int(ccall(np, Int64, ())) - 1)]
end
readers(m::CNLPModel) = readers(m.lib, Symbol(m.prefix))

"""
    default_reader(lib, model::Symbol) -> Union{Symbol,Nothing}

The instance reader the model uses for a bare path, if it names one.
"""
function default_reader(lib::CLib, model::Symbol)
    fp = Libdl.dlsym(lib.handle, Symbol(model, "_default_reader"); throw_error = false)
    fp === nothing && return nothing
    name = _copyout(fp)
    return isempty(name) ? nothing : Symbol(name)
end

function _reader_of_kind(lib, prefix, name::Symbol, kind::Symbol)
    rs = readers(lib, Symbol(prefix))
    any(==(name => kind), rs) || throw(ArgumentError(
        "`$prefix` has no $kind reader `$name`; its readers are $(isempty(rs) ? "none" : rs)"))
    return nothing
end

function _instantiate_reader(lib::CLib, prefix::AbstractString, name::Symbol, path::String)
    _reader_of_kind(lib, prefix, name, :instance)
    fp = Libdl.dlsym(lib.handle, Symbol(prefix, "_new_from_", name))
    id = ccall(fp, Int64, (Cstring,), path)
    id > 0 || error("`$(prefix)_new_from_$(name)` could not create an instance from $path")
    return id
end

"""
    read_start(m, reader::Symbol, path) -> (; x0, y0, zl, zu)

A starting point read from the model's own file format by one of its start
readers, as host vectors, with the multipliers in the NLPModels sign convention
(converted from the cnlp one for a maximizing model) so they go straight to an
NLPModels solver. Use it to warm-start a solve, e.g. `madnlp(m; x0 = s.x0)` or by
copying into `m.meta.x0`.
"""
function read_start(m::CNLPModel, reader::Symbol, path::AbstractString)
    _reader_of_kind(m.lib, m.prefix, reader, :start)
    n, k = m.meta.nvar, m.meta.ncon
    x0, y0, zl, zu = zeros(n), zeros(k), zeros(n), zeros(n)
    fp = Libdl.dlsym(m.lib.handle, Symbol(m.prefix, "_start_from_", reader))
    _check(ccall(fp, Int64, (Int64, Cstring, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                 m.id, path, x0, y0, zl, zu), Symbol(m.prefix, "_start_from_", reader))
    y0, zl, zu = _from_cnlp_signs(m, y0, zl, zu)
    return (; x0, y0, zl, zu)
end

"""
    read_parameters!(m, reader::Symbol, path) -> m

Set the model's parameters from its own file format, with one of its parameter
readers. Live, like [`set_value!`](@ref).
"""
function read_parameters!(m::CNLPModel, reader::Symbol, path::AbstractString)
    _reader_of_kind(m.lib, m.prefix, reader, :parameters)
    fp = Libdl.dlsym(m.lib.handle, Symbol(m.prefix, "_set_from_", reader))
    _check(ccall(fp, Int64, (Int64, Cstring), m.id, path), Symbol(m.prefix, "_set_from_", reader))
    return m
end


# What a first use compiles (a traced load, construction and callbacks of a host model),
# compiled here so it is in the package image. A statement naming something a dependency
# no longer has is skipped.
# The model API on the host storage type, compiled into the package image so the first
# model of a session runs without compiling.
let M = CNLPModel{Vector{Float64}, Vector{Int}}, V = Vector{Float64}, W = NamedTuple{(:obj_weight,), Tuple{Float64}}
    precompile(CNLPModel, (CLib, Symbol, Int))
    precompile(CNLPModel, (CLib, Int))
    precompile(Core.kwcall, (NamedTuple{(:prefix,), Tuple{String}}, Type{CNLPModel}, CLib, Int))
    for (f, args) in (
        (NLPModels.obj, (M, V)), (NLPModels.grad, (M, V)), (NLPModels.grad!, (M, V, V)),
        (NLPModels.cons, (M, V)), (NLPModels.cons!, (M, V, V)), (NLPModels.cons_nln!, (M, V, V)),
        (NLPModels.jac_structure, (M,)), (NLPModels.hess_structure, (M,)),
        (NLPModels.jac_coord, (M, V)), (NLPModels.jac_coord!, (M, V, V)),
        (NLPModels.hess_coord, (M, V)), (NLPModels.hess_coord, (M, V, V)),
        (NLPModels.hess_coord!, (M, V, V)), (NLPModels.hess_coord!, (M, V, V, V)),
        (NLPModels.jprod, (M, V, V)), (NLPModels.jprod!, (M, V, V, V)),
        (NLPModels.jtprod, (M, V, V)), (NLPModels.jtprod!, (M, V, V, V)),
        (NLPModels.hprod, (M, V, V)), (NLPModels.hprod, (M, V, V, V)),
        (NLPModels.hprod!, (M, V, V, V)), (NLPModels.hprod!, (M, V, V, V, V)),
    )
        precompile(f, args)
        precompile(Core.kwcall, (W, typeof(f), args...))
    end
end

module PrecompileStatements
    for (id, m) in Base.loaded_modules
        name = Symbol(id.name)
        isdefined(@__MODULE__, name) || Core.eval(@__MODULE__, :(const $name = $m))
    end
    for line in eachline(joinpath(@__DIR__, "precompile_statements.jl"))
        try
            Core.eval(@__MODULE__, Meta.parse(line))
        catch
        end
    end
end

include("solver.jl")

end # module CNLPLib
