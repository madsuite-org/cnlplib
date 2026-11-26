# Check a built library: every model it lists loads; fixed models and models sized by one
# integer are instantiated, evaluated, and (when the library has the device tier and a GPU
# is present) their device callbacks compared with the host ones. Nothing is solved.
#
#   julia scripts/check-library.jl LIBPATH [N]
using Pkg
Pkg.activate(; temp = true)
Pkg.add([Pkg.PackageSpec(url = "https://github.com/madsuite-org/cnlplib", subdir = "interfaces/CNLPLib.jl"),
         Pkg.PackageSpec(name = "NLPModels"), Pkg.PackageSpec(name = "CUDA")]; io = devnull)
using NLPModels, CUDA, Libdl
# CNLPLib (named CNLPModels before its first registration): found by UUID, either name
const CL = let u = Base.UUID("e8ddee8d-ce21-405b-9ec9-45cea2c1b284")
    Base.require(Base.PkgId(u, Pkg.dependencies()[u].name))
end
const CNLPModel = CL.CNLPModel

libpath = ARGS[1]
n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
lib = CL.load(libpath)
@assert ccall(Libdl.dlsym(lib.handle, :cnlp_abi_version), Int64, ()) == 2 "not a cnlp 0.2 library"
gpu = CUDA.functional()
rel(a, b) = maximum(abs, Array(a) .- b; init = 0.0) / max(1.0, maximum(abs, b; init = 0.0))
failed = String[]
for model in CL.available_models(lib)
    # P_argtype (copy-out): "" for a fixed model, else comma-separated `type|description`
    fp = Libdl.dlsym(lib.handle, Symbol(model, "_argtype"))
    buf = zeros(UInt8, 1024)
    len = ccall(fp, Int64, (Ptr{UInt8}, Int64), buf, length(buf))
    spec = String(buf[1:min(len, length(buf))])
    args = isempty(spec) ? () : (startswith(spec, "int|") && !occursin(',', spec)) ? (n,) : nothing
    if args === nothing
        println(rpad(string(model), 24), "loads (instance needs data; not evaluated)")
        continue
    end
    try
        h = CNLPModel(lib, model, args...)
        y = cos.(1:h.meta.ncon)
        evaluate(x) = (obj(h, x), grad(h, x), cons(h, x), jac_coord(h, x), hess_coord(h, x, y))
        finite(v) = all(isfinite, (v[1], Iterators.flatten(v[2:end])...))
        # Near x0 first, so the values do not depend on x0 being special. Some models are
        # defined only on part of the space (a log, a square root), and the perturbed point
        # can leave it; x0 itself is where a solver starts, so it is checked before failing.
        x = h.meta.x0 .+ 0.1 .* sin.(1:h.meta.nvar)
        v = evaluate(x)
        msg = "nvar $(h.meta.nvar), ncon $(h.meta.ncon)"
        if !finite(v)
            x = copy(h.meta.x0)
            v = evaluate(x)
            finite(v) || error("non-finite values at x0 and near it")
            msg *= ", at x0"
        end
        f, g, c, J, H = v
        devp = Libdl.dlsym(lib.handle, Symbol(model, "_device"); throw_error = false)
        if gpu && devp !== nothing
            d = CNLPModel(lib, model, args...; device = :cuda)
            X, Y = CuArray(x), CuArray(y)
            e = max(abs(obj(d, X) - f) / max(1, abs(f)), rel(grad(d, X), g), rel(cons(d, X), c),
                    rel(jac_coord(d, X), J), rel(hess_coord(d, X, Y), H))
            e <= 1e-8 || error("device differs from host: $e")
            msg *= ", device vs host $(round(e; sigdigits = 2))"
        end
        println(rpad(string(model), 24), "ok: ", msg)
    catch err
        push!(failed, string(model))
        println(rpad(string(model), 24), "FAILED: ", sprint(showerror, err)[1:min(end, 300)])
    end
end
isempty(failed) || println("failed: ", join(failed, ", "))
# Leave without running exit hooks: after a bundled library (with its own Julia runtime)
# has been loaded, the host's exit-time GC has crashed intermittently (seen twice on Julia
# 1.12, in `temp_cleanup_atexit`), which would turn a passing check into a failure.
flush(stdout); flush(stderr)
ccall(:_exit, Cvoid, (Cint,), isempty(failed) ? 0 : 1)
