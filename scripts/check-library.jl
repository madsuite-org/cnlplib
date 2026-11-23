# Check a built library: every model it lists loads; fixed models and models sized by one
# integer are instantiated, evaluated, and (when the library has the device tier and a GPU
# is present) their device callbacks compared with the host ones. Nothing is solved.
#
#   julia scripts/check-library.jl LIBPATH [N]
using Pkg
Pkg.activate(; temp = true)
Pkg.add([Pkg.PackageSpec(url = "https://github.com/madsuite-org/CNLPModels.jl"),
         Pkg.PackageSpec(name = "NLPModels"), Pkg.PackageSpec(name = "CUDA")]; io = devnull)
using CNLPModels, NLPModels, CUDA, Libdl

libpath = ARGS[1]
n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
lib = CNLPModels.load(libpath)
@assert ccall(Libdl.dlsym(lib.handle, :cnlp_abi_version), Int64, ()) == 2 "not a cnlp 0.2 library"
gpu = CUDA.functional()
rel(a, b) = maximum(abs, Array(a) .- b; init = 0.0) / max(1.0, maximum(abs, b; init = 0.0))
failed = String[]
for model in CNLPModels.available_models(lib)
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
        x = h.meta.x0 .+ 0.1 .* sin.(1:h.meta.nvar); y = cos.(1:h.meta.ncon)
        f, g, c = obj(h, x), grad(h, x), cons(h, x)
        J, H = jac_coord(h, x), hess_coord(h, x, y)
        all(isfinite, (f, g..., c..., J..., H...)) || error("non-finite values at x0")
        msg = "nvar $(h.meta.nvar), ncon $(h.meta.ncon)"
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
