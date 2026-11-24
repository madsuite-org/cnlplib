# Build a package's cnlp library with `compile_all`, in a fresh environment.
#
#   julia scripts/build-library.jl PACKAGE_DIR NAME OUT [cuda]    -> OUT/NAME/lib/libNAME.so
#
# PACKAGE_DIR is the model package's checkout, NAME the library name (`lv`), OUT the
# directory the bundle is written into (the library is named after its directory). With `cuda` the library also gets the device tier; that
# needs a CUDA GPU on the build machine. EXAMODELSCOMPILER_REV, if set, takes
# ExaModelsCompiler from that revision of ExaModels.jl instead of the registry.
# CNLP_CHUNK=k/N, if set, compiles the k-th of N parts (`compile_all(...; chunk = (k, N))`,
# k from 0) into OUT/NAME_k/lib/libNAME_k.so; scripts/assemble-chunks.sh joins the parts.
using Pkg

pkgdir_, name = ARGS[1], ARGS[2]
chunkspec = get(ENV, "CNLP_CHUNK", "")
chunk = isempty(chunkspec) ? nothing : Tuple(parse.(Int, split(chunkspec, '/')))
out = joinpath(abspath(ARGS[3]), chunk === nothing ? name : "$(name)_$(chunk[1])")
device = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :cpu

Pkg.activate(; temp = true)
Pkg.Registry.update()   # a runner's registry copy can lag a fresh registration
pkgname = Pkg.TOML.parsefile(joinpath(pkgdir_, "Project.toml"))["name"]
rev = get(ENV, "EXAMODELSCOMPILER_REV", "")
Pkg.develop(Pkg.PackageSpec(path = abspath(pkgdir_)))   # the checkout as it is, not its last commit
# 0.4 writes cnlp ABI 0.2 and has the device tier; never fall back to an older one
specs = [isempty(rev) ? Pkg.PackageSpec(name = "ExaModelsCompiler", version = "0.4") :
             Pkg.PackageSpec(url = "https://github.com/madsuite-org/ExaModels.jl",
                             subdir = "ExaModelsCompiler", rev = rev)]
device === :cuda && append!(specs, [Pkg.PackageSpec(name = "CUDACore"), Pkg.PackageSpec(name = "GPUCompiler")])
Pkg.add(specs)
Pkg.status()

device === :cuda && @eval using CUDACore, GPUCompiler
@eval using ExaModelsCompiler
pkg = Base.require(Main, Symbol(pkgname))
rm(out; force = true, recursive = true)
kw = chunk === nothing ? (;) : (; chunk)
t = @elapsed r = ExaModelsCompiler.compile_all(pkg; path = out, device, bundle = true, kw...)
println("built ", r.libpath, " (", join(r.prefixes, ", "), ") in ", round(t; digits = 1), " s")
write(joinpath(out, "MODELS"), join(r.prefixes, "\n") * "\n")
