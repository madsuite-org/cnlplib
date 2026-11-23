# Build a package's cnlp library with `compile_all`, in a fresh environment.
#
#   julia scripts/build-library.jl PACKAGE_DIR NAME OUT [cuda]    -> OUT/NAME/lib/libNAME.so
#
# PACKAGE_DIR is the model package's checkout, NAME the library name (`lv`), OUT the
# directory the bundle is written into (the library is named after its directory). With `cuda` the library also gets the device tier; that
# needs a CUDA GPU on the build machine. EXAMODELSCOMPILER_REV, if set, takes
# ExaModelsCompiler from that revision of ExaModels.jl instead of the registry.
using Pkg

pkgdir_, name = ARGS[1], ARGS[2]
out = joinpath(abspath(ARGS[3]), name)
device = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :cpu

Pkg.activate(; temp = true)
Pkg.Registry.update()   # a runner's registry copy can lag a fresh registration
pkgname = Pkg.TOML.parsefile(joinpath(pkgdir_, "Project.toml"))["name"]
rev = get(ENV, "EXAMODELSCOMPILER_REV", "")
Pkg.develop(Pkg.PackageSpec(path = abspath(pkgdir_)))   # the checkout as it is, not its last commit
# 0.4 writes cnlp ABI 0.2 and has the device tier; never fall back to an older one
specs = [isempty(rev) ? Pkg.PackageSpec(name = "ExaModelsCompiler", version = "0.4") :
             Pkg.PackageSpec(url = "https://github.com/madsuite-org/ExaModels.jl",
                             subdir = "ExaModelsCompiler", rev = rev),
         # TimerOutputs 1 loads PrettyTables 3, which loads REPL, which `juliac --trim`
         # cannot compile; TimerOutputs 0.5 keeps it out of the image.
         Pkg.PackageSpec(name = "TimerOutputs", version = "0.5.29")]
device === :cuda && append!(specs, [Pkg.PackageSpec(name = "CUDACore"), Pkg.PackageSpec(name = "GPUCompiler")])
Pkg.add(specs)
Pkg.status()

device === :cuda && @eval using CUDACore, GPUCompiler
@eval using ExaModelsCompiler
pkg = Base.require(Main, Symbol(pkgname))
rm(out; force = true, recursive = true)
t = @elapsed r = ExaModelsCompiler.compile_all(pkg; path = out, device, bundle = true)
println("built ", r.libpath, " (", join(r.prefixes, ", "), ") in ", round(t; digits = 1), " s")
write(joinpath(out, "MODELS"), join(r.prefixes, "\n") * "\n")
