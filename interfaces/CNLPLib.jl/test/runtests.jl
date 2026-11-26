using Test
using CNLPLib, HDF5
using NLPModels
using LinearAlgebra
using MadNLP

# The shared fixture (conformance/tinyqp.c), compiled against this repository's cnlp.h.
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const FIXDIR = joinpath(ROOT, "conformance")
struct NoSuchBackend end   # a backend no extension handles
const LIBPATH = joinpath(mktempdir(), "libtinyqp.so")
cc = something(Sys.which("cc"), Sys.which("gcc"), Sys.which("clang"))
@test cc !== nothing
run(`$cc -shared -fPIC -O2 -I$ROOT -o $LIBPATH $(joinpath(FIXDIR, "tinyqp.c"))`)
# libcnlpio (solve/cnlpio.c), against HDF5_jll's libhdf5, as the solution-file library
import HDF5_jll
let h5 = dirname(HDF5_jll.libhdf5_path), lib = joinpath(mktempdir(), "libcnlpio.so")
    # HDF5_jll is a parallel build: its headers include the mpi.h of the MPI it loaded
    incs = [joinpath(HDF5_jll.artifact_dir, "include")]
    for m in values(Base.loaded_modules)
        endswith(string(nameof(m)), "_jll") && isdefined(m, :artifact_dir) || continue
        d = joinpath(m.artifact_dir, "include")
        isfile(joinpath(d, "mpi.h")) && push!(incs, d)
    end
    run(`$cc -std=c99 -O2 -fPIC -shared -I$ROOT $(map(d -> "-I" * d, incs))
         -o $lib $(joinpath(ROOT, "solve", "cnlpio.c")) -L$h5 -lhdf5 -Wl,-rpath,$h5`)
    ENV["CNLPIO_LIB"] = lib
end

@testset "CNLPLib" begin
    lib = CNLPLib.load(LIBPATH)

    @testset "init failure surfaces" begin
        @test_throws CNLPLib.CNLPError CNLPModel(lib, 1; prefix = "tq")
    end

    @testset "storage type" begin
        m = CNLPModel(lib, 4; prefix = "tq")
        @test m isa CNLPModel{Vector{Float64}} && m isa AbstractNLPModel{Float64, Vector{Float64}}
        # a device model needs its extension (and a library with the device tier)
        @test_throws ArgumentError CNLPModel(lib, 4; prefix = "tq", backend = NoSuchBackend())
    end

    m = CNLPModel(lib, 4; prefix = "tq")

    @testset "reports: empty for a library without them" begin
        @test report_instance(m) == ""
        @test report_solution(m, (solution = zeros(4), multipliers = zeros(1), multipliers_L = zeros(4),
                                  multipliers_U = zeros(4), objective = 0.0, status = :first_order)) == ""
    end

    @testset "meta" begin
        @test m.meta.nvar == 4
        @test m.meta.ncon == 1
        @test m.meta.nnzj == 2
        @test m.meta.nnzh == 4
        @test m.meta.x0 == zeros(4)
        @test all(==(-Inf), m.meta.lvar)
        @test all(==(Inf), m.meta.uvar)
        @test m.meta.lcon == [0.0]
        @test m.meta.ucon == [0.0]
    end

    @testset "evaluations match closed form" begin
        x = [0.5, 0.25, 2.0, -1.0]
        @test obj(m, x) == 0.25 + 0.5625 + 1.0 + 4.0
        @test grad(m, x) == 2.0 .* (x .- 1.0)
        @test cons(m, x) == [x[1] + x[2] - 1.0]
        @test jac_structure(m) == ([1, 1], [1, 2])
        @test jac_coord(m, x) == [1.0, 1.0]
        @test hess_structure(m) == ([1, 2, 3, 4], [1, 2, 3, 4])
        # y must not contribute (linear constraint); obj_weight must scale.
        @test hess_coord(m, x, [3.0]; obj_weight = 0.7) == fill(1.4, 4)
        @test hess_coord(m, x; obj_weight = 2.0) == fill(4.0, 4)
    end

    @testset "counters" begin
        @test m.counters.neval_obj > 0
        @test m.counters.neval_grad > 0
        @test m.counters.neval_hess > 0
    end

    @testset "BLAS guard is callable and BLAS works after it" begin
        @test CNLPLib.restore_blas!(lib) === lib
        @test ones(2, 2) * ones(2, 2) == fill(2.0, 2, 2)
    end

    @testset "solve with MadNLP" begin
        res = madnlp(m; print_level = MadNLP.ERROR)
        @test res.status == MadNLP.SOLVE_SUCCEEDED
        @test res.objective ≈ 0.5 rtol = 1e-6
        @test res.solution[1] ≈ 0.5 rtol = 1e-6
        @test res.solution[2] ≈ 0.5 rtol = 1e-6
        @test all(res.solution[3:end] .≈ 1.0)
    end

    @testset "multiple coexisting instances" begin
        x = [0.5, 0.25, 2.0, -1.0]
        o_before = obj(m, x)
        m6 = CNLPModel(lib, 6; prefix = "tq")
        @test m6.meta.nvar == 6
        @test obj(m, x) == o_before          # a second instance does not disturb the first
        @test m.id != m6.id
        res6 = madnlp(m6; print_level = MadNLP.ERROR)
        res4 = madnlp(m; print_level = MadNLP.ERROR)
        @test res6.status == MadNLP.SOLVE_SUCCEEDED
        @test res4.status == MadNLP.SOLVE_SUCCEEDED
        @test res6.objective ≈ 0.5 rtol = 1e-6
        @test res4.objective ≈ 0.5 rtol = 1e-6
    end
end

@testset "library path registry" begin
    # a second copy under a models dir, named like a registered model
    mdir = mktempdir()
    cp(LIBPATH, joinpath(mdir, "libtoyqp.so"))
    CNLPLib.set_path!(mdir)
    l = CNLPLib.lib("toyqp")
    @test l isa CNLPLib.CLib
    @test CNLPLib.lib("toyqp") === l          # cached by name
    CNLPLib.set_path!(mdir)                    # setting the path again: names resolve afresh
    @test !haskey(CNLPLib._LIBS, "toyqp")
    m = CNLPModel("@toyqp", 4; prefix = "tq") # name-based construction
    @test m.meta.nvar == 4
    # not found anywhere: the registry (a local, empty index) has no such library either
    withenv("CNLPLIB_INDEX" => "file://" * (f = tempname(); write(f, ""); f)) do
        @test_throws ErrorException CNLPLib.lib("nonexistent")
    end
end

@testset "arguments are positional, one per schema field" begin
    lib = CNLPLib.load(LIBPATH)
    # `tq` declares one scalar field and also exports tq_new, so a lone integer
    # takes the one-knob constructor.
    @test CNLPLib._schema_field_names(CNLPLib.schema_json(lib; prefix = "tq")) == ["n"]
    m5 = CNLPModel(lib, 5; prefix = "tq")
    @test m5.meta.nvar == 5
    r5 = madnlp(m5; print_level = MadNLP.ERROR)
    @test r5.status == MadNLP.SOLVE_SUCCEEDED
    @test r5.objective ≈ 0.5 rtol = 1e-6

    # `sq` is builder-only (no sq_new) and declares three fields, so the
    # arguments bind positionally in the order the schema publishes them.
    @test CNLPLib._schema_field_names(CNLPLib.schema_json(lib; prefix = "sq")) ==
          ["n", "s", "w"]
    n, s, w = 4, 2.0, [1.0, 2.0, 3.0, 4.0]
    ms = CNLPModel(lib, n, s, w; prefix = "sq")
    @test ms.meta.nvar == 4
    @test ms.meta.ncon == 1
    x = [0.5, 0.25, 2.0, -1.0]
    @test obj(ms, x) == sum(w .* (x .- s) .^ 2)          # min Σ wᵢ(xᵢ-s)²
    @test grad(ms, x) == 2.0 .* w .* (x .- s)
    @test hess_coord(ms, x, [3.0]; obj_weight = 0.7) == 1.4 .* w
    @test cons(ms, x) == [x[1] + x[2] - 1.0]

    # Order is load-bearing: the same values in the wrong order are refused by
    # the library's own setters (s is f64-only, w is the array field).
    @test_throws CNLPLib.CNLPError CNLPModel(lib, s, n, w; prefix = "sq")

    # Wrong arity names the schema rather than reaching the builder.
    @test_throws ArgumentError CNLPModel(lib, 4, 2.0; prefix = "sq")
    @test_throws ArgumentError CNLPModel(lib, 5, 6; prefix = "tq")
    # No arguments at all is the no-instance-data case; both schemas want some.
    @test_throws ArgumentError CNLPModel(lib; prefix = "tq")
    @test_throws ArgumentError CNLPModel(lib; prefix = "sq")
    # A value that cannot cross the boundary is reported as such.
    @test_throws ArgumentError CNLPModel(lib, "five"; prefix = "tq")
    # Inconsistent structured data is the library's own call to make.
    @test_throws CNLPLib.CNLPError CNLPModel(lib, 4, 2.0, [1.0, 2.0]; prefix = "sq")
end

@testset "several models in one library, selected by name" begin
    lib = CNLPLib.load(LIBPATH)
    x = [0.5, 0.25, 2.0, -1.0]
    n, s, w = 4, 2.0, [1.0, 2.0, 3.0, 4.0]

    # the name is the symbol prefix the model's ABI functions are exported under
    m = CNLPModel(lib, :tq, 4)
    @test m.meta.nvar == 4
    @test m.meta.name == "tq"                       # the model, not the file

    ms = CNLPModel(lib, :sq, n, s, w)               # builder-only sibling
    @test ms.lib === m.lib                          # one library, one runtime
    @test obj(m, x) == sum((x .- 1.0) .^ 2)
    @test obj(ms, x) == sum(w .* (x .- s) .^ 2)
    @test madnlp(ms; print_level = MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED

    # instances of different models coexist as freely as instances of one
    m6 = CNLPModel(lib, :tq, 6)
    @test m6.meta.nvar == 6
    @test obj(m, x) == sum((x .- 1.0) .^ 2)
    @test obj(ms, x) == sum(w .* (x .- s) .^ 2)

    # A model consuming no instance data still names itself.
    @test CNLPModel(lib, :fx).meta.name == "fx"

    # the same spelling through every library argument
    @test CNLPModel(LIBPATH, :tq, 4).meta.nvar == 4
    @test CNLPModel("@toyqp", :tq, 4).meta.nvar == 4

    # The display name stays overridable, and defaults to the model.
    @test CNLPModel(lib, :tq, 4; name = "mine").meta.name == "mine"

    # Schemas are per model, addressed the same way.
    @test CNLPLib._schema_field_names(CNLPLib.schema_json(lib, :sq)) ==
          ["n", "s", "w"]
    @test CNLPLib._schema_field_names(CNLPLib.schema_json(lib, :tq)) == ["n"]

    # a name the library does not carry is reported at the name
    @test_throws "carries no model named `nosuch`" CNLPModel(lib, :nosuch, 4)
    @test_throws "carries no model named `nosuch`" CNLPModel(lib, :nosuch)
    # naming the model and passing `prefix=` are the same knob
    @test_throws ArgumentError CNLPModel(lib, :tq, 4; prefix = "sq")
end

@testset "a fixed library needs no arguments" begin
    lib = CNLPLib.load(LIBPATH)
    # fx_nargs() == 0: no arguments instantiate it through fx_new, whose integer is ignored
    m0 = CNLPModel(lib; prefix = "fx")
    @test m0.meta.nvar == 3
    @test m0.meta.ncon == 1
    @test obj(m0, [0.5, 0.5, 1.0]) == 0.5
    # An explicit integer still works, and lands on the same fixed model.
    m1 = CNLPModel(lib, 999; prefix = "fx")
    @test m1.meta.nvar == 3
end

@testset "a string is @name on the search path, or a literal path" begin
    # `@name` resolves on the search path and defaults the prefix to the name.
    @test CNLPLib._is_name("@toyqp")
    @test CNLPLib._default_prefix("@toyqp") == "toyqp"

    # Anything else is a path; the prefix defaults from the file name (libtinyqp.so → tinyqp).
    @test !CNLPLib._is_name(LIBPATH)
    @test CNLPLib._default_prefix(LIBPATH) == "tinyqp"
    mp = CNLPModel(LIBPATH, 4; prefix = "tq")
    @test mp.meta.nvar == 4
    # The handle is cached by absolute path: one dlopen per library.
    @test CNLPLib._resolve_spec(LIBPATH) === CNLPLib._resolve_spec(LIBPATH)
    # A fixed model by path needs nothing beyond the path.
    mf = CNLPModel(LIBPATH; prefix = "fx")
    @test mf.meta.nvar == 3

    # A path to a bundle DIRECTORY finds the library inside it, and the
    # prefix defaults from the directory name.
    bdir = mktempdir()
    mkpath(joinpath(bdir, "toyqp2", "lib"))
    cp(LIBPATH, joinpath(bdir, "toyqp2", "lib", "libtoyqp2.so"))
    @test CNLPLib._default_prefix(joinpath(bdir, "toyqp2")) == "toyqp2"
    md = CNLPModel(joinpath(bdir, "toyqp2"), 4; prefix = "tq")
    @test md.meta.nvar == 4

    # A bare string without `@` is a file in the current directory, not a search-path name.
    cd(bdir) do
        cp(LIBPATH, joinpath(bdir, "qp.so"))
        mc = CNLPModel("qp.so", 4; prefix = "tq")
        @test mc.meta.nvar == 4
        # `toyqp` is on the search path, but without `@` it is a missing local file
        @test_throws "no shared library at" CNLPModel("toyqp", 4)
    end

    # A path that is not there fails as a path, never as a search-path miss.
    @test_throws "no shared library at" CNLPModel(joinpath(FIXDIR, "libnope.so"), 1)
end

@testset "search path initializes from the environment" begin
    old = copy(CNLPLib._PATHS)
    withenv("CNLPLIB_PATH" => "/a:/b") do
        empty!(CNLPLib._PATHS)
        @test CNLPLib._paths() == ["/a", "/b"]
    end
    CNLPLib.set_path!(old...)
end

@testset "a directory without a library says what it tried" begin
    @test_throws "no shared library in" CNLPModel(mktempdir(), 1)
end

@testset "a table field binds column by column" begin
    lib = CNLPLib.load(LIBPATH)
    # min Σ_j w_j (x_{i_j} - 1)^2 over 3 vars: variable 1 carries rows 1 and 4.
    pts = [(i = 1, w = 1.0), (i = 2, w = 2.0), (i = 3, w = 3.0), (i = 1, w = 0.5)]
    m = CNLPModel(lib, pts; prefix = "tb")
    @test (m.meta.nvar, m.meta.ncon, m.meta.nnzj, m.meta.nnzh) == (3, 0, 0, 3)
    x = [0.0, 2.0, 4.0]
    @test obj(m, x) == 1.5 + 2.0 + 27.0
    @test grad(m, x) == [-3.0, 4.0, 18.0]
    hr, hc = hess_structure(m)
    @test hr == [1, 2, 3] && hc == [1, 2, 3]
    @test hess_coord(m, x, Float64[]; obj_weight = 0.5) == [1.5, 2.0, 3.0]
    # A column the library refuses surfaces as its setter's status.
    @test_throws "set_col_i64" CNLPModel(lib, [(i = 1, w = 1.0) for _ in 1:9]; prefix = "tb")
    # And a column type nothing can carry is refused before the boundary.
    @test_throws "unsupported column" CNLPModel(lib, [(i = 1, w = "x")]; prefix = "tb")
end

@testset "integer weights widen through the i64 array setter" begin
    lib = CNLPLib.load(LIBPATH)
    m = CNLPModel(lib, 4, 2.0, [1, 2, 3, 4]; prefix = "sq")
    x = [0.5, 0.25, 2.0, -1.0]
    @test obj(m, x) == sum([1.0, 2.0, 3.0, 4.0] .* (x .- 2.0) .^ 2)
    # A field value no ABI slot can carry is refused with its type named.
    @test_throws "unsupported field" CNLPModel(lib, 4, 2.0, "w"; prefix = "sq")
end

@testset "jprod and jtprod go through the cached structure" begin
    lib = CNLPLib.load(LIBPATH)
    m = CNLPModel(lib, 4; prefix = "tq")
    x = [0.5, 0.25, 2.0, -1.0]
    v = [1.0, 2.0, 3.0, 4.0]
    y = [5.0]
    J = zeros(m.meta.ncon, m.meta.nvar)
    rows, cols = jac_structure(m)
    vals = jac_coord(m, x)
    for k in eachindex(vals)
        J[rows[k], cols[k]] += vals[k]
    end
    @test jprod(m, x, v) ≈ J * v
    @test jtprod(m, x, y) ≈ J' * y
    @test NLPModels.jprod_nln!(m, x, v, zeros(1)) ≈ J * v
    @test NLPModels.jtprod_nln!(m, x, y, zeros(4)) ≈ J' * y
    # every constraint is nonlinear: the `_nln` Jacobian is the Jacobian
    @test NLPModels.jac_nln_structure(m) == (rows, cols)
    nj = neval_jac(m)
    @test NLPModels.jac_nln_coord(m, x) == vals
    @test neval_jac_nln(m) == 1 && neval_jac(m) == nj   # counted once, as the nonlinear one
end

@testset "an unbundled juliac library is refused" begin
    # a plain C library links no libjulia, and loads (the whole suite above ran through it)
    @test !CNLPLib._unbundled(LIBPATH)
    @test !CNLPLib._unbundled("/nonexistent/libfoo.so")
    # one that links the standard libjulia (here this process's own) would share the
    # session's runtime: refused, with what to do instead
    if Sys.islinux()
        using Libdl
        libjulia = realpath(Libdl.dlpath("libjulia"))
        linked = joinpath(mktempdir(), "libtinyqp_jl.so")
        run(`$cc -shared -fPIC -O2 -I$ROOT -o $linked $(joinpath(FIXDIR, "tinyqp.c"))
             -Wl,--no-as-needed $libjulia`)
        @test CNLPLib._unbundled(linked)
        @test_throws "build it as a bundle" CNLPLib.load(linked)
        # a library that only mentions the name (a bundle's image does, in its data) is not one
        src = joinpath(mktempdir(), "mention.c")
        write(src, "const char *jl_name = \"libjulia.so.1.13\";\n")
        mention = joinpath(dirname(src), "libmention.so")
        run(`$cc -shared -fPIC -o $mention $src`)
        @test !CNLPLib._unbundled(mention)
    end
end

@testset "named blocks" begin
    lib = CNLPLib.load(LIBPATH)
    # the fixture publishes `x` (variable), `budget` (constraint), `w` (parameter)
    m = CNLPModel(lib, 4; prefix = "tq")
    @test keys(variables(m)) == (:x,)
    @test keys(constraints(m)) == (:budget,)
    @test keys(parameters(m)) == (:w,)

    b = variables(m, :x)
    @test b === variables(m).x
    @test (b.kind, b.offset, b.length, b.dims) == (:var, 0, 4, (4,))
    # lengths follow the instance
    @test variables(CNLPModel(lib, 7; prefix = "tq"), :x).length == 7

    # a name of the wrong kind names the accessor that would work; a typo lists what exists
    @test_throws "is a parameter (parameters), not a var" variables(m, :w)
    @test_throws "is a variable (variables), not a par" parameters(m, :x)
    @test_throws "no named variable `nope`" variables(m, :nope)

    # extraction from a result object and from a bare vector
    res = (solution = [1.0, 2.0, 3.0, 4.0], multipliers = [7.0],
           multipliers_L = [0.1, 0.2, 0.3, 0.4], multipliers_U = [0.5, 0.6, 0.7, 0.8])
    @test solution(res, b) == [1.0, 2.0, 3.0, 4.0]
    @test solution(res.solution, b) == [1.0, 2.0, 3.0, 4.0]
    @test multipliers_L(res.multipliers_L, b) == [0.1, 0.2, 0.3, 0.4]
    @test multipliers(res, constraints(m, :budget)) == [7.0]
    @test multipliers_L(res, b) == [0.1, 0.2, 0.3, 0.4]
    @test multipliers_U(res, b) == [0.5, 0.6, 0.7, 0.8]

    # Parameters are live state, per instance.
    p = parameters(m, :w)
    value!(m, p, [3.0, 4.0])
    @test value(m, p) == [3.0, 4.0]
    m2 = CNLPModel(lib, 4; prefix = "tq")
    value!(m2, parameters(m2, :w), [9.0, 9.0])
    @test value(m, p) == [3.0, 4.0]          # the first is undisturbed
    @test_throws DimensionMismatch value!(m, p, [1.0])
    @test_throws "not a parameter" value(m, b)

    # named blocks are optional: `sq` publishes none, and everything else still works
    ms = CNLPModel(lib, 4, 2.0, [1.0, 2.0, 3.0, 4.0]; prefix = "sq")
    @test keys(variables(ms)) == ()
    @test_throws "publishes no named blocks" variables(ms, :x)
    @test ms.meta.nvar == 4
end

@testset "discoverability" begin
    lib = CNLPLib.load(LIBPATH)
    @test available_models(lib) == [:tq, :sq, :fx, :tb, :ps]
    @test argtype(lib, :tq) == "int|size"
    @test argtype(lib, :sq) == "int|n,f64|s,Vector{f64}|w"
    @test argtype(lib, :tb) == "Table{i::int w::f64}|pts"
    @test argtype(lib, :fx) == ""
    @test argtype(lib, :ps) == "string|size, in decimal"
    @test argtype(lib, :bf) == ""                   # no published signature: "", not an error
    @test schema_json(lib; prefix = "fx") === nothing
    @test schema_json(lib; prefix = "tq") isa AbstractString   # tq has one
end

@testset "string instantiation" begin
    lib = CNLPLib.load(LIBPATH)
    # a lone string routes to `<prefix>_new_str`, as a lone integer routes to `<prefix>_new`
    m = CNLPModel(lib, "5"; prefix = "ps")
    @test m.meta.nvar == 5
    @test NLPModels.obj(m, zeros(5)) == 4.0 * 5     # sum (0-2)^2
    # a model without the string entry point refuses with its signature
    err = try CNLPModel(lib, "5"; prefix = "tq"); nothing catch e; sprint(showerror, e) end
    @test occursin("no string entry point", err) && occursin("int|size", err)
    # The library's own failure (an unparsable string) surfaces as a status.
    @test_throws "ps_new_str" CNLPModel(lib, "not-a-number"; prefix = "ps")
end

@testset "hprod and cons_nln" begin
    lib = CNLPLib.load(LIBPATH)
    m = CNLPModel(lib, 4; prefix = "tq")
    x = [0.5, 0.25, 2.0, -1.0]
    v = [1.0, -2.0, 3.0, 0.5]
    y = [3.0]
    # cons_nln! is cons! here: the ABI draws no linear distinction.
    c1 = zeros(1); c2 = zeros(1)
    NLPModels.cons!(m, x, c1); NLPModels.cons_nln!(m, x, c2)
    @test c1 == c2
    # tq's Hessian is 2*obj_weight*I
    for ow in (1.0, 0.7)
        Hv = zeros(4)
        NLPModels.hprod!(m, x, y, v, Hv; obj_weight = ow)
        @test Hv ≈ 2.0 * ow * v
        Hv0 = zeros(4)
        NLPModels.hprod!(m, x, v, Hv0; obj_weight = ow)   # y-less form
        @test Hv0 ≈ 2.0 * ow * v
    end
end

@testset "block sizes" begin
    m = CNLPModel(CNLPLib.load(LIBPATH), 4; prefix = "tq")
    b = variables(m, :x)
    @test length(b) == 4 && size(b) == (4,)
    @test sum(length, values(variables(m))) == m.meta.nvar
end

@testset "standard solution format (cnlp-solution 0.2)" begin
    m = CNLPModel(CNLPLib.load(LIBPATH), :tq, 4)
    res = madnlp(m; print_level = MadNLP.ERROR)
    path = joinpath(mktempdir(), "sol.h5")
    @test write_solution(m, path, res) == path            # no default writer: the standard format
    HDF5.h5open(path) do f
        a = HDF5.attributes(f)
        @test read(a["format"]) == "cnlp-solution" && read(a["format_version"]) == "0.2"
        @test read(a["nvar"]) == m.meta.nvar && read(a["ncon"]) == m.meta.ncon
        s = f["solution"]
        x, y = read(s["solution"]), read(s["multipliers"])
        zl, zu = read(s["multipliers_L"]), read(s["multipliers_U"])
        @test read(s["status"]) == "solved" && read(s["minimize"]) == 1
        @test read(s["objective"]) ≈ obj(m, x)
        σ = m.meta.minimize ? 1.0 : -1.0
        @test maximum(abs, σ .* grad(m, x) .+ jtprod(m, x, y) .- zl .+ zu) < 1e-6
        @test minimum(zl) >= 0 && minimum(zu) >= 0
        @test haskey(f, "blocks/variables/x")             # tq names its variable x
        @test read(f["blocks/variables/x/value"]) == x
        @test haskey(f, "blocks/constraints/budget/multiplier") && haskey(f, "blocks/parameters/w/value")
    end
    st = CNLPLib.read_start(m, path)                         # the file back, as a start
    @test st.x0 == res.solution && st.y0 ≈ res.multipliers
    @test_throws CNLPLib.CNLPError CNLPLib.read_start(CNLPModel(CNLPLib.load(LIBPATH), :tq, 5), path)
end

@testset "the model's own formats (tq: a txt writer, txtn/txtw/txt readers)" begin
    lib = CNLPLib.load(LIBPATH)
    m = CNLPModel(lib, :tq, 4)
    @test writers(m) == [:txt] && default_writer(m) === nothing
    @test readers(m) == [:txtn => :instance, :txtw => :parameters, :txt => :start]
    res = madnlp(m; print_level = MadNLP.ERROR)
    dir = mktempdir()
    @test write_solution(m, joinpath(dir, "x.txt"), res; format = :txt) == joinpath(dir, "x.txt")
    @test read_start(m, :txt, joinpath(dir, "x.txt")).x0 == res.solution
    write(joinpath(dir, "n.txt"), "6\n")
    @test CNLPModel(lib, :tq; input = :txtn => joinpath(dir, "n.txt")).meta.nvar == 6
    write(joinpath(dir, "w.txt"), "3 4\n")
    read_parameters!(m, :txtw, joinpath(dir, "w.txt"))
    @test value(m, parameters(m, :w)) == [3.0, 4.0]
end
include("solver.jl")
include("callbacks.jl")
include("registry.jl")
