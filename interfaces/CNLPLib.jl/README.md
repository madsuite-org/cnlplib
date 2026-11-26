# CNLPLib.jl

[![julia](https://github.com/madsuite-org/cnlplib/actions/workflows/julia.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/julia.yml)

```julia
using Pkg; Pkg.add(url = "https://github.com/madsuite-org/cnlplib", subdir = "interfaces/CNLPLib.jl")
```

A [cnlp library](https://cnlplib.madsuite.org) (a nonlinear program behind the C ABI of
[`cnlp.h`](../../cnlp.h)) as an `NLPModels.AbstractNLPModel`, for any NLPModels solver,
and the Julia side of the solver interface (`cnlp_solver.h`).

```julia
using CNLPLib, MadNLP

m = CNLPModel("@lv", :wood, 10000)                   # the registry's lv library, model wood
m = CNLPModel("@emp", :acp, "pglib_opf_case118_ieee")  # a data-set file, by name
m = CNLPModel("/path/to/libmymodel.so", 1000)        # a path: library file or bundle directory
res = madnlp(m)
```

`@name` is looked up on `CNLPLIB_PATH` (or `CNLPLib.set_path!(dirs...)`), then among the
libraries the `cnlp` command installed in `$CNLPLIB_HOME` (default `~/.cnlplib`); a name
found nowhere is installed there with `cnlp install`
(`curl -fsSL https://cnlplib.madsuite.org/install | sh`), so one copy serves the `cnlp`
command, Julia and Python. `name@nightly` is the nightly build; `CNLPLib.update!("lv")`
fetches a newer release. `cnlp"name"` is the loaded library itself, and
`CNLPLib.load(path)` bypasses the lookup.

A library may carry several models, each under its own symbol prefix: the symbol selects
one (`:wood` binds `wood_new`, `wood_obj`, ...). Without a symbol the prefix defaults to
the library name. The arguments instantiate the model, one per field of its schema, in
the library's order; a table is a vector of named tuples:

```julia
m = CNLPModel("@mymodel", [(i = 1, pd = 0.4), (i = 2, pd = 0.3)], [0.9, 0.9], 100.0)
```

`available_models(lib)`, `argtype(lib, :wood)` and `schema_json(lib, :wood)` say what a
library carries and what each model takes. `get_vars(m)`, `get_cons(m)` and `get_pars(m)`
are the model's named blocks, for `solution(res, get_vars(m, :pg))`, `multipliers`,
`multipliers_L`, `multipliers_U`, and `get_value` / `set_value!` on parameters.

### GPU

A library that exports the cnlp device tier evaluates on the GPU, with CUDA arrays in and
out:

```julia
using CUDA, cuMadNLP
m = CNLPModel("@mymodel", 100_000; backend = CUDABackend())
madnlp(m)
```

### Solutions and model formats

`write_solution(m, path, result)` writes a solver result as a
[cnlp-solution](../../formats/solution.md) HDF5 file (`using HDF5`), or with the model's
default writer if it names one. Multipliers are converted to the cnlp sign convention for
maximizing models. A library's own formats are reachable by name: `writers(m)`,
`write_solution(m, :matpower, path, result)`, `readers(lib, :acopf)`,
`CNLPModel(lib, :acopf; input = :matpower => "case.m")`, `read_start(m, :matpower, path)`,
`read_parameters!(m, :name, path)`.

### Solvers: the other side of the interface

A solver written in Julia and compiled as a cnlp solver library receives a
`Ptr{CNLPLib.CNLPProblem}`, solves `CNLPLib.ProblemModel(p)` (an `AbstractNLPModel`) with
any NLPModels solver, and reports with `CNLPLib.set_result!`, which writes the multipliers
in the cnlp sign convention:

```julia
Base.@ccallable function cnlp_solve(p::Ptr{CNLPLib.CNLPProblem}, n::Int64, keys::Ptr{Ptr{UInt8}},
                                    vals::Ptr{Ptr{UInt8}}, r::Ptr{CNLPLib.CNLPResult})::Int64
    m = CNLPLib.ProblemModel(p)
    s = madnlp(m)        # options: CNLPLib.solver_options(n, keys, vals)
    CNLPLib.set_result!(r, m, s.solution, s.multipliers, s.multipliers_L, s.multipliers_U,
                        s.objective, CNLPLib.CNLP_SOLVED, s.iter, string(s.status))
    return 0
end
```

A GPU solver also exports `cnlp_solve_device`, which receives the problem's device
callbacks (`Ptr{CNLPLib.CNLPDeviceProblem}`); `CNLPLib.ProblemModel(p, d)` is then a
`ProblemModel{CuVector}` (CUDA extension). The other way, `CNLPLib.problem(m)` hands a
`CNLPModel` to a cnlp solver library, and `CNLPLib.device_problem(problem(m), md)` adds the
device problem of `md`, the same instance on the GPU.

### Notes for libraries carrying their own runtime

A library embedding a Julia runtime re-forwards the process's BLAS trampoline on its first
call; this package restores the host's BLAS forwarding afterwards (`restore_blas!(lib)`
re-applies it). An unbundled juliac library (linked against the standard `libjulia`) cannot
share the host's runtime; on Linux and macOS, `load` gives it a private copy of the
installed Julia runtime (same Julia version required). Elsewhere, compile with
`bundle = true`.

The test suite compiles [`conformance/tinyqp.c`](../../conformance/tinyqp.c), a plain-C
implementation of the ABI, against the repository's `cnlp.h`. Sibling package:
[cnlplib](../cnlplib-py), the same consumer for Python.
