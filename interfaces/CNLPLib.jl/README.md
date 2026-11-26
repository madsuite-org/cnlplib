# CNLPLib.jl

[![julia](https://github.com/madsuite-org/cnlplib/actions/workflows/julia.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/julia.yml)

```julia
using Pkg; Pkg.add(url = "https://github.com/madsuite-org/cnlplib", subdir = "interfaces/CNLPLib.jl")
```

Load a nonlinear program (NLP) exposed by a **shared library through a plain
C interface** (the [cnlp ABI](https://cnlplib.madsuite.org)) and use it as an `NLPModels.AbstractNLPModel` — so any
NLPModels-compatible solver (Ipopt via NLPModelsIpopt, or any other) can
solve it in Julia.

Any library implementing the [cnlp ABI](https://cnlplib.madsuite.org) works, in any source language:
`conformance/tinyqp.c` in this repository is a complete reference
implementation in ~100 lines of plain C.

```julia
using CNLPLib, NLPModelsIpopt

CNLPLib.set_path!("/opt/models")        # or the CNLPLIB_PATH env variable

m = CNLPModel("@mymodel", 1000)            # resolves libmymodel.so, instance of size 1000
# m = CNLPModel("/path/to/mymodel", 1000)  # or a literal path: library file or bundle dir
res = ipopt(m)
```

**Libraries from the [cnlplib registry](https://cnlplib.madsuite.org/registry/)** are
installed on first use: a name that is neither on the path nor installed in `$CNLPLIB_HOME`
(default `~/.cnlplib`) is installed there with `cnlp install`, so one copy, from one
installer, serves the `cnlp` command, Julia and Python. This needs the `cnlp` command
(`curl -fsSL https://cnlplib.madsuite.org/install | sh`). `name@nightly` is the nightly
build; `CNLPLib.update!("lv")` fetches a newer release.

```julia
m = CNLPModel("@lv@nightly", :wood, 10000)
m = CNLPModel("@emp@nightly", :acp, "pglib_opf_case118_ieee")   # a data-set file, by name
```

The `cnlp"..."` literal is the shortcut onto the path — it resolves the
name, checks that the library exists and loads, and hands back the cached
handle, directly usable as the library argument:

```julia
m = CNLPModel(cnlp"mymodel", 1000)
```

(Or bypass the path entirely with `CNLPLib.load("/path/libmymodel.so")`.)
Any number of model instances may coexist per library.

### Several models in one library

A library may carry more than one model, each under its own symbol prefix,
each with its own schema. Name the one you want as a **symbol**:

```julia
m = CNLPModel("@grid", :acopf, bus, 100.0)   # acopf_* inside libgrid.so
d = CNLPModel("@grid", :dcopf, bus)          # dcopf_* in the same library
```

The symbol *is* the prefix, and it works with every library argument — an
`"@name"`, a path, a `cnlp"..."` literal, or a `CLib`. Models in one library
share its address space and, for a library carrying a language runtime, its
single runtime copy (~80 MB, which is the reason to co-package a family of
models at all); they are otherwise independent, and instances of different
models coexist exactly as instances of one model do. Their schemas are
addressed the same way — `schema_json(lib, :acopf)`.

Omitting the symbol is the single-model spelling, where the prefix falls back
to the library name, so a one-model library needs nothing new.

Libraries built from structured data take their arguments positionally, one per schema
field, in the order the library publishes them; a table is a vector of named tuples:

```julia
m = CNLPModel("@mymodel",
    [(i = 1, pd = 0.4), (i = 2, pd = 0.3)],   # a table
    [0.9, 0.9],                               # an array
    100.0,                                    # a scalar
)
```

### GPU

A library that exports the cnlp device tier evaluates on the GPU, with CUDA arrays in and
out, so a GPU solver runs without host round trips:

```julia
using CUDA, cuMadNLP
m = CNLPModel("@mymodel", 100_000; device = :cuda)
madnlp(m)
```

### Solutions and model formats

`write_solution(m, path, result)` writes a solver result as a
[cnlp-solution](https://github.com/madsuite-org/cnlplib/blob/main/formats/solution.md)
HDF5 file (load HDF5.jl), or with the model's default writer if it names one. Multipliers
are converted to the cnlp sign convention for maximizing models. A library's own formats
are reachable by name: `writers(m)`, `write_solution(m, :matpower, path, result)`,
`readers(lib, :acopf)`, `CNLPModel(lib, :acopf; input = :matpower => "case.m")`,
`read_start(m, :matpower, path)`, `read_parameters!(m, :name, path)`.

### Solvers: the other side of the interface

cnlplib hands a model to a solver as a `cnlp_problem` (`cnlp_solver.h`): sizes, bounds,
a starting point and evaluation callbacks. A solver written in Julia and compiled as a cnlp
solver library receives a `Ptr{CNLPLib.CNLPProblem}`, solves `CNLPLib.ProblemModel(p)` (an
`AbstractNLPModel`) with any NLPModels solver, and reports with `CNLPLib.set_result!`, which
writes the multipliers in the cnlp sign convention:

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

A solver that runs on a GPU also exports `cnlp_solve_device`, which receives the problem's
device callbacks as well (`Ptr{CNLPLib.CNLPDeviceProblem}`); `CNLPLib.ProblemModel(p, d)` is
then a `ProblemModel{CuVector}` (CUDA extension), and `set_result!` takes its device arrays.

The other way, `CNLPLib.problem(m)` hands a `CNLPModel` to a cnlp solver library, and
`CNLPLib.device_problem(problem(m), md)` adds the device problem of `md`, the same instance
on the GPU.

## The C ABI

This package implements **cnlp ABI 0.2**:
[`cnlp.h`](https://github.com/madsuite-org/cnlplib/blob/main/cnlp.h) is the
specification. The test fixture, `conformance/tinyqp.c`, is compiled against it and expands
its declarations, so signature drift fails compilation.

## Implementing a compatible library

Follow `cnlp.h`, and check yourself against `conformance/tinyqp.c` and this package's
test suite, which compiles that file and exercises every function against closed-form
solutions. The file carries four models: `tq_` (instantiated from one integer), `sq_`
(built from a three-field schema through the builder), `fx_` (no instantiation data) and
`tb_` (a table field).

## Notes for libraries carrying their own runtime

If the shared library embeds a language runtime that initializes lazily on
first call and re-forwards the process-global BLAS trampoline (symptom:
`Error: no BLAS/LAPACK library loaded for dgemm_()` in the host afterwards),
this package repairs the host's BLAS forwarding automatically after the
library's first call; `restore_blas!(lib)` re-applies it at any time.

An **unbundled** juliac library — one linked against the standard `libjulia`
instead of shipping a privatized runtime copy — cannot share the host's
runtime: loaded as-is into a Julia process, its first call aborts. On Linux,
`load` detects this and provisions the library with a private, salted copy of
the *installed* Julia runtime (matching version required), patched in scratch
at load time — so a single ~2 MB library file works in-process with nothing
bundled. On other platforms the unbundled form is refused with an
explanation; compile with `bundle = true` there.

Sibling package: [cnlplib](../cnlplib-py) — the same consumer for Python (ctypes + numpy).
