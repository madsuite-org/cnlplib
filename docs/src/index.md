# cnlplib

Compiled nonlinear programs: a C ABI for them, a registry of libraries that implement it,
and the `cnlp` command that installs and solves them.

```sh
curl -fsSL https://cnlplib.madsuite.org/install | sh    # installs the cnlp command
cnlp list                                              # available and installed libraries
cnlp install lv                                        # the latest release, into ~/.cnlplib/lib
cnlp install lv@nightly                                # or the latest build of the default branch
cnlp up                                                # update everything installed
cnlp install ipopt                                     # a solver
cnlp solve @lv:wood 10000 -o sol.h5                    # solve a model with it
```

Installed libraries are found by name by every cnlp consumer, because the installer puts
`~/.cnlplib/lib` on `CNLPLIB_PATH`:

```julia
using CNLPLib
m = CNLPModel("@lv", :wood, 100_000)
```

## The ABI

A cnlp library is a shared library that carries one or more optimization models. Each model
exports plain C symbols to instantiate it and to evaluate its objective, gradient,
constraints, sparse Jacobian and Lagrangian Hessian. Any language that can call C can load
the library and solve the models, with no modeling tool installed.

- **[cnlp.h](../../cnlp.h)**: the specification, version 0.2. The header is the normative
  document: each contract is in its comments, and its macros declare every symbol for a
  model prefix, so an implementation is checked against the file it follows.
- **[formats/](../../formats/)**: the files that go with a library: the input a model is
  instantiated from ([input.md](../../formats/input.md)), parameter values
  ([parameters.md](../../formats/parameters.md)) and solutions ([solution.md](../../formats/solution.md)),
  in HDF5, and the number formats a model can be evaluated in
  ([numbers.md](../../formats/numbers.md)).
- **[example/](../../example/)**: a complete model library and a consumer in plain C;
  **[conformance/](../../conformance/)**: reference files and their check.

A model always has the core symbols (sizes, bounds, starting point, evaluation). Optional
tiers add a library-level model list, instantiation from structured data, named blocks of
variables, constraints and parameters, the objective sense, solution writers, readers of
the model's own file formats, evaluation on a GPU with device pointers, evaluation in
binary32, and evaluation in other number formats. A consumer checks one symbol per tier
and works without the tiers a library does not export.

## The registry

- Each library is released by its own project on its GitHub Releases: every version, and
  a rolling `nightly`, as `<name>-<platform>.tar.gz` (the CPU build) and
  `<name>-<platform>-cuda.tar.gz` (with GPU evaluation), with a `SHA256SUMS`. `cnlp install`
  takes the CUDA build on a machine with an NVIDIA GPU (`--device cpu|cuda` chooses).
  Libraries built on one Julia version share its runtime as an artifact, installed once.
  cnlplib does not rehost binaries.
- [libraries/](../../libraries/) lists them, one TOML file each, naming the repository that
  releases it. It lists no versions, so a release needs no change here; `cnlp` checks every
  download against the release's `SHA256SUMS`. [data/](../../data/) lists data sets.
- Model packages build and publish their releases with one reusable workflow,
  [`.github/workflows/library.yml`](../../.github/workflows/library.yml).
- CI validates every entry and publishes the index and the installer to
  `https://cnlplib.madsuite.org/`.

To publish a model library, see [docs/models.md](models.md).

## Solvers

`cnlp solve` finds and instantiates a model, applies a warm start (`-s`), calls a solver,
and writes the solution (`-o`: the model's own format, or the standard HDF5 file). A
solver implements one small interface, [cnlp_solver.h](../../cnlp_solver.h), and never sees a
file or a model format (a GPU solver also receives the model's device callbacks, and
`--device` chooses where a solve runs); [solvers/](../../solvers/) lists the installable ones. Ipopt is the
reference solver ([adapters/ipopt/](../../adapters/ipopt/)), and
[example/solver.c](../../example/solver.c) is a template. To add a solver, see
[docs/solvers.md](solvers.md).

```sh
cnlp solve @emp:acp pglib_opf_case118_ieee --solver ipopt tol=1e-8 -o matpower:sol.m
cnlp solve ./build/libmodels.so:wood 1000 -s previous.h5 -o sol.h5
```

## Producers and consumers

- [ExaModelsCompiler](https://github.com/madsuite-org/ExaModels.jl) compiles ExaModels
  models into cnlp libraries ahead of time, optionally with GPU evaluation.
- [CNLPLib.jl](https://github.com/madsuite-org/CNLPLib.jl) (Julia) and
  [cnlplib-py](https://github.com/madsuite-org/cnlplib-py) (Python, `import cnlplib`) load
  cnlp libraries as NLPModels and Python objects; CNLPLib.jl also carries the Julia side of
  the solver interface.

The ABI does not depend on any of these: anything that exports the symbols is a cnlp library.

## Prior work

The evaluation semantics follow [NLPModels.jl](https://jso.dev/NLPModels.jl/stable/api/),
and the Lagrangian Hessian convention (objective weight and multipliers, lower triangle)
follows the [AMPL Solver Library](https://www.netlib.org/ampl/solvers/asl.h).
[CUTEst](https://github.com/ralna/CUTEst) compiles test problems into libraries;
[CppADCodeGen](https://github.com/joaoleal/CppADCodeGen) introduced the in-library model
list; [FMI](https://fmi-standard.org) is the analogous standard for simulation models.
Solver C APIs such as Ipopt's define the same callbacks in the other direction, so a cnlp
library binds to them directly.

## License

MIT. Each library in the registry carries its own license, recorded in its entry.
