# Adding a solver

This guide is for anyone who wants their solver to run every cnlp model with `cnlp solve`.
A solver implements one small interface, [cnlp_solver.h](../../cnlp_solver.h); `cnlp solve`
does everything around it (finding and instantiating the model, readers, warm starts,
writers, solution files), so the solver never sees a file or a model format.

## 1. The interface

A cnlp solver is a shared library exporting three functions:

```c
int64_t cnlp_solver_version(void);                        /* CNLP_SOLVER_VERSION */
int64_t cnlp_solver_name(uint8_t* buf, int64_t cap);      /* copy-out, e.g. "ipopt" */
int64_t cnlp_solve(const cnlp_problem* problem, int64_t noptions,
                   const char* const* keys, const char* const* values,
                   cnlp_result* result);
```

`cnlp_problem` is one instance of a model: `nvar`, `ncon`, `nnzj`, `nnzh`; `minimize`
(1 or 0); the starting point `x0`, bounds `lvar`, `uvar`, `lcon`, `ucon` (`±INFINITY` for
none); optionally a warm start's multipliers `y0`, `zl0`, `zu0` (or NULL); and the
evaluation callbacks, each called with `problem->ctx` as its first argument:

| callback | computes |
|---|---|
| `obj(ctx, x, &f)` | f(x) |
| `grad(ctx, x, g)` | the gradient of f, length nvar |
| `cons(ctx, x, c)` | c(x), length ncon |
| `jac_structure(ctx, rows, cols)` / `jac(ctx, x, vals)` | the Jacobian, nnzj entries, 1-based COO |
| `hess_structure(ctx, rows, cols)` / `hess(ctx, x, y, w, vals)` | the lower triangle of `w Hess f + sum_i y_i Hess c_i`, nnzh entries |

Each returns 0 on success. The structures are fixed for the instance; ask for them once.

`cnlp_result` has the caller's arrays `x`, `zl`, `zu` (nvar) and `y` (ncon), which the
solver fills, and `objective` (f(x) in the model's sense), `status`, `iterations` (-1 if
unknown) and `solver_status` (your own status, as text).

### Rules

- **Signs.** Return multipliers for which `s grad f + J'y - zL + zU = 0` at the solution,
  with `s = 1` when minimizing and `-1` when maximizing, and `zL, zU >= 0`. A minimizing
  solver handles a maximization by minimizing `-f` (negate `obj` and `grad`, and pass
  `-w` to `hess`); the multipliers it then computes are already in this convention.
  Return `objective` in the model's sense (f, not -f).
- **Status.** One of `CNLP_SOLVED`, `CNLP_ACCEPTABLE`, `CNLP_INFEASIBLE`,
  `CNLP_UNBOUNDED`, `CNLP_ITERATION_LIMIT`, `CNLP_TIME_LIMIT`, `CNLP_ERROR`.
- **Return value of cnlp_solve.** 0 when the solver ran (whatever `status` says); 1 for an
  option it does not accept (say which in `solver_status`); 2 for a failure before or
  outside the solve.
- **Options** arrive as the user typed them (`tol=1e-8` gives key `tol`, value `1e-8`).
  Parse them as your solver expects; reject what you do not know.
- **Bounds.** `±INFINITY` means none; translate to your solver's own value if it has one.
- **Memory.** Everything in `problem` belongs to the caller and is valid during the call;
  the result arrays are the caller's. Do not keep pointers after returning.
- **Never** exit, abort or throw across the boundary; report through the return value.

### On a GPU (interface 0.2)

A solver that runs on a GPU also exports

```c
int64_t cnlp_solve_device(const cnlp_problem* problem, const cnlp_device_problem* device,
                          int64_t noptions, const char* const* keys, const char* const* values,
                          cnlp_result* result);
```

`device->device` is the memory space (`CNLP_DEVICE_CUDA`), and `device->obj_d`, `grad_d`,
`cons_d`, `jac_d`, `hess_d` are the model's evaluations on device pointers (`x`, `y` and
the outputs live on the device; `obj_d` writes one host double), called with the same `ctx`
as the host callbacks. Sizes, bounds, the starting point and the sparsity structures come
from `problem` on the host, and the result goes to `result` on the host, as in
`cnlp_solve`. Return 1 for a device you do not run on.

`cnlp solve` calls `cnlp_solve_device` when the model has a CUDA tier (its library's
`-cuda` build) and the solver exports it; `--device cpu` keeps a solve on the host and
`--device cuda` requires the device. A solver without `cnlp_solve_device` is always called
through `cnlp_solve`, so a CPU solver needs no change for 0.2.

## 2. Writing the adapter

Start from [example/solver.c](../../example/solver.c), a complete solver in about 120 lines
(Newton's method on the KKT conditions, for equality-constrained problems without bounds).
A real adapter replaces its iteration with a call into the solver's own API, converting
the callbacks one to one. [adapters/ipopt/cnlp_ipopt.c](../../adapters/ipopt/cnlp_ipopt.c)
does this for Ipopt's C interface in under 200 lines, including option passing, warm
starts and the status mapping.

Any language that produces a C shared library works: C, C++, Rust, Fortran
(`iso_c_binding`), or Julia compiled with `juliac`.

## 3. Checking it

[conformance/check_solver.py](../../conformance/check_solver.py) solves the reference QP and
its maximization twin and checks the objective, feasibility, the known solution, and the
multipliers in the cnlp sign convention:

```sh
python3 conformance/check_solver.py ./libcnlp_mysolver.so      # needs h5py and cnlp-solve
```

Then try real models: `cnlp solve @lv:wood 10000 --solver ./libcnlp_mysolver.so`.

## 4. Using it without the registry

`--solver` takes a path, so a solver works as soon as it is built:

```sh
cnlp solve @emp:acp pglib_opf_case118_ieee --solver ./libcnlp_mysolver.so tol=1e-8 -o sol.m
```

## 5. Packing, releasing and registering

As for model libraries ([models.md](models.md)), with the solver's file names:

```
<name>-<platform>.tar.gz
  <name>/lib/libcnlp_<name>.so   the adapter
  <name>/lib/...                 the solver and everything it loads, found through $ORIGIN
  <name>/LICENSE, THIRD_PARTY_NOTICES, VERSION
```

attached to your repository's releases with a `SHA256SUMS` (a `v<version>` release, a
rolling `nightly`, or both), with the corresponding source of any copyleft part. Then open
a pull request adding `solvers/<name>.toml`:

```toml
name             = "ipopt"
description      = "Ipopt (interior point), with MUMPS"
homepage         = "https://github.com/coin-or/Ipopt"
license          = "EPL-2.0"
maintainers      = ["github-handle"]
repo             = "madsuite-org/cnlplib"   # where the releases are
nightly          = true
rolling          = true                     # only the nightly release: `cnlp install ipopt` takes it
platforms        = ["linux-x86_64"]
solver_interface = "0.2"
backends         = ["cpu"]                  # where it evaluates the model: cpu, cuda, rocm, oneapi, metal
```

`cnlp install <name>` then puts it in `~/.cnlplib/solvers/<name>`, and
`cnlp solve ... --solver <name>` uses it. `cnlp list` and the registry page show each
solver's backends: `cpu` for `cnlp_solve`, and `cuda` for a solver that exports
`cnlp_solve_device`.
