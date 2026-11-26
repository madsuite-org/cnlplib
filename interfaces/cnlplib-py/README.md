# cnlplib

[![python](https://github.com/madsuite-org/cnlplib/actions/workflows/python.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/python.yml)

```sh
pip install "git+https://github.com/madsuite-org/cnlplib#subdirectory=interfaces/cnlplib-py"
```

A [cnlp library](https://cnlplib.madsuite.org) (a nonlinear program behind the C ABI of
[`cnlp.h`](../../cnlp.h)) as a numpy-backed model in Python, through ctypes: evaluate it,
or solve it with any solver that takes NLP callbacks (Ipopt via `cyipopt`, the `[ipopt]`
extra; `scipy.optimize`).

```python
import cnlplib

m = cnlplib.CModel("@lv", "wood", 10000)                    # the registry's lv library, model wood
m = cnlplib.CModel("@emp", "acp", "pglib_opf_case118_ieee")  # a data-set file, by name
m = cnlplib.CModel("/path/to/libmymodel.so", 1000)          # a path: library file or bundle dir
x, info = cnlplib.solve_ipopt(m, print_level=0)
```

`m` exposes `nvar`, `ncon`, `nnzj`, `nnzh`, `x0`, `lvar`, `uvar`, `lcon`, `ucon`,
`minimize`, and the methods `obj(x)`, `grad(x)`, `cons(x)`, `jac_structure()`, `jac(x)`,
`hess_structure()`, `hess(x, y, obj_weight=1.0)`: numpy in, numpy out, structure indices
0-based.

`@name` is looked up on `CNLPLIB_PATH` (or `cnlplib.set_path(*dirs)`), then among the
libraries the `cnlp` command installed in `$CNLPLIB_HOME` (default `~/.cnlplib`); a name
found nowhere is installed there with `cnlp install`
(`curl -fsSL https://cnlplib.madsuite.org/install | sh`), so one copy serves the `cnlp`
command, Julia and Python. `name@nightly` is the nightly build; `cnlplib.update("lv")`
fetches a newer release. `cnlplib.load(path)` bypasses the lookup.

A library may carry several models, each under its own symbol prefix: a leading string
argument selects one (`"wood"` binds `wood_new`, `wood_obj`, ...); without it the prefix
defaults to the library name, and `prefix=` overrides. The remaining arguments
instantiate the model, one per field of its schema, in the library's order, each checked
against the kind and type the schema declares; a table is a dict of equal-length columns:

```python
m = cnlplib.CModel("@mymodel",
    {"i": np.array([1, 2]), "pd": np.array([0.4, 0.3])},   # a table
    np.array([0.9, 0.9]),                                  # an array
    100.0,                                                 # a scalar
)
```

`available_models(lib)`, `argtype(lib, "wood")` and `schema(lib, "wood")` say what a
library carries and what each model takes. `m.get_vars()`, `m.get_cons()` and
`m.get_pars()` are the model's named blocks, for `solution(x, m.get_vars("pg"))`,
`multipliers`, `multipliers_L`, `multipliers_U`, and `m.get_value` / `m.set_value` on
parameters.

`m.report_instance()` and `m.report_solution(x, y, zl, zu)` are the library's reports (text
about the instance and about a solution; `""` when it has none). With the library's CUDA
build, `CModel(..., backend="cuda")` evaluates on the GPU: `obj`, `grad`, `cons`, `jac` and
`hess` take and return CuPy arrays.

The test suite compiles [`conformance/tinyqp.c`](../../conformance/tinyqp.c), a plain-C
implementation of the ABI, against the repository's `cnlp.h`. Sibling package:
[CNLPLib.jl](../CNLPLib.jl), the same consumer for Julia.
