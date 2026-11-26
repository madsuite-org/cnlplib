# cnlplib

[![python](https://github.com/madsuite-org/cnlplib/actions/workflows/python.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/python.yml)

Load a nonlinear program (NLP) exposed by a **shared library through a plain
C interface** and use it as a numpy-backed model in Python — evaluate it, or
solve it with any solver that takes dense/sparse NLP callbacks (Ipopt via
`cyipopt` works out of the box; `scipy.optimize` too).

```sh
pip install "git+https://github.com/madsuite-org/cnlplib#subdirectory=interfaces/cnlplib-py"
```

No language runtime is required beyond Python itself: the wrapper is
ctypes + numpy, and any library implementing the ABI below works, whatever
language produced it. `conformance/tinyqp.c` in this repository is a
complete reference implementation in ~100 lines of plain C, and the test
suite compiles and exercises it end to end.

```python
import cnlplib

cnlplib.set_path("/opt/models")          # or the CNLPLIB_PATH env variable

m = cnlplib.CModel("@mymodel", 1000)     # resolves libmymodel.so
# m = cnlplib.CModel("/path/to/mymodel", 1000)   # or a literal path
x, info = cnlplib.solve_ipopt(m)         # cyipopt, if installed
```

`m` exposes `nvar`, `ncon`, `nnzj`, `nnzh`, `x0`, `lvar`, `uvar`, `lcon`,
`ucon`, and methods `obj(x)`, `grad(x)`, `cons(x)`, `jac_structure()`,
`jac(x)`, `hess_structure()`, `hess(x, y, obj_weight=1.0)` — numpy in,
numpy out, structure indices 0-based per Python convention. Any number of
model instances may coexist per library.

**Libraries from the [cnlplib registry](https://cnlplib.madsuite.org/registry/)** are
installed on first use: a name that is neither on the path nor installed in `$CNLPLIB_HOME`
(default `~/.cnlplib`) is installed there with `cnlp install`, so one copy, from one
installer, serves the `cnlp` command, Julia and Python. This needs the `cnlp` command
(`curl -fsSL https://cnlplib.madsuite.org/install | sh`). `name@nightly` is the nightly
build; `cnlplib.update("lv")` fetches a newer release.

```python
m = cnlplib.CModel("@lv@nightly", "wood", 10000)
m = cnlplib.CModel("@emp@nightly", "acp", "pglib_opf_case118_ieee")   # a data-set file, by name
```

## The C ABI

This package consumes **cnlp ABI 0.2**: [`cnlp.h`](https://github.com/madsuite-org/cnlplib/blob/main/cnlp.h)
is the specification (every integer is `int64_t`, indices are 1-based inside the ABI and
0-based here). The test fixture, `conformance/tinyqp.c`, is compiled against it, so
signature drift fails the build.

`m.minimize` is the objective sense (`P_minimize`; a library without it minimizes).

Libraries built from structured data take their arguments positionally, one per schema
field, in the order the library publishes them; a table is a dict of equal-length columns:

```python
m = cnlplib.CModel("@mymodel",
    {"i": np.array([1, 2]), "pd": np.array([0.4, 0.3])},   # a table (columns)
    np.array([0.9, 0.9]),                                  # an array
    100.0,                                                 # a scalar
)
m = cnlplib.CModel("@scalable", 1000)  # one integer: <prefix>_new when exported
```

Each value is checked against the kind and type the schema declares; nothing is coerced.

### Several models in one library

One shared library may export any number of models, each under its own symbol
prefix with its own schema and its own instances. A leading string argument
selects one by name — the name **is** the prefix its ABI functions are
exported under (unambiguous, since a model argument is never a string):

```python
m = cnlplib.CModel("@grid", "acopf", bus, 100.0)  # acopf_* inside libgrid.so
d = cnlplib.CModel("@grid", "dcopf", bus)         # dcopf_* in the same file
sch = cnlplib.schema(lib, "acopf")                # schemas are per model
```

A mistyped name is refused at selection, with the witness symbol named, rather
than surfacing as a raw `undefined symbol` several calls later. Omitting the
name keeps the single-model spelling, where the prefix falls back to the
library name — a one-model library is unaffected. This is the same selection
spelling as CNLPLib.jl's `CNLPModel(lib, :acopf, ...)`.

## Implementing a compatible library

Follow `cnlp.h`, and validate against `conformance/tinyqp.c` and this test suite,
which compiles that file and checks every function against closed-form values.

Sibling package: [CNLPLib.jl](../CNLPLib.jl) — the same consumer for Julia.
