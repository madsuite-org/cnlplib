"""A cnlp library (cnlp ABI 0.2, ``cnlp.h``) as a numpy-backed model, through ctypes.

``CModel`` exposes ``nvar``, ``ncon``, ``nnzj``, ``nnzh``, ``x0``, ``lvar``, ``uvar``,
``lcon``, ``ucon``, ``minimize`` and the methods ``obj(x)``, ``grad(x)``, ``cons(x)``,
``jac_structure()``, ``jac(x)``, ``hess_structure()``, ``hess(x, y, obj_weight=1.0)``:
numpy in, numpy out, structure indices 0-based (the ABI's are 1-based).
"""
import ctypes
import os

import numpy as np

__all__ = [
    "BlockRef",
    "CModel",
    "Result",
    "Solver",
    "argtype",
    "available_models",
    "data_file",
    "install",
    "lib",
    "load",
    "load_solver",
    "multipliers",
    "multipliers_L",
    "multipliers_U",
    "schema",
    "set_path",
    "solution",
    "solve",
    "update",
]

from . import registry as _registry
from .registry import data_file, install, update
from .solve import Result, Solver, load_solver, solve

_c_int = ctypes.c_int64
_c_dbl = ctypes.c_double
_pd = ctypes.POINTER(_c_dbl)
_pi = ctypes.POINTER(_c_int)
_pu8 = ctypes.POINTER(ctypes.c_uint8)
_PTR = {"f64": _pd, "i64": ctypes.POINTER(ctypes.c_longlong)}


_PATHS = []
_LIBS = {}


def set_path(*dirs):
    """Set the library search path for name-based loading (`lib("acopf")`,
    `CModel("@acopf", ...)`). Initialized from the colon-separated `CNLPLIB_PATH`
    environment variable; calling this replaces it."""
    _PATHS[:] = [os.fspath(d) for d in dirs]
    _LIBS.clear()
    return list(_PATHS)


def _paths():
    if not _PATHS:
        env = os.environ.get("CNLPLIB_PATH", "")
        if env:
            _PATHS[:] = env.split(":")
    return _PATHS


def lib(name):
    """The library `name`, loaded and cached by name: `lib<name>.so` on the search path
    (also under `<dir>/<name>/lib/` and `<dir>/lib/`), else among the libraries the `cnlp`
    command installed (`$CNLPLIB_HOME/lib`), else installed there now from the registry
    (`install`). `name@nightly` takes the registry's nightly build."""
    if name not in _LIBS:
        base = name.partition("@")[0]
        fname = f"lib{base}{_registry._ext()}"
        if base == name:
            for d in _paths():
                for cand in (os.path.join(d, fname),
                             os.path.join(d, name, "lib", fname),
                             os.path.join(d, "lib", fname)):
                    if os.path.isfile(cand):
                        _LIBS[name] = load(cand)
                        return _LIBS[name]
        _LIBS[name] = load(_registry.registry_lib(name))
    return _LIBS[name]


def load(path):
    """dlopen the shared library (RTLD_LOCAL | RTLD_DEEPBIND, so its bundled
    runtime does not leak symbols into, or take them from, the process)."""
    flags = os.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)
    return ctypes.CDLL(os.fspath(path), mode=flags)


# `@name` is a library name; any other string is a filesystem path, as written.
def _is_name(spec):
    return spec.startswith("@")


# `librosen.so` → `rosen`; a bundle directory or another file name keeps its stem.
def _default_prefix(spec):
    if _is_name(spec):
        return spec[1:].partition("@")[0]
    base = os.path.splitext(os.path.basename(spec.rstrip("/")))[0]
    return base[3:] if base.startswith("lib") and len(base) > 3 else base


# A shared library, or a bundle directory with the library inside. Absolute, since dlopen
# treats a slash-free relative path like `qp.so` as a soname to search for.
def _resolve_path(spec):
    if os.path.isfile(spec):
        return os.path.abspath(spec)
    if os.path.isdir(spec):
        fname = f"lib{os.path.basename(spec.rstrip('/'))}{_registry._ext()}"
        for cand in (os.path.join(spec, "lib", fname), os.path.join(spec, fname)):
            if os.path.isfile(cand):
                return os.path.abspath(cand)
        raise FileNotFoundError(
            f"no shared library in {spec} (tried lib/{fname} and {fname})")
    raise FileNotFoundError(f"no shared library at {spec}")


# Cached with the name-resolved libraries: absolute paths cannot collide with names.
def _resolve_spec(spec):
    if _is_name(spec):
        return lib(spec[1:])
    key = os.path.abspath(spec)
    if key not in _LIBS:
        _LIBS[key] = load(_resolve_path(spec))
    return _LIBS[key]


def _sym(lib, name, argtypes):
    """The library's function `name`, typed (int64 return). AttributeError if absent."""
    fn = getattr(lib, name)
    fn.restype = _c_int
    fn.argtypes = argtypes
    return fn


def _f(lib, prefix, name, argtypes):
    return _sym(lib, f"{prefix}_{name}", argtypes)


def _copyout(fn, *args):
    """The ABI's string convention: call once with a null buffer to size, once to fill.
    `fn` is typed with `_pu8, _c_int` after `args`."""
    need = int(fn(*args, ctypes.cast(0, _pu8), 0))
    if need <= 0:
        return ""
    buf = (ctypes.c_uint8 * need)()
    fn(*args, buf, need)
    return bytes(buf).decode()


def _arr(a):
    return a.ctypes.data_as(_pd)


def _iarr(a):
    return a.ctypes.data_as(_pi)


def _check(st, what):
    if st != 0:
        raise RuntimeError(f"{what} returned nonzero status {st}")


def _carries_model(lib, name):
    """Whether `lib` exports a model under this name (`<name>_nvar`, which the ABI
    requires of every model however it is instantiated)."""
    return hasattr(lib, f"{name}_nvar")


def _takes_string(lib, prefix):
    """Whether the model under `prefix` instantiates from a string, per its signature."""
    return argtype(lib, prefix).split("|")[0].split(",")[0] == "string"


def _require_model(lib, model):
    if not _carries_model(lib, model):
        where = getattr(lib, "_name", "this library")
        raise ValueError(
            f"{where} carries no model named {model!r} (it exports no {model}_nvar)")


def schema(lib, model=None, *, prefix="rec"):
    """The model's data schema (`<prefix>_schema`), as a dict. In a library carrying
    several models name the one you want, `schema(lib, "acopf")`, as in `CModel`."""
    import json
    if model is not None:
        _require_model(lib, model)
        prefix = model
    return json.loads(_copyout(_f(lib, prefix, "schema", [_pu8, _c_int])))


# The schema says what a slot holds; a value of the wrong kind is refused rather than
# coerced (`int(2.0)` would silently build a different model from the one asked for).
def _as_scalar(name, v, kind):
    if isinstance(v, (bool, np.bool_)):
        raise TypeError(f"field {name!r}: a bool is not a model argument")
    if kind == "i64":
        if not isinstance(v, (int, np.integer)):
            raise TypeError(f"field {name!r} is an int64 in the library's schema, "
                            f"got {type(v).__name__}")
        return int(v)
    if not isinstance(v, (int, float, np.integer, np.floating)):
        raise TypeError(f"field {name!r} is a float64 in the library's schema, "
                        f"got {type(v).__name__}")
    return float(v)


def _as_col(v, kind, name="value"):
    a = np.asarray(v)
    if kind == "i64" and not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"{name} is an int64 array in the library's schema, "
                        f"got dtype {a.dtype}")
    if kind == "f64" and not np.issubdtype(a.dtype, np.number):
        raise TypeError(f"{name} is a float64 array in the library's schema, "
                        f"got dtype {a.dtype}")
    a = np.ascontiguousarray(a, dtype=np.float64 if kind == "f64" else np.int64)
    if a.ndim != 1:
        raise ValueError("columns and arrays must be one-dimensional")
    return a


def _fill_data(lib, prefix, data):
    """Fill a builder from a dict: scalars are numbers, arrays are numpy arrays, tables are
    dicts of columns. Validated against the library's schema."""
    fields = {f["name"]: f for f in schema(lib, prefix=prefix)["fields"]}
    unknown = set(data) - set(fields)
    missing = set(fields) - set(data)
    if unknown or missing:
        raise ValueError(f"data keys do not match schema: unknown={sorted(unknown)}, "
                         f"missing={sorted(missing)}")
    b = _f(lib, prefix, "data_begin", [])()
    if b <= 0:
        raise RuntimeError("data_begin failed")
    for name, spec in fields.items():
        v, t = data[name], spec.get("type")
        if spec["kind"] == "scalar":
            setter = f"set_scalar_{t}"
            ctype = _c_dbl if t == "f64" else ctypes.c_longlong
            _check(_f(lib, prefix, setter, [_c_int, ctypes.c_char_p, ctype])(
                b, name.encode(), _as_scalar(name, v, t)), f"{setter}({name})")
        elif spec["kind"] == "array":
            a = _as_col(v, t, name)
            setter = f"set_array_{t}"
            _check(_f(lib, prefix, setter, [_c_int, ctypes.c_char_p, _PTR[t], _c_int])(
                b, name.encode(), a.ctypes.data_as(_PTR[t]), len(a)), f"{setter}({name})")
        else:  # table
            cols = {c["name"]: c for c in spec["columns"]}
            if set(v) != set(cols):
                raise ValueError(f"table {name!r}: columns {sorted(v)} != schema {sorted(cols)}")
            lens = {len(np.asarray(col)) for col in v.values()}
            if len(lens) != 1:
                raise ValueError(f"table {name!r}: column lengths differ: {lens}")
            for cname, cspec in cols.items():
                ct = cspec["type"]
                a = _as_col(v[cname], ct, f"{name}.{cname}")
                setter = f"set_col_{ct}"
                _check(_f(lib, prefix, setter,
                          [_c_int, ctypes.c_char_p, ctypes.c_char_p, _PTR[ct], _c_int])(
                    b, name.encode(), cname.encode(), a.ctypes.data_as(_PTR[ct]), len(a)),
                    f"{setter}({name}.{cname})")
    if _f(lib, prefix, "data_ready", [_c_int])(b) != 1:
        raise RuntimeError("library reports data incomplete after all fields were set")
    mid = _f(lib, prefix, "new_from_data", [_c_int])(b)
    if mid <= 0:
        raise RuntimeError("new_from_data failed")
    return mid


def _bind(lib, prefix, args):
    """Positional arguments against the schema's field order, checked for arity here
    rather than by the library several calls later."""
    if not hasattr(lib, f"{prefix}_data_begin"):
        raise RuntimeError(
            "this library has no builder surface: it instantiates from a "
            "single integer, CModel(lib, n)")
    try:
        sch = schema(lib, prefix=prefix)
    except AttributeError:
        raise RuntimeError(
            f"{prefix}_data_begin exists but {prefix}_schema does not, so "
            "there is nothing to bind the arguments against") from None
    names = [f["name"] for f in sch["fields"]]
    if len(names) != len(args):
        raise ValueError(
            f"given {len(args)} argument{'' if len(args) == 1 else 's'} but the "
            f"library's schema declares {len(names)} "
            f"field{'' if len(names) == 1 else 's'}: {names}")
    return dict(zip(names, args))


def _instantiate(lib, prefix, args):
    """Positional arguments -> model id. One value per schema field, in the library's
    order. A lone string goes to `<prefix>_new_str`, a lone integer to `<prefix>_new` when
    the library exports it, no arguments to `<prefix>_new(0)` when `<prefix>_nargs()` is 0;
    everything else goes through the builder."""
    if any(isinstance(a, bool) for a in args):
        raise TypeError("a bool is not a model argument")
    if len(args) == 1 and isinstance(args[0], str):
        try:
            fn = _f(lib, prefix, "new_str", [ctypes.c_char_p])
        except AttributeError:
            raise ValueError(
                f"{prefix!r} has no string entry point ({prefix}_new_str); "
                f"its signature is {argtype(lib, prefix)!r}") from None
        # a data-set file by name, as `cnlp solve` takes one; any other string as given
        mid = int(fn(_registry.string_arg(args[0]).encode()))
        if mid <= 0:
            raise RuntimeError(f"{prefix}_new_str returned {mid}")
        return mid
    if len(args) == 1 and isinstance(args[0], (int, np.integer)):
        try:
            new = _f(lib, prefix, "new", [_c_int])
        except AttributeError:
            pass
        else:
            mid = new(_c_int(int(args[0])))
            if mid <= 0:
                raise RuntimeError(f"{prefix}_new({args[0]}) failed (returned {mid})")
            return mid
    if not args:
        try:
            nargs = _f(lib, prefix, "nargs", [])
        except AttributeError:
            pass
        else:
            if int(nargs()) == 0:
                mid = _f(lib, prefix, "new", [_c_int])(_c_int(0))
                if mid <= 0:
                    raise RuntimeError(f"{prefix}_new(0) failed (returned {mid})")
                return mid
    return _fill_data(lib, prefix, _bind(lib, prefix, args))


def argtype(library, model):
    """What the model instantiates from: the types its entry point takes, in order, each
    optionally followed by `|` and a description; `""` for a model that takes nothing or a
    library that publishes no signature.

        cnlplib.argtype("@grid", "acopf")
        # "int|arg1,Vector{f64}|v0,Table{i::int pd::f64}|bus"
    """
    lib_ = _resolve_spec(library) if isinstance(library, str) else library
    try:
        fp = _f(lib_, model, "argtype", [_pu8, _c_int])
    except AttributeError:
        return ""
    return _copyout(fp)


def available_models(library):
    """The models the library carries, by name (`cnlp_nmodels`, `cnlp_model_name`); empty
    for a library that publishes no catalogue."""
    lib_ = _resolve_spec(library) if isinstance(library, str) else library
    try:
        n = int(_sym(lib_, "cnlp_nmodels", [])())
    except AttributeError:
        return []
    if n <= 0:
        return []
    npf = _sym(lib_, "cnlp_model_name", [_c_int, _pu8, _c_int])
    return [_copyout(npf, k) for k in range(n)]


class BlockRef:
    """A named block of a model (a variable, constraint or parameter): `offset` (0-based)
    and `length` of its slice, `dims` its shape, `index` the library's own numbering, used
    by `CModel.get_value` / `CModel.set_value`."""

    __slots__ = ("dims", "index", "kind", "length", "name", "offset")

    def __init__(self, name, kind, offset, length, dims, index):
        self.name, self.kind = name, kind
        self.offset, self.length, self.dims, self.index = offset, length, dims, index

    def __repr__(self):
        return (f"BlockRef({self.name!r}, {self.kind}, "
                f"{'×'.join(map(str, self.dims))} at {self.offset})")


def _shaped(v, b):
    return v.reshape(b.dims) if len(b.dims) > 1 else v


def _slice(vec, b):
    return _shaped(np.asarray(vec)[b.offset:(b.offset + b.length)], b)


def solution(x, block):
    """The part of solution vector `x` belonging to variable `block`, reshaped
    to the block's own dimensions."""
    return _slice(x, block)


def multipliers(y, block):
    """The dual variables in `y` for constraint `block`."""
    return _slice(y, block)


def multipliers_L(z, block):
    """The lower-bound dual variables in `z` for variable `block`."""
    return _slice(z, block)


def multipliers_U(z, block):
    """The upper-bound dual variables in `z` for variable `block`."""
    return _slice(z, block)


def _read_layout(lib, prefix, mid):
    """The library's published named blocks, or empty dicts when it publishes none
    (named blocks are an optional tier)."""
    try:
        nb = _f(lib, prefix, "nblocks", [_c_int])
    except AttributeError:
        return {}, {}, {}
    n = int(nb(_c_int(mid)))
    if n <= 0:
        return {}, {}, {}
    bp = _f(lib, prefix, "block", [_c_int, _c_int, _pi])
    npf = _f(lib, prefix, "block_name", [_c_int, _c_int, _pu8, _c_int])
    blocks = {"var": {}, "con": {}, "par": {}}
    for k in range(n):
        out = np.zeros(12, dtype=np.int64)
        _check(bp(_c_int(mid), _c_int(k), _iarr(out)), f"{prefix}_block")
        b = BlockRef(_copyout(npf, mid, k),
                     ("var", "con", "par")[int(out[0])],
                     int(out[1]), int(out[2]),
                     tuple(int(out[4 + i]) for i in range(int(out[3]))),
                     k)
        blocks[b.kind][b.name] = b
    return blocks["var"], blocks["con"], blocks["par"]


class CModel:
    """A model instance of a library.

        lib = cnlplib.load("liblv.so")
        m = cnlplib.CModel(lib, 1000, prefix="lv")          # lv_new(1000)
        m = cnlplib.CModel("@acopf", bus, vmin, 100.0)      # search path or registry
        m = cnlplib.CModel("/opt/models/rosen", 1000)       # a path: library file or bundle dir
        m = cnlplib.CModel("@grid", "acopf", bus, 100.0)    # the model acopf of libgrid.so

    In a library carrying several models a leading string selects one: the name is the
    symbol prefix its functions are exported under (as CNLPLib.jl's
    `CNLPModel(lib, :acopf, ...)`). A model's own argument can also be a string (a case
    file), so the library is asked: a name it exports selects, anything else is an
    argument, and an explicit `prefix=` settles selection.

    The remaining arguments instantiate the model: one value per field of its schema, in
    the library's order, each a number, an array, or a table (a dict of equal-length
    columns), checked against the kind and type the schema declares. A lone integer or
    string goes to the model's own entry point when it has one; no arguments instantiate a
    model that takes none. Any number of instances may coexist per library.

    `backend="cuda"` evaluates through the library's device tier (its CUDA build): `obj`,
    `grad`, `cons`, `jac` and `hess` take and return CuPy arrays. The bounds, starting point
    and structures stay NumPy arrays on the host.
    """

    def __init__(self, lib, *args, prefix=None, backend=None):
        spec_prefix = None
        if isinstance(lib, str):
            spec_prefix = _default_prefix(lib)
            lib = _resolve_spec(lib)
        model = None
        if (args and isinstance(args[0], str)
                and (prefix is None or args[0] == prefix)):
            if _carries_model(lib, args[0]):
                model, args = args[0], args[1:]
                prefix = model
            elif not _takes_string(lib, prefix or spec_prefix or "rec"):
                # neither a model this library has nor an argument the selected model
                # could take: the name is what the caller got wrong
                _require_model(lib, args[0])
        if prefix is None:
            prefix = spec_prefix if spec_prefix is not None else "rec"
        if model is not None:
            _require_model(lib, model)
        # instantiate before resolving the evaluation table, so a failure to build the
        # model surfaces as what it is
        self._id = _instantiate(lib, prefix, args)
        self._prefix = prefix
        self._lib = lib
        self._vars, self._cons, self._pars = _read_layout(lib, prefix, self._id)
        self._fn = {
            name: _f(lib, prefix, name, argtypes)
            for name, argtypes in (
                ("nvar", [_c_int]), ("ncon", [_c_int]),
                ("nnzj", [_c_int]), ("nnzh", [_c_int]),
                ("meta", [_c_int, _pd, _pd, _pd, _pd, _pd]),
                ("obj", [_c_int, _pd, _pd]),
                ("grad", [_c_int, _pd, _pd]),
                ("cons", [_c_int, _pd, _pd]),
                ("jac_structure", [_c_int, _pi, _pi]),
                ("jac", [_c_int, _pd, _pd]),
                ("hess_structure", [_c_int, _pi, _pi]),
                ("hess", [_c_int, _pd, _pd, _c_dbl, _pd]),
            )
        }
        self.nvar = int(self._fn["nvar"](self._id))
        self.ncon = int(self._fn["ncon"](self._id))
        self.nnzj = int(self._fn["nnzj"](self._id))
        self.nnzh = int(self._fn["nnzh"](self._id))
        self.x0 = np.zeros(self.nvar)
        self.lvar = np.zeros(self.nvar)
        self.uvar = np.zeros(self.nvar)
        self.lcon = np.zeros(self.ncon)
        self.ucon = np.zeros(self.ncon)
        _check(self._fn["meta"](self._id, _arr(self.x0), _arr(self.lvar),
                                _arr(self.uvar), _arr(self.lcon), _arr(self.ucon)),
               "meta")
        self.backend = backend
        if backend is not None:
            self._bind_device(backend)
        # the objective sense is optional in the ABI: a library without it minimizes
        try:
            mn = _f(lib, prefix, "minimize", [_c_int])
        except AttributeError:
            self.minimize = True
        else:
            self.minimize = int(mn(self._id)) != 0

    def _bind_device(self, backend):
        if backend != "cuda":
            raise ValueError(f"backend is None (the host) or 'cuda', not {backend!r}")
        try:
            import cupy
        except ImportError:
            raise ImportError("backend='cuda' needs CuPy (pip install cupy-cuda12x)") from None
        try:
            dev = _f(self._lib, self._prefix, "device", [_c_int])
        except AttributeError:
            raise ValueError(f"{self._prefix!r} has no device tier: install the library's "
                             "CUDA build") from None
        if int(dev(self._id)) != 1:
            raise ValueError(f"{self._prefix}_device reports no CUDA memory space")
        self._xp = cupy
        for name, argtypes in (
            ("obj", [_c_int, ctypes.c_void_p, _pd]),
            ("grad", [_c_int, ctypes.c_void_p, ctypes.c_void_p]),
            ("cons", [_c_int, ctypes.c_void_p, ctypes.c_void_p]),
            ("jac", [_c_int, ctypes.c_void_p, ctypes.c_void_p]),
            ("hess", [_c_int, ctypes.c_void_p, ctypes.c_void_p, _c_dbl, ctypes.c_void_p]),
        ):
            self._fn[name] = _f(self._lib, self._prefix, name + "_d", argtypes)

    def _x(self, x):
        if self.backend is not None:
            x = self._xp.ascontiguousarray(x, dtype=self._xp.float64)
        else:
            x = np.ascontiguousarray(x, dtype=np.float64)
        if x.shape != (self.nvar,):
            raise ValueError(f"x must have shape ({self.nvar},), got {x.shape}")
        return x

    # the pointer a callback takes, and a new output array, where the model evaluates
    def _p(self, a):
        if self.backend is None:
            return _arr(a)
        return ctypes.c_void_p(a.data.ptr)

    def _out(self, n, out):
        if out is not None:
            return out
        return self._xp.zeros(n) if self.backend is not None else np.zeros(n)

    def _sync(self):
        # the library works on the legacy default stream; finish this thread's stream first
        if self.backend is not None:
            self._xp.cuda.get_current_stream().synchronize()

    def obj(self, x):
        out = _c_dbl(0.0)
        x = self._x(x)
        self._sync()
        _check(self._fn["obj"](self._id, self._p(x), ctypes.byref(out)), "obj")
        return out.value

    def grad(self, x, out=None):
        x, g = self._x(x), self._out(self.nvar, out)
        self._sync()
        _check(self._fn["grad"](self._id, self._p(x), self._p(g)), "grad")
        return g

    def cons(self, x, out=None):
        x, c = self._x(x), self._out(self.ncon, out)
        self._sync()
        _check(self._fn["cons"](self._id, self._p(x), self._p(c)), "cons")
        return c

    def _structure(self, name, nnz):
        rows = np.zeros(nnz, dtype=np.int64)
        cols = np.zeros(nnz, dtype=np.int64)
        _check(self._fn[name](self._id, _iarr(rows), _iarr(cols)), name)
        return rows - 1, cols - 1        # 0-based, per Python convention

    def jac_structure(self):
        return self._structure("jac_structure", self.nnzj)

    def jac(self, x, out=None):
        x, v = self._x(x), self._out(self.nnzj, out)
        self._sync()
        _check(self._fn["jac"](self._id, self._p(x), self._p(v)), "jac")
        return v

    def hess_structure(self):
        return self._structure("hess_structure", self.nnzh)

    def hess(self, x, y, obj_weight=1.0, out=None):
        xp = self._xp if self.backend is not None else np
        y = xp.ascontiguousarray(y, dtype=xp.float64)
        if y.shape != (self.ncon,):
            raise ValueError(f"y must have shape ({self.ncon},), got {y.shape}")
        x, v = self._x(x), self._out(self.nnzh, out)
        self._sync()
        _check(self._fn["hess"](self._id, self._p(x), self._p(y),
                                _c_dbl(float(obj_weight)), self._p(v)), "hess")
        return v

    # ── Reports (cnlp.h, the reports tier) ─────────────────────────────────────

    def report_instance(self):
        """The library's text about this instance (what it was built from, its size), or
        "" when the library has none."""
        try:
            fn = _f(self._lib, self._prefix, "report_instance", [_c_int, _pu8, _c_int])
        except AttributeError:
            return ""
        return _copyout(fn, self._id)

    def report_solution(self, x, y=None, zl=None, zu=None, objective=None, status=""):
        """The library's text about a solution (for an optimal power flow, the generation
        cost and the constraint violations by group), or "" when the library has none.
        Host arrays; multipliers in the cnlp sign convention (formats/solution.md), zeros
        when not given; `objective` defaults to f(x)."""
        try:
            fn = _f(self._lib, self._prefix, "report_solution",
                    [_c_int, _pd, _pd, _pd, _pd, _c_dbl, ctypes.c_char_p, _pu8, _c_int])
        except AttributeError:
            return ""
        host = lambda v, n: np.zeros(n) if v is None else np.ascontiguousarray(
            v.get() if hasattr(v, "get") else v, dtype=np.float64)
        x = host(x, self.nvar)
        y, zl, zu = host(y, self.ncon), host(zl, self.nvar), host(zu, self.nvar)
        if objective is None:
            objective = self.obj(x if self.backend is None else self._xp.asarray(x))
        return _copyout(fn, self._id, _arr(x), _arr(y), _arr(zl), _arr(zu),
                        _c_dbl(float(objective)), str(status).encode())

    # ── Named blocks ─────────────────────────────────────────────────────────

    def get_vars(self, name=None):
        """The model's named variable blocks as a dict of `BlockRef`s, or one by name;
        empty for a library that publishes none.

            m = cnlplib.CModel("@grid", "acopf", bus)
            m.get_vars()                       # {"pg": BlockRef, ...}
            cnlplib.solution(x, m.get_vars("pg"))
        """
        return self._named("var", name)

    def get_cons(self, name=None):
        """The model's named constraint blocks; see `get_vars`."""
        return self._named("con", name)

    def get_pars(self, name=None):
        """The model's named parameter blocks; see `get_vars`."""
        return self._named("par", name)

    def _named(self, kind, name):
        d = {"var": self._vars, "con": self._cons, "par": self._pars}[kind]
        if name is None:
            return dict(d)
        if name in d:
            return d[name]
        # a name of another kind names the accessor that would work; a typo lists what exists
        for k, other in (("var", self._vars), ("con", self._cons), ("par", self._pars)):
            if name in other:
                what = {"var": "variable (get_vars)", "con": "constraint (get_cons)",
                        "par": "parameter (get_pars)"}[k]
                raise ValueError(f"{name!r} is a {what}, not a {kind}")
        what = {"var": "variable", "con": "constraint", "par": "parameter"}[kind]
        raise ValueError(
            f"this model has no named {what} {name!r}; it has {sorted(d)}"
            + ("" if d else " (none: this library publishes no named blocks)"))

    def get_value(self, block):
        """A copy of a parameter block's values (a `BlockRef`, or its name)."""
        b = self._par_block(block)
        out = np.zeros(b.length)
        fn = _f(self._lib, self._prefix, "get_value", [_c_int, _c_int, _pd, _c_int])
        _check(fn(self._id, _c_int(b.index), _arr(out), _c_int(b.length)),
               f"{self._prefix}_get_value")
        return _shaped(out, b)

    def set_value(self, block, values):
        """Update a parameter block's values, for every later evaluation of this instance."""
        b = self._par_block(block)
        v = np.ascontiguousarray(np.ravel(values), dtype=np.float64)
        if v.size != b.length:
            raise ValueError(
                f"parameter {b.name!r} has {b.length} elements, got {v.size}")
        fn = _f(self._lib, self._prefix, "set_value", [_c_int, _c_int, _pd, _c_int])
        _check(fn(self._id, _c_int(b.index), _arr(v), _c_int(b.length)),
               f"{self._prefix}_set_value")
        return self

    def _par_block(self, block):
        b = self.get_pars(block) if isinstance(block, str) else block
        if b.kind != "par":
            raise ValueError(f"{b.name!r} is a {b.kind}, not a parameter")
        return b
