#!/usr/bin/env python3
"""Check the conformance files against the spec and the example library.

    python3 conformance/check.py        (needs h5py)

Builds example/qp.c and runs example/solve.c, then checks conformance/qp-4: identification
and shapes, that the input instantiates the model, that the solution is the model's
(objective, feasibility, stationarity in the sign convention of formats/solution.md), and
that the example's JSON writer, an override of the HDF5 format, writes the same content.
Exits nonzero on any failure.
"""
import ctypes, json, os, subprocess, sys, tempfile

import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
I64, F64 = ctypes.c_int64, ctypes.c_double
P64 = ctypes.POINTER(F64)
STATUSES = {"solved", "acceptable", "infeasible", "unbounded", "iteration_limit", "time_limit", "error"}
failures, passed = [], 0


def check(cond, what):
    global passed
    if cond:
        passed += 1
    else:
        failures.append(what)
        print("FAIL", what)


def fn(lib, name, res, *args):
    f = getattr(lib, name)
    f.restype, f.argtypes = res, list(args)
    return f


def build(tmp):
    lib, exe, ex = os.path.join(tmp, "libqp.so"), os.path.join(tmp, "solve"), os.path.join(ROOT, "example")
    cc = ["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2"]
    subprocess.run(cc + ["-shared", "-fPIC", "-o", lib, os.path.join(ex, "qp.c"), "-lm"], check=True)
    subprocess.run(cc + ["-o", exe, os.path.join(ex, "solve.c"), "-L" + tmp, "-lqp", "-Wl,-rpath," + tmp, "-lm"], check=True)
    check(subprocess.run([exe], capture_output=True).returncode == 0, "example/solve runs and checks its results")
    return ctypes.CDLL(lib)


def read_h5(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            out[k] = v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)
        for sec in f:
            out[sec] = {}
            for k, ds in f[sec].items():
                v = ds[()]
                out[sec][k] = v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)
    return out


def check_ident(d, fmt, where):
    check(d.get("format") == fmt, f"{where}: format is {fmt}")
    check(d.get("format_version") == "0.2", f"{where}: format_version is 0.2")


def check_solution(lib, d, where):
    check_ident(d, "cnlp-solution", where)
    iid = fn(lib, "qp_new", I64, I64)(4)
    n, m = fn(lib, "qp_nvar", I64, I64)(iid), fn(lib, "qp_ncon", I64, I64)(iid)
    check((d.get("nvar"), d.get("ncon")) == (n, m), f"{where}: nvar, ncon match the model")
    s = d["solution"]
    for k, L in (("solution", n), ("multipliers", m), ("multipliers_L", n), ("multipliers_U", n)):
        check(len(s[k]) == L, f"{where}: {k} has length {L}")
    check(s["status"] in STATUSES, f"{where}: status is solver-neutral")
    check(s["minimize"] in (0, 1), f"{where}: minimize is 0 or 1")
    arr = lambda v: (F64 * len(v))(*v)
    x, y, zl, zu = (arr(s[k]) for k in ("solution", "multipliers", "multipliers_L", "multipliers_U"))
    f = F64(); fn(lib, "qp_obj", I64, I64, P64, P64)(iid, x, ctypes.byref(f))
    check(abs(f.value - s["objective"]) <= 1e-12, f"{where}: objective = f(x)")
    x0, lv, uv, lc, uc = (F64 * n)(), (F64 * n)(), (F64 * n)(), (F64 * m)(), (F64 * m)()
    fn(lib, "qp_meta", I64, I64, *[P64] * 5)(iid, x0, lv, uv, lc, uc)
    c = (F64 * m)(); fn(lib, "qp_cons", I64, I64, P64, P64)(iid, x, c)
    check(all(lc[i] - 1e-12 <= c[i] <= uc[i] + 1e-12 for i in range(m)), f"{where}: constraints hold")
    check(all(lv[i] <= x[i] <= uv[i] for i in range(n)), f"{where}: bounds hold")
    check(min(list(zl) + list(zu)) >= 0, f"{where}: bound multipliers are nonnegative")
    g = (F64 * n)(); fn(lib, "qp_grad", I64, I64, P64, P64)(iid, x, g)
    nnz = fn(lib, "qp_nnzj", I64, I64)(iid)
    rows, cols, vals = (I64 * nnz)(), (I64 * nnz)(), (F64 * nnz)()
    fn(lib, "qp_jac_structure", I64, I64, ctypes.POINTER(I64), ctypes.POINTER(I64))(iid, rows, cols)
    fn(lib, "qp_jac", I64, I64, P64, P64)(iid, x, vals)
    sigma = 1.0 if s["minimize"] else -1.0
    r = [sigma * g[i] - zl[i] + zu[i] for i in range(n)]          # sigma*grad f + J'y - zL + zU
    for k in range(nnz):
        r[cols[k] - 1] += vals[k] * y[rows[k] - 1]
    check(max(abs(v) for v in r) <= 1e-12, f"{where}: stationarity in the cnlp sign convention")
    return iid, (x, y, zl, zu)


def main():
    d = os.path.join(HERE, "qp-4")
    with tempfile.TemporaryDirectory() as tmp:
        lib = build(tmp)
        inp = read_h5(os.path.join(d, "input.h5"))
        check_ident(inp, "cnlp-input", "input.h5")
        check(fn(lib, "qp_new", I64, I64)(inp["input"]["n"]) > 0, "input.h5: instantiates qp")
        sol = read_h5(os.path.join(d, "solution.h5"))
        iid, (x, y, zl, zu) = check_solution(lib, sol, "solution.h5")
        # the override: the library's JSON writer must write the same solution
        path = os.path.join(tmp, "solution.json")
        w = fn(lib, "qp_write_json", I64, I64, ctypes.c_char_p, P64, P64, P64, P64, F64, ctypes.c_char_p)
        check(w(iid, path.encode(), x, y, zl, zu, sol["solution"]["objective"], sol["solution"]["status"].encode()) == 0,
              "qp_write_json writes")
        j = json.load(open(path))
        check_solution(lib, j, "JSON written by qp_write_json")
        check(all(j["solution"][k] == sol["solution"][k] for k in sol["solution"]), "JSON holds the HDF5 solution's values")
    print(f"{passed} passed, {len(failures)} failed")
    sys.exit(1 if failures or passed == 0 else 0)


if __name__ == "__main__":
    main()
