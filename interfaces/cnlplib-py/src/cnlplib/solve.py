"""Solve a model with a cnlp solver, through the `cnlp solve` command.

The solver is any one installed in the store (`cnlp install madnlp`, `cnlp install ipopt`),
on the CPU or, when the model's library has a CUDA build and the solver runs there, on the
GPU. It runs in its own process, so its runtime never meets this one; the solution comes
back through the cnlp-solution file `cnlp solve` writes (h5py reads it).
"""
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np

from . import registry

__all__ = ["Result", "solve"]


@dataclass
class Result:
    """A solve: the cnlp status (`solved`, `acceptable`, `infeasible`, ...), the objective,
    the iteration count, the solution and multipliers (cnlp sign convention), the solver's
    own status, where the time went (`cnlp solve`'s `times:` line, in seconds) and its output."""
    status: str
    objective: float
    iterations: int
    solution: np.ndarray
    multipliers: np.ndarray
    multipliers_L: np.ndarray
    multipliers_U: np.ndarray
    solver: str
    solver_status: str
    device: str
    times: dict = field(default_factory=dict)
    output: str = ""


def _cnlp_solve():
    c = os.path.join(registry._home(), "bin", "cnlp-solve")
    if os.path.isfile(c):
        return [c]
    return [registry._cnlp(), "solve"]


def _cli_arg(a):
    if isinstance(a, bool) or not isinstance(a, (int, float, str, np.integer, np.floating)):
        raise TypeError(f"{type(a).__name__} cannot be given to `cnlp solve`: give numbers or a file")
    return str(a)


def solve(model, *args, solver="ipopt", device="auto", start=None, outputs=(), **options):
    """Solve with the cnlp solver `solver` (a name installed in the store, or a path to its
    library), through `cnlp solve`. `model` is a `CModel` (built from numbers or a file) or a
    spec as `cnlp solve` takes it (`"@emp:acp"`, with `args` the instance). `device` is
    `"auto"` (the GPU when the library and the solver both run there), `"cpu"`, `"cuda"` or
    `"cuda:N"`; `start` is `reader:path` or a cnlp-solution file; `outputs` are written as with
    `-o`; other keywords are solver options (`tol=1e-8`, `max_iter=100`). Returns a `Result`.

        r = cnlplib.solve(cnlplib.CModel("@emp", "acp", "pglib_opf_case118_ieee"), solver="madnlp", device="cuda")
        r = cnlplib.solve("@lv:wood", 10000, solver="madnlp")
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("cnlplib.solve reads the solution with h5py: pip install h5py") from None
    if hasattr(model, "_cli"):
        if args:
            raise TypeError("a CModel carries its instance; give no more arguments")
        spec, args = model._cli
    else:
        spec = str(model)
    with tempfile.TemporaryDirectory(prefix="cnlplib-") as tmp:
        sol = os.path.join(tmp, "solution.h5")
        cmd = _cnlp_solve() + [spec, *map(_cli_arg, args), "--solver", solver, "--device", device,
                               "-o", f"cnlp:{sol}"]
        if start is not None:
            cmd += ["-s", str(start)]
        for o in outputs:
            cmd += ["-o", str(o)]
        cmd += [f"{k}={v}" for k, v in options.items()]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = p.stdout + p.stderr
        if not os.path.isfile(sol):
            raise RuntimeError(f"`cnlp solve` failed (exit {p.returncode}):\n{out.strip()}")
        with h5py.File(sol, "r") as f:
            g = f["solution"]
            s = lambda k: g[k][()].decode() if isinstance(g[k][()], bytes) else str(g[k][()])
            res = Result(
                status=s("status"), objective=float(g["objective"][()]), iterations=int(g["iterations"][()]),
                solution=g["solution"][()], multipliers=g["multipliers"][()],
                multipliers_L=g["multipliers_L"][()], multipliers_U=g["multipliers_U"][()],
                solver=s("solver"), solver_status=s("solver_status"), device="cpu", output=out)
    m = re.search(r"\((cpu|cuda)\)", out)
    if m:
        res.device = m[1]
    m = re.search(r"^times: (.*)$", out, re.MULTILINE)
    if m:
        res.times = {k.strip(): float(v) for k, v in re.findall(r"([a-z ]+?) ([0-9.]+) s", m[1])}
    return res
