#!/usr/bin/env python3
"""Check a cnlp solver: solve the example QP and its maximization twin with it.

    python3 conformance/check_solver.py SOLVER [CNLP_SOLVE]     (needs h5py)

SOLVER is a solver name installed with `cnlp install`, or a path to a solver library;
CNLP_SOLVE the driver (default: `cnlp-solve` on PATH). Each solution file is checked as in
check.py (objective, feasibility, stationarity in the cnlp sign convention), and against
the known solution x = (0.5, 0.5, 1, 1), y = 1, which the maximization shares.
"""
import ctypes, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check  # noqa: E402

solver = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
driver = sys.argv[2] if len(sys.argv) > 2 else (shutil.which("cnlp-solve") or "cnlp-solve")
with tempfile.TemporaryDirectory() as tmp:
    for name, src, f in (("min", "../example/qp.c", 0.5), ("max", "qpmax.c", -0.5)):
        lib = os.path.join(tmp, f"libqp_{name}.so")
        subprocess.run(["cc", "-std=c99", "-O2", "-shared", "-fPIC", "-o", lib, os.path.join(HERE, src), "-lm"], check=True)
        out = os.path.join(tmp, f"{name}.h5")
        r = subprocess.run([driver, f"{lib}:qp", "4", "--solver", solver, "-o", out, "print_level=0", "sb=yes"],
                           capture_output=True, text=True)
        print(r.stdout.strip())
        check.check(r.returncode == 0, f"{name}: the solve succeeds ({r.stderr.strip()})")
        if r.returncode:
            continue
        d = check.read_h5(out)
        check.check_solution(ctypes.CDLL(lib), d, f"{name}")
        s = d["solution"]
        check.check(s["minimize"] == (name == "min"), f"{name}: the sense is recorded")
        check.check(max(abs(a - b) for a, b in zip(s["solution"], [0.5, 0.5, 1, 1])) < 1e-6, f"{name}: x is the solution")
        check.check(abs(s["multipliers"][0] - 1) < 1e-6, f"{name}: y = 1 (cnlp convention)")
        check.check(abs(s["objective"] - f) < 1e-8, f"{name}: objective {f}")
print(f"{check.passed} passed, {len(check.failures)} failed")
sys.exit(1 if check.failures else 0)
