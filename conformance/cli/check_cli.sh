#!/bin/sh
# End-to-end check of the cnlp command, as a user meets it: the installer, `cnlp install`
# (a library that brings a data set, and a solver), every way `cnlp solve` takes an
# instance, a start and an output, `cnlp up` and `cnlp remove`. Everything is served from
# local files laid out as the releases are; the answers are checked against the closed form
# of the test models (testlib.c).
#
#   conformance/cli/check_cli.sh DIST PYTHON
#
# DIST holds cnlp-solve-linux-x86_64.tar.gz and ipopt-linux-x86_64.tar.gz; PYTHON has h5py.
set -eu
DIST=$(cd "$1" && pwd) PY=$2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
PLAT=linux-x86_64
pass=0; fail=0
ok()  { pass=$((pass + 1)); echo "ok    $*"; }
bad() { fail=$((fail + 1)); echo "FAIL  $*"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

# ── releases, as GitHub serves them ─────────────────────────────────────────────
R=$W/releases; mkdir -p "$R/cnlplib" "$R/cnlptest" "$R/data"
cp "$DIST/cnlp-solve-$PLAT.tar.gz" "$DIST/ipopt-$PLAT.tar.gz" "$R/cnlplib/"
(cd "$R/cnlplib" && sha256sum -- *.tar.gz > SHA256SUMS)
mkdir -p "$W/b/cnlptest/lib"
cc -std=c99 -O2 -shared -fPIC -o "$W/b/cnlptest/lib/libcnlptest.so" "$ROOT/conformance/cli/testlib.c" -lm
printf 'name = cnlptest\nversion = 0.0.1\ncnlp_abi = 0.2\n' > "$W/b/cnlptest/VERSION"
tar -C "$W/b" -czf "$R/cnlptest/cnlptest-$PLAT.tar.gz" cnlptest
(cd "$R/cnlptest" && sha256sum -- *.tar.gz > SHA256SUMS)
mkdir -p "$W/d/cases"; printf '5 2.0\n' > "$W/d/cases/case_small.txt"
tar -C "$W/d" -czf "$R/data/cnlptestdata.tar.gz" cases
cp "$ROOT/bin/cnlp" "$R/cnlp"
T=$(printf '\t')
{
    echo "cnlptest${T}latest${T}$PLAT${T}file://$R/cnlptest/cnlptest-$PLAT.tar.gz${T}-${T}test models${T}lib${T}cnlptestdata${T}-${T}cpu"
    echo "ipopt${T}nightly${T}$PLAT${T}file://$R/cnlplib/ipopt-$PLAT.tar.gz${T}-${T}Ipopt${T}solver${T}${T}-${T}cpu"
    echo "cnlptestdata${T}1${T}any${T}file://$R/data/cnlptestdata.tar.gz${T}$(sha "$R/data/cnlptestdata.tar.gz")${T}test data${T}data${T}${T}-${T}-"
} > "$R/index.tsv"

# ── the installer, into an empty home ───────────────────────────────────────────
H=$W/home; mkdir -p "$H"
env -i HOME="$H" PATH=/usr/bin:/bin CNLPLIB_BASE="file://$R" CNLPLIB_RELEASES="file://$R/cnlplib" \
    sh "$ROOT/install.sh" -y > "$W/install.log" 2>&1 && ok "the installer runs" || bad "the installer runs: $(cat "$W/install.log")"
[ -x "$H/.cnlplib/bin/cnlp" ] && [ -x "$H/.cnlplib/bin/cnlp-solve" ] && ok "it installs cnlp and cnlp-solve" || bad "it installs cnlp and cnlp-solve"
grep -q 'CNLPLIB_PATH' "$H/.profile" && ok "it puts the library directory on CNLPLIB_PATH" || bad "CNLPLIB_PATH in ~/.profile"
export HOME="$H" PATH="$H/.cnlplib/bin:/usr/bin:/bin" CNLPLIB_HOME="$H/.cnlplib" CNLPLIB_INDEX="file://$R/index.tsv"

# ── cnlp install / list ─────────────────────────────────────────────────────────
cnlp install cnlptest ipopt > "$W/i.log" 2>&1 && ok "cnlp install cnlptest ipopt" || bad "cnlp install: $(cat "$W/i.log")"
[ -f "$CNLPLIB_HOME/data/cnlptestdata/case_small.txt" ] && ok "a library brings the data set it needs" || bad "the data set is installed"
[ -f "$CNLPLIB_HOME/solvers/ipopt/lib/libcnlp_ipopt.so" ] && ok "the solver is in solvers/" || bad "the solver is in solvers/"
cnlp list | grep -q 'cpu' && ok "cnlp list shows backends" || bad "cnlp list shows backends"
cnlp install cnlptest 2>&1 | grep -q 'up to date' && ok "a second install is a no-op" || bad "a second install is a no-op"

# ── cnlp solve ───────────────────────────────────────────────────────────────────
h5() {   # FILE FIELD -> value
    "$PY" -c "import h5py,sys; f=h5py.File(sys.argv[1]); v=f['solution'][sys.argv[2]][()]; print(v.decode() if isinstance(v,bytes) else v)" "$1" "$2"
}
close() { "$PY" -c "import sys; sys.exit(0 if abs(float(sys.argv[1]) - float(sys.argv[2])) < 1e-6 else 1)" "$1" "$2"; }
O=$W/out; mkdir -p "$O"
S="print_level=0 sb=yes"

# one integer (P_new), the standard HDF5 output
cnlp solve @cnlptest:pn 6 $S -o "$O/pn.h5" > "$W/s.log" 2>&1 && ok "an integer instance (P_new)" || bad "an integer instance: $(cat "$W/s.log")"
close "$(h5 "$O/pn.h5" objective)" 0.5 && [ "$(h5 "$O/pn.h5" status)" = solved ] && ok "  objective 0.5, status solved" || bad "  pn objective"
# the builder: one value per field
cnlp solve @cnlptest:pb 5 3.0 $S -o "$O/pb.h5" > "$W/s.log" 2>&1 && ok "builder values (n t)" || bad "builder values: $(cat "$W/s.log")"
close "$(h5 "$O/pb.h5" objective)" 12.5 && close "$(h5 "$O/pb.h5" multipliers | tr -d '[]')" 5.0 && ok "  objective 12.5, y = 5" || bad "  pb objective or y"
# a string instance: a file in an installed data set, found by name; the default writer
cnlp solve @cnlptest:ps case_small $S -o "$O/ps.txt" > "$W/s.log" 2>&1 && ok "a data-set file by name (P_new_str)" || bad "a data-set file by name: $(cat "$W/s.log")"
grep -q '^objective 4.5' "$O/ps.txt" && ok "  the model's default writer wrote it (objective 4.5)" || bad "  default writer output"
# reader:path, and cnlp: forcing the standard format over the default writer
cnlp solve @cnlptest:ps "txt:$CNLPLIB_HOME/data/cnlptestdata/case_small.txt" $S -o "cnlp:$O/ps.h5" -o "txt:$O/ps2.txt" > "$W/s.log" 2>&1 \
    && ok "an instance reader (reader:path), two outputs" || bad "an instance reader: $(cat "$W/s.log")"
close "$(h5 "$O/ps.h5" objective)" 4.5 && grep -q '^objective 4.5' "$O/ps2.txt" && ok "  cnlp:path is HDF5, writer:path the model's format" || bad "  forced formats"
# starts: a start reader, and a solution file
printf '0.5\n0.5\n2\n2\n2\n' > "$W/x.txt"
cnlp solve @cnlptest:ps case_small -s "xstart:$W/x.txt" $S -o "cnlp:$O/w.h5" > "$W/s.log" 2>&1 && ok "a start reader (-s reader:path)" || bad "a start reader: $(cat "$W/s.log")"
[ "$(h5 "$O/w.h5" iterations)" -le 1 ] && ok "  started at the solution: $(h5 "$O/w.h5" iterations) iterations" || bad "  start reader iterations"
cnlp solve @cnlptest:pn 6 -s "$O/pn.h5" $S -o "$O/pn2.h5" > "$W/s.log" 2>&1 && ok "a solution file as the start (-s file.h5)" || bad "a solution start: $(cat "$W/s.log")"
# a solver by path, and options
cnlp solve @cnlptest:pn 6 --solver "$CNLPLIB_HOME/solvers/ipopt/lib/libcnlp_ipopt.so" tol=1e-10 $S > "$W/s.log" 2>&1 && ok "a solver by path, with an option" || bad "a solver by path: $(cat "$W/s.log")"
# errors are reported, not crashes
cnlp solve @cnlptest:pn 6 no_such_option=1 $S > "$W/e.log" 2>&1 && bad "an unknown option is refused" || { grep -q 'no_such_option' "$W/e.log" && ok "an unknown option is refused, by name" || bad "unknown option message: $(cat "$W/e.log")"; }
cnlp solve @cnlptest:nope 6 > "$W/e.log" 2>&1 && bad "an unknown model is refused" || { grep -q 'no model' "$W/e.log" && ok "an unknown model is refused" || bad "unknown model message: $(cat "$W/e.log")"; }
grep -q '^  @cnlptest:pn' "$W/e.log" && ok "an unknown model lists the library's models" || bad "no model list: $(cat "$W/e.log")"
cnlp solve @cnlptest:ps small $S > "$W/e.log" 2>&1 && bad "only the exact data-set file name is taken" || { grep -q 'no file `small`' "$W/e.log" && grep -q '^  case_small  (cnlptestdata/case_small.txt)' "$W/e.log" && ok "a partial data-set file name is refused, with the file names that contain it" || bad "partial name message: $(cat "$W/e.log")"; }
cnlp solve @cnlptest:pn 6 --device cuda $S > "$W/e.log" 2>&1 && bad "--device cuda without a device tier is refused" || { grep -q 'no CUDA tier' "$W/e.log" && ok "--device cuda without a device tier is refused" || bad "device message: $(cat "$W/e.log")"; }
cnlp solve @cnlptest:pn 6 --device tpu $S > "$W/e.log" 2>&1 && bad "an unknown device is refused" || { grep -q 'auto, cpu or cuda' "$W/e.log" && ok "an unknown device is refused" || bad "unknown device message: $(cat "$W/e.log")"; }
cnlp solve @cnlptest:pn 6 $S > "$W/s.log" 2>&1 && grep -q '(cpu)' "$W/s.log" && ok "without a device tier the solve runs on the cpu, and says so" || bad "cpu solve: $(cat "$W/s.log")"
cnlp solve @cnlptest:pb 5 > "$W/e.log" 2>&1 && bad "a wrong number of arguments is refused" || { grep -q 'takes 2' "$W/e.log" && ok "a wrong number of arguments is refused" || bad "argument count message: $(cat "$W/e.log")"; }
cnlp solve @nolib:x 1 > "$W/e.log" 2>&1 && bad "a missing library is refused" || { grep -q 'cnlp install' "$W/e.log" && ok "a missing library is refused, with what to install" || bad "missing library message: $(cat "$W/e.log")"; }

# ── cnlp up / remove ────────────────────────────────────────────────────────────
cnlp up > "$W/u.log" 2>&1 && ok "cnlp up" || bad "cnlp up: $(cat "$W/u.log")"
cnlp remove cnlptest > /dev/null && [ ! -d "$CNLPLIB_HOME/lib/cnlptest" ] && ok "cnlp remove" || bad "cnlp remove"

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
