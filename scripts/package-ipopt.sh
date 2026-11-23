#!/bin/sh
# Bundle the Ipopt solver: libcnlp_ipopt.so with Ipopt and its dependencies, from the
# Julia binary builds (Ipopt_jll, OpenBLAS32_jll), in one lib/ with $ORIGIN rpaths.
#
#   scripts/package-ipopt.sh ADAPTER.so LIBPATH OPENBLAS32 PLATFORM  -> ipopt-PLATFORM.tar.gz
#
# LIBPATH is Ipopt_jll's library path (colon-separated); OPENBLAS32 the LP64 libopenblas.so,
# which the adapter forwards libblastrampoline to.
set -eu
AD=$1 LP=$2 OB=$3 PLAT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
S=$(mktemp -d); L=$S/ipopt/lib; mkdir -p "$L"
cp "$AD" "$L/"; cp -L "$OB" "$L/libopenblas.so"
# every library they load that comes from LIBPATH (not the system's libc and friends)
deps() { LD_LIBRARY_PATH="$LP:$(dirname "$OB")" ldd "$1" | awk '$3 ~ /^\// { print $1, $3 }'; }
todo="$L/$(basename "$AD") $L/libopenblas.so"; seen=""
while [ -n "$todo" ]; do
    f=${todo%% *}; [ "$todo" = "$f" ] && todo="" || todo=${todo#* }
    deps "$f" | while read -r name path; do
        case "$path" in /lib/*|/lib64/*|/usr/lib/*) continue ;; esac
        [ -e "$L/$name" ] || { cp -L "$path" "$L/$name"; echo "$L/$name"; }
    done > "$S/new"
    todo="$todo $(tr '\n' ' ' < "$S/new")"; todo=$(echo $todo)
done
chmod u+w "$L"/*; for f in "$L"/*.so*; do patchelf --set-rpath '$ORIGIN' "$f"; done
cp "$HERE/../LICENSE" "$S/ipopt/LICENSE"
cat > "$S/ipopt/THIRD_PARTY_NOTICES" <<'N'
This bundle carries Ipopt and the libraries it loads, from the Julia binary builds (Yggdrasil):
  Ipopt         EPL-2.0      https://github.com/coin-or/Ipopt
  MUMPS         CeCILL-C     https://mumps-solver.org
  METIS         Apache-2.0   https://github.com/KarypisLab/METIS
  OpenBLAS      BSD-3-Clause https://github.com/OpenMathLib/OpenBLAS
  SPRAL         BSD-3-Clause https://github.com/ralna/spral
  hwloc         BSD-3-Clause https://github.com/open-mpi/hwloc
  libblastrampoline  MIT     https://github.com/JuliaLinearAlgebra/libblastrampoline
  libgfortran, libquadmath, libgcc_s, libstdc++, libgomp  GPL-3.0 with the GCC Runtime Library Exception
N
printf 'name = ipopt\nversion = nightly\nsolver_interface = 0.1\nplatform = %s\n' "$PLAT" > "$S/ipopt/VERSION"
tar -C "$S" -czf "ipopt-$PLAT.tar.gz" ipopt
ls "$L" | tr '\n' ' '; echo; du -h "ipopt-$PLAT.tar.gz" | cut -f1
rm -rf "$S"
