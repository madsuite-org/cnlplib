#!/bin/sh
# Build the cnlp-solve driver and the Ipopt solver, and pack them:
#   scripts/build-cnlp.sh PLATFORM OUT   -> OUT/cnlp-solve-PLATFORM.tar.gz, OUT/ipopt-PLATFORM.tar.gz,
#                                           OUT/ipopt-sources.tar
#
# Needs a C compiler, cmake, curl, patchelf and julia (for the Ipopt and OpenBLAS binaries).
# HDF5 is built from source as a static library, so cnlp-solve depends only on the C library.
set -eu
PLAT=$1 OUT=$(mkdir -p "$2" && cd "$2" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=${CNLP_BUILD_DIR:-$ROOT/build}; mkdir -p "$W"
H5V=1.14.6
if [ ! -f "$W/hdf5/lib/libhdf5.a" ]; then
    curl -fsSL -o "$W/hdf5.tar.gz" "https://github.com/HDFGroup/hdf5/releases/download/hdf5_$H5V/hdf5-$H5V.tar.gz"
    tar -xzf "$W/hdf5.tar.gz" -C "$W"
    cmake -S "$W/hdf5-$H5V" -B "$W/hdf5-build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$W/hdf5" \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_HL_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF -DHDF5_BUILD_CPP_LIB=OFF \
        -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF -DHDF5_ENABLE_SZIP_SUPPORT=OFF > "$W/hdf5-cmake.log"
    cmake --build "$W/hdf5-build" -j "$(nproc)" > "$W/hdf5-build.log"
    cmake --install "$W/hdf5-build" > /dev/null
fi
# the driver
D=$W/stage/cnlp-solve; rm -rf "$D"; mkdir -p "$D/bin" "$D/lib"
cc -std=c99 -O2 -Wall -Wextra -Wno-format-truncation -I"$ROOT" -I"$W/hdf5/include" -o "$D/bin/cnlp-solve" \
    "$ROOT/solve/cnlp-solve.c" "$W/hdf5/lib/libhdf5.a" -ldl -lm -lpthread
# libcnlpio, the solution file for Julia and Python (cnlpio.h), with HDF5 linked in
cc -std=c99 -O2 -Wall -Wextra -fPIC -shared -I"$ROOT" -I"$W/hdf5/include" -o "$D/lib/libcnlpio.so" \
    "$ROOT/solve/cnlpio.c" "$W/hdf5/lib/libhdf5.a" -Wl,--exclude-libs,ALL -ldl -lm -lpthread   # HDF5 kept private
cp "$ROOT/LICENSE" "$D/LICENSE"
printf 'cnlp-solve links HDF5 %s statically: BSD-3-Clause, https://github.com/HDFGroup/hdf5\n' "$H5V" > "$D/THIRD_PARTY_NOTICES"
tar -C "$W/stage" -czf "$OUT/cnlp-solve-$PLAT.tar.gz" cnlp-solve
# the Ipopt solver, against the Julia binary builds of Ipopt and OpenBLAS32
J=$W/julia-env; mkdir -p "$J"
julia --startup-file=no --project="$J" -e '
    using Pkg; Pkg.add(["Ipopt_jll", "OpenBLAS32_jll"]; io = devnull)
    using Ipopt_jll, OpenBLAS32_jll
    # Yggdrasil encodes upstream version a.b.c as (100a).(100b).(100c)
    up2(name) = (v = only(p.version for p in values(Pkg.dependencies()) if p.name == name);
        join((v.major ÷ 100, v.minor ÷ 100, v.patch ÷ 100), "."))
    write(ARGS[1], join([dirname(dirname(Ipopt_jll.libipopt_path)), Ipopt_jll.LIBPATH[], OpenBLAS32_jll.libopenblas_path,
                         up2("Ipopt_jll"), up2("MUMPS_seq_jll")], "\n"))' "$W/ipopt-paths"
IP=$(sed -n 1p "$W/ipopt-paths"); LP=$(sed -n 2p "$W/ipopt-paths"); OB=$(sed -n 3p "$W/ipopt-paths")
IPV=$(sed -n 4p "$W/ipopt-paths"); MUV=$(sed -n 5p "$W/ipopt-paths")
cc -std=c99 -O2 -Wall -Wextra -shared -fPIC -I"$ROOT" -I"$IP/include" -o "$W/libcnlp_ipopt.so" \
    "$ROOT/adapters/ipopt/cnlp_ipopt.c" -L"$IP/lib" -lipopt -ldl
(cd "$OUT" && sh "$ROOT/scripts/package-ipopt.sh" "$W/libcnlp_ipopt.so" "$LP" "$OB" "$PLAT")
# the corresponding source of the copyleft parts of the Ipopt bundle (Ipopt EPL-2.0, MUMPS CeCILL-C)
SRC=$W/stage/ipopt-sources; rm -rf "$SRC"; mkdir -p "$SRC"
curl -fsSL -o "$SRC/Ipopt-$IPV.tar.gz" "https://github.com/coin-or/Ipopt/archive/refs/tags/releases/$IPV.tar.gz"
curl -fsSL -o "$SRC/MUMPS_$MUV.tar.gz" "https://mumps-solver.org/MUMPS_$MUV.tar.gz"
printf '# Source of Ipopt %s and MUMPS %s in the ipopt release archive

Built by the Yggdrasil recipes I/Ipopt and M/MUMPS (https://github.com/JuliaPackaging/Yggdrasil),
which record any patches applied. The GCC runtime libraries in the archive are under the
GCC Runtime Library Exception, and the rest under permissive licenses (THIRD_PARTY_NOTICES).
' "$IPV" "$MUV" > "$SRC/README.md"
tar -C "$W/stage" -cf "$OUT/ipopt-sources.tar" ipopt-sources
ls -la "$OUT"
