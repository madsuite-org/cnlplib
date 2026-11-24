#!/bin/sh
# Link a shared runtime into an unpacked library, as `cnlp install` does.
#
#   scripts/link-runtime.sh LIBRARY_DIR RUNTIME_DIR
#
# LIBRARY_DIR is an unpacked NAME-PLATFORM.tar.gz (NAME/), RUNTIME_DIR the unpacked
# runtime-HASH.tar.gz its RUNTIME file names (runtime-HASH/). Each file of the runtime's julia/
# is linked into the library's lib/julia/, so the library's runpaths find it there.
set -eu
L=$1 R=$(cd "$2" && pwd)
want=$(sed -n 's/^runtime = //p' "$L/RUNTIME")
[ "$want" = "$(basename "$R")" ] || { echo "$L needs $want, not $(basename "$R")"; exit 1; }
mkdir -p "$L/lib/julia"
for f in $(ls -A "$R/julia"); do ln -sfn "$R/julia/$f" "$L/lib/julia/$f"; done
