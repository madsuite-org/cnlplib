#!/bin/sh
# Pack a library bundle with its notices, and the runtime libraries it shares.
#
#   scripts/package-library.sh BUNDLE NAME VERSION PLATFORM LICENSE_FILE [DEVICE]
#       -> NAME-PLATFORM.tar.gz and runtime-HASH.tar.gz
#
# The file name carries no version, so each release's asset has a stable URL; the version
# is in the VERSION file inside. The bundled Julia runtime is kept whole: its module
# initializers load every library it carries (SuiteSparse among them), so a pruned bundle
# fails in any process that is not Julia already (C, Python, cnlp solve). The GPL and LGPL
# sources go in a separate archive (scripts/gpl-sources.sh).
#
# The runtime is split in two. Each library's own copy of libjulia and libjulia-internal is
# renamed with a random prefix (privatized), so libraries loaded in one process, or into
# Julia, never share one; those stay in the library. Everything else in lib/julia (OpenBLAS,
# libgfortran, SuiteSparse, ...) is the same file in every library built with the same Julia,
# so it goes in runtime-HASH.tar.gz, HASH being over the files' names and bytes, and the
# library records HASH in its RUNTIME file. `cnlp install` fetches that runtime once into
# $CNLPLIB_HOME/artifacts/runtime-HASH and links its files into the library's lib/julia
# (scripts/link-runtime.sh does the same for a check).
set -eu
B=$1 NAME=$2 VERSION=$3 PLAT=$4 LIC=$5 DEVICE=${6:-}
HERE=$(cd "$(dirname "$0")" && pwd)
S=$(mktemp -d); mkdir "$S/$NAME"; cp -a "$B"/. "$S/$NAME"/
rm -rf "$S/$NAME/share"
cp "$LIC" "$S/$NAME/LICENSE"
cp "$HERE/THIRD_PARTY_NOTICES" "$S/$NAME/THIRD_PARTY_NOTICES"
# the runtime the bundle carries: lib/..._libjulia.so.X.Y.Z
JULIA=$(ls "$B"/lib/*libjulia.so.*.*.* 2>/dev/null | sed -n 's/.*libjulia\.so\.//p' | head -1)
printf 'name = %s\nversion = %s\ncnlp_abi = 0.2\njulia = %s\nplatform = %s\ndevice = %s\n' \
    "$NAME" "$VERSION" "$JULIA" "$PLAT" "$DEVICE" > "$S/$NAME/VERSION"

# the privatizing prefixes: one per runtime in the bundle (a joined chunked library has several)
SALTS=$(ls "$S/$NAME/lib" | sed -n 's/^\(.*\)_libjulia\.so$/\1/p')
[ -n "$SALTS" ] || { echo "no privatized libjulia in $B/lib"; exit 1; }
private() { for s in $SALTS; do case "$1" in "${s}"_*) return 0 ;; esac; done; return 1; }
R=$(mktemp -d); mkdir "$R/julia"
for f in $(ls -A "$S/$NAME/lib/julia"); do
    private "$f" || mv "$S/$NAME/lib/julia/$f" "$R/julia/$f"
done
# HASH: over each file's name, kind and bytes (a symlink by its target), in a fixed order
HASH=$(cd "$R/julia" && for f in $(ls -A | LC_ALL=C sort); do
           if [ -L "$f" ]; then printf 'L %s %s\n' "$f" "$(readlink "$f")"
           else printf 'F %s %s\n' "$f" "$(sha256sum < "$f" | cut -d' ' -f1)"; fi
       done | sha256sum | cut -c1-16)
RT="runtime-$HASH"
mv "$R" "$S/$RT"
printf 'runtime = %s\njulia = %s\nfiles = %s\n' "$RT" "$JULIA" "$(ls -A "$S/$RT/julia" | wc -l)" > "$S/$NAME/RUNTIME"

OUT="$NAME-$PLAT.tar.gz"
tar -C "$S" -czf "$OUT" "$NAME"
tar -C "$S" -czf "$RT.tar.gz" "$RT"
rm -rf "$S"
echo "$OUT $(du -h "$OUT" | cut -f1) julia=$JULIA"
echo "$RT.tar.gz $(du -h "$RT.tar.gz" | cut -f1)"
