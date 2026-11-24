#!/bin/sh
# Pack a library bundle with its notices, and the shared artifact of its runtime.
#
#   JULIA_LIBDIR=... scripts/package-library.sh BUNDLE NAME VERSION PLATFORM LICENSE_FILE [DEVICE]
#       -> NAME-PLATFORM.tar.gz
#
# The file name carries no version, so each release's asset has a stable URL; the version
# is in the VERSION file inside. The bundled Julia runtime is kept whole: its module
# initializers load every library it carries (SuiteSparse among them), so a pruned bundle
# fails in any process that is not Julia already (C, Python, cnlp solve). The GPL and LGPL
# sources go in a separate archive (scripts/gpl-sources.sh).
#
# A library can share files with other libraries through artifacts: archives named by a
# hash of their content, listed in the library's ARTIFACTS file as `NAME-HASH TARGET` lines.
# `cnlp install` fetches each once, from the library's release (NAME-HASH.artifact.tar.gz),
# into $CNLPLIB_HOME/artifacts/NAME-HASH, and links its files into the library's TARGET
# directory (scripts/link-artifacts.sh does the same for a check).
#
# The Julia runtime is shared that way, from the official Julia release itself. Each
# library's own copy of libjulia and libjulia-internal is renamed with a random prefix
# (privatized), so libraries loaded in one process, or into Julia, never share one; those stay
# in the library. Every other file of lib/julia that is byte for byte the official Julia's
# (JULIA_LIBDIR, default: the lib/julia of the `julia` on PATH, which must be the version the
# bundle was built with) is left out, and ARTIFACTS names `julia-VERSION lib/julia`: `cnlp
# install` takes that Julia's lib/julia from its official release (or a local juliaup copy of
# it) and links in what the library does not carry. A file that differs (a newer libstdc++
# that a dependency brings) stays in the library.
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
JL=${JULIA_LIBDIR:-$(julia --startup-file=no -e 'print(abspath(Sys.BINDIR, "..", "lib", "julia"))')}
V=$(ls "$JL"/../libjulia.so.*.*.* 2>/dev/null | sed -n 's/.*libjulia\.so\.//p' | head -1)
[ "$V" = "$JULIA" ] || { echo "$JL is Julia $V, but the bundle carries $JULIA"; exit 1; }
same() {   # the same file as the official one: a symlink to the same target, or the same bytes
    if [ -L "$1" ]; then [ -L "$2" ] && [ "$(readlink "$1")" = "$(readlink "$2")" ]
    else [ -f "$2" ] && [ ! -L "$2" ] && cmp -s "$1" "$2"; fi
}
n=0
for f in $(ls -A "$S/$NAME/lib/julia"); do
    private "$f" && continue
    same "$S/$NAME/lib/julia/$f" "$JL/$f" && { rm "$S/$NAME/lib/julia/$f"; n=$((n + 1)); }
done
printf 'julia-%s lib/julia\n' "$JULIA" > "$S/$NAME/ARTIFACTS"

OUT="$NAME-$PLAT.tar.gz"
tar -C "$S" -czf "$OUT" "$NAME"
rm -rf "$S"
echo "$OUT $(du -h "$OUT" | cut -f1) julia=$JULIA ($n runtime files from the official Julia)"
