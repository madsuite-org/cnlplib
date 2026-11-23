#!/bin/sh
# Prune a library bundle to what it loads, add its notices, and pack it.
#
#   scripts/package-library.sh BUNDLE NAME VERSION PLATFORM LICENSE_FILE [DEVICE] -> NAME-VERSION-PLATFORM.tar.gz
#
# A juliac bundle carries the whole Julia library directory; the library itself loads only
# the runtime and a few system libraries, which are kept. Everything else (including the
# GPL-licensed SuiteSparse parts and OpenSSL) is removed.
set -eu
B=$1 NAME=$2 VERSION=$3 PLAT=$4 LIC=$5 DEVICE=${6:-}
HERE=$(cd "$(dirname "$0")" && pwd)
KEEP='_libjulia-internal\.so|libunwind\.so|libz\.so|libatomic\.so|libstdc\+\+\.so|libgcc_s\.so|libopenlibm\.so|libpcre2-8\.so'
S=$(mktemp -d); mkdir "$S/$NAME"; cp -a "$B"/. "$S/$NAME"/
for f in "$S/$NAME"/lib/julia/*; do basename "$f" | grep -Eq "$KEEP" || rm -rf "$f"; done
rm -rf "$S/$NAME/share"
cp "$LIC" "$S/$NAME/LICENSE"
cp "$HERE/THIRD_PARTY_NOTICES" "$S/$NAME/THIRD_PARTY_NOTICES"
# the runtime the bundle carries: lib/julia/..._libjulia-internal.so.X.Y.Z
JULIA=$(ls "$B"/lib/*libjulia.so.*.*.* 2>/dev/null | sed -n 's/.*libjulia\.so\.//p' | head -1)
printf 'name = %s\nversion = %s\ncnlp_abi = 0.2\njulia = %s\nplatform = %s\ndevice = %s\n' \
    "$NAME" "$VERSION" "$JULIA" "$PLAT" "$DEVICE" > "$S/$NAME/VERSION"
OUT="$NAME-$VERSION-$PLAT.tar.gz"
tar -C "$S" -czf "$OUT" "$NAME"
rm -rf "$S"
echo "$OUT $(du -h "$OUT" | cut -f1)"
