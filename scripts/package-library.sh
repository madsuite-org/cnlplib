#!/bin/sh
# Pack a library bundle with its notices.
#
#   scripts/package-library.sh BUNDLE NAME VERSION PLATFORM LICENSE_FILE [DEVICE] -> NAME-PLATFORM.tar.gz
#
# The file name carries no version, so each release's asset has a stable URL; the version
# is in the VERSION file inside. The bundled Julia runtime is kept whole: its module
# initializers load every library it carries (SuiteSparse among them), so a pruned bundle
# fails in any process that is not Julia already (C, Python, cnlp solve). The GPL and LGPL
# sources go in a separate archive (scripts/gpl-sources.sh).
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
OUT="$NAME-$PLAT.tar.gz"
tar -C "$S" -czf "$OUT" "$NAME"
rm -rf "$S"
echo "$OUT $(du -h "$OUT" | cut -f1) julia=$JULIA"
