#!/bin/sh
# Join chunk bundles into one library bundle.
#
#   scripts/assemble-chunks.sh NAME OUT CHUNK_BUNDLE...   -> OUT/lib/libNAME.so
#
# Each CHUNK_BUNDLE is a bundle built by scripts/build-library.jl with CNLP_CHUNK set
# (NAME_k/lib/libNAME_k.so, its Julia runtime, and MODELS). The bundles are merged into OUT
# by moving their files (a file present in two of them must be the same file; the chunk
# bundles are consumed), and libNAME.so is written: a small C
# library that answers the library-level calls (cnlp_abi_version, cnlp_nmodels,
# cnlp_model_name) for all the models, and lists every libNAME_k.so as a dependency. A symbol
# looked up through a handle is also searched in the libraries that handle depends on, so
# each model's symbols resolve into its chunk, and consumers load the joined library exactly
# as a single one. Each chunk keeps its own (privatized) Julia runtime, which starts the first
# time one of its models is called.
set -eu
NAME=$1 OUT=$2; shift 2
mkdir -p "$OUT/lib"
: > "$OUT/MODELS"
LIBS=""
for B in "$@"; do
    C=$(basename "$B")
    [ -f "$B/lib/lib$C.so" ] || { echo "$B: no lib/lib$C.so"; exit 1; }
    # the dependency is recorded under the library's SONAME, which must be its file name
    S=$(readelf -d "$B/lib/lib$C.so" | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p')
    [ -z "$S" ] || [ "$S" = "lib$C.so" ] || { echo "$B: SONAME $S is not lib$C.so"; exit 1; }
    # the list first: the loop moves files out of the tree find would still be walking
    L=$(mktemp); (cd "$B" && find . \( -type f -o -type l \) ! -name MODELS ! -path './share/*') > "$L"
    while read -r f; do
        if [ -e "$OUT/$f" ] || [ -L "$OUT/$f" ]; then
            cmp -s "$B/$f" "$OUT/$f" || { echo "$f differs between chunks"; exit 1; }
        else
            mkdir -p "$OUT/$(dirname "$f")"
            mv "$B/$f" "$OUT/$f"
        fi
    done < "$L"
    rm -f "$L"
    cat "$B/MODELS" >> "$OUT/MODELS"
    LIBS="$LIBS -l:lib$C.so"
done
N=$(grep -c . "$OUT/MODELS")
[ "$N" -eq "$(sort -u "$OUT/MODELS" | grep -c .)" ] || { echo "a model is in two chunks"; exit 1; }
SRC=$(mktemp -d)/lib$NAME.c
{
    printf '#include <stdint.h>\n#include <string.h>\nstatic const char* NAMES[%s] = {\n' "$N"
    sed 's/.*/    "&",/' "$OUT/MODELS"
    printf '};\nint64_t cnlp_abi_version(void) { return 2; }\n'
    printf 'int64_t cnlp_nmodels(void) { return %s; }\n' "$N"
    printf 'int64_t cnlp_model_name(int64_t k, uint8_t* buf, int64_t cap) {\n'
    printf '    if (k < 0 || k >= %s) return 0;\n' "$N"
    printf '    int64_t n = (int64_t)strlen(NAMES[k]);\n'
    printf '    if (cap > 0) memcpy(buf, NAMES[k], (size_t)(n < cap ? n : cap));\n'
    printf '    return n;\n}\n'
} > "$SRC"
# shellcheck disable=SC2086
cc -std=c99 -O2 -shared -fPIC -o "$OUT/lib/lib$NAME.so" "$SRC" \
    -L"$OUT/lib" -Wl,--no-as-needed $LIBS -Wl,-rpath,'$ORIGIN'
echo "lib$NAME.so: $N models in $# chunks"
