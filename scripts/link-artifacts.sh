#!/bin/sh
# Link a library's shared artifacts into it, as `cnlp install` does.
#
#   JULIA_LIBDIR=... scripts/link-artifacts.sh LIBRARY_DIR [ARTIFACT_PARENT_DIR]
#
# LIBRARY_DIR is an unpacked library (NAME/) with an ARTIFACTS file of `ARTIFACT TARGET`
# lines. `julia-VERSION` is the official Julia's lib/julia, here JULIA_LIBDIR (default: the
# lib/julia of the `julia` on PATH); any other artifact is ARTIFACT_PARENT_DIR/ARTIFACT, an
# unpacked ARTIFACT.artifact.tar.gz. Each file is linked into the library's TARGET unless the
# library carries a file of that name.
set -eu
L=$1 P=${2:-.}
JL=${JULIA_LIBDIR:-$(julia --startup-file=no -e 'print(abspath(Sys.BINDIR, "..", "lib", "julia"))')}
while read -r a t; do
    [ -n "$a" ] || continue
    case "$a" in
        julia-*.*.*) d=$JL
            [ -f "$d/../libjulia.so.${a#julia-}" ] || { echo "$d is not Julia ${a#julia-}"; exit 1; } ;;
        *) d=$(cd "$P/$a" && pwd) || { echo "$L needs $a, not in $P"; exit 1; } ;;
    esac
    mkdir -p "$L/$t"
    for f in $(ls -A "$d"); do
        [ -e "$L/$t/$f" ] || [ -L "$L/$t/$f" ] || ln -s "$d/$f" "$L/$t/$f"
    done
done < "$L/ARTIFACTS"
