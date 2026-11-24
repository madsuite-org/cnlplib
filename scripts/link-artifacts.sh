#!/bin/sh
# Link a library's shared artifacts into it, as `cnlp install` does.
#
#   scripts/link-artifacts.sh LIBRARY_DIR ARTIFACT_PARENT_DIR
#
# LIBRARY_DIR is an unpacked library (NAME/) with an ARTIFACTS file of `NAME-HASH TARGET`
# lines; ARTIFACT_PARENT_DIR holds each artifact unpacked (NAME-HASH/, from
# NAME-HASH.artifact.tar.gz). Each file of an artifact is linked into the library's TARGET.
set -eu
L=$1 P=$(cd "$2" && pwd)
while read -r a t; do
    [ -n "$a" ] || continue
    [ -d "$P/$a" ] || { echo "$L needs $a, not in $P"; exit 1; }
    mkdir -p "$L/$t"
    for f in $(ls -A "$P/$a"); do ln -sfn "$P/$a/$f" "$L/$t/$f"; done
done < "$L/ARTIFACTS"
