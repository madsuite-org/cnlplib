#!/bin/sh
# Refuse device code newer than a driver may load.
#
#   scripts/check-ptx.sh BUNDLE MAX     (e.g. MAX = 7.8)
#
# A CUDA library carries its kernels as PTX, which the driver compiles when it loads them;
# a driver older than the PTX ISA version refuses them, and every device evaluation then
# fails. The ISA a build emits follows the build host's toolkit unless it is pinned, so the
# check reads every `.version X.Y` in the bundle's libraries and fails on any newer than MAX.
set -eu
B=$1 MAX=$2
V=$(for f in "$B"/lib/lib*.so; do strings -a "$f"; done | sed -n 's/^\.version \([0-9][0-9]*\.[0-9][0-9]*\)$/\1/p' | sort -uV)
[ -n "$V" ] || { echo "no PTX in $B/lib (a device build should carry some)"; exit 1; }
echo "PTX ISA versions: $(echo $V)"
NEW=$(printf '%s\n%s\n' "$V" "$MAX" | sort -uV | tail -1)
[ "$NEW" = "$MAX" ] || { echo "PTX ISA $NEW is newer than $MAX: drivers older than its toolkit cannot load it"; exit 1; }
