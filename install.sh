#!/bin/sh
# Install the cnlplib command:  curl -fsSL https://cnlplib.madsuite.org/install | sh
#
# Options (after `sh -s --`):
#   -y                 do not ask for confirmation
#   --path DIR         install directory (default ~/.cnlplib)
#   --add-to-path no   leave shell startup files alone
#
# It installs DIR/bin/cnlplib and, unless told not to, adds to ~/.bashrc, ~/.zshrc and
# ~/.profile a marked block that puts DIR/bin on PATH and DIR/lib on CNLPMODELS_PATH,
# so cnlp consumers find installed libraries by name. Run it again to update cnlplib.
set -eu

BASE="${CNLPLIB_BASE:-https://cnlplib.madsuite.org}"
PREFIX="$HOME/.cnlplib"; YES=0; ADD=yes
while [ "$#" -gt 0 ]; do
    case "$1" in
        -y) YES=1 ;;
        --path) PREFIX="$2"; shift ;;
        --path=*) PREFIX="${1#--path=}" ;;
        --add-to-path) ADD="$2"; shift ;;
        --add-to-path=*) ADD="${1#--add-to-path=}" ;;
        *) echo "install: unknown option $1" >&2; exit 1 ;;
    esac
    shift
done

echo "cnlplib will be installed in: $PREFIX"
if [ "$YES" = 0 ] && [ -r /dev/tty ]; then
    printf "Proceed? [Y/n] "; read -r ans < /dev/tty || ans=y
    case "$ans" in n|N|no|No) echo "Cancelled."; exit 1 ;; esac
fi

mkdir -p "$PREFIX/bin" "$PREFIX/lib"
tmp="$(mktemp "$PREFIX/bin/.cnlplib.XXXXXX")"
curl -fsSL "$BASE/cnlplib" -o "$tmp" || { rm -f "$tmp"; echo "install: download failed: $BASE/cnlplib" >&2; exit 1; }
chmod +x "$tmp"; mv "$tmp" "$PREFIX/bin/cnlplib"
echo "Installed $PREFIX/bin/cnlplib"

if [ "$ADD" = yes ]; then
    begin="# >>> cnlplib initialize >>>"; end="# <<< cnlplib initialize <<<"
    block="$begin
export PATH=\"$PREFIX/bin:\$PATH\"
export CNLPMODELS_PATH=\"$PREFIX/lib\${CNLPMODELS_PATH:+:\$CNLPMODELS_PATH}\"
$end"
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        [ -f "$rc" ] || [ "$rc" = "$HOME/.profile" ] || continue
        if [ -f "$rc" ] && grep -qF "$begin" "$rc"; then
            # replace the existing block, leave the rest of the file untouched
            awk -v b="$begin" -v e="$end" '$0==b{skip=1} !skip{print} $0==e{skip=0}' "$rc" > "$rc.cnlplib.tmp" && mv "$rc.cnlplib.tmp" "$rc"
        fi
        printf '\n%s\n' "$block" >> "$rc"
        echo "Updated $rc"
    done
    echo "For this terminal:  export PATH=\"$PREFIX/bin:\$PATH\" CNLPMODELS_PATH=\"$PREFIX/lib\${CNLPMODELS_PATH:+:\$CNLPMODELS_PATH}\""
fi
echo "Next:  cnlplib list"
