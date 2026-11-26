#!/bin/sh
# Install the cnlp command:  curl -fsSL https://cnlplib.madsuite.org/install | sh
#
# Options (after `sh -s --`):
#   -y                 do not ask for confirmation
#   --path DIR         install directory (default ~/.cnlplib)
#   --add-to-path no   leave shell startup files alone
#
# It installs DIR/bin/cnlp, the solve driver DIR/bin/cnlp-solve, DIR/lib/libcnlpio.so and, unless told not
# to, adds to ~/.bashrc, ~/.zshrc and ~/.profile a marked block that puts DIR/bin on PATH
# and DIR/lib on CNLPLIB_PATH, so cnlp consumers find installed libraries by name. Run it
# again to update both.
set -eu

BASE="${CNLPLIB_BASE:-https://cnlplib.madsuite.org}"
RELEASES="${CNLPLIB_RELEASES:-https://github.com/madsuite-org/cnlplib/releases/download/nightly}"
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

echo "cnlp will be installed in: $PREFIX"
if [ "$YES" = 0 ] && [ -r /dev/tty ]; then
    printf "Proceed? [Y/n] "; read -r ans < /dev/tty || ans=y
    case "$ans" in n|N|no|No) echo "Cancelled."; exit 1 ;; esac
fi

mkdir -p "$PREFIX/bin" "$PREFIX/lib"
tmp="$(mktemp "$PREFIX/bin/.cnlp.XXXXXX")"
curl -fsSL "$BASE/cnlp" -o "$tmp" || { rm -f "$tmp"; echo "install: download failed: $BASE/cnlp" >&2; exit 1; }
chmod +x "$tmp"; mv "$tmp" "$PREFIX/bin/cnlp"
echo "Installed $PREFIX/bin/cnlp"

# the solve driver: a binary, from cnlplib's rolling release, checked against its SHA256SUMS
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64|Linux-amd64) PLAT=linux-x86_64 ;;
    *) PLAT="" ;;
esac
if [ -n "$PLAT" ]; then
    d="$(mktemp -d "$PREFIX/.cnlp-solve.XXXXXX")"
    a="cnlp-solve-$PLAT.tar.gz"
    if curl -fsSL "$RELEASES/$a" -o "$d/$a" && curl -fsSL "$RELEASES/SHA256SUMS" -o "$d/SHA256SUMS"; then
        want="$(awk -v f="$a" '$2 == f { print $1 }' "$d/SHA256SUMS")"
        got="$( (sha256sum "$d/$a" 2>/dev/null || shasum -a 256 "$d/$a") | cut -d' ' -f1)"
        if [ -n "$want" ] && [ "$want" = "$got" ]; then
            tar -xzf "$d/$a" -C "$d" && mv "$d/cnlp-solve/bin/cnlp-solve" "$PREFIX/bin/cnlp-solve" && echo "Installed $PREFIX/bin/cnlp-solve"
            # the solution file library Julia and Python use (cnlpio.h)
            [ -f "$d/cnlp-solve/lib/libcnlpio.so" ] && mkdir -p "$PREFIX/lib" && mv "$d/cnlp-solve/lib/libcnlpio.so" "$PREFIX/lib/libcnlpio.so"
        else
            echo "install: checksum mismatch for $a; cnlp solve is not installed" >&2
        fi
    else
        echo "install: could not download $a; cnlp solve is not installed" >&2
    fi
    rm -rf "$d"
else
    echo "install: no cnlp-solve build for $(uname -s)-$(uname -m) yet; everything but cnlp solve works" >&2
fi

if [ "$ADD" = yes ]; then
    begin="# >>> cnlplib initialize >>>"; end="# <<< cnlplib initialize <<<"
    block="$begin
export PATH=\"$PREFIX/bin:\$PATH\"
export CNLPLIB_HOME=\"$PREFIX\"
export CNLPLIB_PATH=\"$PREFIX/lib\${CNLPLIB_PATH:+:\$CNLPLIB_PATH}\"
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
    echo "For this terminal:  export PATH=\"$PREFIX/bin:\$PATH\" CNLPLIB_PATH=\"$PREFIX/lib\${CNLPLIB_PATH:+:\$CNLPLIB_PATH}\""
fi
echo "Next:  cnlp list;  cnlp install ipopt lv;  cnlp solve @lv:wood 1000"
