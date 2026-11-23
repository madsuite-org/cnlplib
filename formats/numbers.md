# cnlp-numbers: number formats

Version 0.2. The formats a model can be evaluated in, for the binary32 family (`P_s*`) and
the number-format tier (`P_x*`) of `cnlp.h`.

## Names

Canonical names are the standards' spellings where a standard exists; aliases are
accepted. Names are compared as exact ASCII strings, never change meaning, and an unknown
name is never guessed at. Formats defined elsewhere use a reverse-DNS prefix
(`org.example.posit32`). Parameters follow a colon: `name:key=value`.

## Formats

Native byte order; arrays are contiguous.

| Name | Alias | Bytes | Value | Defined by |
|---|---|---|---|---|
| `binary32` | `f32` | 4 | IEEE binary32 | IEEE 754-2019; C23 `_Float32` |
| `binary64` | `f64` | 8 | IEEE binary64 | IEEE 754-2019; C23 `_Float64` |
| `binary128` | `f128` | 16 | IEEE binary128, never another `long double` | IEEE 754-2019; C23 `_Float128` |
| `double_binary64` | `dd` | 16 | `{hi, lo}` binary64, value `hi + lo`, `\|lo\| <= ulp(hi)/2` | layout of IBM extended double (`__ibm128`) |
| `quad_binary64` | `qd` | 32 | four binary64 of decreasing magnitude, value their sum, each `<= ulp/2` of the previous | layout of QD's `double[4]` |
| `apfloat:prec=N` | | 16 + 8⌈N/64⌉ | arbitrary precision, below | MPFR's number model |

In `dd` and `qd`, infinities and NaNs are in the first term and the rest are zero.

**apfloat.** `struct { int32_t kind; int32_t sign; int64_t exp; uint64_t limb[L]; }`,
`L = ceil(N/64)`, `N >= 2`. `kind`: 0 finite nonzero, 1 zero, 2 infinity, 3 NaN. `sign`:
+1 or −1. A finite nonzero value is `sign × m × 2^exp`, `1/2 <= m < 1`, with `m` the
binary fraction `limb[L-1] … limb[0]` (top bit of `limb[L-1]` set, bits below `N` zero);
otherwise `exp` and `limb` are zero. Evaluation uses `N`-bit arithmetic with each operation
correctly rounded to nearest, as MPFR does. Libraries convert to MPFR or `BigFloat`
internally; no foreign type or allocation crosses the ABI.

## Reserved

`binary16`, `binaryN` (N >= 128, multiple of 32) and `decimalN` (N >= 32, multiple of 32)
from IEEE 754 / C23; `bfloat16` and the 8-, 6- and 4-bit formats under DLPack's spellings
(`float8_e4m3fn`, …).

## DLPack

`binary32`, `binary64`, `binary128` are DLPack `{kDLFloat, 32|64|128, 1}`
(`cnlp_format_to_dlpack`); the others have no DLPack form (`dd` is not `lanes = 2`).
