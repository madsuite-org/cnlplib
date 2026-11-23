# cnlp-input: what a model is instantiated from

Version 0.2. Conventions: [README.md](README.md).

`/input` holds the values for the model's entry point (`P_argtype`):

| Entry point | `/input` holds |
|---|---|
| fixed (`""`) | nothing; may be absent |
| `P_new(n)` | integer `n` |
| `P_new_str(s)` | string `s`; a path is relative to the file's directory |
| builder (`P_schema`) | one entry per schema field, named and typed as in the schema |

Builder fields, by schema kind: `scalar` a scalar dataset `/input/<name>`; `array` a 1-D
dataset; `table` a group `/input/<name>` with one dataset per column, of equal lengths.
Types (i64, f64) must match the schema exactly; nothing is converted.

**Reading.** Check that the file has exactly the entries the entry point needs, then call
it: `P_new`, `P_new_str`, or `P_data_begin`, one `P_set_*` per field, `P_data_ready`,
`P_new_from_data`. Checks run before any instance is created; `P_data_ready` may still
reject inconsistent data.

**Writing.** Use the names and types of the library's schema.

Example: `/input/N` (scalar 3), `/input/bus/i` (`[1, 2, 3]`), `/input/bus/pd`
(`[0.0, 0.9, 1.2]`).
