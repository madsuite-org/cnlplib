# cnlp data formats

Version 0.2. The files that go with a cnlp library, defined only through what `cnlp.h`
publishes, so any consumer reads what another wrote.

| Format | Document | Holds |
|---|---|---|
| `cnlp-input` | [input.md](input.md) | the values a model is instantiated from |
| `cnlp-parameters` | [parameters.md](parameters.md) | parameter values, by block name |
| `cnlp-solution` | [solution.md](solution.md) | a solution and its multipliers; also a warm start |

[numbers.md](numbers.md) defines the number formats a model can be evaluated in.

## Conventions

**Encoding.** HDF5 (`.h5`). Other encodings, JSON for example, are overrides: a library
provides them as readers and writers (below; `example/qp.c` writes JSON), or a consumer
converts.

**Identification.** Root attributes:

| Attribute | Meaning |
|---|---|
| `format` | `"cnlp-input"`, `"cnlp-parameters"` or `"cnlp-solution"` |
| `format_version` | `"0.2"`; a reader rejects an unknown major version and ignores unknown members of a newer minor one |
| `library`, `model` | optional: the library, and the model's symbol prefix |

**Sections.** Content lives in the groups `/input`, `/parameters` or `/solution`. A file may
carry several; readers ignore those they do not need.

**Names** come from the ABI: input fields from `P_schema`, blocks from `P_block_name`.

**Values.** Floats are binary64, integers int64, booleans int8 0/1. A block is stored with
its `dims` (an attribute), column-major (first index fastest).

**Reading against a model.** Names must exist with the right kind, type and length, and a
solution's sizes must match the model's. A failed check is an error unless the consumer
offers an explicit override.

## Model-specific formats

A library may read and write its own formats (an OPF model reads MATPOWER cases), through
the readers and writers tiers: `P_new_from_<name>`, `P_set_from_<name>`,
`P_start_from_<name>`, `P_write_<name>`. They add to the HDF5 formats and never replace
them; `P_default_reader` / `P_default_writer` name the default. Consumers write a model
format as `name:path`.

Reference files: [conformance/](../conformance/).
