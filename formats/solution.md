# cnlp-solution: a solution of a model

Version 0.2. Conventions: [README.md](README.md).

For `min (or max) f(x)  s.t.  lcon <= c(x) <= ucon,  lvar <= x <= uvar`, with `nvar`
variables and `ncon` constraints.

## 1. Content

Root attributes: `nvar`, `ncon`.

Group `/solution` (as ExaModelsIO writes a stored solution):

| Field | Type | Meaning |
|---|---|---|
| `solution` | f64[nvar] | `x` |
| `multipliers` | f64[ncon] | `y` |
| `multipliers_L`, `multipliers_U` | f64[nvar] | `zL, zU >= 0` |
| `objective` | f64 | `f(x)` in the model's sense |
| `minimize` | bool | `P_minimize` |
| `status` | string | `solved`, `acceptable`, `infeasible`, `unbounded`, `iteration_limit`, `time_limit` or `error` |
| `iterations` | integer | `-1` if unknown |
| `solver`, `solver_status` | string | optional |

With the layout tier, `/blocks/<kind>/<name>/` may add each block's slice in its shape,
with `dims`:
variables `value`, `multiplier_L`, `multiplier_U`; constraints `value` (= `c(x)`),
`multiplier`; parameters `value`. The flat arrays are the reference. Parameters may
instead be a `/parameters` section ([parameters.md](parameters.md)).

## 2. Signs

The multipliers make this Lagrangian stationary:

    L = σ f(x) + yᵀ c(x) − zLᵀ (x − lvar) + zUᵀ (x − uvar),   σ = 1 (min), −1 (max),

the multipliers of `min σf` and of the ABI Hessian with `obj_weight = σ`. Writers and
readers convert (measured on HS71, both senses):

| Source | Stationarity | Conversion |
|---|---|---|
| NLPModels results (MadNLP, NLPModelsIpopt) | `∇f + Jᵀy − zL + zU = 0` | maximization: negate `y`, `zL`, `zU` |
| MathOptInterface duals | `σ∇f − Jᵀy − zL − zU = 0` | `y = −y_MOI`, `zU = −zU_MOI` |

## 3. Use

With no format given, a consumer writes with the model's `P_default_writer` if it has one,
otherwise this format. [cnlpio.h](../cnlpio.h) writes and reads it (libcnlpio, installed with
the `cnlp` command), for any consumer.

A solution read against a model gives a warm start (`x`, `y`, `zL`, `zU`); its `nvar` and
`ncon` must match the model's. Further checks are the consumer's.
