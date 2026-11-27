/* cnlpio: the standard solution file of a cnlp model (formats/solution.md), for every
 * consumer: the `cnlp` command, CNLPLib.jl and the Python cnlplib call this one library
 * (libcnlpio, installed with `cnlp` as $CNLPLIB_HOME/lib/libcnlpio.so), so a solution
 * written by one is the same file from all three.
 *
 * Multipliers are in the cnlp sign convention (formats/solution.md, section 2). Functions
 * return 0 on success, else nonzero with a message copied to `err` (up to `cap` bytes,
 * NUL-terminated; `err` may be NULL). */
#ifndef CNLPIO_H
#define CNLPIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CNLPIO_VERSION 1

int64_t cnlpio_version(void);

/* Write a solution of instance `id` of the model `model` (its symbol prefix) of the library
 * opened as `lib` (a dlopen handle, or NULL), named `library` in the file. With a handle and
 * the model's named blocks, each block's slice is written too, under /blocks/<kind>/<name>
 * (constraint values from the model's P_cons at x, parameter values from P_get_value).
 * `status` is one of formats/solution.md's status words. */
int64_t cnlpio_write_solution(const char* path, void* lib, const char* library, const char* model,
                              int64_t id, int64_t nvar, int64_t ncon, int64_t minimize,
                              const double* x, const double* y, const double* zl, const double* zu,
                              double objective, const char* status, int64_t iterations,
                              const char* solver, const char* solver_status, char* err, int64_t cap);

/* Read a solution file as a start for a model of `nvar` variables and `ncon` constraints:
 * x, y, zl, zu (caller-allocated), checked against the file's sizes. */
int64_t cnlpio_read_start(const char* path, int64_t nvar, int64_t ncon,
                          double* x, double* y, double* zl, double* zu, char* err, int64_t cap);

#ifdef __cplusplus
}
#endif
#endif
