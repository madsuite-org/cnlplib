/* cnlp_solver.h: the cnlp solver interface, version 0.1.
 *
 * A cnlp solver is a shared library that solves one instance of a cnlp model. The caller
 * (`cnlp solve`, or any program) loads the model, instantiates it, and hands the solver a
 * cnlp_problem: the instance's sizes, bounds and starting point, and its evaluation
 * callbacks. The solver fills a cnlp_result. Everything else (finding and instantiating
 * models, readers, writers, solution files) is the caller's, so a solver implements only
 * this.
 *
 * CONVENTIONS are cnlp.h's: int64_t integers, 1-based sparse structures, the Hessian as
 * the lower triangle of obj_weight * Hess f + sum_i y_i Hess c_i, status 0 = success.
 * Multipliers follow formats/solution.md: the Lagrangian
 *     L = s f(x) + y'c(x) - zL'(x - lvar) + zU'(x - uvar),   s = 1 (min), -1 (max),
 * is stationary at the solution, with zL, zU >= 0. ±INFINITY marks a missing bound.
 */
#ifndef CNLP_SOLVER_H
#define CNLP_SOLVER_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CNLP_SOLVER_VERSION 1   /* major * 1000 + minor: 0.1 */

/* Solver-neutral statuses; the solver's own is in cnlp_result.solver_status. */
enum {
    CNLP_SOLVED = 0, CNLP_ACCEPTABLE = 1, CNLP_INFEASIBLE = 2, CNLP_UNBOUNDED = 3,
    CNLP_ITERATION_LIMIT = 4, CNLP_TIME_LIMIT = 5, CNLP_ERROR = 6
};

/* One instance. `ctx` is passed back to every callback; the callbacks return 0 on success. */
typedef struct cnlp_problem {
    void* ctx;
    int64_t nvar, ncon, nnzj, nnzh;
    int64_t minimize;                     /* 1 minimize, 0 maximize */
    const double *x0, *lvar, *uvar;       /* nvar */
    const double *lcon, *ucon;            /* ncon */
    const double *y0, *zl0, *zu0;         /* a warm start's multipliers, or NULL */
    int64_t (*obj)(void* ctx, const double* x, double* f);
    int64_t (*grad)(void* ctx, const double* x, double* g);
    int64_t (*cons)(void* ctx, const double* x, double* c);
    int64_t (*jac_structure)(void* ctx, int64_t* rows, int64_t* cols);
    int64_t (*jac)(void* ctx, const double* x, double* vals);
    int64_t (*hess_structure)(void* ctx, int64_t* rows, int64_t* cols);
    int64_t (*hess)(void* ctx, const double* x, const double* y, double obj_weight, double* vals);
} cnlp_problem;

/* Filled by the solver. The arrays are the caller's (x, zl, zu: nvar; y: ncon). */
typedef struct cnlp_result {
    double *x, *y, *zl, *zu;
    double objective;                     /* f(x), in the model's sense */
    int64_t status;                       /* CNLP_SOLVED, ... */
    int64_t iterations;                   /* -1 if unknown */
    char solver_status[128];              /* the solver's own status, NUL-terminated */
} cnlp_result;

/* A cnlp solver library exports these three. Options are key=value strings, passed to
 * the solver as they are (an unknown option is an error, status 1). cnlp_solve returns 0
 * when it ran (whatever result->status says), 1 for bad options, 2 for a failure. */
#define CNLP_DECLARE_SOLVER                                                   \
    int64_t cnlp_solver_version(void);                                        \
    int64_t cnlp_solver_name(uint8_t* buf, int64_t cap);                      \
    int64_t cnlp_solve(const cnlp_problem* problem, int64_t noptions,         \
                       const char* const* keys, const char* const* values,     \
                       cnlp_result* result);

#ifdef __cplusplus
}
#endif
#endif
