/* cnlp_solver.h: the cnlp solver interface, version 0.2.
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

#define CNLP_SOLVER_VERSION 2   /* major * 1000 + minor: 0.2 */

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

/* Device evaluation (0.2; optional). With a problem whose model has a device tier (cnlp.h,
 * P##_device), the caller may also hand the solver the device callbacks: the same
 * evaluations on device pointers (x, g, c, vals, y on the device; obj_d's `f` is one host
 * double). The structures, bounds, starting point and the result stay on the host, through
 * cnlp_problem and cnlp_result. `device` is the memory space, as P##_device reports it. */
enum { CNLP_DEVICE_CUDA = 1 };

typedef struct cnlp_device_problem {
    int64_t device;                       /* CNLP_DEVICE_CUDA */
    int64_t (*obj_d)(void* ctx, const double* x, double* f);
    int64_t (*grad_d)(void* ctx, const double* x, double* g);
    int64_t (*cons_d)(void* ctx, const double* x, double* c);
    int64_t (*jac_d)(void* ctx, const double* x, double* vals);
    int64_t (*hess_d)(void* ctx, const double* x, const double* y, double obj_weight, double* vals);
} cnlp_device_problem;

/* A cnlp solver library exports these three. Options are key=value strings, passed to
 * the solver as they are (an unknown option is an error, status 1). cnlp_solve returns 0
 * when it ran (whatever result->status says), 1 for bad options, 2 for a failure. */
#define CNLP_DECLARE_SOLVER                                                   \
    int64_t cnlp_solver_version(void);                                        \
    int64_t cnlp_solver_name(uint8_t* buf, int64_t cap);                      \
    int64_t cnlp_solve(const cnlp_problem* problem, int64_t noptions,         \
                       const char* const* keys, const char* const* values,     \
                       cnlp_result* result);

/* A solver that runs on a device also exports cnlp_solve_device: the problem (host
 * callbacks for the structures; ctx is shared), its device callbacks, and the options and
 * result of cnlp_solve. It returns 1, like an invalid option, for a device it does not
 * support. A caller uses it only when the solver exports it. */
#define CNLP_DECLARE_DEVICE_SOLVER                                            \
    int64_t cnlp_solve_device(const cnlp_problem* problem,                    \
                              const cnlp_device_problem* device,              \
                              int64_t noptions, const char* const* keys,      \
                              const char* const* values, cnlp_result* result);

#ifdef __cplusplus
}
#endif
#endif
