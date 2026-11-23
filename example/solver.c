/* solver.c: a cnlp solver, as a template for wrapping your own (see cnlp_solver.h).
 *
 *   cc -std=c99 -O2 -shared -fPIC -I.. -o libcnlp_newton.so solver.c -lm
 *   cnlp solve ./libqp.so:qp 4 --solver ./libcnlp_newton.so
 *   python3 ../conformance/check_solver.py ./libcnlp_newton.so
 *
 * The method is Newton's method on the KKT conditions of an equality-constrained problem
 * with no variable bounds, with dense linear algebra: enough to solve the example QP, and
 * small enough to read. A real adapter replaces solve_kkt with a call to its solver.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../cnlp_solver.h"

CNLP_DECLARE_SOLVER

int64_t cnlp_solver_version(void) { return CNLP_SOLVER_VERSION; }

int64_t cnlp_solver_name(uint8_t* buf, int64_t cap) {
    static const char name[] = "newton";
    int64_t n = (int64_t)strlen(name);
    if (buf && cap > 0) memcpy(buf, name, (size_t)(n < cap ? n : cap));
    return n;
}

/* Solve A z = b in place (A is k x k, row-major) by Gaussian elimination with partial pivoting. */
static int lu_solve(int64_t k, double* A, double* b) {
    for (int64_t c = 0; c < k; c++) {
        int64_t p = c;
        for (int64_t r = c + 1; r < k; r++) if (fabs(A[r * k + c]) > fabs(A[p * k + c])) p = r;
        if (fabs(A[p * k + c]) < 1e-14) return 1;
        if (p != c) {
            for (int64_t j = 0; j < k; j++) { double t = A[c * k + j]; A[c * k + j] = A[p * k + j]; A[p * k + j] = t; }
            double t = b[c]; b[c] = b[p]; b[p] = t;
        }
        for (int64_t r = c + 1; r < k; r++) {
            double f = A[r * k + c] / A[c * k + c];
            for (int64_t j = c; j < k; j++) A[r * k + j] -= f * A[c * k + j];
            b[r] -= f * b[c];
        }
    }
    for (int64_t r = k - 1; r >= 0; r--) {
        for (int64_t j = r + 1; j < k; j++) b[r] -= A[r * k + j] * b[j];
        b[r] /= A[r * k + r];
    }
    return 0;
}

int64_t cnlp_solve(const cnlp_problem* p, int64_t nopt, const char* const* keys, const char* const* vals,
                   cnlp_result* r) {
    int64_t n = p->nvar, m = p->ncon, k = n + m, maxit = 50;
    double tol = 1e-10, s = p->minimize ? 1.0 : -1.0;   /* s f is minimized */
    for (int64_t i = 0; i < nopt; i++) {
        if (!strcmp(keys[i], "max_iter")) maxit = atoll(vals[i]);
        else if (!strcmp(keys[i], "tol")) tol = atof(vals[i]);
        else if (strcmp(keys[i], "print_level") && strcmp(keys[i], "sb")) {   /* accepted and ignored */
            snprintf(r->solver_status, sizeof r->solver_status, "unknown option %s", keys[i]);
            return 1;
        }
    }
    for (int64_t i = 0; i < n; i++)
        if (!isinf(p->lvar[i]) || !isinf(p->uvar[i])) {
            snprintf(r->solver_status, sizeof r->solver_status, "this method takes no variable bounds");
            return 2;
        }
    for (int64_t i = 0; i < m; i++)
        if (p->lcon[i] != p->ucon[i]) {
            snprintf(r->solver_status, sizeof r->solver_status, "this method takes equality constraints only");
            return 2;
        }
    int64_t *jr = malloc((p->nnzj + 1) * 8), *jc = malloc((p->nnzj + 1) * 8);
    int64_t *hr = malloc((p->nnzh + 1) * 8), *hc = malloc((p->nnzh + 1) * 8);
    double *jv = malloc((p->nnzj + 1) * 8), *hv = malloc((p->nnzh + 1) * 8);
    double *g = malloc((n + 1) * 8), *c = malloc((m + 1) * 8), *K = malloc((k * k + 1) * 8), *rhs = malloc((k + 1) * 8);
    p->jac_structure(p->ctx, jr, jc);
    p->hess_structure(p->ctx, hr, hc);
    memcpy(r->x, p->x0, n * 8);
    memset(r->y, 0, m * 8);
    r->status = CNLP_ITERATION_LIMIT;
    int64_t it;
    for (it = 0; it < maxit; it++) {
        /* residuals: s grad f + J'y (stationarity) and c - lcon (feasibility) */
        p->grad(p->ctx, r->x, g); p->cons(p->ctx, r->x, c); p->jac(p->ctx, r->x, jv);
        for (int64_t i = 0; i < n; i++) rhs[i] = -s * g[i];
        for (int64_t e = 0; e < p->nnzj; e++) rhs[jc[e] - 1] -= jv[e] * r->y[jr[e] - 1];
        for (int64_t i = 0; i < m; i++) rhs[n + i] = -(c[i] - p->lcon[i]);
        double res = 0;
        for (int64_t i = 0; i < k; i++) res = fmax(res, fabs(rhs[i]));
        if (res <= tol) { r->status = CNLP_SOLVED; break; }
        /* the KKT matrix [H J'; J 0], H the Hessian of the Lagrangian of s f */
        memset(K, 0, k * k * 8);
        p->hess(p->ctx, r->x, r->y, s, hv);
        for (int64_t e = 0; e < p->nnzh; e++) {
            int64_t a = hr[e] - 1, b = hc[e] - 1;
            K[a * k + b] += hv[e];
            if (a != b) K[b * k + a] += hv[e];
        }
        for (int64_t e = 0; e < p->nnzj; e++) {
            int64_t a = n + jr[e] - 1, b = jc[e] - 1;
            K[a * k + b] += jv[e]; K[b * k + a] += jv[e];
        }
        if (lu_solve(k, K, rhs)) { r->status = CNLP_ERROR; break; }
        for (int64_t i = 0; i < n; i++) r->x[i] += rhs[i];
        for (int64_t i = 0; i < m; i++) r->y[i] += rhs[n + i];
    }
    memset(r->zl, 0, n * 8); memset(r->zu, 0, n * 8);   /* no bounds, so no bound multipliers */
    p->obj(p->ctx, r->x, &r->objective);
    r->iterations = it;
    snprintf(r->solver_status, sizeof r->solver_status, "newton: %s after %lld iterations",
             r->status == CNLP_SOLVED ? "converged" : "stopped", (long long)it);
    free(jr); free(jc); free(hr); free(hc); free(jv); free(hv); free(g); free(c); free(K); free(rhs);
    return 0;
}
