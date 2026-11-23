/* The cnlp solver interface for Ipopt (through its C interface).
 *
 *   cc -std=c99 -O2 -shared -fPIC -I../.. $(pkg-config --cflags ipopt) \
 *      -o libcnlp_ipopt.so cnlp_ipopt.c $(pkg-config --libs ipopt)
 *
 * A maximization is solved as the minimization of -f. Ipopt's multipliers for that
 * problem satisfy -grad f + J'y - zL + zU = 0, which is the cnlp convention with s = -1,
 * so they are returned as they are.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <coin-or/IpStdCInterface.h>
#include "cnlp_solver.h"

CNLP_DECLARE_SOLVER

typedef struct {
    const cnlp_problem* p;
    double sign;                  /* 1 minimize, -1 maximize */
    ipindex *jr, *jc, *hr, *hc;   /* the structures, as Ipopt's index type */
    int64_t iterations;
    int failed;                   /* a callback reported an error */
} adapter;

static bool f_cb(ipindex n, ipnumber* x, bool nx, ipnumber* f, UserDataPtr u) {
    (void)n; (void)nx; adapter* a = u;
    double v;
    if (a->p->obj(a->p->ctx, x, &v)) { a->failed = 1; return false; }
    *f = a->sign * v;
    return true;
}
static bool g_cb(ipindex n, ipnumber* x, bool nx, ipnumber* g, UserDataPtr u) {
    (void)nx; adapter* a = u;
    if (a->p->grad(a->p->ctx, x, g)) { a->failed = 1; return false; }
    if (a->sign < 0) for (ipindex i = 0; i < n; i++) g[i] = -g[i];
    return true;
}
static bool c_cb(ipindex n, ipnumber* x, bool nx, ipindex m, ipnumber* c, UserDataPtr u) {
    (void)n; (void)nx; (void)m; adapter* a = u;
    if (a->p->cons(a->p->ctx, x, c)) { a->failed = 1; return false; }
    return true;
}
static bool j_cb(ipindex n, ipnumber* x, bool nx, ipindex m, ipindex nnz, ipindex* r, ipindex* c,
                 ipnumber* v, UserDataPtr u) {
    (void)n; (void)nx; (void)m; adapter* a = u;
    if (!v) { memcpy(r, a->jr, nnz * sizeof *r); memcpy(c, a->jc, nnz * sizeof *c); return true; }
    if (a->p->jac(a->p->ctx, x, v)) { a->failed = 1; return false; }
    return true;
}
static bool h_cb(ipindex n, ipnumber* x, bool nx, ipnumber w, ipindex m, ipnumber* y, bool ny,
                 ipindex nnz, ipindex* r, ipindex* c, ipnumber* v, UserDataPtr u) {
    (void)n; (void)nx; (void)m; (void)ny; adapter* a = u;
    if (!v) { memcpy(r, a->hr, nnz * sizeof *r); memcpy(c, a->hc, nnz * sizeof *c); return true; }
    if (a->p->hess(a->p->ctx, x, y, a->sign * w, v)) { a->failed = 1; return false; }
    return true;
}
static bool iter_cb(ipindex mode, ipindex k, ipnumber f, ipnumber inf_pr, ipnumber inf_du, ipnumber mu,
                    ipnumber d_norm, ipnumber reg, ipnumber a_du, ipnumber a_pr, ipindex ls, UserDataPtr u) {
    (void)mode; (void)f; (void)inf_pr; (void)inf_du; (void)mu; (void)d_norm; (void)reg;
    (void)a_du; (void)a_pr; (void)ls;
    ((adapter*)u)->iterations = k;
    return true;
}

/* Ipopt built for Julia (Ipopt_jll) calls BLAS through libblastrampoline, which serves
 * nothing until a BLAS is forwarded to it. Julia does that when it starts; outside Julia
 * the adapter does it, once, with the LP64 OpenBLAS bundled next to it (lib/), unless
 * LBT_DEFAULT_LIBS already names one. */
static void forward_blas(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    const char* e = getenv("LBT_DEFAULT_LIBS");
    if (e && *e) return;
    int32_t (*fwd)(const char*, int32_t, int32_t, const char*) = (int32_t (*)(const char*, int32_t, int32_t, const char*))
        dlsym(RTLD_DEFAULT, "lbt_forward");
    if (!fwd) return;   /* an Ipopt linked to a BLAS directly */
    Dl_info info;
    if (!dladdr((void*)forward_blas, &info) || !info.dli_fname) return;
    char path[4096];
    snprintf(path, sizeof path, "%s", info.dli_fname);
    char* slash = strrchr(path, '/');
    if (!slash) return;
    snprintf(slash + 1, sizeof path - (size_t)(slash + 1 - path), "libopenblas.so");
    fwd(path, 0, 0, NULL);   /* add to what is forwarded: a model library in the same process may have forwarded its own */
}

int64_t cnlp_solver_version(void) { return CNLP_SOLVER_VERSION; }

int64_t cnlp_solver_name(uint8_t* buf, int64_t cap) {
    static const char name[] = "ipopt";
    int64_t n = (int64_t)strlen(name);
    if (buf && cap > 0) memcpy(buf, name, (size_t)(n < cap ? n : cap));
    return n;
}

static ipindex* structure(int64_t nnz, int64_t (*f)(void*, int64_t*, int64_t*), void* ctx, ipindex** cols) {
    int64_t* r = malloc((nnz ? nnz : 1) * sizeof *r);
    int64_t* c = malloc((nnz ? nnz : 1) * sizeof *c);
    ipindex* ri = malloc((nnz ? nnz : 1) * sizeof *ri);
    *cols = malloc((nnz ? nnz : 1) * sizeof **cols);
    if (nnz && f(ctx, r, c)) { free(r); free(c); free(ri); free(*cols); return NULL; }
    for (int64_t k = 0; k < nnz; k++) { ri[k] = (ipindex)r[k]; (*cols)[k] = (ipindex)c[k]; }
    free(r); free(c);
    return ri;
}

static double clip(double b) { return isinf(b) ? (b > 0 ? 2e19 : -2e19) : b; }

/* An option value is passed as an integer, else a number, else a string: Ipopt rejects a
 * value of the wrong type, so the first kind it accepts is the right one. */
static int set_option(IpoptProblem ip, const char* k, const char* v) {
    char* end;
    long long i = strtoll(v, &end, 10);
    if (*v && !*end && AddIpoptIntOption(ip, (char*)k, (ipindex)i)) return 1;
    double d = strtod(v, &end);
    if (*v && !*end && AddIpoptNumOption(ip, (char*)k, d)) return 1;
    return AddIpoptStrOption(ip, (char*)k, (char*)v);
}

int64_t cnlp_solve(const cnlp_problem* p, int64_t nopt, const char* const* keys, const char* const* vals,
                   cnlp_result* r) {
    forward_blas();
    adapter a = { p, p->minimize ? 1.0 : -1.0, NULL, NULL, NULL, NULL, -1, 0 };
    int64_t n = p->nvar, m = p->ncon, ret = 2;
    double *lv = malloc((n ? n : 1) * sizeof(double)), *uv = malloc((n ? n : 1) * sizeof(double));
    double *lc = malloc((m ? m : 1) * sizeof(double)), *uc = malloc((m ? m : 1) * sizeof(double));
    for (int64_t i = 0; i < n; i++) { lv[i] = clip(p->lvar[i]); uv[i] = clip(p->uvar[i]); }
    for (int64_t i = 0; i < m; i++) { lc[i] = clip(p->lcon[i]); uc[i] = clip(p->ucon[i]); }
    IpoptProblem ip = NULL;
    a.jr = structure(p->nnzj, p->jac_structure, p->ctx, &a.jc);
    a.hr = structure(p->nnzh, p->hess_structure, p->ctx, &a.hc);
    if (!a.jr || !a.hr) goto done;
    ip = CreateIpoptProblem((ipindex)n, lv, uv, (ipindex)m, lc, uc, (ipindex)p->nnzj, (ipindex)p->nnzh,
                            1 /* 1-based */, f_cb, c_cb, g_cb, j_cb, h_cb);
    if (!ip) goto done;
    SetIntermediateCallback(ip, iter_cb);
    for (int64_t k = 0; k < nopt; k++)
        if (!set_option(ip, keys[k], vals[k])) {
            snprintf(r->solver_status, sizeof r->solver_status, "unknown or invalid option %s=%s", keys[k], vals[k]);
            ret = 1; goto done;
        }
    if (p->y0 && p->zl0 && p->zu0) AddIpoptStrOption(ip, "warm_start_init_point", "yes");
    memcpy(r->x, p->x0, n * sizeof(double));
    if (p->y0) memcpy(r->y, p->y0, m * sizeof(double)); else memset(r->y, 0, m * sizeof(double));
    if (p->zl0) memcpy(r->zl, p->zl0, n * sizeof(double)); else memset(r->zl, 0, n * sizeof(double));
    if (p->zu0) memcpy(r->zu, p->zu0, n * sizeof(double)); else memset(r->zu, 0, n * sizeof(double));
    double fs = 0;
    enum ApplicationReturnStatus st = IpoptSolve(ip, r->x, NULL, &fs, r->y, r->zl, r->zu, &a);
    r->objective = a.sign * fs;
    r->iterations = a.iterations;
    switch (st) {
        case Solve_Succeeded: r->status = CNLP_SOLVED; break;
        case Solved_To_Acceptable_Level: r->status = CNLP_ACCEPTABLE; break;
        case Infeasible_Problem_Detected: r->status = CNLP_INFEASIBLE; break;
        case Diverging_Iterates: r->status = CNLP_UNBOUNDED; break;
        case Maximum_Iterations_Exceeded: r->status = CNLP_ITERATION_LIMIT; break;
        case Maximum_CpuTime_Exceeded: case Maximum_WallTime_Exceeded: r->status = CNLP_TIME_LIMIT; break;
        default: r->status = CNLP_ERROR;
    }
    snprintf(r->solver_status, sizeof r->solver_status, "Ipopt status %d%s", (int)st,
             a.failed ? " (a model callback failed)" : "");
    ret = 0;
done:
    if (ip) FreeIpoptProblem(ip);
    free(a.jr); free(a.jc); free(a.hr); free(a.hc);
    free(lv); free(uv); free(lc); free(uc);
    return ret;
}
