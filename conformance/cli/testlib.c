/* A cnlp library for the end-to-end test of `cnlp`: three models with the same problem,
 *     min sum_i (x_i - t)^2   s.t.   x_1 + x_2 = 1      (n >= 2 variables),
 * whose solution is x_1 = x_2 = 1/2, x_i = t otherwise, f = 2 (1/2 - t)^2 and y = 2 (t - 1/2),
 * instantiated three ways:
 *   pn   P_new(n)                         t = 1
 *   pb   the builder: n (int), t (f64)
 *   ps   P_new_str(path), a text file "n t"; readers txt (instance, start), writer txt (default)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int64_t N[256], T_count = 0; static double T[256];
static int64_t make(int64_t n, double t) {
    if (n < 2 || T_count >= 256) return 0;
    N[T_count] = n; T[T_count] = t; return ++T_count;
}
static int ok(int64_t id) { return id >= 1 && id <= T_count; }
static int64_t copyout(const char* s, uint8_t* buf, int64_t cap) {
    int64_t n = (int64_t)strlen(s);
    if (buf && cap > 0) memcpy(buf, s, (size_t)(n < cap ? n : cap));
    return n;
}

/* the shared evaluation, defined once per prefix */
#define CORE(P)                                                                         \
    int64_t P##_nvar(int64_t id) { return ok(id) ? N[id - 1] : -1; }                   \
    int64_t P##_ncon(int64_t id) { return ok(id) ? 1 : -1; }                            \
    int64_t P##_nnzj(int64_t id) { return ok(id) ? 2 : -1; }                            \
    int64_t P##_nnzh(int64_t id) { return ok(id) ? N[id - 1] : -1; }                    \
    int64_t P##_meta(int64_t id, double* x0, double* lv, double* uv, double* lc,        \
                     double* uc) {                                                      \
        if (!ok(id)) return 1;                                                          \
        for (int64_t i = 0; i < N[id - 1]; i++) {                                       \
            x0[i] = 0; lv[i] = -INFINITY; uv[i] = INFINITY;                             \
        }                                                                               \
        lc[0] = uc[0] = 1;                                                              \
        return 0;                                                                       \
    }                                                                                   \
    int64_t P##_obj(int64_t id, const double* x, double* f) {                           \
        if (!ok(id)) return 1;                                                          \
        double s = 0;                                                                   \
        for (int64_t i = 0; i < N[id - 1]; i++)                                         \
            s += (x[i] - T[id - 1]) * (x[i] - T[id - 1]);                               \
        *f = s;                                                                         \
        return 0;                                                                       \
    }                                                                                   \
    int64_t P##_grad(int64_t id, const double* x, double* g) {                          \
        if (!ok(id)) return 1;                                                          \
        for (int64_t i = 0; i < N[id - 1]; i++) g[i] = 2 * (x[i] - T[id - 1]);          \
        return 0;                                                                       \
    }                                                                                   \
    int64_t P##_cons(int64_t id, const double* x, double* c) {                          \
        if (!ok(id)) return 1;                                                          \
        c[0] = x[0] + x[1];                                                             \
        return 0;                                                                       \
    }                                                                                   \
    int64_t P##_jac_structure(int64_t id, int64_t* r, int64_t* c) {                     \
        if (!ok(id)) return 1;                                                          \
        r[0] = r[1] = 1; c[0] = 1; c[1] = 2;                                            \
        return 0;                                                                       \
    }                                                                                   \
    int64_t P##_jac(int64_t id, const double* x, double* v) {                           \
        (void)x;                                                                        \
        if (!ok(id)) return 1;                                                          \
        v[0] = v[1] = 1;                                                                \
        return 0;                                                                       \
    }                                                                                   \
    int64_t P##_hess_structure(int64_t id, int64_t* r, int64_t* c) {                    \
        if (!ok(id)) return 1;                                                          \
        for (int64_t i = 0; i < N[id - 1]; i++) r[i] = c[i] = i + 1;                    \
        return 0;                                                                       \
    }                                                                                   \
    int64_t P##_hess(int64_t id, const double* x, const double* y, double w,            \
                     double* v) {                                                       \
        (void)x; (void)y;                                                               \
        if (!ok(id)) return 1;                                                          \
        for (int64_t i = 0; i < N[id - 1]; i++) v[i] = 2 * w;                           \
        return 0;                                                                       \
    }

CORE(pn)
CORE(pb)
CORE(ps)

/* pn: one integer */
int64_t pn_nargs(void) { return 1; }
int64_t pn_argtype(uint8_t* b, int64_t c) { return copyout("int|size", b, c); }
int64_t pn_new(int64_t n) { return make(n, 1.0); }

/* pb: the builder, two scalar fields */
static int64_t B_n[64]; static double B_t[64]; static int B_set[64]; static int64_t B_count = 0;
int64_t pb_nargs(void) { return 2; }
int64_t pb_argtype(uint8_t* b, int64_t c) { return copyout("int|size,f64|target", b, c); }
int64_t pb_schema(uint8_t* b, int64_t c) {
    return copyout("{\"fields\":[{\"name\":\"n\",\"kind\":\"scalar\",\"type\":\"i64\"},"
                   "{\"name\":\"t\",\"kind\":\"scalar\",\"type\":\"f64\"}]}", b, c);
}
int64_t pb_data_begin(void) { if (B_count >= 64) return 0; B_set[B_count] = 0; return ++B_count; }
int64_t pb_set_scalar_i64(int64_t b, const char* f, int64_t v) {
    if (b < 1 || b > B_count || strcmp(f, "n")) return 1;
    B_n[b - 1] = v; B_set[b - 1] |= 1;
    return 0;
}
int64_t pb_set_scalar_f64(int64_t b, const char* f, double v) {
    if (b < 1 || b > B_count || strcmp(f, "t")) return 1;
    B_t[b - 1] = v; B_set[b - 1] |= 2;
    return 0;
}
int64_t pb_data_ready(int64_t b) { return b >= 1 && b <= B_count && B_set[b - 1] == 3; }
int64_t pb_new_from_data(int64_t b) { return pb_data_ready(b) ? make(B_n[b - 1], B_t[b - 1]) : 0; }

/* ps: a text file "n t"; readers and a writer of the same format */
static int64_t read_nt(const char* path) {
    FILE* f = fopen(path, "r"); long long n; double t;
    if (!f) return 0;
    int k = fscanf(f, "%lld %lf", &n, &t); fclose(f);
    return k == 2 ? make((int64_t)n, t) : 0;
}
int64_t ps_nargs(void) { return 1; }
int64_t ps_argtype(uint8_t* b, int64_t c) { return copyout("string|case file", b, c); }
int64_t ps_new_str(const char* path) { return read_nt(path); }
int64_t ps_nreaders(void) { return 2; }
int64_t ps_reader_name(int64_t k, uint8_t* b, int64_t c) { return k == 0 ? copyout("txt", b, c) : k == 1 ? copyout("xstart", b, c) : -1; }
int64_t ps_reader_kind(int64_t k) { return k == 0 ? 0 : k == 1 ? 2 : -1; }
int64_t ps_default_reader(uint8_t* b, int64_t c) { return copyout("txt", b, c); }
int64_t ps_new_from_txt(const char* path) { return read_nt(path); }
/* a start: the x values, one per line; multipliers 0 */
int64_t ps_start_from_xstart(int64_t id, const char* path, double* x0, double* y0, double* zl, double* zu) {
    if (!ok(id)) return 1;
    FILE* f = fopen(path, "r"); if (!f) return 2;
    int64_t n = N[id - 1];
    for (int64_t i = 0; i < n; i++) if (fscanf(f, "%lf", &x0[i]) != 1) { fclose(f); return 3; }
    fclose(f); y0[0] = 0; for (int64_t i = 0; i < n; i++) zl[i] = zu[i] = 0;
    return 0;
}
int64_t ps_nwriters(void) { return 1; }
int64_t ps_writer_name(int64_t k, uint8_t* b, int64_t c) { return k == 0 ? copyout("txt", b, c) : -1; }
int64_t ps_default_writer(int64_t id, uint8_t* b, int64_t c) { return ok(id) ? copyout("txt", b, c) : -1; }
int64_t ps_write_txt(int64_t id, const char* path, const double* x, const double* y, const double* zl,
                     const double* zu, double obj, const char* status) {
    (void)zl; (void)zu; if (!ok(id)) return 1;
    FILE* f = fopen(path, "w"); if (!f) return 2;
    fprintf(f, "status %s\nobjective %.12g\ny %.12g\nx", status ? status : "", obj, y[0]);
    for (int64_t i = 0; i < N[id - 1]; i++) fprintf(f, " %.12g", x[i]);
    fprintf(f, "\n"); fclose(f); return 0;
}

int64_t ps_report_instance(int64_t id, uint8_t* b, int64_t c) {
    if (!ok(id)) return -1;
    char t[128]; snprintf(t, sizeof t, "ps instance: n = %lld\n", (long long)N[id - 1]);
    return copyout(t, b, c);
}
int64_t ps_report_solution(int64_t id, const double* x, const double* y, const double* zl, const double* zu,
                           double obj, const char* status, uint8_t* b, int64_t c) {
    (void)y; (void)zl; (void)zu; if (!ok(id)) return -1;
    double sx = 0; for (int64_t i = 0; i < N[id - 1]; i++) sx += x[i];
    char t[256]; snprintf(t, sizeof t, "ps solution: %s, objective %.6g, sum(x) = %.6g\n", status ? status : "?", obj, sx);
    return copyout(t, b, c);
}

/* the library level */
int64_t cnlp_abi_version(void) { return 2; }
int64_t cnlp_nmodels(void) { return 3; }
int64_t cnlp_model_name(int64_t k, uint8_t* b, int64_t c) {
    return k == 0 ? copyout("pn", b, c) : k == 1 ? copyout("pb", b, c) : k == 2 ? copyout("ps", b, c) : -1;
}
