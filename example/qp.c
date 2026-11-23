/* qp.c — a complete cnlp model library in plain C.
 *
 * One size-parametric model under the prefix `qp`:
 *
 *     minimize   sum_i (x_i - 1)^2       over x in R^n
 *     subject to x_1 + x_2 = 1
 *
 * It also shows two optional tiers: evaluation in double-double (formats/numbers.md,
 * `double_binary64`), a format C has no type for, and a solution writer for JSON, an
 * encoding the standard formats (HDF5) leave to libraries and consumers.
 *
 * Build:    cc -shared -fPIC -O2 -o libqp.so qp.c -lm
 * Consume:  see solve.c next to this file — or any cnlp consumer:
 *           CNLPModels.jl:  CNLPModel("path/to/libqp.so", 4)
 *           cnlpmodels-py:  cnlpmodels.CModel("path/to/libqp.so", 4)
 *
 * Including cnlp.h and expanding the X-macros first is what buys prototype
 * checking: if a definition below drifts from the ABI, the compiler says so.
 * CNLP_DECLARE_MODEL(qp) expands P##_new, P##_obj, ... to qp_new, qp_obj, ...
 */
#include <math.h>
#include <stdio.h>
#include "../cnlp.h"

CNLP_DECLARE_MODEL(qp)
CNLP_DECLARE_FORMATS(qp)
CNLP_DECLARE_WRITERS(qp)
int64_t qp_write_json(int64_t id, const char* path, const double* x, const double* y,
                      const double* zl, const double* zu, double objective, const char* status);

/* Instances are just sizes behind ids; a static table suffices. */
#define QP_MAX_INSTANCES 32
static int64_t qp_sizes[QP_MAX_INSTANCES];
static int64_t qp_count = 0;

static int64_t valid(int64_t id) { return id >= 1 && id <= qp_count; }

/* ── instantiation ──────────────────────────────────────────────────────── */

int64_t qp_nargs(void) { return 1; }             /* one value closes it     */

int64_t qp_argtype(uint8_t* buf, int64_t cap) {  /* "int|size", copy-out    */
    static const char sig[] = "int|size";
    int64_t len = (int64_t)sizeof(sig) - 1;
    for (int64_t i = 0; i < cap && i < len; i++) buf[i] = (uint8_t)sig[i];
    return len;
}

int64_t qp_new(int64_t n) {                      /* -> id > 0; 0 = failure  */
    if (n < 2 || qp_count >= QP_MAX_INSTANCES) return 0;
    qp_sizes[qp_count++] = n;
    return qp_count;
}

/* ── metadata ───────────────────────────────────────────────────────────── */

int64_t qp_nvar(int64_t id) { return valid(id) ? qp_sizes[id - 1] : -1; }
int64_t qp_ncon(int64_t id) { return valid(id) ? 1 : -1; }
int64_t qp_nnzj(int64_t id) { return valid(id) ? 2 : -1; }  /* d(x1+x2)     */
int64_t qp_nnzh(int64_t id) { return valid(id) ? qp_sizes[id - 1] : -1; }

int64_t qp_meta(int64_t id, double* x0, double* lvar, double* uvar,
                double* lcon, double* ucon) {
    if (!valid(id)) return 1;
    int64_t n = qp_sizes[id - 1];
    for (int64_t i = 0; i < n; i++) {
        x0[i] = 0.0;
        lvar[i] = -INFINITY;
        uvar[i] = INFINITY;
    }
    lcon[0] = 1.0;                               /* x1 + x2 = 1: both bounds */
    ucon[0] = 1.0;
    return 0;
}

/* ── evaluation ─────────────────────────────────────────────────────────── */

int64_t qp_obj(int64_t id, const double* x, double* out) {
    if (!valid(id)) return 1;
    int64_t n = qp_sizes[id - 1];
    double f = 0.0;
    for (int64_t i = 0; i < n; i++) f += (x[i] - 1.0) * (x[i] - 1.0);
    *out = f;
    return 0;
}

int64_t qp_grad(int64_t id, const double* x, double* g) {
    if (!valid(id)) return 1;
    int64_t n = qp_sizes[id - 1];
    for (int64_t i = 0; i < n; i++) g[i] = 2.0 * (x[i] - 1.0);
    return 0;
}

int64_t qp_cons(int64_t id, const double* x, double* c) {
    if (!valid(id)) return 1;
    c[0] = x[0] + x[1];
    return 0;
}

/* Indices are 1-based inside the ABI. */
int64_t qp_jac_structure(int64_t id, int64_t* rows, int64_t* cols) {
    if (!valid(id)) return 1;
    rows[0] = 1; cols[0] = 1;                    /* dc1/dx1 */
    rows[1] = 1; cols[1] = 2;                    /* dc1/dx2 */
    return 0;
}

int64_t qp_jac(int64_t id, const double* x, double* vals) {
    (void)x;
    if (!valid(id)) return 1;
    vals[0] = 1.0;
    vals[1] = 1.0;
    return 0;
}

/* Lower triangle of obj_weight*grad^2 f + y1*grad^2 c1.  The objective's
 * Hessian is 2I and the constraint's is zero, so the multiplier drops out —
 * but the signature still takes y, as the ABI requires. */
int64_t qp_hess_structure(int64_t id, int64_t* rows, int64_t* cols) {
    if (!valid(id)) return 1;
    int64_t n = qp_sizes[id - 1];
    for (int64_t i = 0; i < n; i++) { rows[i] = i + 1; cols[i] = i + 1; }
    return 0;
}

int64_t qp_hess(int64_t id, const double* x, const double* y,
                double obj_weight, double* vals) {
    (void)x; (void)y;
    if (!valid(id)) return 1;
    int64_t n = qp_sizes[id - 1];
    for (int64_t i = 0; i < n; i++) vals[i] = 2.0 * obj_weight;
    return 0;
}

/* ── double-double (number-format tier) ─────────────────────────────────────
 * A double-double is two doubles {hi, lo} whose exact sum is the value, with
 * |lo| <= ulp(hi)/2: about 106 bits of significand.  The arithmetic below is
 * the classic error-free transformations (Dekker, Knuth); fma gives the exact
 * product error. */

typedef struct { double hi, lo; } dd;

static dd two_sum(double a, double b) {
    double s = a + b, bb = s - a;
    return (dd){ s, (a - (s - bb)) + (b - bb) };
}
static dd quick_two_sum(double a, double b) {     /* requires |a| >= |b| */
    double s = a + b;
    return (dd){ s, b - (s - a) };
}
static dd dd_add(dd a, dd b) {
    dd s = two_sum(a.hi, b.hi);
    return quick_two_sum(s.hi, s.lo + a.lo + b.lo);
}
static dd dd_mul(dd a, dd b) {
    double p = a.hi * b.hi, e = fma(a.hi, b.hi, -p);
    return quick_two_sum(p, e + a.hi * b.lo + a.lo * b.hi);
}
static dd dd_scale2(dd a) { return (dd){ 2.0 * a.hi, 2.0 * a.lo }; }   /* exact */
static dd dd_of(double a) { return (dd){ a, 0.0 }; }

/* The model offers one extra format, index 0. */
int64_t qp_nformats(int64_t id) { return valid(id) ? 1 : -1; }

int64_t qp_format_name(int64_t id, int64_t k, uint8_t* buf, int64_t cap) {
    if (!valid(id) || k != 0) return -1;
    static const char nm[] = "double_binary64";
    int64_t len = (int64_t)sizeof(nm) - 1;
    for (int64_t i = 0; i < cap && i < len; i++) buf[i] = (uint8_t)nm[i];
    return len;
}

/* Every buffer is an array of dd elements; prec is ignored (fixed format). */
#define QP_X_CHECK(id, k) do { if (!valid(id)) return 1; if ((k) != 0) return 1; } while (0)

int64_t qp_xmeta(int64_t id, int64_t k, int64_t prec, void* x0, void* lvar,
                 void* uvar, void* lcon, void* ucon) {
    (void)prec; QP_X_CHECK(id, k);
    int64_t n = qp_sizes[id - 1];
    dd *x = x0, *lv = lvar, *uv = uvar, *lc = lcon, *uc = ucon;
    for (int64_t i = 0; i < n; i++) {
        x[i] = dd_of(0.0); lv[i] = dd_of(-INFINITY); uv[i] = dd_of(INFINITY);
    }
    lc[0] = dd_of(1.0); uc[0] = dd_of(1.0);
    return 0;
}

int64_t qp_xobj(int64_t id, int64_t k, int64_t prec, const void* xv, void* out) {
    (void)prec; QP_X_CHECK(id, k);
    const dd* x = xv;
    dd f = dd_of(0.0);
    for (int64_t i = 0; i < qp_sizes[id - 1]; i++) {
        dd d = dd_add(x[i], dd_of(-1.0));
        f = dd_add(f, dd_mul(d, d));
    }
    *(dd*)out = f;
    return 0;
}

int64_t qp_xgrad(int64_t id, int64_t k, int64_t prec, const void* xv, void* gv) {
    (void)prec; QP_X_CHECK(id, k);
    const dd* x = xv; dd* g = gv;
    for (int64_t i = 0; i < qp_sizes[id - 1]; i++) g[i] = dd_scale2(dd_add(x[i], dd_of(-1.0)));
    return 0;
}

int64_t qp_xcons(int64_t id, int64_t k, int64_t prec, const void* xv, void* cv) {
    (void)prec; QP_X_CHECK(id, k);
    const dd* x = xv;
    ((dd*)cv)[0] = dd_add(x[0], x[1]);
    return 0;
}

int64_t qp_xjac(int64_t id, int64_t k, int64_t prec, const void* xv, void* vals) {
    (void)prec; (void)xv; QP_X_CHECK(id, k);
    ((dd*)vals)[0] = dd_of(1.0);
    ((dd*)vals)[1] = dd_of(1.0);
    return 0;
}

int64_t qp_xhess(int64_t id, int64_t k, int64_t prec, const void* xv, const void* yv,
                 const void* obj_weight, void* vals) {
    (void)prec; (void)xv; (void)yv; QP_X_CHECK(id, k);
    dd w2 = dd_scale2(*(const dd*)obj_weight);
    for (int64_t i = 0; i < qp_sizes[id - 1]; i++) ((dd*)vals)[i] = w2;
    return 0;
}

/* ── a solution writer: JSON (writers tier) ─────────────────────────────────
 * The cnlp-solution content in JSON: an encoding the spec leaves to libraries
 * and consumers.  Same members as the HDF5 file; non-finite numbers are null. */

static const char* qp_writers[] = { "json" };

int64_t qp_nwriters(void) { return 1; }

int64_t qp_writer_name(int64_t k, uint8_t* buf, int64_t cap) {
    if (k != 0) return -1;
    const char* nm = qp_writers[0];
    int64_t len = 4;
    for (int64_t i = 0; i < cap && i < len; i++) buf[i] = (uint8_t)nm[i];
    return len;
}

int64_t qp_default_writer(int64_t id, uint8_t* buf, int64_t cap) {   /* none */
    (void)buf; (void)cap;
    return valid(id) ? 0 : -1;
}

static void json_num(FILE* f, double v) {
    if (isfinite(v)) fprintf(f, "%.17g", v); else fputs("null", f);
}
static void json_arr(FILE* f, const char* key, const double* v, int64_t n) {
    fprintf(f, "\"%s\": [", key);
    for (int64_t i = 0; i < n; i++) { if (i) fputs(", ", f); json_num(f, v[i]); }
    fputs("]", f);
}

int64_t qp_write_json(int64_t id, const char* path, const double* x, const double* y,
                      const double* zl, const double* zu, double objective, const char* status) {
    if (!valid(id)) return 1;
    FILE* f = fopen(path, "w");
    if (!f) return 2;
    int64_t n = qp_sizes[id - 1];
    fprintf(f, "{\"format\": \"cnlp-solution\", \"format_version\": \"0.2\", \"model\": \"qp\",\n");
    fprintf(f, " \"nvar\": %lld, \"ncon\": 1,\n \"solution\": {", (long long)n);
    json_arr(f, "solution", x, n); fputs(", ", f);
    json_arr(f, "multipliers", y, 1); fputs(", ", f);
    json_arr(f, "multipliers_L", zl, n); fputs(", ", f);
    json_arr(f, "multipliers_U", zu, n);
    fputs(", \"objective\": ", f); json_num(f, objective);
    fprintf(f, ", \"minimize\": true, \"status\": \"%s\", \"iterations\": -1}}\n",
            status ? status : "error");
    return fclose(f) == 0 ? 0 : 2;
}

/* ── the library-level catalogue (optional tier, three lines here) ──────── */

int64_t cnlp_abi_version(void) { return 2; }   /* 0.2 */
int64_t cnlp_nmodels(void) { return 1; }

int64_t cnlp_model_name(int64_t k, uint8_t* buf, int64_t cap) {
    if (k != 0) return -1;
    static const char nm[] = "qp";
    int64_t len = (int64_t)sizeof(nm) - 1;
    for (int64_t i = 0; i < cap && i < len; i++) buf[i] = (uint8_t)nm[i];
    return len;
}
