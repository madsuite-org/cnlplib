/* solve.c — a plain-C consumer of the model in qp.c.
 *
 * Linking against the library and declaring the symbols through cnlp.h is
 * all it takes — no dlopen needed when the prefix is known at compile time:
 *
 *     cc -shared -fPIC -O2 -o libqp.so qp.c -lm
 *     cc -O2 -o solve solve.c -L. -lqp -Wl,-rpath,'$ORIGIN' -lm
 *     ./solve
 *
 * (A consumer that discovers models at run time uses dlopen/dlsym with the
 * names from cnlp_model_name instead — that is what the Julia and Python
 * consumers do.)
 */
#include <stdio.h>
#include <stdlib.h>
#include "../cnlp.h"

CNLP_DECLARE_MODEL(qp)
CNLP_DECLARE_FORMATS(qp)

typedef struct { double hi, lo; } dd;   /* formats/numbers.md: double_binary64 */

int main(void) {
    /* Instantiate at a size of our choosing — the library was compiled
     * knowing no particular n. */
    int64_t id = qp_new(4);
    if (id <= 0) { fprintf(stderr, "qp_new failed\n"); return 1; }

    int64_t n = qp_nvar(id), m = qp_ncon(id);
    printf("qp instance %lld: n = %lld, m = %lld\n", (long long)id, (long long)n, (long long)m);

    double *x0 = malloc(n * sizeof *x0), *lv = malloc(n * sizeof *lv),
           *uv = malloc(n * sizeof *uv), lc[1], uc[1];
    if (qp_meta(id, x0, lv, uv, lc, uc) != 0) return 1;

    /* Evaluate at the optimum x* = (0.5, 0.5, 1, 1): f = 0.5, c = 1. */
    double xs[] = { 0.5, 0.5, 1.0, 1.0 };
    double f, c[1], *g = malloc(n * sizeof *g);
    qp_obj(id, xs, &f);
    qp_cons(id, xs, c);
    qp_grad(id, xs, g);
    printf("f(x*) = %g   c(x*) = %g   g(x*) = [%g %g %g %g]\n",
           f, c[0], g[0], g[1], g[2], g[3]);

    /* The same objective in double-double, at x = 1 + 2^-60 in every
     * coordinate.  In double that point rounds to x = 1, where f = 0; in
     * double-double the objective is 4 * 2^-120 exactly. */
    uint8_t name[64];
    int64_t len = qp_format_name(id, 0, name, sizeof name - 1);
    name[len] = 0;
    dd xd[4], fd;
    for (int i = 0; i < 4; i++) xd[i] = (dd){ 1.0, 0x1p-60 };
    if (qp_xobj(id, 0, 0, xd, &fd) != 0) return 1;
    double xr[] = { 1.0 + 0x1p-60, 1.0 + 0x1p-60, 1.0 + 0x1p-60, 1.0 + 0x1p-60 }, fr;
    qp_obj(id, xr, &fr);
    printf("format 0 = %s:  f(1 + 2^-60) = %g + %g  (double gives %g)\n",
           (char*)name, fd.hi, fd.lo, fr);

    free(x0); free(lv); free(uv); free(g);
    return f == 0.5 && c[0] == 1.0 && fd.hi == 0x1p-118 && fr == 0.0 ? 0 : 1;
}
