/* Load a packed cnlp library in a plain C process (no Julia in it) and evaluate every model
 * that can be instantiated from one integer or from nothing: the way C, Python and
 * `cnlp solve` use it. A library that needs anything its bundle lacks fails here.
 *
 *   cc -std=c99 -o smoke smoke.c -ldl && ./smoke path/to/libNAME.so [N]
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: smoke LIB [N]\n"); return 2; }
    int64_t size = argc > 2 ? atoll(argv[2]) : 10;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "cannot load %s: %s\n", argv[1], dlerror()); return 1; }
    int64_t (*ver)(void) = dlsym(h, "cnlp_abi_version");
    int64_t (*nm)(void) = dlsym(h, "cnlp_nmodels");
    int64_t (*mn)(int64_t, uint8_t*, int64_t) = dlsym(h, "cnlp_model_name");
    if (!ver || ver() != 2 || !nm || !mn) { fprintf(stderr, "not a cnlp 0.2 library with a model list\n"); return 1; }
    int failed = 0, evaluated = 0;
    for (int64_t k = 0; k < nm(); k++) {
        char p[128], s[256], a[1024];
        int64_t L = mn(k, (uint8_t*)p, sizeof p - 1); p[L > 0 ? L : 0] = 0;
        #define SYM(name) (snprintf(s, sizeof s, "%s_%s", p, name), dlsym(h, s))
        int64_t (*at)(uint8_t*, int64_t) = SYM("argtype");
        L = at ? at((uint8_t*)a, sizeof a - 1) : -1; a[L > 0 ? L : 0] = 0;
        int64_t (*nw)(int64_t) = SYM("new");
        if (!nw || (L > 0 && (strncmp(a, "int|", 4) || strchr(a, ',')))) { printf("%-24s loads\n", p); continue; }
        int64_t id = nw(L > 0 ? size : 0);
        int64_t n = id > 0 ? ((int64_t (*)(int64_t))SYM("nvar"))(id) : -1;
        int64_t m = id > 0 ? ((int64_t (*)(int64_t))SYM("ncon"))(id) : -1;
        if (n < 0 || m < 0) { printf("%-24s FAILED to instantiate\n", p); failed = 1; continue; }
        double *x = calloc(n + 1, 8), *l = calloc(n + 1, 8), *u = calloc(n + 1, 8), *lc = calloc(m + 1, 8), *uc = calloc(m + 1, 8), f = NAN;
        int64_t st = ((int64_t (*)(int64_t, double*, double*, double*, double*, double*))SYM("meta"))(id, x, l, u, lc, uc);
        st |= ((int64_t (*)(int64_t, const double*, double*))SYM("obj"))(id, x, &f);
        if (st || !isfinite(f)) { printf("%-24s FAILED to evaluate\n", p); failed = 1; }
        else { printf("%-24s ok: nvar %lld, ncon %lld, f(x0) = %g\n", p, (long long)n, (long long)m, f); evaluated++; }
        free(x); free(l); free(u); free(lc); free(uc);
    }
    printf("%d evaluated%s\n", evaluated, failed ? ", some FAILED" : "");
    return failed;
}
