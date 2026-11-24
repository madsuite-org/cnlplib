/* cnlp-solve: solve a model of a cnlp library with a cnlp solver (`cnlp solve`).
 *
 *   cnlp solve LIB[:MODEL] [INSTANCE...] [-s START] [-o OUTPUT]... [--solver NAME] [key=value...]
 *
 * LIB is @name (an installed library, found on CNLPLIB_PATH and in $CNLPLIB_HOME/lib) or a
 * path (a library file or bundle directory); MODEL may be left out when the library has one.
 * INSTANCE is what the model is instantiated from, by its P_argtype: an integer, a file
 * (a path, `reader:path` for one of the model's readers, or the name of a file in an
 * installed data set), or one value per builder field. START is `reader:path` or a
 * cnlp-solution file; OUTPUT is `writer:path`, or a path written with the model's default
 * writer, else as a cnlp-solution HDF5 file. key=value options go to the solver.
 *
 *   cc -std=c99 -O2 -I.. -o cnlp-solve cnlp-solve.c -ldl -lhdf5   (includes cnlpio.c)
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <hdf5.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "cnlp_solver.h"

static void die(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fputs("cnlp solve: ", stderr); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap); exit(2);
}
static int isfile(const char* p) { struct stat s; return stat(p, &s) == 0 && S_ISREG(s.st_mode); }
static int isdir(const char* p) { struct stat s; return stat(p, &s) == 0 && S_ISDIR(s.st_mode); }
static const char* home_dir(void) {
    static char h[PATH_MAX];
    const char* e = getenv("CNLPLIB_HOME");
    if (e && *e) return e;
    snprintf(h, sizeof h, "%s/.cnlplib", getenv("HOME") ? getenv("HOME") : ".");
    return h;
}

/* ── the model library ─────────────────────────────────────────────────────────── */

static void* LIB;
static char PRE[128];
static void* sym(const char* s) {
    char n[256]; snprintf(n, sizeof n, "%s_%s", PRE, s);
    return dlsym(LIB, n);
}
static void* need(const char* s) {
    void* f = sym(s);
    if (!f) die("`%s` has no %s_%s", PRE, PRE, s);
    return f;
}
/* a copy-out string (`P_name(buf, cap) -> length`), or "" */
static char* copyout(int64_t (*f)(uint8_t*, int64_t)) {
    static char b[4096];
    int64_t n = f ? f((uint8_t*)b, sizeof b - 1) : 0;
    b[n < 0 ? 0 : (n < (int64_t)sizeof b - 1 ? n : (int64_t)sizeof b - 1)] = 0;
    return b;
}

/* @name or a path -> the library file */
static void resolve_lib(const char* spec, char* out) {
    if (spec[0] == '@') {
        const char* name = spec + 1;
        char dirs[8192]; const char* e = getenv("CNLPLIB_PATH");
        snprintf(dirs, sizeof dirs, "%s%s%s/lib", e && *e ? e : "", e && *e ? ":" : "", home_dir());
        for (char* d = strtok(dirs, ":"); d; d = strtok(NULL, ":")) {
            const char* pats[] = { "%s/%s/lib/lib%s.so", "%s/lib%s.so", "%s/lib/lib%s.so" };
            for (int k = 0; k < 3; k++) {
                if (k == 0) snprintf(out, PATH_MAX, pats[0], d, name, name);
                else snprintf(out, PATH_MAX, pats[k], d, name);
                if (isfile(out)) return;
            }
        }
        die("no installed library `%s` (see: cnlp list; install with: cnlp install %s)", name, name);
    }
    if (isfile(spec)) { snprintf(out, PATH_MAX, "%s", spec); return; }
    if (isdir(spec)) {
        char b[PATH_MAX]; snprintf(b, sizeof b, "%s", spec);
        size_t L = strlen(b); while (L > 1 && b[L - 1] == '/') b[--L] = 0;
        const char* base = strrchr(b, '/') ? strrchr(b, '/') + 1 : b;
        snprintf(out, PATH_MAX, "%s/lib/lib%s.so", b, base);
        if (isfile(out)) return;
    }
    die("no library at `%s`", spec);
}

/* ── data files: a path, or a file in an installed data set ─────────────────────── */

/* A data-set file by name: the exact file name, or the name without its extension
 * (`pglib_opf_case118_ieee` finds pglib/pglib_opf_case118_ieee.m). Nothing looser: a name
 * that matches no file is an error, with the files whose names contain it as suggestions. */
static int find_in(const char* dir, const char* want, char* out, int depth) {
    DIR* d = opendir(dir); if (!d) return 0;
    struct dirent* e; int found = 0;
    char p[PATH_MAX];
    while (!found && (e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        if (isdir(p)) { if (depth < 6) found = find_in(p, want, out, depth + 1); continue; }
        char stem[NAME_MAX + 1]; snprintf(stem, sizeof stem, "%s", e->d_name);
        char* dot = strrchr(stem, '.'); if (dot) *dot = 0;
        if (!strcmp(e->d_name, want) || !strcmp(stem, want)) { snprintf(out, PATH_MAX, "%s", p); found = 1; }
    }
    closedir(d); return found;
}
/* up to *left files under dir whose names contain want, as "  stem  (relative path)" lines */
static char SUG[10][PATH_MAX + NAME_MAX + 8];
static void suggest_in(const char* dir, const char* root, const char* want, int* left, int depth) {
    DIR* d = opendir(dir); if (!d) return;
    struct dirent* e; char p[PATH_MAX];
    while (*left > 0 && (e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        if (isdir(p)) { if (depth < 6) suggest_in(p, root, want, left, depth + 1); continue; }
        if (strstr(e->d_name, want)) {
            char stem[NAME_MAX + 1]; snprintf(stem, sizeof stem, "%s", e->d_name);
            char* dot = strrchr(stem, '.'); if (dot) *dot = 0;
            snprintf(SUG[10 - *left], sizeof SUG[0], "  %s  (%s)\n", stem, p + strlen(root) + 1); (*left)--;
        }
    }
    closedir(d);
}
static int cmp_str(const void* a, const void* b) { return strcmp((const char*)a, (const char*)b); }
static const char* resolve_file(const char* a) {
    static char p[PATH_MAX];
    if (isfile(a) || strchr(a, '/')) return a;
    char d[PATH_MAX]; snprintf(d, sizeof d, "%s/data", home_dir());
    if (find_in(d, a, p, 0)) return p;
    int left = 10; suggest_in(d, d, a, &left, 0);
    qsort(SUG, 10 - left, sizeof SUG[0], cmp_str);
    fprintf(stderr, "cnlp solve: no file `%s`: not a path, and no installed data-set file of that name%s\n",
            a, left < 10 ? "; did you mean" : "");
    for (int k = 0; k < 10 - left; k++) fputs(SUG[k], stderr);
    exit(1);
}

/* `name:rest` where name is one of the model's readers or writers of `kind`, else NULL */
static int has_reader(const char* name, int64_t kind) {
    int64_t (*nr)(void) = sym("nreaders");
    int64_t (*rn)(int64_t, uint8_t*, int64_t) = sym("reader_name");
    int64_t (*rk)(int64_t) = sym("reader_kind");
    if (!nr || !rn || !rk) return 0;
    for (int64_t k = 0; k < nr(); k++) {
        char b[256]; int64_t n = rn(k, (uint8_t*)b, sizeof b - 1); b[n > 0 ? n : 0] = 0;
        if (!strcmp(b, name) && rk(k) == kind) return 1;
    }
    return 0;
}
static int has_writer(int64_t id, const char* name) {
    int64_t (*nw)(void) = sym("nwriters");
    int64_t (*wn)(int64_t, uint8_t*, int64_t) = sym("writer_name");
    if (!nw || !wn) return 0;
    for (int64_t k = 0; k < nw(); k++) {
        char b[256]; int64_t n = wn(k, (uint8_t*)b, sizeof b - 1); b[n > 0 ? n : 0] = 0;
        if (!strcmp(b, name)) return 1;
    }
    (void)id; return 0;
}
static char* split_prefix(const char* s, char* name) {
    const char* c = strchr(s, ':');
    if (!c || c == s || (size_t)(c - s) >= 128) return NULL;
    memcpy(name, s, c - s); name[c - s] = 0;
    return (char*)c + 1;
}

/* ── instantiation ────────────────────────────────────────────────────────────── */

static int64_t instantiate(int nargs, char** args) {
    char spec[4096]; snprintf(spec, sizeof spec, "%s", copyout(need("argtype")));
    int items = spec[0] ? 1 : 0;
    for (char* c = spec; *c; c++) items += *c == ',';
    char rd[128]; const char* rest;
    if (nargs == 1 && (rest = split_prefix(args[0], rd)) && has_reader(rd, 0)) {
        char f[256]; snprintf(f, sizeof f, "new_from_%s", rd);
        int64_t (*nf)(const char*) = need(f);
        return nf(resolve_file(rest));
    }
    if (items == 0) {
        if (nargs) die("`%s` is a fixed model: it takes no instance arguments", PRE);
        int64_t (*nw)(int64_t) = need("new"); return nw(0);
    }
    if (nargs != items) die("`%s` takes %d instance argument%s (%s), got %d", PRE, items, items == 1 ? "" : "s", spec, nargs);
    if (items == 1 && !strncmp(spec, "int|", 4) && sym("new")) {
        char* end; long long n = strtoll(args[0], &end, 10);
        if (*end) die("`%s` takes an integer (%s), got `%s`", PRE, spec + 4, args[0]);
        int64_t (*nw)(int64_t) = need("new"); return nw(n);
    }
    if (items == 1 && !strncmp(spec, "string|", 7)) {
        int64_t (*ns)(const char*) = need("new_str"); return ns(resolve_file(args[0]));
    }
    /* the builder, for scalar fields: names from P_schema, in order */
    char* schema = copyout(need("schema"));
    int64_t (*begin)(void) = need("data_begin");
    int64_t (*si)(int64_t, const char*, int64_t) = need("set_scalar_i64");
    int64_t (*sf)(int64_t, const char*, double) = need("set_scalar_f64");
    int64_t b = begin();
    char* q = schema; char* item = strtok(spec, ",");
    for (int k = 0; k < nargs; k++, item = strtok(NULL, ",")) {
        q = strstr(q, "\"name\""); if (!q) die("cannot read the schema of `%s`", PRE);
        q = strchr(q + 6, '"') + 1; char field[256]; size_t L = strcspn(q, "\"");
        memcpy(field, q, L); field[L] = 0; q += L;
        char* end;
        if (!strncmp(item, "int|", 4)) {
            long long v = strtoll(args[k], &end, 10);
            if (*end) die("field `%s` of `%s` takes an integer, got `%s`", field, PRE, args[k]);
            if (si(b, field, v)) die("`%s` refused %s = %s", PRE, field, args[k]);
        } else if (!strncmp(item, "f64|", 4)) {
            double v = strtod(args[k], &end);
            if (*end) die("field `%s` of `%s` takes a number, got `%s`", field, PRE, args[k]);
            if (sf(b, field, v)) die("`%s` refused %s = %s", PRE, field, args[k]);
        } else die("field `%s` of `%s` (%s) cannot be given on the command line", field, PRE, item);
    }
    int64_t (*ready)(int64_t) = need("data_ready"); int64_t (*nfd)(int64_t) = need("new_from_data");
    if (ready(b) != 1) die("the instance data of `%s` is incomplete", PRE);
    return nfd(b);
}

/* ── the problem handed to the solver ─────────────────────────────────────────── */

typedef struct { int64_t id; void *obj, *grad, *cons, *js, *jac, *hs, *hess, *obj_d, *grad_d, *cons_d, *jac_d, *hess_d; } instance;
typedef int64_t (*ev)(int64_t, const double*, double*);
typedef int64_t (*st)(int64_t, int64_t*, int64_t*);
typedef int64_t (*hv)(int64_t, const double*, const double*, double, double*);
static int64_t m_obj(void* c, const double* x, double* f) { instance* m = c; return ((ev)m->obj)(m->id, x, f); }
static int64_t m_grad(void* c, const double* x, double* g) { instance* m = c; return ((ev)m->grad)(m->id, x, g); }
static int64_t m_cons(void* c, const double* x, double* v) { instance* m = c; return ((ev)m->cons)(m->id, x, v); }
static int64_t m_js(void* c, int64_t* r, int64_t* k) { instance* m = c; return ((st)m->js)(m->id, r, k); }
static int64_t m_jac(void* c, const double* x, double* v) { instance* m = c; return ((ev)m->jac)(m->id, x, v); }
static int64_t m_hs(void* c, int64_t* r, int64_t* k) { instance* m = c; return ((st)m->hs)(m->id, r, k); }
static int64_t m_hess(void* c, const double* x, const double* y, double w, double* v) {
    instance* m = c; return ((hv)m->hess)(m->id, x, y, w, v);
}
static int64_t m_obj_d(void* c, const double* x, double* f) { instance* m = c; return ((ev)m->obj_d)(m->id, x, f); }
static int64_t m_grad_d(void* c, const double* x, double* g) { instance* m = c; return ((ev)m->grad_d)(m->id, x, g); }
static int64_t m_cons_d(void* c, const double* x, double* v) { instance* m = c; return ((ev)m->cons_d)(m->id, x, v); }
static int64_t m_jac_d(void* c, const double* x, double* v) { instance* m = c; return ((ev)m->jac_d)(m->id, x, v); }
static int64_t m_hess_d(void* c, const double* x, const double* y, double w, double* v) {
    instance* m = c; return ((hv)m->hess_d)(m->id, x, y, w, v);
}

/* ── cnlp-solution HDF5 (formats/solution.md): libcnlpio, compiled in ───────────── */

#include "cnlpio.c"
static const char* STATUS[] = { "solved", "acceptable", "infeasible", "unbounded", "iteration_limit", "time_limit", "error" };

/* ── the solver ───────────────────────────────────────────────────────────────── */

static void* load_solver(const char* s, char* path) {
    if (strchr(s, '/')) snprintf(path, PATH_MAX, "%s", s);
    else {
        snprintf(path, PATH_MAX, "%s/solvers/%s/lib/libcnlp_%s.so", home_dir(), s, s);
        if (!isfile(path)) die("no solver `%s` (install one with: cnlp install %s)", s, s);
    }
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) die("cannot load the solver %s: %s", path, dlerror());
    int64_t (*ver)(void) = dlsym(h, "cnlp_solver_version");
    if (!ver || ver() / 1000 != CNLP_SOLVER_VERSION / 1000) die("%s is not a cnlp solver of interface %d.x", path, CNLP_SOLVER_VERSION / 1000);
    return h;
}

/* a model's report (CNLP_DECLARE_REPORTS), printed when it has one; `call` copies it out */
static void print_report(int64_t (*call)(uint8_t*, int64_t, void*), void* arg) {
    int64_t L = call(NULL, 0, arg);
    if (L <= 0) return;
    uint8_t* b = malloc(L + 1);
    int64_t n = call(b, L, arg);
    if (n > 0) { b[n < L ? n : L] = 0; fputs((char*)b, stdout); if (b[(n < L ? n : L) - 1] != '\n') putchar('\n'); }
    free(b);
}
typedef struct { int64_t id; void* f; } rep_inst;
static int64_t call_inst(uint8_t* b, int64_t c, void* a) {
    rep_inst* r = a; return ((int64_t (*)(int64_t, uint8_t*, int64_t))r->f)(r->id, b, c);
}
typedef struct { int64_t id; void* f; cnlp_result* R; const char* status; } rep_sol;
static int64_t call_sol(uint8_t* b, int64_t c, void* a) {
    rep_sol* r = a;
    return ((int64_t (*)(int64_t, const double*, const double*, const double*, const double*, double, const char*, uint8_t*, int64_t))r->f)(
        r->id, r->R->x, r->R->y, r->R->zl, r->R->zu, r->R->objective, r->status, b, c);
}

/* the library's models, one per line, with the instance arguments each takes */
static void list_models(const char* spec, int64_t (*nm)(void), int64_t (*mn)(int64_t, uint8_t*, int64_t)) {
    if (!nm || !mn) return;
    fprintf(stderr, "models:\n");
    for (int64_t k = 0; k < nm(); k++) {
        char name[128]; int64_t L = mn(k, (uint8_t*)name, sizeof name - 1); name[L > 0 ? L : 0] = 0;
        snprintf(PRE, sizeof PRE, "%s", name);
        const char* at = copyout(sym("argtype"));
        fprintf(stderr, "  %s:%s%s%s\n", spec, name, at[0] ? "  " : "", at);
    }
}

static void usage(void) {
    puts("usage: cnlp solve LIB[:MODEL] [INSTANCE...] [-s START] [-o OUTPUT]... [--solver NAME] [--device D] [key=value...]\n"
         "  LIB      @name (installed) or a path      MODEL   left out when the library has one\n"
         "  INSTANCE an integer, a file (path, reader:path, or a data-set file name), or builder values\n"
         "  START    reader:path or a cnlp-solution file\n"
         "  OUTPUT   writer:path, or a path (the model's default writer, else cnlp-solution HDF5)\n"
         "  --solver ipopt (default) or a path to a cnlp solver library; key=value options go to it\n"
         "  --device auto (default: the GPU when the model and the solver both run there), cpu or cuda");
}

#include "privatize.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char** argv) {
    if (argc >= 2 && !strcmp(argv[1], "--privatize")) return privatize(argc - 2, argv + 2);
    double t_start = now();
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) { usage(); return argc < 2; }
    const char *libspec = NULL, *start = NULL, *solver = "ipopt", *device = "auto";
    char *pos[64], *outs[16], *keys[128], *vals[128]; int npos = 0, nout = 0, nopt = 0;
    for (int i = 1; i < argc; i++) {
        char* a = argv[i];
        if (!strcmp(a, "-s") || !strcmp(a, "--start")) { if (++i == argc) die("%s needs a value", a); start = argv[i]; }
        else if (!strcmp(a, "-o") || !strcmp(a, "--output")) { if (++i == argc) die("%s needs a value", a); if (nout < 16) outs[nout++] = argv[i]; }
        else if (!strcmp(a, "--solver")) { if (++i == argc) die("--solver needs a value"); solver = argv[i]; }
        else if (!strcmp(a, "--device")) { if (++i == argc) die("--device needs a value"); device = argv[i]; }
        else if (a[0] == '-' && a[1]) die("unknown option %s (see: cnlp solve --help)", a);
        else if (!libspec) libspec = a;
        else if (strchr(a, '=') && a[0] != '=') { char* e = strchr(a, '='); *e = 0; keys[nopt] = a; vals[nopt++] = e + 1; }
        else if (npos < 64) pos[npos++] = a;
    }
    /* LIB[:MODEL], split at the last colon when what follows names no path */
    char spec[PATH_MAX], model[128] = "", lib[PATH_MAX];
    snprintf(spec, sizeof spec, "%s", libspec);
    char* c = strrchr(spec, ':');
    if (c && !strchr(c, '/')) { snprintf(model, sizeof model, "%s", c + 1); *c = 0; }
    resolve_lib(spec, lib);
    double t_lib0 = now();
    LIB = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!LIB) die("cannot load %s: %s", lib, dlerror());
    int64_t (*abi)(void) = dlsym(LIB, "cnlp_abi_version");
    if (!abi || abi() != 2) die("%s is not a cnlp 0.2 library", lib);
    int64_t (*nm)(void) = dlsym(LIB, "cnlp_nmodels");
    int64_t (*mn)(int64_t, uint8_t*, int64_t) = dlsym(LIB, "cnlp_model_name");
    if (!model[0]) {
        if (!nm || !mn || nm() != 1) {
            fprintf(stderr, "cnlp solve: %s has several models; give one as LIB:MODEL\n", spec);
            list_models(spec, nm, mn); exit(1);
        }
        int64_t L = mn(0, (uint8_t*)model, sizeof model - 1); model[L > 0 ? L : 0] = 0;
    }
    snprintf(PRE, sizeof PRE, "%s", model);
    if (!sym("nvar")) {
        fprintf(stderr, "cnlp solve: %s has no model `%s`\n", spec, model);
        list_models(spec, nm, mn); exit(1);
    }

    double t_lib = now() - t_lib0, t_inst0 = now();
    int64_t id = instantiate(npos, pos);
    if (id <= 0) die("`%s` could not be instantiated from the given arguments", PRE);
    instance M = { id, need("obj"), need("grad"), need("cons"), need("jac_structure"), need("jac"),
                need("hess_structure"), need("hess"), NULL, NULL, NULL, NULL, NULL };
    int64_t n = ((int64_t (*)(int64_t))need("nvar"))(id), m = ((int64_t (*)(int64_t))need("ncon"))(id);
    int64_t nj = ((int64_t (*)(int64_t))need("nnzj"))(id), nh = ((int64_t (*)(int64_t))need("nnzh"))(id);
    int64_t (*mini)(int64_t) = sym("minimize");
    int minimize = mini ? mini(id) != 0 : 1;
    double *x0 = calloc(n + 1, 8), *lv = calloc(n + 1, 8), *uv = calloc(n + 1, 8), *lc = calloc(m + 1, 8), *uc = calloc(m + 1, 8);
    if (((int64_t (*)(int64_t, double*, double*, double*, double*, double*))need("meta"))(id, x0, lv, uv, lc, uc))
        die("%s_meta failed", PRE);
    double *y0 = NULL, *zl0 = NULL, *zu0 = NULL;
    if (start) {
        y0 = calloc(m + 1, 8); zl0 = calloc(n + 1, 8); zu0 = calloc(n + 1, 8);
        char rd[128]; const char* rest = split_prefix(start, rd);
        if (rest && has_reader(rd, 2)) {
            char f[256]; snprintf(f, sizeof f, "start_from_%s", rd);
            int64_t (*sf)(int64_t, const char*, double*, double*, double*, double*) = need(f);
            if (sf(id, resolve_file(rest), x0, y0, zl0, zu0)) die("reader `%s` could not read %s", rd, rest);
        } else {
            char e[512];
            if (cnlpio_read_start(start, n, m, x0, y0, zl0, zu0, e, sizeof e)) die("cannot read the start: %s", e);
        }
    }
    cnlp_problem P = { &M, n, m, nj, nh, minimize, x0, lv, uv, lc, uc, y0, zl0, zu0,
                       m_obj, m_grad, m_cons, m_js, m_jac, m_hs, m_hess };
    cnlp_result R = { calloc(n + 1, 8), calloc(m + 1, 8), calloc(n + 1, 8), calloc(n + 1, 8), 0, CNLP_ERROR, -1, "" };

    double t_inst = now() - t_inst0, t_sol0 = now();
    char spath[PATH_MAX]; void* sh = load_solver(solver, spath);
    double t_sol = now() - t_sol0;
    int64_t (*solve)(const cnlp_problem*, int64_t, const char* const*, const char* const*, cnlp_result*) = dlsym(sh, "cnlp_solve");
    char sname[64]; int64_t (*sn)(uint8_t*, int64_t) = dlsym(sh, "cnlp_solver_name");
    int64_t L = sn ? sn((uint8_t*)sname, sizeof sname - 1) : 0; sname[L > 0 ? L : 0] = 0;

    /* the device: the model's device tier with a solver that runs on it. `auto` takes it
     * when both have it (CUDA); `cuda` requires it; `cpu` never takes it. */
    if (strcmp(device, "auto") && strcmp(device, "cpu") && strcmp(device, "cuda"))
        die("--device is auto, cpu or cuda, not `%s`", device);
    int64_t (*solve_d)(const cnlp_problem*, const cnlp_device_problem*, int64_t, const char* const*,
                       const char* const*, cnlp_result*) = dlsym(sh, "cnlp_solve_device");
    int64_t (*dev)(int64_t) = sym("device");
    int64_t space = dev ? dev(id) : 0;
    int on_device = 0;
    if (strcmp(device, "cpu")) {
        on_device = solve_d && space == CNLP_DEVICE_CUDA;
        if (!on_device && !strcmp(device, "cuda"))
            die("--device cuda: %s", space != CNLP_DEVICE_CUDA ? "the model has no CUDA tier (install the library's -cuda build)"
                                                               : "the solver does not run on a device");
    }
    cnlp_device_problem D = { CNLP_DEVICE_CUDA, m_obj_d, m_grad_d, m_cons_d, m_jac_d, m_hess_d };
    if (on_device) {
        M.obj_d = need("obj_d"); M.grad_d = need("grad_d"); M.cons_d = need("cons_d");
        M.jac_d = need("jac_d"); M.hess_d = need("hess_d");
    }
    void* rep_i = sym("report_instance");
    if (rep_i) { rep_inst a = { id, rep_i }; print_report(call_inst, &a); }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int64_t rc = on_device ? solve_d(&P, &D, nopt, (const char* const*)keys, (const char* const*)vals, &R)
                           : solve(&P, nopt, (const char* const*)keys, (const char* const*)vals, &R);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (rc == 1) die("%s", R.solver_status);
    if (rc) die("the solver failed: %s", R.solver_status);
    double secs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("%s:%s  nvar %lld  ncon %lld  (%s)\n%s  objective %.15g  iterations %lld  %.3f s  (%s)\n", spec, PRE,
           (long long)n, (long long)m, on_device ? "cuda" : "cpu", STATUS[R.status >= 0 && R.status <= 6 ? R.status : 6], R.objective,
           (long long)R.iterations, secs, R.solver_status);
    void* rep_s = sym("report_solution");
    if (rep_s) { rep_sol a = { id, rep_s, &R, STATUS[R.status >= 0 && R.status <= 6 ? R.status : 6] }; print_report(call_sol, &a); }

    double t_out0 = now();
    for (int k = 0; k < nout; k++) {
        char wr[128] = ""; const char* path = outs[k]; const char* rest = split_prefix(outs[k], wr);
        if (rest && strcmp(wr, "cnlp") && has_writer(id, wr)) path = rest;
        else if (rest && !strcmp(wr, "cnlp")) { path = rest; wr[0] = 0; }
        else {
            wr[0] = 0;
            int64_t (*dw)(int64_t, uint8_t*, int64_t) = sym("default_writer");
            if (dw) { int64_t l = dw(id, (uint8_t*)wr, sizeof wr - 1); wr[l > 0 ? l : 0] = 0; }
        }
        if (wr[0]) {
            char f[256]; snprintf(f, sizeof f, "write_%s", wr);
            int64_t (*w)(int64_t, const char*, const double*, const double*, const double*, const double*, double, const char*) = need(f);
            if (w(id, path, R.x, R.y, R.zl, R.zu, R.objective, STATUS[R.status])) die("writer `%s` could not write %s", wr, path);
        } else {
            char e[512];
            if (cnlpio_write_solution(path, LIB, lib, PRE, id, n, m, minimize, R.x, R.y, R.zl, R.zu, R.objective,
                                      STATUS[R.status >= 0 && R.status <= 6 ? R.status : 6], R.iterations, sname,
                                      R.solver_status, e, sizeof e))
                die("%s", e);
        }
        printf("wrote %s%s%s\n", path, wr[0] ? " with " : "", wr);
    }
    /* where the time went: the model library's load (dlopen; a runtime it carries starts on
     * its first call, in instantiate), instantiating the model (and reading a start), loading
     * the solver (and what it does when it loads, such as warming up a GPU), the solve, and
     * writing the outputs; total from the start of this program to here */
    double t_out = now() - t_out0;
    printf("times: load model library %.3f s, instantiate %.3f s, load solver %.3f s, solve %.3f s, write %.3f s, total %.3f s\n",
           t_lib, t_inst, t_sol, secs, t_out, now() - t_start);
    return R.status <= CNLP_ACCEPTABLE ? 0 : 1;
}
