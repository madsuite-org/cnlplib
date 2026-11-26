/* libcnlpio: the standard solution file (cnlpio.h, formats/solution.md), with HDF5 linked in.
 *
 *   cc -std=c99 -O2 -fPIC -shared -I.. -o libcnlpio.so cnlpio.c libhdf5.a -ldl -lm -lpthread
 *
 * cnlp-solve compiles the same file in. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <hdf5.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnlpio.h"

int64_t cnlpio_version(void) { return CNLPIO_VERSION; }

static int64_t io_fail(char* err, int64_t cap, const char* fmt, ...) {
    if (err && cap > 0) { va_list ap; va_start(ap, fmt); vsnprintf(err, (size_t)cap, fmt, ap); va_end(ap); }
    return 1;
}

static void io_attr_str(hid_t o, const char* k, const char* v) {
    hid_t t = H5Tcopy(H5T_C_S1); H5Tset_size(t, strlen(v) ? strlen(v) : 1);
    hid_t s = H5Screate(H5S_SCALAR), a = H5Acreate2(o, k, t, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, t, v); H5Aclose(a); H5Sclose(s); H5Tclose(t);
}
static void io_attr_i64(hid_t o, const char* k, int64_t v) {
    hid_t s = H5Screate(H5S_SCALAR), a = H5Acreate2(o, k, H5T_STD_I64LE, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT64, &v); H5Aclose(a); H5Sclose(s);
}
/* a dataset of `nd` dimensions (0: a scalar); `dims` in the file's (row-major) order */
static void io_ds(hid_t g, const char* k, hid_t ft, hid_t mt, int nd, const hsize_t* dims, const void* v) {
    hid_t s = nd == 0 ? H5Screate(H5S_SCALAR) : H5Screate_simple(nd, dims, NULL);
    hid_t x = H5Dcreate2(g, k, ft, s, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(x, mt, H5S_ALL, H5S_ALL, H5P_DEFAULT, v); H5Dclose(x); H5Sclose(s);
}
static void io_vec(hid_t g, const char* k, int64_t n, const double* v) {
    hsize_t d = (hsize_t)n; io_ds(g, k, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, 1, &d, v);
}
static void io_str(hid_t g, const char* k, const char* v) {
    hid_t t = H5Tcopy(H5T_C_S1); H5Tset_size(t, strlen(v) ? strlen(v) : 1);
    io_ds(g, k, t, t, 0, NULL, v); H5Tclose(t);
}

static void* io_sym(void* lib, const char* model, const char* s) {
    char n[256]; snprintf(n, sizeof n, "%s_%s", model, s);
    return dlsym(lib, n);
}

/* A block's slice in its shape: the vector is column-major (first index fastest), which is
 * the file's row-major layout of the reversed dims, as Julia's HDF5.jl writes an array. */
static void io_slice(hid_t h, const char* k, const double* v, int nd, const int64_t* dims) {
    hsize_t d[8];
    for (int i = 0; i < nd; i++) d[i] = (hsize_t)dims[nd - 1 - i];
    io_ds(h, k, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, nd, d, v);
}

/* /blocks/<kind>/<name>/: variables value, multiplier_L, multiplier_U; constraints value
 * (c(x)), multiplier; parameters value */
static int64_t io_blocks(hid_t f, void* lib, const char* model, int64_t id, int64_t ncon,
                         const double* x, const double* y, const double* zl, const double* zu,
                         char* err, int64_t cap) {
    int64_t (*nb)(int64_t) = io_sym(lib, model, "nblocks");
    int64_t (*bk)(int64_t, int64_t, int64_t*) = io_sym(lib, model, "block");
    int64_t (*bn)(int64_t, int64_t, uint8_t*, int64_t) = io_sym(lib, model, "block_name");
    int64_t (*gv)(int64_t, int64_t, double*, int64_t) = io_sym(lib, model, "get_value");
    int64_t (*cons)(int64_t, const double*, double*) = io_sym(lib, model, "cons");
    if (!nb || !bk || !bn) return 0;
    int64_t n = nb(id);
    if (n <= 0) return 0;
    double* c = calloc((size_t)ncon + 1, sizeof(double));
    if (!c) return io_fail(err, cap, "out of memory");
    int have_c = ncon == 0 || (cons && cons(id, x, c) == 0);
    hid_t blocks = H5Gcreate2(f, "blocks", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    static const char* KIND[] = { "variables", "constraints", "parameters" };
    int64_t rc = 0;
    for (int64_t k = 0; k < n && !rc; k++) {
        int64_t out[4 + 8] = { 0 };
        if (bk(id, k, out)) { rc = io_fail(err, cap, "%s_block(%lld) failed", model, (long long)k); break; }
        int64_t kind = out[0], off = out[1], len = out[2], nd = out[3];
        if (kind < 0 || kind > 2 || nd < 0 || nd > 8) { rc = io_fail(err, cap, "%s_block(%lld): bad block", model, (long long)k); break; }
        char name[256]; int64_t L = bn(id, k, (uint8_t*)name, sizeof name - 1); name[L > 0 && L < (int64_t)sizeof name ? L : 0] = 0;
        hid_t g = H5Lexists(blocks, KIND[kind], H5P_DEFAULT) > 0 ? H5Gopen2(blocks, KIND[kind], H5P_DEFAULT)
                                                                : H5Gcreate2(blocks, KIND[kind], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t h = H5Gcreate2(g, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dn = (hsize_t)nd; hid_t s = H5Screate_simple(1, &dn, NULL);
        hid_t a = H5Acreate2(h, "dims", H5T_STD_I64LE, s, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a, H5T_NATIVE_INT64, out + 4); H5Aclose(a); H5Sclose(s);
        if (kind == 0) {
            io_slice(h, "value", x + off, (int)nd, out + 4);
            io_slice(h, "multiplier_L", zl + off, (int)nd, out + 4);
            io_slice(h, "multiplier_U", zu + off, (int)nd, out + 4);
        } else if (kind == 1) {
            if (have_c) io_slice(h, "value", c + off, (int)nd, out + 4);
            io_slice(h, "multiplier", y + off, (int)nd, out + 4);
        } else if (gv) {
            double* v = calloc((size_t)len + 1, sizeof(double));
            if (v && gv(id, k, v, len) == 0) io_slice(h, "value", v, (int)nd, out + 4);
            free(v);
        }
        H5Gclose(h); H5Gclose(g);
    }
    H5Gclose(blocks); free(c);
    return rc;
}

static const char* io_base(const char* p) { const char* b = strrchr(p, '/'); return b ? b + 1 : p; }

int64_t cnlpio_write_solution(const char* path, void* lib, const char* library, const char* model,
                              int64_t id, int64_t nvar, int64_t ncon, int64_t minimize,
                              const double* x, const double* y, const double* zl, const double* zu,
                              double objective, const char* status, int64_t iterations,
                              const char* solver, const char* solver_status, char* err, int64_t cap) {
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);   /* failures are returned, not printed */
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0) return io_fail(err, cap, "cannot write %s", path);
    io_attr_str(f, "format", "cnlp-solution"); io_attr_str(f, "format_version", "0.2");
    io_attr_str(f, "library", io_base(library ? library : "")); io_attr_str(f, "model", model ? model : "");
    io_attr_i64(f, "nvar", nvar); io_attr_i64(f, "ncon", ncon);
    hid_t g = H5Gcreate2(f, "solution", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    io_vec(g, "solution", nvar, x); io_vec(g, "multipliers", ncon, y);
    io_vec(g, "multipliers_L", nvar, zl); io_vec(g, "multipliers_U", nvar, zu);
    io_ds(g, "objective", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, 0, NULL, &objective);
    int8_t mn = (int8_t)(minimize != 0); io_ds(g, "minimize", H5T_STD_I8LE, H5T_NATIVE_INT8, 0, NULL, &mn);
    io_str(g, "status", status ? status : "error");
    io_ds(g, "iterations", H5T_STD_I64LE, H5T_NATIVE_INT64, 0, NULL, &iterations);
    io_str(g, "solver", solver ? solver : ""); io_str(g, "solver_status", solver_status ? solver_status : "");
    H5Gclose(g);
    int64_t rc = lib && model ? io_blocks(f, lib, model, id, ncon, x, y, zl, zu, err, cap) : 0;
    if (H5Fclose(f) < 0 && !rc) rc = io_fail(err, cap, "cannot write %s", path);
    return rc;
}

int64_t cnlpio_read_start(const char* path, int64_t nvar, int64_t ncon,
                          double* x, double* y, double* zl, double* zu, char* err, int64_t cap) {
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) return io_fail(err, cap, "cannot read %s (not an HDF5 file?)", path);
    const char* k[] = { "solution/solution", "solution/multipliers", "solution/multipliers_L", "solution/multipliers_U" };
    double* v[] = { x, y, zl, zu }; int64_t L[] = { nvar, ncon, nvar, nvar };
    for (int i = 0; i < 4; i++) {
        hid_t d = H5Dopen2(f, k[i], H5P_DEFAULT);
        if (d < 0) { H5Fclose(f); return io_fail(err, cap, "%s has no %s", path, k[i]); }
        hid_t s = H5Dget_space(d); hssize_t c = H5Sget_simple_extent_npoints(s); H5Sclose(s);
        if (c != L[i]) {
            H5Dclose(d); H5Fclose(f);
            return io_fail(err, cap, "%s: %s has %lld entries, the model needs %lld", path, k[i], (long long)c, (long long)L[i]);
        }
        H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v[i]); H5Dclose(d);
    }
    H5Fclose(f);
    return 0;
}
