/* The example QP as a maximization: max -f(x) over the same feasible set, so the solution
 * is the same x with objective -f(x). In the cnlp convention (s = -1) its multipliers equal
 * the minimization's, which is what a solver's sign handling is checked against. */
#define qp_obj qpmin_obj
#define qp_grad qpmin_grad
#define qp_hess qpmin_hess
#include "../example/qp.c"
#undef qp_obj
#undef qp_grad
#undef qp_hess

int64_t qp_minimize(int64_t id) { return valid(id) ? 0 : -1; }

int64_t qp_obj(int64_t id, const double* x, double* out) {
    int64_t s = qpmin_obj(id, x, out);
    if (s == 0) *out = -*out;
    return s;
}
int64_t qp_grad(int64_t id, const double* x, double* g) {
    int64_t s = qpmin_grad(id, x, g);
    if (s == 0) for (int64_t i = 0; i < qp_sizes[id - 1]; i++) g[i] = -g[i];
    return s;
}
int64_t qp_hess(int64_t id, const double* x, const double* y, double obj_weight, double* vals) {
    return qpmin_hess(id, x, y, -obj_weight, vals);   /* the constraint is linear */
}
