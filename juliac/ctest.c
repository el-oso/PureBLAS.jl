/* Mode-1 validation: a NON-Julia (C) host loading libpureblas.so. The trimmed library
 * self-initializes its embedded Julia runtime on first @ccallable call (via
 * ijl_autoinit_and_adopt_thread), so a plain C program can use PureBLAS as a BLAS library.
 *
 *   julia juliac/build.jl                      # produce juliac/build/libpureblas.so
 *   gcc juliac/ctest.c -o /tmp/ctest -ldl && /tmp/ctest
 *   # expect: daxpy: 12.0 24.0 36.0 48.0 / dnrm2: 5.4772
 *
 * NOTE: forwarding this .so via BLAS.lbt_forward from INSIDE a running Julia process aborts
 * (double-init of the shared libjulia) — a current juliac limitation. Use the native API
 * (Mode 2) inside Julia; this .so is for non-Julia hosts and trim-compatibility verification.
 * See ../ROADMAP.md.
 */
#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>

typedef void (*daxpy_t)(int64_t *, double *, double *, int64_t *, double *, int64_t *);
typedef double (*dnrm2_t)(int64_t *, double *, int64_t *);
/* dgemm_64_: two char args by reference, then by-ref scalars/arrays, then two hidden Fortran
 * string-length longs (the trailing `1, 1`). */
typedef void (*dgemm_t)(char *, char *, int64_t *, int64_t *, int64_t *, double *, double *,
                        int64_t *, double *, int64_t *, double *, double *, int64_t *, long, long);

int main(void) {
    void *h = dlopen("juliac/build/libpureblas.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen fail: %s\n", dlerror()); return 1; }
    daxpy_t daxpy = (daxpy_t)dlsym(h, "daxpy_64_");
    dnrm2_t dnrm2 = (dnrm2_t)dlsym(h, "dnrm2_64_");
    if (!daxpy || !dnrm2) { printf("dlsym fail\n"); return 1; }

    int64_t n = 4, i1 = 1; double a = 2.0;
    double x[4] = {1, 2, 3, 4}, y[4] = {10, 20, 30, 40};
    daxpy(&n, &a, x, &i1, y, &i1);          /* y += 2*x -> 12 24 36 48 */
    printf("daxpy: %.1f %.1f %.1f %.1f\n", y[0], y[1], y[2], y[3]);
    printf("dnrm2: %.4f\n", dnrm2(&n, x, &i1));  /* sqrt(30) = 5.4772 */

    /* dgemm C := alpha*A*B + beta*C, column-major, transA=transB='N'. 2x2:
     *   A = [1 3; 2 4], B = [5 7; 6 8]  (col-major {1,2,3,4}, {5,6,7,8})
     *   A*B = [23 31; 34 46];  C0 = [1 1; 1 1], alpha=2, beta=3 -> 2*A*B + 3
     *   -> [49 65; 71 95]  (col-major {49,71,65,95}) */
    dgemm_t dgemm = (dgemm_t)dlsym(h, "dgemm_64_");
    if (!dgemm) { printf("dlsym dgemm fail\n"); return 1; }
    char N = 'N';
    int64_t m2 = 2, n2 = 2, k2 = 2, ld = 2;
    double alpha = 2.0, beta = 3.0;
    double A[4] = {1, 2, 3, 4}, B[4] = {5, 6, 7, 8}, Cm[4] = {1, 1, 1, 1};
    dgemm(&N, &N, &m2, &n2, &k2, &alpha, A, &ld, B, &ld, &beta, Cm, &ld, 1, 1);
    printf("dgemm: %.1f %.1f %.1f %.1f\n", Cm[0], Cm[1], Cm[2], Cm[3]); /* 49 71 65 95 */
    return 0;
}
