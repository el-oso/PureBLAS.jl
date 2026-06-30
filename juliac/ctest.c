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
    return 0;
}
