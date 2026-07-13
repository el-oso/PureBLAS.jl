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
#include <stdlib.h>
#include <math.h>

typedef void (*daxpy_t)(int64_t *, double *, double *, int64_t *, double *, int64_t *);
typedef double (*dnrm2_t)(int64_t *, double *, int64_t *);
/* dgemm_64_: two char args by reference, then by-ref scalars/arrays, then two hidden Fortran
 * string-length longs (the trailing `1, 1`). */
typedef void (*dgemm_t)(char *, char *, int64_t *, int64_t *, int64_t *, double *, double *,
                        int64_t *, double *, int64_t *, double *, double *, int64_t *, long, long);
/* dgemv_64_ (L2): one char arg (trans) + one trailing hidden length. y := alpha*op(A)*x + beta*y. */
typedef void (*dgemv_t)(char *, int64_t *, int64_t *, double *, double *, int64_t *, double *,
                        int64_t *, double *, double *, int64_t *, long);
/* dgesvd_64_ (LAPACK): two char args (jobu, jobvt) + info out-arg + two trailing hidden lengths. */
typedef void (*dgesvd_t)(char *, char *, int64_t *, int64_t *, double *, int64_t *, double *,
                         double *, int64_t *, double *, int64_t *, double *, int64_t *, int64_t *,
                         long, long);

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

    /* dgemv (L2, char-ABI): y := 1*A*x + 0*y, A = [1 3; 2 4] (col-major), x = {1,1}
     *   A*x = [1+3; 2+4] = [4; 6] */
    dgemv_t dgemv = (dgemv_t)dlsym(h, "dgemv_64_");
    if (!dgemv) { printf("dlsym dgemv fail\n"); return 1; }
    double one = 1.0, zero = 0.0, xv[2] = {1, 1}, yv[2] = {0, 0};
    dgemv(&N, &m2, &n2, &one, A, &ld, xv, &i1, &zero, yv, &i1, 1);
    printf("dgemv: %.1f %.1f\n", yv[0], yv[1]); /* 4.0 6.0 */

    /* dgesvd (LAPACK, 2-char + info ABI): full SVD of A = [1 3; 2 4] (col-major {1,2,3,4}).
     *   singular values ~ 5.4650, 0.3660; verify U*diag(S)*VT reconstructs A. jobu=jobvt='A'. */
    dgesvd_t dgesvd = (dgesvd_t)dlsym(h, "dgesvd_64_");
    if (!dgesvd) { printf("dlsym dgesvd fail\n"); return 1; }
    char Aj = 'A';
    int64_t lwork = -1, info = -99;
    double As[4] = {1, 2, 3, 4}, S[2], Um[4], VTm[4], work[16];
    /* workspace query then compute (PureBLAS manages its own workspace, but honor the protocol) */
    dgesvd(&Aj, &Aj, &m2, &n2, As, &ld, S, Um, &ld, VTm, &ld, work, &lwork, &info, 1, 1);
    lwork = (int64_t)work[0];
    dgesvd(&Aj, &Aj, &m2, &n2, As, &ld, S, Um, &ld, VTm, &ld, work, &lwork, &info, 1, 1);
    printf("dgesvd: info=%lld  S= %.4f %.4f\n", (long long)info, S[0], S[1]); /* 5.4650 0.3660 */
    /* reconstruct R = U*diag(S)*VT (col-major) and compare to the original {1,2,3,4} */
    double A0[4] = {1, 2, 3, 4}, rerr = 0.0;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            double r = 0.0;
            for (int k = 0; k < 2; k++) r += Um[i + 2*k] * S[k] * VTm[k + 2*j];
            double d = r - A0[i + 2*j]; if (d < 0) d = -d;
            if (d > rerr) rerr = d;
        }
    printf("dgesvd recon max|err| = %.2e\n", rerr); /* ~1e-15 */

    /* larger SVD (n=160) — exercises the divide-and-conquer singular-VECTOR path (n > _SVD_DC_CROSS=96,
     * the crossover just retuned). Deterministic diagonally-dominant matrix; verify full reconstruction
     * U*diag(S)*VT ≈ A and orthonormality of U (max|UᵀU − I|). This is the real drop-in SVD path. */
    {
        int64_t Ng = 160, ldn = 160;
        double *Ag = malloc(Ng*Ng*sizeof(double)), *A0g = malloc(Ng*Ng*sizeof(double));
        double *Ug = malloc(Ng*Ng*sizeof(double)), *VTg = malloc(Ng*Ng*sizeof(double));
        double *Sg = malloc(Ng*sizeof(double)), wq[1];
        for (int64_t j = 0; j < Ng; j++)
            for (int64_t i = 0; i < Ng; i++) {
                double v = (double)(((i*131 + j*17 + 1) % 101)) / 50.0 - 1.0;
                if (i == j) v += (double)Ng;                 /* diagonal dominance → well-conditioned */
                Ag[i + Ng*j] = v; A0g[i + Ng*j] = v;
            }
        int64_t lw = -1, inf = -99;
        dgesvd(&Aj, &Aj, &Ng, &Ng, Ag, &ldn, Sg, Ug, &ldn, VTg, &ldn, wq, &lw, &inf, 1, 1);
        lw = (int64_t)wq[0]; if (lw < 8*Ng) lw = 8*Ng;
        double *wk = malloc(lw*sizeof(double));
        dgesvd(&Aj, &Aj, &Ng, &Ng, Ag, &ldn, Sg, Ug, &ldn, VTg, &ldn, wk, &lw, &inf, 1, 1);
        double rerr2 = 0.0, oerr = 0.0;
        for (int64_t j = 0; j < Ng; j++)
            for (int64_t i = 0; i < Ng; i++) {
                double r = 0.0;
                for (int64_t k = 0; k < Ng; k++) r += Ug[i + Ng*k] * Sg[k] * VTg[k + Ng*j];
                double d = fabs(r - A0g[i + Ng*j]); if (d > rerr2) rerr2 = d;
            }
        for (int64_t a = 0; a < Ng; a++)
            for (int64_t b = 0; b < Ng; b++) {
                double s = 0.0;
                for (int64_t i = 0; i < Ng; i++) s += Ug[i + Ng*a] * Ug[i + Ng*b];
                double d = fabs(s - (a == b ? 1.0 : 0.0)); if (d > oerr) oerr = d;
            }
        printf("dgesvd n=160: info=%lld  S[0]=%.3f S[159]=%.3f  recon|err|=%.2e  ortho|err|=%.2e\n",
               (long long)inf, Sg[0], Sg[Ng-1], rerr2, oerr);   /* info=0, err ~1e-12 */
        free(Ag); free(A0g); free(Ug); free(VTg); free(Sg); free(wk);
    }
    return 0;
}
