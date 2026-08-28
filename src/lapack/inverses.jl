# LAPACK EXPLICIT INVERSES — trtri / potri / getri, native (Mode-2) entry points.
#
# Same story as `solves.jl`, one step later. These existed ONLY inside the `@ccallable` shims in
# cabi_lapack.jl, with the same two consequences: no AD-traceable Mode-2 API, and — because
# bench/plots.jl compares `PureBLAS.foo!` against `LinearAlgebra.LAPACK.foo!` — no way to GATE them.
# All three are LBT-forwarded into user code by `cabi_forward.jl`, so they ship; they were simply never
# measured. A user's PureOSQP run found `potri` at 0.67x (322 us vs OpenBLAS 216 us at n=200) that no
# red cell could report, because the op had no cell to be red.
#
# THE SHIMS CALL THESE, so there is exactly one implementation of each.
#
# ── Why these were slow, and it was the ALGORITHM, not the kernels ────────────────────────────────
# Both `trtri` and `potri` used to solve against a DENSE n×n identity:
#     trtri:  one trsm  vs I   ->  n³      (LAPACK's blocked trtri: n³/3)
#     potri:  two trsms vs I   ->  2n³     (LAPACK: trtri + lauum   = 2n³/3)
# i.e. 3x the necessary work. potri losing at only 1.49x the time while doing 3x the flops says the
# underlying kernels were already ~2x more efficient per flop; the algorithm was the whole gap.
#
# `_trtri!` (blas3/level3.jl) is the blocked recursive triangular inverse that trsm already uses
# internally — n³/3, because it never touches the zero triangle. It was one file away the whole time.
# This is the "wire the fastest path" rule: a C-ABI shim must route to the FASTEST correct kernel, and
# these two declined to.

# ── trtri: in-place inverse of a triangular A (dtrtri.f) ──────────────────────────────────────────
# Overwrites the `uplo` triangle of A with A⁻¹. `diag='U'` treats the diagonal as unit and leaves it.
# `_trtri!` writes out-of-place (it reads A while building the inverse), so it needs one n×n scratch;
# the triangle is then copied back. A fresh allocation rather than the shared L3 scratch, because the
# recursion's own trsm bases reach for that.
function trtri!(A::AbstractMatrix{T}; uplo::AbstractChar = 'L', diag::AbstractChar = 'N') where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("trtri!: A must be square"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("trtri!: uplo must be 'L' or 'U'"))
    (diag == 'N' || diag == 'U') || throw(ArgumentError("trtri!: diag must be 'N' or 'U'"))
    n == 0 && return A
    up = uplo == 'U'
    X = Matrix{T}(undef, n, n)
    Xv = view(X, 1:n, 1:n)
    _trtri!(Xv, A, n, up, diag == 'U')
    if up
        @inbounds for j in 1:n, i in 1:j
            A[i, j] = Xv[i, j]
        end
    else
        @inbounds for j in 1:n, i in j:n
            A[i, j] = Xv[i, j]
        end
    end
    return A
end

# ── lauum: in-place triangular product (dlauum.f) ────────────────────────────────────────────────
# Lower: M := Mᴴ·M.  Upper: M := M·Mᴴ.  Only the `uplo` triangle is read and written.
#
# WHY THIS EXISTS RATHER THAN JUST CALLING syrk. The product of a triangular matrix with its own
# adjoint is n³/3, but syrk on the same operand treats it as DENSE and pays n³ — the zero triangle is
# multiplied anyway. Measured: potri built on a dense syrk gates at n=256 (1.33) and then FALLS OVER
# at n=1024 (0.59 vs OpenBLAS), because at large n the flop ratio is the whole story — 4n³/3 against
# LAPACK's 2n³/3 is 2x the work and ≈0.5x the ratio, which is what the number says. Small n hides it
# behind per-call overhead, which is exactly how a shortcut like that survives a spot check.
#
# The recursion, for lower M = [M11 0; M21 M22] and R = MᴴM:
#     R11 = M11ᴴM11 + M21ᴴM21     (recurse, then syrk accumulating with beta=1)
#     R21 = M22ᴴM21               (trmm, side='L')
#     R22 = M22ᴴM22               (recurse)
# The ORDER matters: R11's syrk must read M21 before the trmm overwrites it. Upper is the mirror with
# M = [M11 M12; 0 M22], R = MMᴴ, trmm on the right.
#
# The base does one dense syrk on a b×b block through a copy (syrk cannot alias C and A). That block's
# 3x waste is bounded by 2b²/n² of the total — 0.2% at b=16, n=1024 — so the base stays small. It
# reuses `_trtri_base()`: the same recursion-base criterion for the same triangular shape, and that
# knob is already documented FLAT across 8/16/32/64 on all three µarchs.
function _lauum!(M::AbstractMatrix{T}, up::Bool) where {T}
    n = size(M, 1)
    n == 0 && return M
    if n <= _trtri_base()
        Tmp = Matrix{T}(undef, n, n)
        copyto!(Tmp, M)
        if up                                   # zero the strict lower so the dense product is exact
            @inbounds for j in 1:n, i in (j + 1):n
                Tmp[i, j] = zero(T)
            end
        else
            @inbounds for j in 1:n, i in 1:(j - 1)
                Tmp[i, j] = zero(T)
            end
        end
        if T <: Complex
            herk!(M, Tmp; uplo = up ? 'U' : 'L', trans = up ? 'N' : 'C', alpha = true, beta = false)
        else
            syrk!(M, Tmp; uplo = up ? 'U' : 'L', trans = up ? 'N' : 'T',
                  alpha = one(T), beta = zero(T))
        end
        return M
    end
    ct = T <: Complex ? 'C' : 'T'
    h = n ÷ 2
    M11 = view(M, 1:h, 1:h)
    M22 = view(M, (h + 1):n, (h + 1):n)
    if up
        M12 = view(M, 1:h, (h + 1):n)
        _lauum!(M11, true)                                        # R11 = M11·M11ᴴ
        if T <: Complex
            herk!(M11, M12; uplo = 'U', trans = 'N', alpha = true, beta = true)   # += M12·M12ᴴ
        else
            syrk!(M11, M12; uplo = 'U', trans = 'N', alpha = one(T), beta = one(T))
        end
        trmm!(M12, M22; side = 'R', uplo = 'U', transA = ct, alpha = true)        # R12 = M12·M22ᴴ
        _lauum!(M22, true)
    else
        M21 = view(M, (h + 1):n, 1:h)
        _lauum!(M11, false)                                       # R11 = M11ᴴ·M11
        if T <: Complex
            herk!(M11, M21; uplo = 'L', trans = 'C', alpha = true, beta = true)   # += M21ᴴ·M21
        else
            syrk!(M11, M21; uplo = 'L', trans = 'T', alpha = one(T), beta = one(T))
        end
        trmm!(M21, M22; side = 'L', uplo = 'L', transA = ct, alpha = true)        # R21 = M22ᴴ·M21
        _lauum!(M22, false)
    end
    return M
end

# ── potri: A⁻¹ from the Cholesky factor already in A (dpotri.f) ───────────────────────────────────
# Overwrites the `uplo` triangle of A with A⁻¹ (which is Hermitian, so LAPACK stores one triangle).
#
# M := the factor's inverse (`_trtri!`). Then A = L·Lᴴ ⇒ A⁻¹ = L⁻ᴴ·L⁻¹ = Mᴴ·M, a rank-k update — so
# syrk (real) / herk (complex), both of which gate. Upper stores A = Uᴴ·U ⇒ A⁻¹ = U⁻¹·U⁻ᴴ = M·Mᴴ,
# hence trans='N' there.
#
# A dense syrk here was tried first and MEASURED INADEQUATE: it gates at n=256 (1.33) and collapses at
# n=1024 (0.59), because treating the triangular M as dense costs n³ instead of n³/3 and at large n the
# flop ratio is the whole story. `_lauum!` above does the triangular product properly, which brings the
# total to trtri + lauum = 2n³/3 — LAPACK's own count.
function potri!(A::AbstractMatrix{T}; uplo::AbstractChar = 'L') where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potri!: A must be square"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("potri!: uplo must be 'L' or 'U'"))
    n == 0 && return A
    up = uplo == 'U'
    trtri!(A; uplo = uplo, diag = 'N')     # A := M, the factor's inverse (in place, n³/3)
    _lauum!(A, up)                         # A := MᴴM (lower) / MMᴴ (upper), n³/3
    return A
end
