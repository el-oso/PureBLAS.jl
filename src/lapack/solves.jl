# LAPACK triangular-factor SOLVES — potrs / getrs / trtrs, native (Mode-2) entry points.
#
# These were previously implemented ONLY inside the `@ccallable` C-ABI shims in cabi_lapack.jl, which
# had two consequences worth naming: there was no AD-traceable Mode-2 API for the solve step of `\`
# (the whole point of the native path), and — because bench/plots.jl compares `PureBLAS.foo!` against
# `LinearAlgebra.LAPACK.foo!` — they could not be GATED at all. `getrs`/`potrs` back `lu(A) \ b` and
# `cholesky(A) \ b`, i.e. some of the most-executed LAPACK in practice, and neither had ever appeared
# on a gate chart. The shims now call these, so there is exactly one implementation.
#
# All three are compositions of the already-gated `trsm!` (plus `_laswp!` for getrs), operating on
# caller-supplied factors in standard LAPACK convention. That makes them self-consistent under a mixed
# backend: forwarding them is correct even when the factorization itself ran on OpenBLAS.

# ── potrs: A·X = B given the Cholesky factor of A (dpotrs.f) ──────────────────────────────────────
# uplo='L': A = L·Lᴴ ⇒ solve L·Y = B then Lᴴ·X = Y.  uplo='U': A = Uᴴ·U ⇒ Uᴴ·Y = B then U·X = Y.
# Overwrites B with X.
function potrs!(A::AbstractMatrix{T}, B::AbstractVecOrMat; uplo::AbstractChar = 'L') where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potrs!: A must be square"))
    Bm = _gt_asmat(B)
    size(Bm, 1) == n || throw(DimensionMismatch("potrs!: size(B,1) must equal n"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("potrs!: uplo must be 'L' or 'U'"))
    ct = T <: Complex ? 'C' : 'T'
    if uplo == 'L'
        trsm!(Bm, A; side = 'L', uplo = 'L', transA = 'N', alpha = one(T))
        trsm!(Bm, A; side = 'L', uplo = 'L', transA = ct, alpha = one(T))
    else
        trsm!(Bm, A; side = 'L', uplo = 'U', transA = ct, alpha = one(T))
        trsm!(Bm, A; side = 'L', uplo = 'U', transA = 'N', alpha = one(T))
    end
    return B
end

# ── trtrs: op(A)·X = B, A triangular (dtrtrs.f) — a single trsm ───────────────────────────────────
function trtrs!(
        A::AbstractMatrix{T}, B::AbstractVecOrMat;
        uplo::AbstractChar = 'U', trans::AbstractChar = 'N', diag::AbstractChar = 'N'
    ) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("trtrs!: A must be square"))
    Bm = _gt_asmat(B)
    size(Bm, 1) == n || throw(DimensionMismatch("trtrs!: size(B,1) must equal n"))
    trsm!(Bm, A; side = 'L', uplo = uplo, transA = trans, diag = diag, alpha = one(T))
    return B
end

# ── getrs: A·X = B given P·A = L·U from getrf (dgetrs.f) ──────────────────────────────────────────
# trans='N': apply the interchanges to B, then L\ (unit diagonal) then U\.
# trans='T'/'C': Uᵀ\ then Lᵀ\, then the interchanges in REVERSE order — LAPACK's own ordering, and the
# reversal is load-bearing (the permutation is applied on the other side of the transposed solve).
# `ipiv` is LAPACK 1-based.
function getrs!(
        A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, B::AbstractVecOrMat;
        trans::AbstractChar = 'N'
    ) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("getrs!: A must be square"))
    length(ipiv) >= n || throw(DimensionMismatch("getrs!: length(ipiv) < n"))
    Bm = _gt_asmat(B)
    size(Bm, 1) == n || throw(DimensionMismatch("getrs!: size(B,1) must equal n"))
    nrhs = size(Bm, 2)
    if trans == 'N'
        _laswp!(Bm, ipiv, 1, n, 1, nrhs)                                              # P·B
        trsm!(Bm, A; side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = one(T)) # L·Y = P·B
        trsm!(Bm, A; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = one(T)) # U·X = Y
    elseif trans == 'T' || trans == 'C'
        trsm!(Bm, A; side = 'L', uplo = 'U', transA = trans, diag = 'N', alpha = one(T))
        trsm!(Bm, A; side = 'L', uplo = 'L', transA = trans, diag = 'U', alpha = one(T))
        @inbounds for i in n:-1:1
            q = Int(ipiv[i])
            if q != i
                for j in 1:nrhs
                    Bm[i, j], Bm[q, j] = Bm[q, j], Bm[i, j]
                end
            end
        end
    else
        throw(ArgumentError("getrs!: trans must be 'N', 'T' or 'C'"))
    end
    return B
end
