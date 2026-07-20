# LAPACK trrfs (dtrrfs/ztrrfs): componentwise backward error `Berr[j]` and a forward-error bound
# `Ferr[j]` for a computed solution `X` of the triangular system `op(A)·X = B`. Direct port of
# Reference-LAPACK dtrrfs.f/ztrrfs.f, composed from PureBLAS's own `trmv!`/`trsv!` (native.jl) and
# the Hager–Higham 1-norm estimator `_lacn2_estimate` (trsen.jl, DLACN2/ZLACN2 port) — so this file
# needs `trsyl.jl` + `trsen.jl` already in scope (both are already `include`d by PureBLAS.jl ahead of
# where this file would sit, so `using PureBLAS` is sufficient).
#
#   Berr[j] = max_i |r_i| / (|op(A)|·|X[:,j]| + |B[:,j]|)_i ,   r = B[:,j] − op(A)·X[:,j]
#   Ferr[j] = ‖ |op(A)⁻¹|·( |r| + nz·eps·(|op(A)|·|X[:,j]|+|B[:,j]|) ) ‖_∞ / ‖X[:,j]‖_∞   (1-norm-
#             estimated via DLACN2 on the operator `inv(op(A))·diag(w)`, exactly as dtrrfs does).
# Generic over T (s/d/c/z); the complex path uses CABS1 = |re|+|im| (via bunchkaufman.jl's
# `_bk_cabs1`, already in scope) in place of `abs`, matching ztrrfs exactly.

"""
    trrfs!(uplo, trans, diag, A, B, X, Ferr, Berr) -> (Ferr, Berr)

Componentwise backward error (`Berr`) and a forward-error bound (`Ferr`) for the computed solution
`X` of the triangular system `op(A)·X = B` (`op` selected by `trans` ∈ `'N'`,`'T'`,`'C'`; `uplo` ∈
`'U'`,`'L'`; `diag` ∈ `'N'`,`'U'`). `Ferr`, `Berr` (length `size(B,2)`) are filled in place and
returned. Mirrors `LinearAlgebra.LAPACK.trrfs!`. Generic over `T<:Number` (s/d/c/z); `A`, `B`, `X`
are not modified.
"""
function trrfs!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
        A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, X::AbstractVecOrMat{T},
        Ferr::AbstractVector{<:Real} = Vector{real(T)}(undef, size(B, 2)),
        Berr::AbstractVector{<:Real} = Vector{real(T)}(undef, size(B, 2))) where {T}
    R = real(T)
    n = size(A, 2)
    size(A, 1) == n || throw(DimensionMismatch("trrfs!: A must be square"))
    nrhs = size(B, 2)
    size(X, 2) == nrhs || throw(DimensionMismatch("trrfs!: second dimensions of B and X must match"))
    (uplo == 'U' || uplo == 'L') || throw(ArgumentError("trrfs!: uplo must be 'U' or 'L'"))
    (trans == 'N' || trans == 'T' || trans == 'C') ||
        throw(ArgumentError("trrfs!: trans must be 'N', 'T' or 'C'"))
    (diag == 'N' || diag == 'U') || throw(ArgumentError("trrfs!: diag must be 'N' or 'U'"))
    if n == 0 || nrhs == 0
        fill!(Ferr, zero(R)); fill!(Berr, zero(R))
        return Ferr, Berr
    end
    notran = trans == 'N'
    upper = uplo == 'U'
    nounit = diag == 'N'
    # LACN2 alternates op and its (conjugate-)transpose. dtrrfs/ztrrfs use TRANST for kase=1 and
    # TRANSN for kase=2. For complex, non-notran maps to TRANSN='C' (NOT the user's 'T'/'C') — the
    # conjugate-transpose the estimator needs; mirroring `trans` directly is wrong for trans='T'.
    op = T <: Complex ? 'C' : 'T'
    transt = notran ? op : 'N'
    transn = notran ? 'N' : op
    nz = n + 1
    eps_p = eps(R)
    safmin = _syl_safmin(R)
    safe1 = R(nz) * safmin
    safe2 = safe1 / eps_p

    r = Vector{T}(undef, n)          # residual  r = op(A)·x − b
    wabs = Vector{R}(undef, n)       # |op(A)|·|x| + |b|
    wt = Vector{R}(undef, n)         # LACN2 weight  |r| + nz·eps·wabs (+safe1 as needed)

    @inbounds for j in 1:nrhs
        # residual: r := op(A)·X[:,j] − B[:,j]
        for i in 1:n; r[i] = X[i, j]; end
        trmv!(A, r; uplo = uplo, trans = trans, diag = diag)
        for i in 1:n; r[i] -= B[i, j]; end

        for i in 1:n; wabs[i] = _bk_cabs1(B[i, j]); end
        if notran
            if upper
                for k in 1:n
                    xk = _bk_cabs1(X[k, j])
                    hi = nounit ? k : (k - 1)
                    for i in 1:hi; wabs[i] += _bk_cabs1(A[i, k]) * xk; end
                    !nounit && (wabs[k] += xk)
                end
            else
                for k in 1:n
                    xk = _bk_cabs1(X[k, j])
                    lo = nounit ? k : (k + 1)
                    for i in lo:n; wabs[i] += _bk_cabs1(A[i, k]) * xk; end
                    !nounit && (wabs[k] += xk)
                end
            end
        else
            if upper
                for k in 1:n
                    s = nounit ? zero(R) : _bk_cabs1(X[k, j])
                    hi = nounit ? k : (k - 1)
                    for i in 1:hi; s += _bk_cabs1(A[i, k]) * _bk_cabs1(X[i, j]); end
                    wabs[k] += s
                end
            else
                for k in 1:n
                    s = nounit ? zero(R) : _bk_cabs1(X[k, j])
                    lo = nounit ? k : (k + 1)
                    for i in lo:n; s += _bk_cabs1(A[i, k]) * _bk_cabs1(X[i, j]); end
                    wabs[k] += s
                end
            end
        end
        s = zero(R)
        for i in 1:n
            s = wabs[i] > safe2 ? max(s, _bk_cabs1(r[i]) / wabs[i]) :
                                   max(s, (_bk_cabs1(r[i]) + safe1) / (wabs[i] + safe1))
        end
        Berr[j] = s

        for i in 1:n
            wt[i] = wabs[i] > safe2 ? _bk_cabs1(r[i]) + R(nz) * eps_p * wabs[i] :
                                       _bk_cabs1(r[i]) + R(nz) * eps_p * wabs[i] + safe1
        end
        applyf = function (xv, kase)
            if kase == 1
                trsv!(A, xv; uplo = uplo, trans = transt, diag = diag)
                for i in 1:n; xv[i] *= wt[i]; end
            else
                for i in 1:n; xv[i] *= wt[i]; end
                trsv!(A, xv; uplo = uplo, trans = transn, diag = diag)
            end
            return nothing
        end
        ferr_j = _lacn2_estimate(n, applyf, T)
        lstres = zero(R)
        for i in 1:n; lstres = max(lstres, _bk_cabs1(X[i, j])); end
        Ferr[j] = lstres != zero(R) ? ferr_j / lstres : ferr_j
    end
    return Ferr, Berr
end
