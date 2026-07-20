# LAPACK gesvx — the expert general-solve driver: A·X = B (or AᵀX=B / AᴴX=B) with optional
# equilibration, an LU factorization, iterative refinement, and forward/backward error bounds +
# a reciprocal condition number. Composed from PureBLAS's own getrf!/trsm!/_laswp!/gecon! (correctness-
# first — self-consistent on standard-convention factors). Generic over s/d/c/z via T<:BlasFloat.
#
# ferr is a VALID (LAPACK-faithful in form, slightly conservative) forward-error bound:
#   ferr_j ≤ ‖A⁻¹‖_∞ · ‖ |r| + (n+1)·eps·(|A||x|+|b|) ‖_∞ / ‖x‖_∞
# with ‖A⁻¹‖_∞ from the Higham–Hager estimator (gecon! norm='I'). berr is LAPACK's exact componentwise
# backward error. This bounds the true error (correctness-first); it is not bit-identical to OpenBLAS.

# ── geequ: row/col equilibration scale factors (LAPACK dgeequ). R_i = 1/max_j|A_ij|; then
# C_j = 1/max_i|R_i·A_ij|. Returns (rowcnd, colcnd, amax); R, C filled in place. ────────────────────────
function _geequ!(A::AbstractMatrix{T}, R::AbstractVector{Tr}, C::AbstractVector{Tr}) where {T,Tr<:Real}
    m, n = size(A)
    sfmin = floatmin(Tr)
    @inbounds for i in 1:m; R[i] = zero(Tr); end
    @inbounds for j in 1:n, i in 1:m
        a = abs(A[i, j]); a > R[i] && (R[i] = a)
    end
    rcmin = Tr(Inf); rcmax = zero(Tr); amax = zero(Tr)
    @inbounds for i in 1:m
        amax = max(amax, R[i])
        R[i] = R[i] > sfmin ? one(Tr) / R[i] : one(Tr)
        rcmin = min(rcmin, R[i]); rcmax = max(rcmax, R[i])
    end
    rowcnd = rcmax > 0 ? max(rcmin, sfmin) / min(rcmax, one(Tr) / sfmin) : one(Tr)
    @inbounds for j in 1:n; C[j] = zero(Tr); end
    @inbounds for j in 1:n, i in 1:m
        a = abs(A[i, j]) * R[i]; a > C[j] && (C[j] = a)
    end
    ccmin = Tr(Inf); ccmax = zero(Tr)
    @inbounds for j in 1:n
        C[j] = C[j] > sfmin ? one(Tr) / C[j] : one(Tr)
        ccmin = min(ccmin, C[j]); ccmax = max(ccmax, C[j])
    end
    colcnd = ccmax > 0 ? max(ccmin, sfmin) / min(ccmax, one(Tr) / sfmin) : one(Tr)
    return rowcnd, colcnd, amax
end

# dlaqge: decide the equilibration type from (rowcnd, colcnd, amax) and scale A in place. Returns equed.
function _laqge!(A::AbstractMatrix{T}, R::AbstractVector{Tr}, C::AbstractVector{Tr},
                 rowcnd::Tr, colcnd::Tr, amax::Tr) where {T,Tr<:Real}
    m, n = size(A)
    small = floatmin(Tr) / eps(Tr); large = one(Tr) / small
    (amax == 0 || m == 0 || n == 0) && return 'N'
    dorow = !(rowcnd >= Tr(0.1) && amax >= small && amax <= large)
    docol = !(colcnd >= Tr(0.1))
    if dorow && docol
        @inbounds for j in 1:n, i in 1:m; A[i, j] *= R[i] * C[j]; end
        return 'B'
    elseif dorow
        @inbounds for j in 1:n, i in 1:m; A[i, j] *= R[i]; end
        return 'R'
    elseif docol
        @inbounds for j in 1:n, i in 1:m; A[i, j] *= C[j]; end
        return 'C'
    end
    return 'N'
end

# ── gerfs: iterative refinement of the solution X (columns), returning ferr, berr per RHS. `A` is the
# ORIGINAL (possibly-equilibrated) matrix; `AF`/`ipiv` its LU factors. `ainv_inf` = ‖A⁻¹‖_∞ estimate. ──
function _gerfs!(trans::Char, A::AbstractMatrix{T}, AF::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
                 B::AbstractMatrix{T}, X::AbstractMatrix{T}, ferr::AbstractVector{Tr},
                 berr::AbstractVector{Tr}, ainv_inf::Tr) where {T,Tr<:Real}
    n = size(A, 1); nrhs = size(B, 2)
    epsm = eps(Tr); sfmin = floatmin(Tr)
    safe1 = (n + 1) * sfmin; safe2 = safe1 / epsm
    notran = trans == 'N'
    opel(i, k) = notran ? A[i, k] : (trans == 'T' ? A[k, i] : conj(A[k, i]))  # op(A)[i,k]
    r = Vector{T}(undef, n); wrk = Vector{Tr}(undef, n); dx = Vector{T}(undef, n)
    @inbounds for j in 1:nrhs
        x = view(X, :, j); b = view(B, :, j)
        lstres = Tr(3)
        for _iter in 1:5
            for i in 1:n                                    # r = b - op(A)·x  ; wrk = |b| + |op(A)|·|x|
                acc = b[i]; w = abs(b[i])
                for k in 1:n
                    aik = opel(i, k); acc -= aik * x[k]; w += abs(aik) * abs(x[k])
                end
                r[i] = acc; wrk[i] = w
            end
            s = zero(Tr)                                     # componentwise backward error
            for i in 1:n
                s = wrk[i] > safe2 ? max(s, abs(r[i]) / wrk[i]) :
                    max(s, (abs(r[i]) + safe1) / (wrk[i] + safe1))
            end
            berr[j] = s
            (s <= epsm || s > lstres / 2) && break
            lstres = s
            for i in 1:n; dx[i] = r[i]; end                  # solve op(A)·δx = r on the LU factors
            _lu_solve_vec!(trans, AF, ipiv, dx)
            for i in 1:n; x[i] += dx[i]; end
        end
        # ferr: w_i = |r_i| + (n+1)·eps·(|A||x|+|b|)_i ; ferr ≤ ‖A⁻¹‖_∞·‖w‖_∞ / ‖x‖_∞. r/wrk hold the
        # last refinement step's residual and |b|+|A||x| — reuse wrk directly (it IS |b|+|op(A)||x|).
        wmax = zero(Tr)
        for i in 1:n
            wv = abs(r[i]) + (n + 1) * epsm * wrk[i]
            wmax = max(wmax, wv)
        end
        xmax = zero(Tr); for i in 1:n; xmax = max(xmax, abs(x[i])); end
        ferr[j] = xmax > 0 ? ainv_inf * wmax / xmax : ainv_inf * wmax
    end
    return ferr, berr
end

# op(A)·δx = r on the LU factors (mirrors getrs_64_'s composition).
function _lu_solve_vec!(trans::Char, AF::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
                        d::AbstractVector{T}) where {T}
    n = length(d)
    if trans == 'N'
        @inbounds for i in 1:n                               # P·r
            q = Int(ipiv[i]); (q != i) && ((d[i], d[q]) = (d[q], d[i]))
        end
        trsv!(AF, d; uplo = 'L', trans = 'N', diag = 'U')
        trsv!(AF, d; uplo = 'U', trans = 'N', diag = 'N')
    else
        trsv!(AF, d; uplo = 'U', trans = trans, diag = 'N')
        trsv!(AF, d; uplo = 'L', trans = trans, diag = 'U')
        @inbounds for i in n:-1:1
            q = Int(ipiv[i]); (q != i) && ((d[i], d[q]) = (d[q], d[i]))
        end
    end
    return d
end

_anorm1(A) = (m = size(A, 1); n = size(A, 2); v = zero(real(eltype(A)));
    @inbounds for j in 1:n; s = zero(real(eltype(A))); for i in 1:m; s += abs(A[i, j]); end; v = max(v, s); end; v)
_anorm_inf(A) = (m = size(A, 1); n = size(A, 2); v = zero(real(eltype(A)));
    @inbounds for i in 1:m; s = zero(real(eltype(A))); for j in 1:n; s += abs(A[i, j]); end; v = max(v, s); end; v)
_amax(A) = (v = zero(real(eltype(A))); @inbounds for a in A; v = max(v, abs(a)); end; v)

"""
    gesvx!(fact, trans, A, AF, ipiv, equed, R, C, B) -> (X, equed, rcond, ferr, berr, rpgf)

Expert general solver (LAPACK `gesvx`). `fact ∈ {'F','N','E'}` (factored/notyet/equilibrate),
`trans ∈ {'N','T','C'}`. Optionally equilibrates (`equed`), LU-factors into `AF`/`ipiv`, solves,
iteratively refines, and returns error bounds + reciprocal condition number `rcond` and reciprocal
pivot growth `rpgf`. `A`, `AF`, `ipiv`, `R`, `C`, `B` may be modified in place. Generic over s/d/c/z.
"""
function gesvx!(fact::Char, trans::Char, A::AbstractMatrix{T}, AF::AbstractMatrix{T},
                ipiv::AbstractVector{<:Integer}, equed::Char, R::AbstractVector{Tr},
                C::AbstractVector{Tr}, B::AbstractMatrix{T}) where {T<:BlasFloat,Tr<:Real}
    n = size(A, 1); nrhs = size(B, 2)
    X = Matrix{T}(undef, n, nrhs)
    ferr = Vector{Tr}(undef, nrhs); berr = Vector{Tr}(undef, nrhs)
    n == 0 && return X, equed, one(Tr), ferr, berr, one(Tr)

    # 1) equilibrate (fact='E'): compute R,C, scale A, choose equed.
    if fact == 'E'
        rowcnd, colcnd, amax = _geequ!(A, R, C)
        equed = _laqge!(A, R, C, rowcnd, colcnd, amax)
    end
    rowequ = equed == 'R' || equed == 'B'
    colequ = equed == 'C' || equed == 'B'

    # 2) reciprocal pivot growth = maxabs(A) / maxabs(U) (LAPACK dgesvx work[1]).
    # 3) factor (fact='N' or 'E'): AF ← A, getrf!.
    if fact == 'N' || fact == 'E'
        @inbounds for j in 1:n, i in 1:n; AF[i, j] = A[i, j]; end
        _, _, info = getrf!(AF, ipiv)
        info != 0 && return X, equed, zero(Tr), ferr, berr, zero(Tr)   # singular
    end
    umax = zero(Tr); @inbounds for j in 1:n, i in 1:j; umax = max(umax, abs(AF[i, j])); end
    amx = _amax(A)
    rpgf = umax == 0 ? one(Tr) : amx / umax

    # 4) scale B (row-equilibration for trans='N', col for trans≠'N').
    scaleB = notran = trans == 'N'
    if (notran && rowequ) || (!notran && colequ)
        f = notran ? R : C
        @inbounds for j in 1:nrhs, i in 1:n; B[i, j] *= f[i]; end
    end

    # 5) solve AF·X = B (X ← B, then triangular solves).
    @inbounds for j in 1:nrhs, i in 1:n; X[i, j] = B[i, j]; end
    if notran
        _laswp!(X, ipiv, 1, n, 1, nrhs)
        trsm!(X, AF; side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = one(T))
        trsm!(X, AF; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = one(T))
    else
        trsm!(X, AF; side = 'L', uplo = 'U', transA = trans, diag = 'N', alpha = one(T))
        trsm!(X, AF; side = 'L', uplo = 'L', transA = trans, diag = 'U', alpha = one(T))
        @inbounds for i in n:-1:1
            q = Int(ipiv[i])
            (q != i) && (for j in 1:nrhs; (X[i, j], X[q, j]) = (X[q, j], X[i, j]); end)
        end
    end

    # 6) rcond (‖A‖₁ of the equilibrated matrix) + ‖A⁻¹‖_∞ estimate for ferr.
    a1 = _anorm1(A); ainf = _anorm_inf(A)
    rcond = gecon!(a1, AF, ipiv; norm = '1')
    rcondI = gecon!(ainf, AF, ipiv; norm = 'I')
    ainv_inf = (rcondI > 0 && ainf > 0) ? one(Tr) / (rcondI * ainf) : zero(Tr)

    # 7) iterative refinement → ferr, berr (on the equilibrated system).
    _gerfs!(trans, A, AF, ipiv, B, X, ferr, berr, ainv_inf)

    # 8) unscale X (col-equilibration for trans='N', row for trans≠'N'); rescale ferr accordingly.
    if (notran && colequ) || (!notran && rowequ)
        f = notran ? C : R
        @inbounds for j in 1:nrhs, i in 1:n; X[i, j] *= f[i]; end
    end
    return X, equed, rcond, ferr, berr, rpgf
end
