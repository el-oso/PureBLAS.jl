# Condition-number estimation: gecon / trcon / pocon.
#
# rcond = 1 / (‖A‖ · ‖A⁻¹‖) estimated from an existing factorization. ‖A⁻¹‖ is estimated WITHOUT
# forming the inverse, via the Higham–Hager block-1-norm estimator (LAPACK's DLACON/ZLACON, thread-
# safe variant DLACN2/ZLACN2): it needs only the action of op = A⁻¹ (and opᴴ) on a vector, which we
# supply through PureBLAS's triangular solves (trsv!) over the stored factors + the LU row pivots.
#
# The estimator loop below is a DIRECT transcription of LAPACK's reference DLACON (real) and ZLACON
# (complex) — the sign-vector iteration, the est≤estold cycling break, the real-only sign-repeat
# convergence test, and the final alternate-vector estimate 2·‖x‖₁/(3n). Reverse-communication (the
# Fortran KASE/JUMP state machine) is replaced by a callback `applyop!(x, adjoint)`; the arithmetic
# is byte-for-byte the same. Generic over Float64/Float32/ComplexF64/ComplexF32.
#
# Source mirrored: LAPACK 3.x reference DLACON.f / ZLACON.f (netlib), and the DGECON/DTRCON/DPOCON
# (+ Z variants) drivers for the solve wiring and the 1-norm↔∞-norm operator swap.

# 1-norm of a vector: Σ|xᵢ| (real DASUM; complex DZSUM1 = Σ modulus, NOT cabs1 dzasum). abs(z) = |z|
# covers both, matching the exact reference metric for each type.
@inline function _cond_absum(x::AbstractVector{T}) where {T}
    s = zero(real(T))
    @inbounds for i in eachindex(x); s += abs(x[i]); end
    return s
end

# Index of first element of maximum modulus (real IDAMAX; complex IZMAX1 — modulus, not cabs1).
@inline function _cond_iamax(x::AbstractVector{T}) where {T}
    R = real(T); m = -one(R); idx = 1
    @inbounds for i in eachindex(x)
        a = abs(x[i]); if a > m; m = a; idx = i; end
    end
    return idx
end

# Complex/real "sign": SIGN(1,x)=±1 (real, DLACON), x/|x| or 1 if |x|≤safmin (complex, ZLACON).
@inline _cond_sign(x::T) where {T<:Real} = x >= zero(T) ? one(T) : -one(T)
@inline function _cond_sign(x::T) where {T<:Complex}
    ax = abs(x)
    return ax > floatmin(real(T)) ? x / ax : one(T)
end

# ── _lacn2!: Higham–Hager 1-norm estimator of ‖op‖₁. `applyop!(x, adj)` overwrites x with op·x when
# adj=false and opᴴ·x when adj=true. Returns the estimate. Allocates its own x/v/isgn workspace
# (not a hot path). EXACT DLACON/ZLACON transcription — do not "simplify" the iteration.
function _lacn2!(applyop!, ::Type{T}, n::Int) where {T}
    R = real(T)
    x = Vector{T}(undef, n)
    v = Vector{T}(undef, n)
    isgn = Vector{Int}(undef, T <: Real ? n : 0)   # real-only sign-repeat test vector
    itmax = 5

    fill!(x, one(T) / n)
    applyop!(x, false)                              # x ← op·x   (KASE 1, JUMP 1)
    if n == 1
        v[1] = x[1]
        return abs(v[1])
    end
    est = _cond_absum(x)
    @inbounds for i in eachindex(x); x[i] = _cond_sign(x[i]); end
    if T <: Real
        @inbounds for i in eachindex(x); isgn[i] = round(Int, real(x[i])); end
    end
    applyop!(x, true)                               # x ← opᴴ·x  (KASE 2, JUMP 2)
    j = _cond_iamax(x)
    iter = 2

    final = false
    while true                                      # MAIN LOOP (iterations 2..ITMAX)
        fill!(x, zero(T)); @inbounds x[j] = one(T)
        applyop!(x, false)                          # x ← op·eⱼ  (KASE 1, JUMP 3)
        copyto!(v, x)
        estold = est
        est = _cond_absum(v)

        if T <: Real                                # real: repeated sign vector ⇒ converged (→ final)
            same = true
            @inbounds for i in eachindex(x)
                if round(Int, real(_cond_sign(x[i]))) != isgn[i]; same = false; break; end
            end
            same && (final = true)
        end
        if !final && est <= estold                  # cycling test (both real & complex) ⇒ final
            final = true
        end
        final && break

        @inbounds for i in eachindex(x); x[i] = _cond_sign(x[i]); end
        if T <: Real
            @inbounds for i in eachindex(x); isgn[i] = round(Int, real(x[i])); end
        end
        applyop!(x, true)                           # x ← opᴴ·x  (KASE 2, JUMP 4)
        jlast = j
        j = _cond_iamax(x)
        if abs(x[jlast]) != abs(x[j]) && iter < itmax
            iter += 1
            continue
        end
        break
    end

    # FINAL STAGE: alternate signed test vector xᵢ = ±(1 + (i-1)/(n-1)); take max with est.
    altsgn = one(R)
    @inbounds for i in 1:n
        x[i] = T(altsgn * (one(R) + R(i - 1) / R(n - 1)))
        altsgn = -altsgn
    end
    applyop!(x, false)                              # x ← op·x   (KASE 1, JUMP 5)
    temp = 2 * (_cond_absum(x) / (3 * n))
    if temp > est
        copyto!(v, x)
        est = temp
    end
    return est
end

@inline _cond_onenorm(norm::Char) = (norm == '1' || norm == 'O' || norm == 'o')

# Apply the LU row permutation P (getrf ipiv, LAPACK convention) to x: forward for A/Aᴴ solves.
@inline function _cond_perm_fwd!(x, ipiv)
    @inbounds for i in eachindex(ipiv)
        p = ipiv[i]; if p != i; x[i], x[p] = x[p], x[i]; end
    end
    return x
end
@inline function _cond_perm_bwd!(x, ipiv)          # Pᵀ (reverse order), for the adjoint solve
    @inbounds for i in length(ipiv):-1:1
        p = ipiv[i]; if p != i; x[i], x[p] = x[p], x[i]; end
    end
    return x
end

"""
    gecon!(normA, A_lu, ipiv; norm='1') -> rcond

Estimate the reciprocal condition number `1/(‖A‖·‖A⁻¹‖)` of a general matrix `A` from its LU
factorization (`A_lu`, `ipiv` as returned by `getrf!`). `normA` is `‖A‖₁` (norm='1'/'O') or `‖A‖_∞`
(norm='I') of the ORIGINAL `A`. Estimates `‖A⁻¹‖` with the Higham–Hager estimator, applying `A⁻¹`
via getrs-style solves (Px → L⁻¹ → U⁻¹) through the stored factors. Generic over s/d/c/z.
"""
function gecon!(normA::Real, A_lu::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer};
                norm::Char = '1') where {T<:BlasFloat}
    n = size(A_lu, 1)
    size(A_lu, 2) == n || throw(DimensionMismatch("gecon!: A_lu must be square"))
    (n == 0 || normA <= 0) && return zero(real(T))

    # A⁻¹·y = U⁻¹ L⁻¹ P y ; A⁻ᴴ·y = Pᵀ L⁻ᴴ U⁻ᴴ y
    solveA!(x) = (_cond_perm_fwd!(x, ipiv);
                  trsv!(A_lu, x; uplo='L', trans='N', diag='U');
                  trsv!(A_lu, x; uplo='U', trans='N', diag='N'); x)
    solveAH!(x) = (trsv!(A_lu, x; uplo='U', trans='C', diag='N');
                   trsv!(A_lu, x; uplo='L', trans='C', diag='U');
                   _cond_perm_bwd!(x, ipiv); x)

    one1 = _cond_onenorm(norm)
    # norm='1': op=A⁻¹. norm='I': ‖A⁻¹‖_∞ = ‖A⁻ᴴ‖₁ ⇒ op=A⁻ᴴ. adj flips which solve runs.
    applyop!(x, adj) = ((one1 ? !adj : adj) ? solveA!(x) : solveAH!(x))

    ainv = _lacn2!(applyop!, T, n)
    return iszero(ainv) ? zero(real(T)) : (one(real(T)) / ainv) / real(normA)
end

# 1-norm / ∞-norm of a triangular matrix (with unit-diagonal handling), for trcon's own ‖A‖.
function _cond_trnorm(A::AbstractMatrix{T}, uplo::Char, diag::Char, one1::Bool) where {T}
    n = size(A, 1); R = real(T); up = uplo == 'U'; unit = diag == 'U'
    val = zero(R)
    @inbounds if one1                               # max column sum
        for j in 1:n
            s = unit ? one(R) : zero(R)
            rng = up ? (unit ? (1:j-1) : (1:j)) : (unit ? (j+1:n) : (j:n))
            for i in rng; s += abs(A[i, j]); end
            s > val && (val = s)
        end
    else                                            # max row sum
        for i in 1:n
            s = unit ? one(R) : zero(R)
            rng = up ? (unit ? (i+1:n) : (i:n)) : (unit ? (1:i-1) : (1:i))
            for j in rng; s += abs(A[i, j]); end
            s > val && (val = s)
        end
    end
    return val
end

"""
    trcon!(A; uplo='U', diag='N', norm='1') -> rcond

Estimate the reciprocal condition number of the triangular matrix `A` (upper if `uplo='U'`, unit
diagonal if `diag='U'`). `‖A‖` is computed directly; `‖A⁻¹‖` via the Higham–Hager estimator driving
`trsv!` solves. Generic over s/d/c/z.
"""
function trcon!(A::AbstractMatrix{T}; uplo::Char = 'U', diag::Char = 'N', norm::Char = '1') where {T<:BlasFloat}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("trcon!: A must be square"))
    n == 0 && return one(real(T))
    one1 = _cond_onenorm(norm)
    anorm = _cond_trnorm(A, uplo, diag, one1)
    anorm <= 0 && return zero(real(T))

    solveA!(x)  = trsv!(A, x; uplo=uplo, trans='N', diag=diag)
    solveAH!(x) = trsv!(A, x; uplo=uplo, trans='C', diag=diag)
    applyop!(x, adj) = ((one1 ? !adj : adj) ? solveA!(x) : solveAH!(x))

    ainv = _lacn2!(applyop!, T, n)
    return iszero(ainv) ? zero(real(T)) : (one(real(T)) / ainv) / anorm
end

"""
    pocon!(normA, A_chol; uplo='L', norm='1') -> rcond

Estimate the reciprocal condition number of a Hermitian positive-definite `A` from its Cholesky
factor (`A_chol`: `A = L·Lᴴ` if `uplo='L'`, `A = Uᴴ·U` if `uplo='U'`). `normA` is `‖A‖₁ = ‖A‖_∞`
(equal for Hermitian). `A⁻¹` is Hermitian, so op = opᴴ and norm='1'/'I' coincide. Generic s/d/c/z.
"""
function pocon!(normA::Real, A_chol::AbstractMatrix{T}; uplo::Char = 'L', norm::Char = '1') where {T<:BlasFloat}
    n = size(A_chol, 1)
    size(A_chol, 2) == n || throw(DimensionMismatch("pocon!: A_chol must be square"))
    (n == 0 || normA <= 0) && return zero(real(T))

    # A⁻¹ = L⁻ᴴ L⁻¹ (lower) or U⁻¹ U⁻ᴴ (upper); Hermitian ⇒ same operator for op and opᴴ.
    solve! = uplo == 'L' ?
        (x -> (trsv!(A_chol, x; uplo='L', trans='N', diag='N');
               trsv!(A_chol, x; uplo='L', trans='C', diag='N'); x)) :
        (x -> (trsv!(A_chol, x; uplo='U', trans='C', diag='N');
               trsv!(A_chol, x; uplo='U', trans='N', diag='N'); x))
    applyop!(x, _adj) = solve!(x)

    ainv = _lacn2!(applyop!, T, n)
    return iszero(ainv) ? zero(real(T)) : (one(real(T)) / ainv) / real(normA)
end
