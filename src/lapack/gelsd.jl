# Rank-deficient least squares via the SINGULAR VALUE DECOMPOSITION (LAPACK gelsd). Solves
# min‖A·X − B‖₂ for a possibly rank-deficient A, returning the MINIMUM-NORM solution
# X = V·Σ⁺·Uᴴ·B, where singular values ≤ rcond·σ_max are treated as zero (their reciprocals dropped).
# Composed from PureBLAS's own economy SVD (`gesvd!`, svd.jl — gebrd bidiagonalization + bdsqr/bdsdc)
# and `gemm!` (gemm.jl) for the two back-projections. Returns the singular values too.
# Generic over Float64/ComplexF32/ComplexF64 (native gesvd! kernels); Float32-real is computed in
# Float64 (see below). Mirrors dgelsd/zgelsd: bidiagonalize → SVD → rcond-threshold → solve.

# Default rank-cut when the caller passes rcond ∉ (0,1). PureBLAS's gelsd composes a FULL SVD and then
# thresholds σᵢ ≤ rcond·σ₁ (vs LAPACK dlalsd's D&C-integrated deflation, which collapses null σ's to
# ~machine-zero). For an exactly rank-deficient A the compose-SVD path leaves the null σ's at a FLOOR of
# ~a-few·eps·σ₁ — above LAPACK's eps·σ₁ cut (measured up to 3.1e-16·σ₁ > eps) — so a fixed eps·σ₁ threshold
# splits the null cluster, keeps a ~3e-16 σ, and divides by it → a ‖x‖~1e14 garbage solution (Fable
# adversarial review). Scale the default cut with the problem size — `min(m,n)·eps·σ₁`, exactly Julia's
# own `pinv` rtol convention — which clears the O(√n)·eps SVD null floor while retaining every genuine
# singular value (the spectrum shows a >10-order gap between the smallest true σ and the null cluster).
@inline _gelsd_eps(::Type{R}, mn::Int) where {R <: Real} = R(max(mn, 1)) * eps(R)

# Solve min‖A·X − B‖₂ (A m×n). B is size ≥ max(m,n) × nrhs (LAPACK ldb): input rows 1:m hold b,
# output rows 1:n hold X. rcond thresholds the singular values (∉(0,1) ⇒ machine precision, per
# dlalsd). Overwrites A and B. Returns (B, rank, s) with s the descending singular values (length
# min(m,n)).
#
# IN-PLACE FORM (caller-owned `s` last, mirroring getrf!(A, ipiv)): `s` needs length ≥ min(m,n) and
# holds the descending singular values on exit — EXCEPT when min(m,n)==0, where the routine returns
# immediately and leaves `s` untouched. `s` must not alias `A` or `B`: gesvd! fills `s` before the
# `Uᴴ·b` gemm! reads B, so an aliased `s` would destroy the right-hand side (and `A` is copied to
# workspace first, so that half is only a latent hazard — both are rejected).
function gelsd!(
        A::AbstractMatrix{T}, B::AbstractMatrix{T}, rcond::Real,
        s::AbstractVector{<:Real}
    ) where {T <: BlasFloat}
    m, n = size(A); mn = min(m, n); R = real(T); nrhs = size(B, 2)
    size(B, 1) >= max(m, n) || _throw_brows_mn(:gelsd!, size(B, 1), max(m, n))
    length(s) >= mn || throw(ArgumentError("gelsd!: length(s) must be ≥ min(m,n)"))
    (Base.mightalias(s, A) || Base.mightalias(s, B)) &&
        throw(ArgumentError("gelsd!: `s` must not alias `A` or `B`"))
    mn == 0 && return B, 0, s
    sv = view(s, 1:mn)
    # economy SVD  A = U·diag(s)·Vᴴ  (U m×mn, s descending, Vt = Vᴴ mn×n)
    U, Vt, C, Ac = _gelsd_work(T, m, n, mn, nrhs)
    @inbounds for j in 1:n, i in 1:m           # gesvd! destroys its A; A itself stays the caller's
        Ac[i, j] = A[i, j]
    end
    gesvd!(Ac, U, sv, Vt)
    # effective rank: σ_i ≤ tol treated as zero (dlalsd: rcond∉(0,1) ⇒ rounding unit)
    rcnd = (rcond <= 0 || rcond >= 1) ? _gelsd_eps(R, mn) : R(rcond)
    tol = rcnd * sv[1]
    rank = 0
    @inbounds for i in 1:mn
        sv[i] > tol && (rank += 1)
    end
    # c := Uᴴ·b  (mn × nrhs)
    gemm!(C, U, view(B, 1:m, :); transA = 'C', alpha = one(T), beta = zero(T))
    # apply Σ⁺ (drop the reciprocals of the thresholded-to-zero singular values)
    @inbounds for jc in 1:nrhs, i in 1:mn
        C[i, jc] = sv[i] > tol ? C[i, jc] / sv[i] : zero(T)
    end
    # X := V·c = (Vᴴ)ᴴ·c  (n × nrhs) → B(1:n)   (C already holds Uᴴb, so overwriting B(1:m) is safe)
    gemm!(view(B, 1:n, :), Vt, C; transA = 'C', alpha = one(T), beta = zero(T))
    return B, rank, s
end

# Allocating convenience form — identical behaviour and return value to the pre-workspace gelsd!.
function gelsd!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, rcond::Real) where {T <: BlasFloat}
    s = Vector{real(T)}(undef, min(size(A, 1), size(A, 2)))
    return gelsd!(A, B, rcond, s)
end

# Float32-real path: PureBLAS's gesvd! has no Float32-real kernel (svd.jl covers Float64 + complex).
# Compute in Float64 — MORE accurate than sgelsd but the same min-norm LS solution to Float32 tolerance.
# ponytail: promote-to-Float64; add a native Float32 SVD kernel if Float32 gelsd perf ever matters.
# The Float64 staging lives on `_l3ws(Float64)` — a different owner object from the Float32 workspace,
# so it cannot alias the Float32 caller's buffers. `_gelsd_promote_work`'s first argument sizes BOTH the
# A staging and the B staging, so it is called with max(m,n) (B's ldb-mandated row count) and the A
# staging is then narrowed to its own m×n leading block.
function gelsd!(A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32}, rcond::Real, s::AbstractVector{<:Real})
    m, n = size(A); mn = min(m, n); nrhs = size(B, 2); mb = max(m, n)
    size(B, 1) >= mb || _throw_brows_mn(:gelsd!, size(B, 1), mb)
    length(s) >= mn || throw(ArgumentError("gelsd!: length(s) must be ≥ min(m,n)"))
    (Base.mightalias(s, A) || Base.mightalias(s, B)) &&
        throw(ArgumentError("gelsd!: `s` must not alias `A` or `B`"))
    mn == 0 && return B, 0, s
    Ad0, Bd, sd = _gelsd_promote_work(mb, n, mn, nrhs)
    Ad = view(Ad0, 1:m, 1:n)
    @inbounds for j in 1:n, i in 1:m
        Ad[i, j] = Float64(A[i, j])
    end
    @inbounds for j in 1:nrhs, i in 1:mb
        Bd[i, j] = Float64(B[i, j])
    end
    _, rank, _ = gelsd!(Ad, Bd, rcond, sd)
    @inbounds for j in 1:nrhs, i in 1:mb
        B[i, j] = Float32(Bd[i, j])
    end
    @inbounds for i in 1:mn
        s[i] = Float32(sd[i])
    end
    return B, rank, s
end

function gelsd!(A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32}, rcond::Real)
    s = Vector{Float32}(undef, min(size(A, 1), size(A, 2)))
    return gelsd!(A, B, rcond, s)
end
