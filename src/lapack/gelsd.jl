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
function gelsd!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, rcond::Real) where {T <: BlasFloat}
    m, n = size(A); mn = min(m, n); R = real(T); nrhs = size(B, 2)
    size(B, 1) >= max(m, n) ||
        throw(DimensionMismatch("gelsd!: size(B,1)=$(size(B, 1)) must be ≥ max(m,n)=$(max(m, n))"))
    s = Vector{R}(undef, mn)
    mn == 0 && return B, 0, s
    # economy SVD  A = U·diag(s)·Vᴴ  (U m×mn, s descending, Vt = Vᴴ mn×n)
    U = Matrix{T}(undef, m, mn); Vt = Matrix{T}(undef, mn, n)
    gesvd!(copy(A), U, s, Vt)
    # effective rank: σ_i ≤ tol treated as zero (dlalsd: rcond∉(0,1) ⇒ rounding unit)
    rcnd = (rcond <= 0 || rcond >= 1) ? _gelsd_eps(R, mn) : R(rcond)
    tol = rcnd * s[1]
    rank = 0
    @inbounds for i in 1:mn
        s[i] > tol && (rank += 1)
    end
    # c := Uᴴ·b  (mn × nrhs)
    C = Matrix{T}(undef, mn, nrhs)
    gemm!(C, U, view(B, 1:m, :); transA = 'C', alpha = one(T), beta = zero(T))
    # apply Σ⁺ (drop the reciprocals of the thresholded-to-zero singular values)
    @inbounds for jc in 1:nrhs, i in 1:mn
        C[i, jc] = s[i] > tol ? C[i, jc] / s[i] : zero(T)
    end
    # X := V·c = (Vᴴ)ᴴ·c  (n × nrhs) → B(1:n)   (C already holds Uᴴb, so overwriting B(1:m) is safe)
    gemm!(view(B, 1:n, :), Vt, C; transA = 'C', alpha = one(T), beta = zero(T))
    return B, rank, s
end

# Float32-real path: PureBLAS's gesvd! has no Float32-real kernel (svd.jl covers Float64 + complex).
# Compute in Float64 — MORE accurate than sgelsd but the same min-norm LS solution to Float32 tolerance.
# ponytail: promote-to-Float64; add a native Float32 SVD kernel if Float32 gelsd perf ever matters.
function gelsd!(A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32}, rcond::Real)
    m, n = size(A); mn = min(m, n)
    Bd = Matrix{Float64}(undef, size(B, 1), size(B, 2)); copyto!(Bd, B)
    _, rank, sd = gelsd!(Matrix{Float64}(A), Bd, rcond)
    copyto!(B, Bd)
    s = Vector{Float32}(undef, mn)
    @inbounds for i in 1:mn
        s[i] = Float32(sd[i])
    end
    return B, rank, s
end
