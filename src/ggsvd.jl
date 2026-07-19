# LAPACK generalized SVD of a matrix pair (ggsvd3):  for A (m×n), B (p×n), find orthogonal U (m×m),
# V (p×p), Q (n×n) and a nonsingular common right factor such that
#     Uᵀ·A·Q = Σ₁·R,   Vᵀ·B·Q = Σ₂·R,   with  Σ₁ = diag(α), Σ₂ = diag(β),  αᵢ² + βᵢ² = 1,
# R upper-triangular. The generalized singular values are αᵢ/βᵢ. This is the FULL-RANK path
# (rank([A;B]) = n): the "[0 R]" of the general LAPACK form degenerates to the full n×n R.
#
# Method — GSVD via QR + CS decomposition (Van Loan / Golub–Van Loan §8.7.4), NOT the LAPACK
# dggsvp3+dtgsja (pivoted-preprocessing + Kogbetliantz-Jacobi) path:
#   1. QR of the stack:  [A; B] = Q_M·Rup   (Q_M (m+p)×n orthonormal, Rup n×n upper).
#   2. Split Q_M = [Q₁; Q₂] (Q₁ m×n, Q₂ p×n) and CS-decompose the pair:
#        SVD  Q₁ = U·C·Zᵀ   (C = diag αᵢ);  then W = Q₂·Z has orthogonal columns with ‖W[:,i]‖ = βᵢ,
#        so V = normalize(W) completed to orthonormal, and αᵢ² + βᵢ² = 1 holds by construction.
#   3. Common factor  X = Zᵀ·Rup;  A = U·C·X, B = V·S·X.  RQ of X = R·Qᵀ gives the upper-tri R and Q.
#
# LIMITATIONS (honest):
#  - Float64 only — the CS step needs an SVD WITH singular vectors, and this build's gesvd! provides
#    vectors for Float64 only (complex gesvd! with vectors is unimplemented here).
#  - rank([A;B]) < n (deficient common factor) is not handled — needs the dggsvp3 zero-block partitioning.
#  - NORM BALANCE (Fable adversarial review): the QR of the raw stack [A;B] has backward error ~eps·‖[A;B]‖
#    ≈ eps·max(‖A‖,‖B‖), so the A-side identity ‖UᵀAQ−Σ₁R‖/‖A‖ degrades as ~eps·‖B‖/‖A‖ when ‖A‖≪‖B‖ (and
#    symmetrically). Machine precision holds when ‖A‖ and ‖B‖ are comparable; extreme imbalance (many orders
#    of magnitude) loses the smaller matrix's row space. The robust fix (dggsvp3-style pre-scaling of A,B to
#    comparable norms with a matching α/β/R rescale) is a follow-up; do NOT feed pairs with ‖A‖/‖B‖ ≫ 1e6.
# ponytail: complex + rank-deficient + norm-balancing are follow-ups; the full-rank, comparably-scaled real
# GSVD is the substantive, validated deliverable.

# Unblocked orgqr (LAPACK dorg2r): overwrite A (holding _ggl_geqr2! reflectors) with the first `ncol`
# columns of Q = H(1)···H(k). Generic over T (dorg2r/zung2r share this shape).
function _ggl_org2r!(A::AbstractMatrix{T}, tau::AbstractVector{T}, ncol::Int) where {T}
    mm, n = size(A); k = length(tau)
    @inbounds for i in k:-1:1
        if i < ncol
            A[i, i] = one(T)
            _house_left!(view(A, i:mm, i+1:ncol), view(A, i:mm, i), tau[i])
        end
        for l in i+1:mm; A[l, i] *= -tau[i]; end
        A[i, i] = one(T) - tau[i]
        for l in 1:i-1; A[l, i] = zero(T); end
    end
    return A
end

"""
    ggsvd!(A, B) -> NamedTuple

Generalized SVD of the real pair `(A, B)` (`A` m×n, `B` p×n), full-rank path (`rank([A;B]) == n`).
Returns a NamedTuple `(U, V, Q, alpha, beta, R, Sigma1, Sigma2, k, l)`: orthogonal `U` (m×m), `V`
(p×p), `Q` (n×n); cosine/sine vectors `alpha`, `beta` (length n, `alpha².+beta².==1`); upper-triangular
`R` (n×n); and the "generalized-diagonal" scaling matrices `Sigma1` (m×n) and `Sigma2` (p×n) for which
`Uᵀ·A·Q == Sigma1·R`  and  `Vᵀ·B·Q == Sigma2·R`. These identities hold to machine precision **when `‖A‖`
and `‖B‖` are comparable**; under extreme norm imbalance the smaller matrix's identity degrades as
`~eps·max(‖A‖,‖B‖)/min(‖A‖,‖B‖)` (see the file-header NORM BALANCE limitation). `k`, `l` are the block
sizes (`k+l == n` here). Generalized singular values are `alpha ./ beta`. Float64 only (see file
header). `A`, `B` are NOT overwritten.
"""
function ggsvd!(A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64})
    m, n = size(A); p = size(B, 1)
    size(B, 2) == n || throw(DimensionMismatch("ggsvd!: A and B must have the same number of columns"))
    mm = m + p
    mm >= n || throw(DimensionMismatch("ggsvd!: need rows(A)+rows(B) ≥ cols (full-rank path)"))

    # 1) QR of the stack  M = [A; B] = Q_M·Rup.
    M = Matrix{Float64}(undef, mm, n)
    @inbounds for j in 1:n
        for i in 1:m; M[i, j] = A[i, j]; end
        for i in 1:p; M[m + i, j] = B[i, j]; end
    end
    tau = zeros(Float64, n)
    _ggl_geqr2!(M, tau)
    Rup = zeros(Float64, n, n)                              # copy R before org2r destroys the upper part
    @inbounds for j in 1:n, i in 1:j; Rup[i, j] = M[i, j]; end
    _ggl_org2r!(M, tau, n)                                  # M := Q_M (mm×n)
    Q1 = M[1:m, :]; Q2 = M[m+1:mm, :]

    # 2) CS decomposition.  SVD  Q₁ = U·C·Zᵀ.
    U = Matrix{Float64}(undef, m, m); Sc = Vector{Float64}(undef, min(m, n)); Vt = Matrix{Float64}(undef, n, n)
    gesvd!(copy(Q1), U, Sc, Vt; full_u = true, full_v = true)
    Z = permutedims(Vt)                                     # n×n, columns = right singular vectors z_i
    alpha = zeros(Float64, n)
    @inbounds for i in 1:length(Sc); alpha[i] = Sc[i]; end  # cosines (descending; padded 0 for i>min(m,n))

    # W = Q₂·Z  (p×n); columns orthogonal, ‖W[:,i]‖ = βᵢ.
    W = Matrix{Float64}(undef, p, n)
    gemm!(W, Q2, Z)
    beta = zeros(Float64, n)
    @inbounds for i in 1:n
        s = 0.0; for r in 1:p; s += W[r, i]^2; end
        beta[i] = sqrt(s)
    end
    # Numerical hygiene: enforce αᵢ²+βᵢ²=1 exactly via the more reliable of the two (columns of [Q1;Q2]
    # are orthonormal so cos²+sin²=1 analytically; round-off makes them drift by ~1e-16).
    @inbounds for i in 1:n
        nrm = hypot(alpha[i], beta[i])
        if nrm > 0; alpha[i] /= nrm; beta[i] /= nrm; end
    end

    # V: normalize the βᵢ>tol columns of W, complete to a p×p orthonormal basis.
    tol = sqrt(eps(Float64)) * max(1.0, maximum(beta; init = 0.0))
    V = zeros(Float64, p, p)
    col = 0
    @inbounds for i in 1:n
        if beta[i] > tol && col < p
            col += 1
            inv = 1.0 / sqrt(sum(abs2, @view W[:, i]))
            for r in 1:p; V[r, col] = W[r, i] * inv; end
        end
    end
    _complete_orthonormal!(V, col)                          # fill columns col+1:p

    # 3) Common factor  X = Zᵀ·Rup = Vt·Rup, then RQ:  X = R·Qᵀ  (Q = G).
    X = Matrix{Float64}(undef, n, n)
    gemm!(X, Vt, Rup)
    G = Matrix{Float64}(undef, n, n)
    _ggl_gerq2_accumG!(X, G)                                # X[1:n,1:n] → R (upper); X_orig = R·Gᵀ
    R = zeros(Float64, n, n)
    @inbounds for j in 1:n, i in 1:j; R[i, j] = X[i, j]; end
    Q = G                                                   # X = R·Qᵀ ⟹ Uᵀ A Q = Σ₁·R, Vᵀ B Q = Σ₂·R

    # Σ₁ = Uᵀ·(Q₁·Z) = C (m×n, α on the diagonal);  Σ₂ = Vᵀ·W (p×n, β on a generalized diagonal —
    # a single β per row, column-shifted when B is rank-deficient / n>p). Built exactly (not from the
    # normalized α,β) so the returned identities UᵀAQ=Σ₁R, VᵀBQ=Σ₂R hold to machine precision.
    QZ = Matrix{Float64}(undef, m, n); gemm!(QZ, Q1, Z)
    Sigma1 = Matrix{Float64}(undef, m, n); gemm!(Sigma1, U, QZ; transA = 'T')
    Sigma2 = Matrix{Float64}(undef, p, n); gemm!(Sigma2, V, W; transA = 'T')

    # block sizes: k = #{αᵢ ≈ 1} (uncoupled cosine block), l = rest.  k+l = n on the full-rank path.
    k = 0; @inbounds for i in 1:n; alpha[i] >= 1.0 - tol && (k += 1); end
    l = n - k
    return (U = U, V = V, Q = Q, alpha = alpha, beta = beta, R = R,
            Sigma1 = Sigma1, Sigma2 = Sigma2, k = k, l = l)
end

# Complete columns `filled+1:end` of the p×p matrix V (whose first `filled` columns are orthonormal)
# to a full orthonormal basis via modified Gram–Schmidt against random directions.
function _complete_orthonormal!(V::AbstractMatrix{Float64}, filled::Int)
    p = size(V, 1)
    filled >= p && return V
    @inbounds for j in filled+1:p
        # start from a coordinate direction, orthogonalize against existing columns, renormalize
        for attempt in 1:p+1
            for r in 1:p; V[r, j] = 0.0; end
            V[((j + attempt - 2) % p) + 1, j] = 1.0
            for c in 1:j-1
                d = 0.0; for r in 1:p; d += V[r, c] * V[r, j]; end
                for r in 1:p; V[r, j] -= d * V[r, c]; end
            end
            nrm = sqrt(sum(abs2, @view V[:, j]))
            if nrm > 1e-8
                for r in 1:p; V[r, j] /= nrm; end
                break
            end
        end
    end
    return V
end
