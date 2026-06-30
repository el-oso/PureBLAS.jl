using LinearAlgebra: PosDefException

# LAPACK-level routines built on the gated Level-3 BLAS. First: Cholesky (potrf).
# Real symmetric positive-definite A = L·Lᵀ (uplo='L') or A = Uᵀ·U (uplo='U'), factored in place into
# the `uplo` triangle. Right-looking BLOCKED algorithm: each NB diagonal block is factored by the
# unblocked `_potf2` base, then the gated trsm (panel solve) + syrk (trailing rank-NB update) carry the
# bulk. Generic over T<:Real (the unblocked base + the generic trsm/syrk path make it ForwardDiff-
# traceable); BlasReal hits the SIMD trsm/syrk. ponytail: NB hand-set for Zen4; lift to a knob if tuning.

const _POTRF_BASE = 512    # recurse above this; below, the unblocked base (potf2, vectorized inner loop).
# Measured sweet spot on Zen4: smaller bases pay more small-k trsm/syrk overhead, larger pay a
# memory-bound unblocked panel. ponytail: hand-set; revisit when tuning Cholesky to the gate.

# Unblocked right-looking Cholesky of an n×n block, lower triangle. Throws PosDefException at the first
# non-positive pivot (LAPACK's info>0). Reads/writes only the lower triangle.
function _potf2_lower!(A, n::Int)
    @inbounds for j in 1:n
        d = A[j, j]
        d > 0 || throw(PosDefException(j))
        ajj = sqrt(d); A[j, j] = ajj; invd = inv(ajj)
        for i in (j + 1):n; A[i, j] *= invd; end                 # scale column j below the diagonal
        for k in (j + 1):n                                        # rank-1 update of the lower trailing
            akj = A[k, j]
            for i in k:n; A[i, k] -= A[i, j] * akj; end
        end
    end
    return A
end
# Unblocked, upper triangle: A = Uᵀ·U.
function _potf2_upper!(A, n::Int)
    @inbounds for j in 1:n
        d = A[j, j]
        d > 0 || throw(PosDefException(j))
        ajj = sqrt(d); A[j, j] = ajj; invd = inv(ajj)
        for i in (j + 1):n; A[j, i] *= invd; end                 # scale row j right of the diagonal
        for k in (j + 1):n                                        # rank-1 update of the upper trailing
            ajk = A[j, k]
            for i in (j + 1):k; A[i, k] -= A[j, i] * ajk; end
        end
    end
    return A
end

# Recursive (cache-oblivious) Cholesky. Lower: split 2×2 — factor A11, solve the off-diagonal panel
# A21·L11⁻ᵀ (trsm), downdate A22 -= A21·A21ᵀ (syrk), recurse A22. The top-level trsm/syrk are large-k
# (half-matrix → the gated packed L3 paths); only the ≤_POTRF_BASE diagonal base is scalar potf2.
function _potrf_lower!(A, n::Int, base::Int = _POTRF_BASE)
    n <= base && return _potf2_lower!(A, n)
    h = n ÷ 2
    _potrf_lower!(view(A, 1:h, 1:h), h, base)
    A21 = view(A, (h + 1):n, 1:h)
    trsm!(A21, view(A, 1:h, 1:h); side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = true)
    syrk!(view(A, (h + 1):n, (h + 1):n), A21; uplo = 'L', trans = 'N', alpha = -1, beta = 1)
    _potrf_lower!(view(A, (h + 1):n, (h + 1):n), n - h, base)
    return A
end
# Upper: A = Uᵀ·U. Off-diagonal panel A12 = U11⁻ᵀ·A12 (trsm side-L), downdate A22 -= A12ᵀ·A12 (syrk).
function _potrf_upper!(A, n::Int)
    n <= _POTRF_BASE && return _potf2_upper!(A, n)
    h = n ÷ 2
    _potrf_upper!(view(A, 1:h, 1:h), h)
    A12 = view(A, 1:h, (h + 1):n)
    trsm!(A12, view(A, 1:h, 1:h); side = 'L', uplo = 'U', transA = 'T', diag = 'N', alpha = true)
    syrk!(view(A, (h + 1):n, (h + 1):n), A12; uplo = 'U', trans = 'T', alpha = -1, beta = 1)
    _potrf_upper!(view(A, (h + 1):n, (h + 1):n), n - h)
    return A
end

# Public: Cholesky factor A in place into its `uplo` triangle. Returns A. Throws PosDefException if A is
# not positive definite. (Native, AD-traceable; the C-ABI `potrf_64_` wrapper lands with the L3 ABI.)
function potrf!(A::AbstractMatrix; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potrf!: A must be square"))
    uplo == 'L' ? _potrf_lower!(A, n) : _potrf_upper!(A, n)
    return A
end
