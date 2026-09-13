# Symmetric-indefinite / Hermitian LAPACK drivers built on the Bunch-Kaufman factors (bunchkaufman.jl):
#
#   sysv! / hesv!  : one-shot solve  A·X = B         = sytrf!/hetrf! then sytrs!/hetrs!  (trivial compose)
#   sytri! / hetri!: matrix inverse  A⁻¹ from D & L/U = dsytri / zsytri (complex-symmetric) / zhetri (Herm)
#
# `sytri!`/`hetri!` are the real new code: the block back-inversion. From A = L·D·Lᵀ (uplo='L') /
# U·D·Uᵀ (uplo='U'), inv(A) = (Lᵀ)⁻¹·D⁻¹·L⁻¹. LAPACK does it in place, column by column, WITHOUT ever
# forming L⁻¹: invert each D block, then walk k from the trailing end inward computing the k-th column
# of the inverse as  A[·,k] ← -A_trailing·(old A[·,k]),  A[k,k] ← D⁻¹[k,k] - old·new.  The symv against
# the already-inverted trailing block + the two dot products carry the (Lᵀ)⁻¹ and L⁻¹ applications.
# Ported ONE-TO-ONE from reference dsytri/zsytri/zhetri: same loop direction, same interchange (with the
# Hermitian conjugated crossing-strip swap), reusing PureBLAS's own `_symv!`/`_hemv!` + `_dotu`/`_dotc`.
#
# `herm` unifies the three: herm=false covers BOTH real-symmetric (dsytri) and complex-symmetric (zsytri)
# — real diagonal falls out for free, and the 2×2 uses the raw pivot value T (signed real ≡ |·| here; the
# complex square must NOT be an abs). herm=true (zhetri) forces the real diagonal (real(A[k,k]), real of
# the diagonal-correcting dot) and the conjugated (Hermitian) dot/symv/swap.
#
# `syconv!` (dsyconv, the LDLᵀ-packed ⇄ separated-D storage converter) is NOT implemented: neither the
# classic dsytri nor the dsytrs/sysv solve path needs it — they read D straight out of A's stored
# triangle + ipiv. syconv only feeds the Aasen (dsytrf_aa) / rook variants, which PureBLAS does not use.
#
# ponytail: scalar-generic driver reusing the gated L2 symv/hemv; correctness-first, matching
# bunchkaufman.jl. The symv/dot per column are the hot spots (already the gated kernels).

# ── inverse from the factors ──────────────────────────────────────────────────────────────────────

# uplo='L': invert in place from A = L·D·Lᵀ (herm=false) or L·D·Lᴴ (herm=true). `work` is an n-length
# scratch (the LAPACK WORK array). Only the lower triangle is read/written; result is the lower triangle
# of the symmetric/Hermitian inverse.
# THE CROSSING-STRIP SWAP, and why it is not the obvious one-element loop.
#
# The strip exchanges a COLUMN segment `A[jlo:jhi, kc]` (stride 1, contiguous) with a ROW segment
# `A[kr, jlo:jhi]` (stride `lda`). At a power-of-two `lda` the row side is pathological: on Zen4,
# lda=512 doubles is 4096 B, so EVERY element of the row sits on its own page and in the same L1 set.
# Measured in isolation (`bench/probes/sytri_strip_ab.jl`), one full sweep of strips at n=512:
#
#     lda = n      (po2)   1370.7 us        lda = n + 8   56.3 us      -- 24x, from eight elements
#
# That is the whole of `sytri`'s non-symv time: the decomposition
# (`bench/probes/sytri_decompose.jl`) puts symv at 0.99-1.07 of OpenBLAS's while PB's glue is 1295 us
# of a 5362 us total at n=512, and `sytri@512` publishes 0.831.
#
# BATCHING THE SWAPS IS INVALID — checked, do not try it. The tempting fix is to collect every
# interchange and apply one `A <- P A P'` at the end, which would be blocked and tiled and capture the
# full 24x. It cannot be done: each iteration computes its column from `Asub = A[k+1:n, k+1:n]`, and the
# swap writes `A[kp,j]` for j in k+1:kp-1, `A[i,kp]` for i>kp and `A[kp,kp]` — every one of those has
# BOTH indices >= k+1, i.e. inside the trailing block the NEXT iteration reads. The swaps are part of
# the recurrence, not a permutation of a finished result.
#
# Nor can the strip be made contiguous: it exchanges a column segment with a row segment, both in the
# STORED lower (or upper) triangle, so one side is always strided — and `A[j,kp]`, the symmetric
# alternative to `A[kp,j]`, is in the untouched triangle and not stored.
#
# So what is left is the FORMULATION, measured four ways at po2 lda (same probe):
#     base (one-element read-modify-write)   1370.7 us
#     gather / swap / scatter via `work`     1061.9 us   <- 22% better
#     + software prefetch, D=8 / D=32        1487 / 1518 us  -- WORSE; the hw prefetcher already has the
#                                            stride and a hint cannot fix a TLB miss
#     tiled over j (TW=64)                   1381.9 us   -- inert, as it must be: reordering cannot
#                                            reduce the page count when each element owns a page
#
# Three clean passes beat one interleaved read-modify-write because the RMW writes back to a line that
# conflicts with the next read. But it INVERTS off the pathology — 111.9 us vs 56.3 at lda = n+8, 2x
# WORSE — so it is gated on `_alias_ld` (level3.jl), the same full-L1-way-period predicate the trsm and
# hessenberg po2 paths use. `work` is the sytri arena borrow, dead at swap time (its symv and dots are
# done for this k), so staging costs no extra memory.
@inline function _strip_swap!(A, work, jlo::Int, jhi::Int, kc::Int, kr::Int, herm::Bool)
    ns = jhi - jlo + 1
    ns <= 0 && return nothing
    @inbounds if _alias_ld(stride(A, 2))
        for t in 1:ns                                     # gather the strided row side
            work[t] = A[kr, jlo + t - 1]
        end
        for t in 1:ns                                     # swap against the contiguous column side
            tmp = A[jlo + t - 1, kc]
            A[jlo + t - 1, kc] = herm ? conj(work[t]) : work[t]
            work[t] = herm ? conj(tmp) : tmp
        end
        for t in 1:ns                                     # scatter back
            A[kr, jlo + t - 1] = work[t]
        end
    else
        for j in jlo:jhi
            if herm
                tmp = conj(A[j, kc]); A[j, kc] = conj(A[kr, j]); A[kr, j] = tmp
            else
                tmp = A[j, kc]; A[j, kc] = A[kr, j]; A[kr, j] = tmp
            end
        end
    end
    return nothing
end

function _sytri_lower!(
        A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, herm::Bool,
        work::AbstractVector{T}
    ) where {T}
    n = size(A, 1)
    Tr = real(T)
    @inbounds begin
        k = n
        while k >= 1
            if ipiv[k] > 0                                   # 1×1 diagonal block
                A[k, k] = herm ? (one(Tr) / real(A[k, k])) : (one(T) / A[k, k])
                if k < n                                     # compute column k of the inverse
                    m = n - k
                    wv = view(work, 1:m); yv = view(A, (k + 1):n, k)
                    copyto!(wv, yv)
                    Asub = view(A, (k + 1):n, (k + 1):n)
                    herm ? _hemv!(false, m, -one(T), Asub, wv, 1, zero(T), yv, 1) :
                        _symv!(false, m, -one(T), Asub, wv, 1, zero(T), yv, 1)
                    d = herm ? _dotc(m, wv, 1, yv, 1) : _dotu(m, wv, 1, yv, 1)
                    A[k, k] -= herm ? real(d) : d
                end
                kstep = 1
            else                                             # 2×2 diagonal block (columns k-1, k)
                t = herm ? Tr(abs(A[k, k - 1])) : A[k, k - 1]
                ak = herm ? (real(A[k - 1, k - 1]) / t) : (A[k - 1, k - 1] / t)
                akp1 = herm ? (real(A[k, k]) / t) : (A[k, k] / t)
                akkp1 = A[k, k - 1] / t
                d = t * (ak * akp1 - one(Tr))
                A[k - 1, k - 1] = akp1 / d
                A[k, k] = ak / d
                A[k, k - 1] = -akkp1 / d
                if k < n                                     # columns k-1 and k of the inverse
                    m = n - k
                    wv = view(work, 1:m)
                    yk = view(A, (k + 1):n, k); ykm1 = view(A, (k + 1):n, k - 1)
                    Asub = view(A, (k + 1):n, (k + 1):n)
                    copyto!(wv, yk)
                    herm ? _hemv!(false, m, -one(T), Asub, wv, 1, zero(T), yk, 1) :
                        _symv!(false, m, -one(T), Asub, wv, 1, zero(T), yk, 1)
                    dk = herm ? _dotc(m, wv, 1, yk, 1) : _dotu(m, wv, 1, yk, 1)
                    A[k, k] -= herm ? real(dk) : dk
                    A[k, k - 1] -= herm ? _dotc(m, yk, 1, ykm1, 1) : _dotu(m, yk, 1, ykm1, 1)
                    copyto!(wv, ykm1)
                    herm ? _hemv!(false, m, -one(T), Asub, wv, 1, zero(T), ykm1, 1) :
                        _symv!(false, m, -one(T), Asub, wv, 1, zero(T), ykm1, 1)
                    dm = herm ? _dotc(m, wv, 1, ykm1, 1) : _dotu(m, wv, 1, ykm1, 1)
                    A[k - 1, k - 1] -= herm ? real(dm) : dm
                end
                kstep = 2
            end
            # interchange rows/cols k ↔ kp in the trailing block
            kp = abs(ipiv[k])
            if kp != k
                if kp < n
                    for i in (kp + 1):n
                        tmp = A[i, k]; A[i, k] = A[i, kp]; A[i, kp] = tmp
                    end
                end
                _strip_swap!(A, work, k + 1, kp - 1, k, kp, herm)   # A[k+1:kp-1,k] ↔ A[kp,k+1:kp-1]
                herm && (A[kp, k] = conj(A[kp, k]))
                tmp = A[k, k]; A[k, k] = A[kp, kp]; A[kp, kp] = tmp
                if kstep == 2
                    tmp = A[k, k - 1]; A[k, k - 1] = A[kp, k - 1]; A[kp, k - 1] = tmp
                end
            end
            k -= kstep
        end
    end
    return A
end

# uplo='U': invert in place from A = U·D·Uᵀ / U·D·Uᴴ. Mirror of the lower path (loop k=1..n ascending,
# trailing block is the LEADING submatrix A[1:k-1,1:k-1], 2×2 block is columns k, k+1).
function _sytri_upper!(
        A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, herm::Bool,
        work::AbstractVector{T}
    ) where {T}
    n = size(A, 1)
    Tr = real(T)
    @inbounds begin
        k = 1
        while k <= n
            if ipiv[k] > 0                                   # 1×1 diagonal block
                A[k, k] = herm ? (one(Tr) / real(A[k, k])) : (one(T) / A[k, k])
                if k > 1                                     # compute column k of the inverse
                    m = k - 1
                    wv = view(work, 1:m); yv = view(A, 1:m, k)
                    copyto!(wv, yv)
                    Asub = view(A, 1:m, 1:m)
                    herm ? _hemv!(true, m, -one(T), Asub, wv, 1, zero(T), yv, 1) :
                        _symv!(true, m, -one(T), Asub, wv, 1, zero(T), yv, 1)
                    d = herm ? _dotc(m, wv, 1, yv, 1) : _dotu(m, wv, 1, yv, 1)
                    A[k, k] -= herm ? real(d) : d
                end
                kstep = 1
            else                                             # 2×2 diagonal block (columns k, k+1)
                t = herm ? Tr(abs(A[k, k + 1])) : A[k, k + 1]
                ak = herm ? (real(A[k, k]) / t) : (A[k, k] / t)
                akp1 = herm ? (real(A[k + 1, k + 1]) / t) : (A[k + 1, k + 1] / t)
                akkp1 = A[k, k + 1] / t
                d = t * (ak * akp1 - one(Tr))
                A[k, k] = akp1 / d
                A[k + 1, k + 1] = ak / d
                A[k, k + 1] = -akkp1 / d
                if k > 1                                     # columns k and k+1 of the inverse
                    m = k - 1
                    wv = view(work, 1:m)
                    yk = view(A, 1:m, k); ykp1 = view(A, 1:m, k + 1)
                    Asub = view(A, 1:m, 1:m)
                    copyto!(wv, yk)
                    herm ? _hemv!(true, m, -one(T), Asub, wv, 1, zero(T), yk, 1) :
                        _symv!(true, m, -one(T), Asub, wv, 1, zero(T), yk, 1)
                    dk = herm ? _dotc(m, wv, 1, yk, 1) : _dotu(m, wv, 1, yk, 1)
                    A[k, k] -= herm ? real(dk) : dk
                    A[k, k + 1] -= herm ? _dotc(m, yk, 1, ykp1, 1) : _dotu(m, yk, 1, ykp1, 1)
                    copyto!(wv, ykp1)
                    herm ? _hemv!(true, m, -one(T), Asub, wv, 1, zero(T), ykp1, 1) :
                        _symv!(true, m, -one(T), Asub, wv, 1, zero(T), ykp1, 1)
                    dm = herm ? _dotc(m, wv, 1, ykp1, 1) : _dotu(m, wv, 1, ykp1, 1)
                    A[k + 1, k + 1] -= herm ? real(dm) : dm
                end
                kstep = 2
            end
            # interchange rows/cols k ↔ kp in the leading block
            kp = abs(ipiv[k])
            if kp != k
                for i in 1:(kp - 1)
                    tmp = A[i, k]; A[i, k] = A[i, kp]; A[i, kp] = tmp
                end
                _strip_swap!(A, work, kp + 1, k - 1, k, kp, herm)   # A[kp+1:k-1,k] ↔ A[kp,kp+1:k-1]
                herm && (A[kp, k] = conj(A[kp, k]))
                tmp = A[k, k]; A[k, k] = A[kp, kp]; A[kp, kp] = tmp
                if kstep == 2
                    tmp = A[k, k + 1]; A[k, k + 1] = A[kp, k + 1]; A[kp, k + 1] = tmp
                end
            end
            k += kstep
        end
    end
    return A
end

"""
    sytri!(A, ipiv; uplo='L') -> A

Overwrite A with the inverse of the symmetric (or complex-symmetric) matrix whose Bunch-Kaufman factors
(`sytrf!` output) are stored in the `uplo` triangle of A with pivots `ipiv`. Only the `uplo` triangle of
A⁻¹ is written (the inverse is symmetric). Generic over T<:Number. Mirrors LAPACK `dsytri`/`zsytri`.
"""
function sytri!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("sytri!: A must be square"))
    length(ipiv) == n || throw(DimensionMismatch("sytri!: length(ipiv) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("sytri!: uplo must be 'L' or 'U'"))
    # One arena borrow of length n, threaded down as an explicit argument (both engines take `work`),
    # so the only handle that leaves this scope is `A` — the matrix the caller owns.
    @scope arn begin
        work = borrow!(arn, eltype(A), n)
        return uplo == 'L' ? _sytri_lower!(A, ipiv, false, work) : _sytri_upper!(A, ipiv, false, work)
    end
end

"""
    hetri!(A, ipiv; uplo='L') -> A

Overwrite A with the inverse of the Hermitian matrix whose Bunch-Kaufman factors (`hetrf!` output) are in
the `uplo` triangle with pivots `ipiv`. For real `eltype(A)` this is identical to `sytri!`. Only the
`uplo` triangle of A⁻¹ is written. Mirrors LAPACK `zhetri` (real diagonal, conjugated off-diagonals).
"""
function hetri!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("hetri!: A must be square"))
    length(ipiv) == n || throw(DimensionMismatch("hetri!: length(ipiv) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("hetri!: uplo must be 'L' or 'U'"))
    herm = eltype(A) <: Complex
    @scope arn begin                       # same single borrow as sytri! — one field, one role, two entries
        work = borrow!(arn, eltype(A), n)
        return uplo == 'L' ? _sytri_lower!(A, ipiv, herm, work) : _sytri_upper!(A, ipiv, herm, work)
    end
end

# ── one-shot solve  A·X = B  (factor + solve) ───────────────────────────────────────────────────────

# `ipiv` is an OUTPUT (element 2 of the returned tuple — the caller needs it to drive a later
# `sytrs!`/`sytri!`), so it can NOT come from the owned L3Workspace: the next call would silently
# overwrite a view the caller still holds. The 4-argument form below takes it from the caller instead,
# mirroring `getrf!(A, ipiv)` / `gbtrf!(kl, ku, m, AB, ipiv)`; the 3-argument form is the allocating
# convenience and is kept byte-for-byte in behaviour (same tuple, same element types).
#
# Aliasing (lesson 9): `ipiv` is `<:Integer` while A/B carry the numeric element type, and `sytrf!`
# needs division, so an `ipiv` that aliases A or B is not expressible for any T this factorization
# runs on — no guard is added. A and B must stay distinct, which was already true of the 3-arg form.

"""
    sysv!(uplo, A, B) -> (A, ipiv, B)
    sysv!(uplo, A, B, ipiv) -> (A, ipiv, B)

Solve the symmetric-indefinite system A·X = B in place: Bunch-Kaufman–factor A (`sytrf!`, in the `uplo`
triangle) then solve (`sytrs!`). On return A holds the factors, `ipiv` the pivots, and B the solution X.
The 4-argument form writes the pivots into the caller's `ipiv` (length `size(A,1)`, checked by `sytrf!`)
and allocates nothing.
"""
function sysv!(uplo::Char, A::AbstractMatrix, B::AbstractVecOrMat, ipiv::AbstractVector{<:Integer})
    sytrf!(A, ipiv; uplo = uplo)
    sytrs!(A, ipiv, B; uplo = uplo)
    return A, ipiv, B
end
sysv!(uplo::Char, A::AbstractMatrix, B::AbstractVecOrMat) =
    sysv!(uplo, A, B, Vector{Int}(undef, size(A, 1)))

"""
    hesv!(uplo, A, B) -> (A, ipiv, B)
    hesv!(uplo, A, B, ipiv) -> (A, ipiv, B)

Solve the Hermitian system A·X = B in place via the Hermitian Bunch-Kaufman factorization (`hetrf!` then
`hetrs!`). For real `eltype(A)` this equals `sysv!`. The 4-argument form writes the pivots into the
caller's `ipiv` and allocates nothing.
"""
function hesv!(uplo::Char, A::AbstractMatrix, B::AbstractVecOrMat, ipiv::AbstractVector{<:Integer})
    hetrf!(A, ipiv; uplo = uplo)
    hetrs!(A, ipiv, B; uplo = uplo)
    return A, ipiv, B
end
hesv!(uplo::Char, A::AbstractMatrix, B::AbstractVecOrMat) =
    hesv!(uplo, A, B, Vector{Int}(undef, size(A, 1)))
