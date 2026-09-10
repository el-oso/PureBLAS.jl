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
    # SINGLE RHS -> trsv, and hand it the VECTOR rather than `view(Bm, :, 1)`. Both halves matter and
    # both were measured separately on getrs: `trsm!`'s own `nrhs == 1 -> trsv!` path is nested inside
    # its `k <= _trsm_dbase()` tiny-k bypass so it never fires above k~32, and passing a SubArray of
    # `_gt_asmat`'s reshape instead of the plain Vector costs a further 1.6-4.9%.
    #
    # MEASURED FOR THIS SHAPE, on all three boxes (bench/probes/trtrs_nrhs1.jl, gate-exact regime,
    # one size per PROCESS, both arms verified against `LAPACK.trtrs!`), trsv/trsm:
    #   n            100     128     256     1024    2048
    #   wintermute   1.023   1.024   1.047   1.011   1.009 (tie)
    #   galen        1.029   1.023   1.027   1.019     -
    #   neuromancer  1.026   1.045   1.082   1.026     -
    # Wins on every box at every size, tight CIs. Target cells: trtrs@100 0.879 wintermute / 0.883
    # neuromancer, @128 0.954, @256 0.983, @1024 0.975, @2048 0.962 (the @32/@50 cells carry spreads
    # of 0.44-0.58 and are noise, not targets).
    #
    # trtrs's shape is NOT getrs's — one solve, uplo from the caller, versus L/unit then U — so it was
    # measured in its own right rather than inheriting getrs's conclusion. That distinction is why
    # 8d22af1, which put this in `trsm!` and thereby changed trtrs/potrs/sytrs on one routine's
    # evidence, had to be reverted.
    if size(Bm, 2) == 1 && T <: BlasReal && stride(Bm, 1) == 1
        b = B isa AbstractVector ? B : view(Bm, :, 1)
        trsv!(A, b; uplo = uplo, trans = trans, diag = diag)
        return B
    end
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
        # SINGLE RHS: two trsv beat two blocked trsm. `trsm!` has an `nrhs == 1 -> trsv!` fast path
        # already, but it is nested inside the `k <= _trsm_dbase()` tiny-k bypass, so above k~32 a
        # one-column solve pays panel, packing and blocking overhead for a single column. Solvers on
        # given factors are the common caller of that shape — the gate's own `_lufac` passes a vector.
        #
        # Measured on ALL THREE boxes, gate-exact regime (fresh LU factors per sample,
        # `_reps_quadratic` reps in the timed core), full `getrs!` both ways, every arm verified
        # against `LAPACK.getrs!`, one size per PROCESS. trsv/trsm:
        #   wintermute Zen4  n=8 2.176  50 1.063  100 1.040  128 1.041  256 1.051  512 1.025  1024 1.009
        #   galen      Zen3       —     50 1.090  100 1.050        —    256 1.034  512 1.013  1024 1.007
        #   neuromancer Zen5      —     50 1.067  100 1.065        —    256 1.081  512 1.031  1024 1.016
        # Wins on every box at every size up to 1024 and ties at 2048; the gain decays monotonically
        # in n, as an overhead-amortisation story predicts.
        #
        # ⚠ SCOPE IS DELIBERATE. An earlier attempt (8d22af1, reverted in 862e1fd) put this in `trsm!`
        # itself, where it silently reached `trtrs!`, `potrs!`, `sytrs!` and `gesvx` — routines whose
        # shapes (potrs issues L/'N' then L/'T'; trtrs is a single solve) were never measured. It is
        # confined to getrs and to trans='N' here because that is what the evidence covers and what
        # the gate exercises. The 'T'/'C' branch below is deliberately left alone.
        # AFTER the trsv routing the path still measured 1.6-4.9% slower than a hand-rolled solve, and
        # finding the cause took three wrong guesses worth recording so they are not re-tried:
        #   • `_laswp!` — EXONERATED. Specialising it on `j1 == j2` bought nothing (2.66/4.46/0.76%
        #     after, vs 2.1/4.5/1.6 before), and inlining the swap at this call site bought nothing
        #     either (0.997/1.002/1.003 with `_laswp!` restored, i.e. identical). It is not the cost.
        #   • the ENTRY frame (size checks, `_gt_asmat`, the kwarg wrapper) — EXONERATED at 0.1-0.9%,
        #     though `blas2-entry-overhead-blocks-blocked-lapack` made it the prior suspect.
        #   • the `SubArray` — THE CAUSE. See the line below.
        # The decomposition that settled it is `bench/probes/getrs_laswp_nrhs1.jl`: three arms
        # differing in ONE step each. The lesson is that the first probe's hand-rolled arm differed
        # from the shipped path in TWO ways at once, and the whole delta got attributed to the first
        # of them.
        if nrhs == 1 && T <: BlasReal && stride(Bm, 1) == 1
            # PASS THE VECTOR ITSELF, not `view(Bm, :, 1)`. `_gt_asmat` reshapes a vector RHS to n×1,
            # and handing `trsv!` a SubArray of that reshape instead of the plain `Vector` cost the
            # entire residual above: 1.64/4.90/1.03% -> 0.42/0.41/0.13% at n=100/256/512 from this
            # line alone. The branch is on a TYPE, so it resolves at compile time and the vector case
            # carries no runtime test.
            b = B isa AbstractVector ? B : view(Bm, :, 1)
            _laswp!(Bm, ipiv, 1, n, 1, nrhs)                                          # P·B
            trsv!(A, b; uplo = 'L', trans = 'N', diag = 'U')                          # L·Y = P·B
            trsv!(A, b; uplo = 'U', trans = 'N', diag = 'N')                          # U·X = Y
            return B
        end
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
