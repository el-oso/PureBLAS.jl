# LAPACK general-banded LU with partial pivoting — faithful port of Reference-LAPACK
# dgbtf2 (unblocked banded LU) + dgbtrs (banded LU solve), generic over s/d/c/z (and any
# T<:Number, so Mode 2 / ForwardDiff-traceable). `_gbtf2!` depends only on Base, which is what
# keeps the generic-T / ForwardDiff path alive: a `Dual` fails the `T<:BlasFloat` gate in
# `gbtrf!` and lands here unchanged.
#
# ── LAPACK band storage (GB) ────────────────────────────────────────────────────────────────
# An n×n matrix with `kl` subdiagonals and `ku` superdiagonals is held in an `ldab × n` array
# AB with ldab ≥ 2·kl+ku+1. Let KV = ku + kl. The band element A(i,j) lives at
#     AB[KV+1 + i-j, j]     for  max(1, j-ku) ≤ i ≤ min(n, j+kl).
# The diagonal A(j,j) sits at row KV+1. Rows 1..kl are RESERVED workspace for the fill-in that
# partial pivoting creates (an LU factor of a (kl,ku) band has up to kl+ku superdiagonals), so
# the caller supplies data in rows kl+1..2kl+ku+1. On exit U occupies rows 1..KV+1 (KV super-
# diagonals) and the unit-L multipliers occupy rows KV+2..2kl+ku+1.
#
# ── Pivoting ────────────────────────────────────────────────────────────────────────────────
# For each column j the pivot is the row of largest |·| among the diagonal and its ≤kl sub-
# diagonal entries (LAPACK metric: |·| for real, cabs1=|Re|+|Im| for complex — matches idamax/
# izamax). ipiv[j] = global 1-based pivot row. `ju` tracks the last column any pivot has reached
# (bounds the trailing rank-1 update to the live band). Pivoting is a correctness boundary.

@inline _gb_cabs1(x::Real) = abs(x)
@inline _gb_cabs1(z::Complex) = abs(real(z)) + abs(imag(z))

# PDM tier: DERIVE, as a formula over the BAND WIDTH kl — a problem parameter, not a machine one, so
# it const-folds, is trim-safe, and needs nothing pinned in juliac/build.jl.
#
# This replaced a Measure-tier OncePerProcess that selected a single constant, and measurement forced
# the replacement: that ladder probed at kl=192 and picked nb=16, but sweeping kl end to end shows the
# optimum MOVES with kl, so no constant can be right. Zen4 F64, freq-locked, PB/OB, winner starred
# (kl = ku = n/8, the bench shape):
#   kl    16    20    24    32    40    48    64    80    96   128   192   256   384   512
#   nb=8 0.97* 1.00* 1.20* 1.42* 1.73* 1.51* 1.93* 1.72* 1.68* 1.47  1.34  1.16  1.08  0.90
#   nb=16                  1.25              1.81             1.53* 1.42* 1.31  1.21  1.15
#   nb=24                  1.10              1.63             1.45  1.37  1.31* 1.22* 1.18*
# 8 for kl <= 96, 16 for kl in 128..192, 24 from 256 — one step per 128 of band width, which
# `8 * (1 + kl ÷ 128)` reproduces at EVERY measured point. The response is flat within 1-3% either side
# of the winner, so the step positions are not delicate.
#
# The blocking gate `kl >= 2*nb` then admits kl >= 16 (nb=8), which is right: at kl=8 and 12 the
# UNBLOCKED kernel wins outright (1.31 and 1.13, against 0.86/0.95 blocked), so the gate declines
# exactly where it should. i2 = kl-nb >= nb still holds by construction.
#
# RESIDUAL, diagnosed rather than left open: kl=16 reaches only 0.97. The unblocked kernel is 0.796
# there and no nb rescues it past 0.97 — and the reason is that OpenBLAS is ALSO unblocked at that
# width (its ILAENV nb=64 exceeds kl, so dgbtrf takes the dgbtf2 branch). We are losing to OpenBLAS's
# *unblocked* band downdate, not to its blocking, so the remaining lever is the quality of `_gbtf2!`'s
# rank-1 band update against dger — NOT the panel width. Worth stating because the obvious next move,
# routing that loop through `_ger!`, is the shape that has failed three times already on BLAS-2 entry
# overhead; it needs measuring, not assuming.
const _GBTRF_NB_PREF = @load_preference("gbtrf_nb", nothing)
@static if isnothing(_GBTRF_NB_PREF)
    @inline _gbtrf_nb(::Type{T}, kl::Int) where {T} = clamp(8 * (1 + kl ÷ 128), 8, 48)
else
    @inline _gbtrf_nb(::Type{T}, kl::Int) where {T} = _GBTRF_NB_PREF::Int
end

# Minimum band width at which BLOCKING beats the unblocked kernel. PDM tier: MEASURE, and the
# falsification that forced it is worth recording because it was mine.
#
# The nb formula above was validated on Zen4 only, and shipping it there REGRESSED Zen3: at kl=16 the
# blocked path (nb=8) measures 0.93 on Zen4 where unblocked is 0.796 — blocking wins — but on Zen3 the
# same cell is 0.91 blocked against ~1.18 unblocked, so blocking LOSES. bench/plots.jl's gbtrf row went
# from PASS (1.43 / 1.18) to FAIL (1.32 / 0.91) on Zen3 as a direct result. The crossover is therefore
# microarchitecture-dependent, which is the definition of a Measure-tier knob, and this is exactly the
# "derive -> validate on the FLEET -> ship" step of req#8(b) catching a one-box derivation.
#
# Mechanism (consistent with both boxes): at kl=16 OpenBLAS is ALSO unblocked (its ILAENV nb=64 exceeds
# kl, so dgbtrf takes the dgbtf2 branch), so this cell is a scalar-band-downdate race, not a blocking
# race — and which side wins depends on how well each box runs that scalar loop versus a rank-8 gemm on
# a very thin trailing block. Not something a residency model predicts.
#
# Candidate set is derived: multiples of the nb quantum (8). The probe times blocked against unblocked
# at each candidate width and returns the smallest at which blocking actually wins, so the ladder
# adapts to a box we have never measured.
const _GBTRF_CROSS_PREF = @load_preference("gbtrf_cross", nothing)
@static if isnothing(_GBTRF_CROSS_PREF)
    # Width-derived; see `_at_gbtrf_cross` (cpuinfo.jl) for the three-box table. Each box keeps its
    # OWN measured crossover, which is what avoids the failure this file documents above — shipping
    # a single one-box value took Zen3's gbtrf row from PASS (1.43/1.18) to FAIL (1.32/0.91).
    @inline _gbtrf_cross(::Type{Float64}) = _at_gbtrf_cross(_HW, Float64)
    @inline _gbtrf_cross(::Type{Float32}) = _at_gbtrf_cross(_HW, Float32)
    @inline _gbtrf_cross(::Type{ComplexF64}) = _at_gbtrf_cross(_HW, ComplexF64)
    @inline _gbtrf_cross(::Type{ComplexF32}) = _at_gbtrf_cross(_HW, ComplexF32)
    @inline _gbtrf_cross(::Type{T}) where {T} = 32
else
    @inline _gbtrf_cross(::Type{T}) where {T} = _GBTRF_CROSS_PREF::Int   # pinned (trim lands here)
end

# gbtrf!(kl, ku, m, AB) → (AB, ipiv, info).  AB overwritten with L\U in band storage; ipiv the
# min(m,n) pivot rows; info = index of the first exactly-zero pivot (0 if none). Allocates ipiv and
# dispatches; the arity and the 3-tuple are load-bearing (the C-ABI wrapper destructures them).
function gbtrf!(kl::Integer, ku::Integer, m::Integer, AB::AbstractMatrix{T}) where {T}
    _, n = size(AB)
    ipiv = Vector{Int}(undef, min(Int(m), n))
    nb = _gbtrf_nb(T, Int(kl))
    # Blocking only pays when the in-band trailing block A22 is at least as tall as the panel is
    # wide (i2 = kl-nb ≥ nb). Reference dgbtrf's structural condition is the weaker nb ≤ kl, but at
    # nb ≈ kl every trailing flop funnels through the dense corners instead of the band — the same
    # collapse `_pbtrf_nb` documents (banded_chol.jl:173-186), where clamping the width to kd turned
    # a 1.6× win into the routine's only gate miss. Narrow bands stay on the unblocked kernel, where
    # PureBLAS already gates.
    if T <: BlasFloat && _strided1(AB) && Int(kl) >= max(2 * nb, _gbtrf_cross(T)) &&
min(Int(m), n) > nb
        info = _gbtrf_blocked!(kl, ku, m, AB, ipiv, nb)
    else
        info = _gbtf2!(kl, ku, m, AB, ipiv)
    end
    return AB, ipiv, info
end

# _gbtf2!(kl, ku, m, AB, ipiv) → info.  Reference dgbtf2: UNBLOCKED banded LU, one rank-1 downdate
# per column, no BLAS calls. Base-only, so it serves every non-BlasFloat eltype as well as being the
# narrow-band path (blocking cannot pay when the panel does not fit inside kl).
function _gbtf2!(
        kl::Integer, ku::Integer, m::Integer,
        AB::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}
    ) where {T}
    ldab, n = size(AB)
    kl >= 0 || throw(ArgumentError("gbtrf!: kl < 0"))
    ku >= 0 || throw(ArgumentError("gbtrf!: ku < 0"))
    ldab >= 2 * kl + ku + 1 || throw(DimensionMismatch("gbtrf!: ldab must be ≥ 2kl+ku+1"))
    kv = ku + kl
    mn = min(Int(m), n)
    info = 0
    z = zero(T)
    # Zero the fill-in triangle in the leading columns (ku+2 .. min(kv,n)).
    @inbounds for j in (ku + 2):min(kv, n)
        for i in (kv - j + 2):kl
            AB[i, j] = z
        end
    end
    ju = 1
    @inbounds for j in 1:mn
        if j + kv <= n                                   # zero the fill-in of the incoming column
            for i in 1:kl
                AB[i, j + kv] = z
            end
        end
        km = min(kl, Int(m) - j)                          # # subdiagonal entries in this column
        jp = 1; pmax = _gb_cabs1(AB[kv + 1, j])           # partial-pivot argmax (diag + subdiagonals)
        for i in 2:(km + 1)
            a = _gb_cabs1(AB[kv + i, j]); a > pmax && (pmax = a; jp = i)
        end
        ipiv[j] = jp + j - 1
        if AB[kv + jp, j] != z
            ju = max(ju, min(j + ku + jp - 1, n))         # last column the pivot reaches
            if jp != 1                                    # swap A-row j ↔ A-row (jp+j-1) over cols j..ju
                for jj in j:ju
                    r1 = kv + 1 + j - jj                   # A(j, jj)
                    r2 = kv + jp + j - jj                  # A(jp+j-1, jj)
                    AB[r1, jj], AB[r2, jj] = AB[r2, jj], AB[r1, jj]
                end
            end
            if km > 0
                d = one(T) / AB[kv + 1, j]                 # multipliers L(j+i,j) = A(j+i,j)/pivot
                for i in 1:km
                    AB[kv + 1 + i, j] *= d
                end
                for jj in (j + 1):ju                       # rank-1 trailing update within the band
                    ujj = AB[kv + 1 + j - jj, jj]          # U(j,jj)
                    if ujj != z
                        # `ivdep`: the loop STORES into column jj and LOADS from column j, and
                        # jj > j always (jj runs j+1:ju), so they are distinct columns of AB and
                        # cannot alias — but alias analysis cannot see that and otherwise has to
                        # assume a store→load dependency. Measured 1.03–1.19× on the isolated
                        # downdate, which is 87–97% of the whole routine.
                        @simd ivdep for i in 1:km
                            AB[kv + 1 + i + j - jj, jj] -= AB[kv + 1 + i, j] * ujj  # A(j+i,jj) -= L(j+i,j)·U(j,jj)
                        end
                    end
                end
            end
        elseif info == 0
            info = j
        end
    end
    return info
end

# ── blocked kernel (dgbtrf.f DO 180) ──────────────────────────────────────────────────────────────
# Partition at outer column j (reference names; jb = min(nb, min(m,n)-j+1)):
#
#     A11(jb×jb)  A12(jb×j2)  A13(jb×j3)      i2 = min(kl-jb, m-j-jb+1)
#     A21(i2×jb)  A22(i2×j2)  A23(i2×j3)      i3 = min(jb, m-j-kl+1)
#     A31(i3×jb)  A32(i3×j2)  A33(i3×j3)      j2, j3 computed AFTER ju is updated
#
# A13's superdiagonal part and A31's subdiagonal part lie OUTSIDE the stored band: A31[ii,jj] sits at
# storage row kv+kl+1+ii-jj, which is ≤ ldab only for ii ≤ jj, and A13[ii,jj] at row ii-jj+1, ≥ 1
# only for ii ≥ jj. Both fill in completely during the panel — A31 through the pivot swaps, A13
# through the trsm — so each is staged in a dense work array, exactly as the reference does.
# Unlike pbtrf, where the corner only appears past kd, here i3 = jb for essentially the whole sweep
# whenever kl is wide, so the corner path is the COMMON case, not an edge case.
#
# Every band block is read through the ld-1 trick (`_pb_blk`, banded_chol.jl:376): a block whose
# row-col offset is constant along each block diagonal is a plain column-major matrix with leading
# dimension ldab-1. That is the `AB(r,c), LDAB-1` idiom of every BLAS call in dgbtrf.f.
#
# TWO deviations from the reference, both deliberate:
#  (1) L11 is staged into a dense jb×jb scratch instead of being passed as an ld-1 view. The
#      strictly-upper triangle of a view anchored at AB(kv+1,j) IS U11 — live matrix data — and
#      handing that to a triangular kernel is the aliasing class banded_chol.jl:420-428 records
#      costing a 4e-4 factor error. trsm contractually ignores it, but jb²/2 copies per panel is
#      O(n·nb/2) total (noise) and buys certainty.
#  (2) `gemm!` is NOT used: it gates on `C isa StridedMatrix`, which `PtrMatrix` is not, so it would
#      silently fall through to the generic kernel. `_gemm_core!` is the entry that takes PtrMatrix.
#      `trsm!` needs no such care — it gates on `_strided1`, which PtrMatrix satisfies.
#
# The undo loop at the end is easy to mistake for redundant and is not: dgbtf2 stores L UNPERMUTED
# (its swap runs only over columns to the right, and `gbtrs!` compensates by interleaving a row swap
# with each column of multipliers). The blocked panel swaps the full panel width so the trsm/gemm see
# a consistent block, so the left columns must be un-swapped to restore that storage convention.
# Skip it and `gbtrs!` silently solves the wrong system while the factor still looks plausible.
function _gbtrf_blocked!(
        kl::Integer, ku::Integer, m::Integer,
        AB::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, nb::Int
    ) where {T}
    ldab, n = size(AB)
    kl >= 0 || throw(ArgumentError("gbtrf!: kl < 0"))
    ku >= 0 || throw(ArgumentError("gbtrf!: ku < 0"))
    ldab >= 2 * kl + ku + 1 || throw(DimensionMismatch("gbtrf!: ldab must be ≥ 2kl+ku+1"))
    M = Int(m); KL = Int(kl); KU = Int(ku)
    # PRECONDITION, structural and not a tuning choice: the panel must fit inside the subdiagonal
    # band. Above it i2 = kl-jb goes negative and the A11/A21/A31 partition stops describing the
    # storage, so the factor is silently wrong rather than merely slow (verified: every nb > kl cell
    # diverges from _gbtf2!, every nb ≤ kl cell matches bit-exactly). `gbtrf!` gates on the stricter
    # kl ≥ 2·nb for performance reasons; this guard protects direct callers, including the tests.
    nb <= KL || _throw_gbtrf_nb(nb, KL)
    kv = KU + KL
    mn = min(M, n)
    info = 0
    z = zero(T)
    ldabs = stride(AB, 2)                              # NOT 2kl+ku+1: callers may pad
    ld2 = ldabs - 1
    ldw = nb + 1
    W13, W31, S = _gbtrf_work(T, nb)                   # GKH-owned; W13/W31 arrive zeroed
    p0 = pointer(AB); p13 = pointer(W13); p31 = pointer(W31); ps = pointer(S)

    # Fill-in zeroing of the leading columns (dgbtrf DO 60), identical to the unblocked kernel.
    @inbounds for j in (KU + 2):min(kv, n)
        for i in (kv - j + 2):KL
            AB[i, j] = z
        end
    end

    ju = 1                                             # ONE variable across ALL panels (see DO 180)
    GC.@preserve AB W13 W31 S begin
        @inbounds for j in 1:nb:mn
            jb = min(nb, mn - j + 1)
            i2 = min(KL - jb, M - j - jb + 1)
            i3 = min(jb, M - j - KL + 1)

            # ── panel: unblocked factorization of the jb columns (DO 80) ──────────────────────────
            for jj in j:(j + jb - 1)
                if jj + kv <= n                        # zero the incoming column's fill-in
                    for i in 1:KL
                        AB[i, jj + kv] = z
                    end
                end
                km = min(KL, M - jj)
                jp = 1; pmax = _gb_cabs1(AB[kv + 1, jj])
                for i in 2:(km + 1)
                    a = _gb_cabs1(AB[kv + i, jj]); a > pmax && (pmax = a; jp = i)
                end
                ipiv[jj] = jp + jj - j                 # PANEL-LOCAL; +j-1 is applied below
                if AB[kv + jp, jj] != z
                    ju = max(ju, min(jj + KU + jp - 1, n))
                    if jp != 1
                        if jp + jj - 1 < j + KL
                            # swap over the whole panel width; stride ldab-1 walks one column right
                            # and one storage row up, i.e. along a row of A.
                            for t in 0:(jb - 1)
                                r1 = kv + 1 + jj - j - t; r2 = kv + jp + jj - j - t
                                c = j + t
                                AB[r1, c], AB[r2, c] = AB[r2, c], AB[r1, c]
                            end
                        else
                            # the pivot row reaches A31, whose columns j..jj-1 live in W31
                            for t in 0:(jj - j - 1)
                                r1 = kv + 1 + jj - j - t
                                w = jp + jj - j - KL
                                AB[r1, j + t], W31[w, 1 + t] = W31[w, 1 + t], AB[r1, j + t]
                            end
                            for t in 0:(j + jb - jj - 1)
                                r1 = kv + 1 - t; r2 = kv + jp - t
                                c = jj + t
                                AB[r1, c], AB[r2, c] = AB[r2, c], AB[r1, c]
                            end
                        end
                    end
                    if km > 0
                        d = one(T) / AB[kv + 1, jj]
                        for i in 1:km
                            AB[kv + 1 + i, jj] *= d
                        end
                        # rank-1 update confined to the PANEL (jm), not to ju: everything right of
                        # the panel is deferred to the trsm/gemm below.
                        jm = min(ju, j + jb - 1)
                        for c in 1:(jm - jj)
                            ujj = AB[kv + 1 - c, jj + c]
                            if ujj != z
                                @simd ivdep for i in 1:km
                                    AB[kv + 1 + i - c, jj + c] -= AB[kv + 1 + i, jj] * ujj
                                end
                            end
                        end
                    end
                elseif info == 0
                    info = jj
                end
                nw = min(jj - j + 1, i3)               # stash this column of A31 into W31
                for t in 1:nw
                    W31[t, jj - j + 1] = AB[kv + KL + t - jj + j, jj]
                end
            end

            if j + jb <= n
                j2 = min(ju - j + 1, kv) - jb
                j3 = max(0, ju - j - kv + 1)

                # row interchanges on A12/A22/A32 (DLASWP over the ld-1 view at AB(kv+1-jb, j+jb)),
                # applied with the still-LOCAL ipiv values and composed in sequence.
                for t in 1:jb
                    ip = ipiv[j + t - 1]
                    if ip != t
                        for q in 0:(j2 - 1)
                            r1 = kv + 1 - jb + t - 1 - q
                            r2 = kv + 1 - jb + ip - 1 - q
                            c = j + jb + q
                            AB[r1, c], AB[r2, c] = AB[r2, c], AB[r1, c]
                        end
                    end
                end
                for i in j:(j + jb - 1)                # LOCAL → GLOBAL, after the laswp
                    ipiv[i] = ipiv[i] + j - 1
                end

                # A13/A23/A33 columnwise (DO 110): these columns are past the band window, so the
                # swap is written out rather than expressed as a view.
                k2 = j - 1 + jb + j2
                for i in 1:j3
                    jj = k2 + i
                    for ii in (j + i - 1):(j + jb - 1)
                        ip = ipiv[ii]                  # GLOBAL here
                        if ip != ii
                            r1 = kv + 1 + ii - jj; r2 = kv + 1 + ip - jj
                            AB[r1, jj], AB[r2, jj] = AB[r2, jj], AB[r1, jj]
                        end
                    end
                end

                # L11 → dense scratch (deviation (1)): unit lower, strictly-upper zeroed so the
                # kernel never reads live U11 through the aliasing triangle.
                for q in 1:jb
                    for p in 1:(q - 1); S[p, q] = z; end
                    S[q, q] = one(T)
                    for p in (q + 1):jb; S[p, q] = AB[kv + 1 + p - q, j + q - 1]; end
                end
                Lv = PtrMatrix(ps, jb, jb, nb)
                A21v = _pb_blk(p0, ld2, ldabs, kv + 1 + jb, j, i2, jb)

                if j2 > 0
                    A12v = _pb_blk(p0, ld2, ldabs, kv + 1 - jb, j + jb, jb, j2)
                    trsm!(A12v, Lv; side = 'L', uplo = 'L', transA = 'N', diag = 'U',
                        alpha = one(T))
                    if i2 > 0
                        A22v = _pb_blk(p0, ld2, ldabs, kv + 1, j + jb, i2, j2)
                        _gemm_core!(A22v, A21v, A12v, -one(T), one(T), false, false, false, false)
                    end
                    if i3 > 0
                        A32v = _pb_blk(p0, ld2, ldabs, kv + KL + 1 - jb, j + jb, i3, j2)
                        W31v = PtrMatrix(p31, i3, jb, ldw)
                        _gemm_core!(A32v, W31v, A12v, -one(T), one(T), false, false, false, false)
                    end
                end

                if j3 > 0
                    for jj in 1:j3                     # lower triangle of A13 → W13
                        for ii in jj:jb
                            W13[ii, jj] = AB[ii - jj + 1, jj + j + kv - 1]
                        end
                    end
                    W13v = PtrMatrix(p13, jb, j3, ldw)
                    trsm!(W13v, Lv; side = 'L', uplo = 'L', transA = 'N', diag = 'U',
                        alpha = one(T))
                    if i2 > 0
                        A23v = _pb_blk(p0, ld2, ldabs, 1 + jb, j + kv, i2, j3)
                        _gemm_core!(A23v, A21v, W13v, -one(T), one(T), false, false, false, false)
                    end
                    if i3 > 0
                        A33v = _pb_blk(p0, ld2, ldabs, 1 + KL, j + kv, i3, j3)
                        W31v = PtrMatrix(p31, i3, jb, ldw)
                        _gemm_core!(A33v, W31v, W13v, -one(T), one(T), false, false, false, false)
                    end
                    for jj in 1:j3                     # and back into the band
                        for ii in jj:jb
                            AB[ii - jj + 1, jj + j + kv - 1] = W13[ii, jj]
                        end
                    end
                end
            else
                for i in j:(j + jb - 1)                # the reference adjusts in BOTH branches
                    ipiv[i] = ipiv[i] + j - 1
                end
            end

            # ── undo the panel interchanges on the LEFT columns (DO 170) ──────────────────────────
            # Restores dgbtf2's unpermuted-L convention (see the header) and, as a side effect,
            # restores A31 to upper-triangular so its triangle fits back inside the band. Strictly
            # DECREASING jj: the swaps are involutions, so reversed order is the exact inverse.
            for jj in (j + jb - 1):-1:j
                jp = ipiv[jj] - jj + 1
                if jp != 1
                    if jp + jj - 1 < j + KL
                        for t in 0:(jj - j - 1)
                            r1 = kv + 1 + jj - j - t; r2 = kv + jp + jj - j - t
                            c = j + t
                            AB[r1, c], AB[r2, c] = AB[r2, c], AB[r1, c]
                        end
                    else
                        for t in 0:(jj - j - 1)
                            r1 = kv + 1 + jj - j - t
                            w = jp + jj - j - KL
                            AB[r1, j + t], W31[w, 1 + t] = W31[w, 1 + t], AB[r1, j + t]
                        end
                    end
                end
                nw = min(i3, jj - j + 1)
                for t in 1:nw
                    AB[kv + KL + t - jj + j, jj] = W31[t, jj - j + 1]
                end
            end
        end
    end
    return info
end

# gbtrs!(trans, kl, ku, m, AB, ipiv, B) → B.  Solve op(A)·X = B in place from gbtrf!'s factors.
# trans ∈ {'N','T','C'}. Mirrors dgbtrs: for 'N', apply L⁻¹·P then a banded upper back-substitution
# (band width kl+ku); for 'T'/'C', a banded upper Uᵀ/Uᴴ forward solve then Lᵀ/Lᴴ with reverse swaps.
function gbtrs!(
        trans::AbstractChar, kl::Integer, ku::Integer, m::Integer,
        AB::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
        B::AbstractVecOrMat{T}
    ) where {T}
    ldab, n = size(AB)
    kv = ku + kl
    kd = kv + 1                                           # diagonal row; multipliers at rows kd+1..
    Bm = B isa AbstractVector ? reshape(B, :, 1) : B
    nrhs = size(Bm, 2)
    lnoti = kl > 0
    if trans == 'N' || trans == 'n'
        if lnoti                                          # forward: L·y = P·b
            @inbounds for j in 1:(n - 1)
                lm = min(kl, n - j)
                lp = ipiv[j]
                if lp != j
                    for c in 1:nrhs
                        Bm[lp, c], Bm[j, c] = Bm[j, c], Bm[lp, c]
                    end
                end
                for c in 1:nrhs
                    bj = Bm[j, c]
                    for i in 1:lm
                        Bm[j + i, c] -= AB[kd + i, j] * bj
                    end
                end
            end
        end
        @inbounds for c in 1:nrhs                          # backward: U·x = y (band kv superdiagonals)
            for j in n:-1:1
                xj = Bm[j, c] / AB[kv + 1, j]
                Bm[j, c] = xj
                for i in max(1, j - kv):(j - 1)
                    Bm[i, c] -= AB[kv + 1 + i - j, j] * xj
                end
            end
        end
    else
        cj = (trans == 'C' || trans == 'c')
        @inbounds for c in 1:nrhs                          # forward: Uᵀ/Uᴴ·z = b
            for j in 1:n
                s = Bm[j, c]
                for i in max(1, j - kv):(j - 1)
                    u = AB[kv + 1 + i - j, j]
                    s -= (cj ? conj(u) : u) * Bm[i, c]
                end
                ujj = AB[kv + 1, j]
                Bm[j, c] = s / (cj ? conj(ujj) : ujj)
            end
        end
        if lnoti                                          # backward: Lᵀ/Lᴴ·w = z, with reverse swaps
            @inbounds for j in (n - 1):-1:1
                lm = min(kl, n - j)
                for c in 1:nrhs
                    s = Bm[j, c]
                    for i in 1:lm
                        mij = AB[kd + i, j]
                        s -= (cj ? conj(mij) : mij) * Bm[j + i, c]
                    end
                    Bm[j, c] = s
                end
                lp = ipiv[j]
                if lp != j
                    for c in 1:nrhs
                        Bm[lp, c], Bm[j, c] = Bm[j, c], Bm[lp, c]
                    end
                end
            end
        end
    end
    return B
end
