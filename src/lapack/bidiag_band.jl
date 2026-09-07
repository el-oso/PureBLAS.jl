# Two-stage bidiagonalization, stage 1: dense → upper BAND of width b.
#
# WHY THIS EXISTS. One-stage `gebrd!` (dgebrd/dlabrd) is bandwidth-bound and cannot be tuned out of it.
# For the panel column at offset j, dlabrd runs TWO gemvs over the whole remaining trailing block (one
# for the Y column, one for the X column), and summing over all n columns that is ~(2/3)·n³ ELEMENTS
# touched — a term that does not depend on the panel width nb, which is why no nb retunes it away.
# MEASURED on wintermute Zen4 (bench/probes/gebrd_roofline.jl), n=1000: gebrd moves ≈6.7 GB in 148.7 ms
# = 44.7 GB/s, against a bare gemv of the same shape at 66-72 GB/s and gemm at 42.6 GF/s. LAPACK's own
# dgebrd lands in the same place (152.1 ms), so this is NOT a cell where the reference demonstrates the
# silicon can do better — one-stage is near its own floor.
#
# Stage 1 replaces that streaming term with a blocked O(n³/b) pass whose work is all gemm: alternate a
# QR on a b-wide column panel with an LQ on a b-tall row panel, applying each through compact-WY. The
# result is upper banded with bandwidth b, and stage 2 (bulge chasing) takes band → bidiagonal in
# O(n²·b) flops. Projection from the rates above: ≈63 ms of stage-1 compute at n=1000 against 148.7 now.
#
# VALUES PATH ONLY, for now. Q and P are NOT accumulated — `gelsd!`'s full-rank fast path and
# `gesvd`'s values path need only the bidiagonal, and the singular-vector back-transform through two
# stages (stage 2's bulge reflectors are many and tiny) is a separate problem. Callers that need
# vectors keep the one-stage `gebrd!`.

# Compact-WY apply, LEFT, transposed:  C := Qᵀ·C  for Q = H(1)·H(2)⋯H(k) = I − V·T·Vᵀ, so Qᵀ = I − V·Tᵀ·Vᵀ
# and C := C − V·(Tᵀ·(Vᵀ·C)). `V` is the (rows)×k panel with the reflectors as its columns, explicit unit
# diagonal and zeros above (the caller stages it — `geqrf!` leaves R in that triangle). This is the same
# algebra `geqrf!`'s blocked driver runs inline; it is spelled out here because that copy is entangled
# with the driver's µarch split for the skinny W gemm and cannot be called.
function _wy_left_T!(
        V::AbstractMatrix{Float64}, tau::AbstractVector{Float64}, C::AbstractMatrix{Float64},
        Tm::AbstractMatrix{Float64}, G::AbstractMatrix{Float64}, W::AbstractMatrix{Float64}
    )
    mv, k = size(V); nc = size(C, 2)
    (k == 0 || nc == 0 || mv == 0) && return C
    Gv = view(G, 1:k, 1:k); Tv = view(Tm, 1:k, 1:k); Wv = view(W, 1:k, 1:nc)
    gemm!(Gv, V, V; transA = 'T', alpha = true, beta = false)          # G = VᵀV
    _wy_tfactor!(Tv, Gv, tau, k)
    gemm!(Wv, V, C; transA = 'T', alpha = true, beta = false)          # W = VᵀC
    trmm!(Wv, Tv; side = 'L', uplo = 'U', transA = 'T')                # W := Tᵀ·W
    gemm!(C, V, Wv; alpha = -1.0, beta = true)                         # C −= V·W
    return C
end

# Compact-WY apply, RIGHT:  C := C·H(1)·H(2)⋯H(k). For the LQ convention (`gelqf!`: Q = H(k)⋯H(1), v_i
# in ROW i of A with v_i[1] ≡ 1) that product is Qᵀ, which is exactly what the unblocked `gelqf!` applies
# to its own trailing rows one reflector at a time. Rowwise-forward compact-WY:
#   H(1)⋯H(k) = I − Vᵀ·T·V,  V the k×N matrix of reflector ROWS  ⇒  C := C − (C·Vᵀ)·T·V.
function _wy_right!(
        V::AbstractMatrix{Float64}, tau::AbstractVector{Float64}, C::AbstractMatrix{Float64},
        Tm::AbstractMatrix{Float64}, G::AbstractMatrix{Float64}, W::AbstractMatrix{Float64}
    )
    k, nv = size(V); nr = size(C, 1)
    (k == 0 || nr == 0 || nv == 0) && return C
    Gv = view(G, 1:k, 1:k); Tv = view(Tm, 1:k, 1:k); Wv = view(W, 1:nr, 1:k)
    gemm!(Gv, V, V; transB = 'T', alpha = true, beta = false)          # G = VVᵀ
    _wy_tfactor!(Tv, Gv, tau, k)
    gemm!(Wv, C, V; transB = 'T', alpha = true, beta = false)          # W = C·Vᵀ
    trmm!(Wv, Tv; side = 'R', uplo = 'U')                              # W := W·T
    gemm!(C, Wv, V; alpha = -1.0, beta = true)                         # C −= W·V
    return C
end

# dlarft's forward triangular factor from the Gram matrix G = VᵀV (or VVᵀ rowwise) and tau. Same
# recurrence `_apply_reflectors_left!` runs; T's strict lower triangle must be ZEROED because the
# `trmm!`/`gemm!` consumers read the full block (the NaN bug recorded in kb `pureblas-svd`).
@inline function _wy_tfactor!(
        Tv::AbstractMatrix{Float64}, Gv::AbstractMatrix{Float64}, tau::AbstractVector{Float64}, k::Int
    )
    @inbounds for j in 1:k, i in 1:k
        Tv[i, j] = 0.0
    end
    @inbounds for c in 1:k
        tc = tau[c]
        Tv[c, c] = tc
        for ii in 1:(c - 1)
            s = 0.0
            for kk in ii:(c - 1)
                s = muladd(Tv[ii, kk], Gv[kk, c], s)
            end
            Tv[ii, c] = -tc * s
        end
    end
    return Tv
end

# Band width. PDM **Derive**: the band panel is a compact-WY block exactly like `geqrf!`'s, so it wants
# the same width for the same reason — deep enough to amortize the rank-b trailing gemm while the b×b
# T factor stays L1-resident. `_qr_nb(n, n)` is that validated ramp, and reusing it means the band
# inherits QR's fleet validation rather than introducing a second, unvalidated literal. Capped at the
# problem size. | tune: no (rides _qr_nb)
@inline _band_b(m::Int, n::Int) = max(8, min(_qr_nb(n, n), n >> 2))

# Stage 1 driver: reduce A (m×n, m ≥ n) to upper band form with bandwidth `b`, in place. On exit
# A[i, i:i+b] holds the band and everything else is scratch (reflectors); Q and P are discarded.
#
# The po2-lda pad is not optional here. MEASURED before adding it: n=1000 (non-po2) ran 1.65x FASTER
# than one-stage gebrd while n=512 and n=2048 ran 0.33x and 0.55x — the aliasing costs more than the
# whole algorithmic win, because every panel's compact-WY apply reads A's columns at that stride. Same
# predicate and remedy as `_gebrd_needs_pad`/`_gebrd_padded!` (svd.jl).
function _gebrd_band!(A::AbstractMatrix{Float64}, b::Int)
    m, n = size(A)
    if _gebrd_needs_pad(A, m, n)
        @scope arn begin
            ldb = _offway_ld(m, Float64)
            p = borrow!(arn, Float64, m, n, ldb)
            lda = stride(A, 2); sz = sizeof(Float64)
            GC.@preserve A p begin
                pa = pointer(A); pp = pointer(p)
                @inbounds for j in 0:(n - 1)
                    unsafe_copyto!(pp + (j * ldb) * sz, pa + (j * lda) * sz, m)
                end
                _gebrd_band_run!(p, b)
                @inbounds for j in 0:(n - 1)
                    unsafe_copyto!(pa + (j * lda) * sz, pp + (j * ldb) * sz, m)
                end
            end
        end
        return A
    end
    return _gebrd_band_run!(A, b)
end

function _gebrd_band_run!(A::AbstractMatrix{Float64}, b::Int)
    m, n = size(A)
    @scope arn begin
        Vq = borrow!(arn, Float64, m, b, _offway_ld(m, Float64))
        Vp = borrow!(arn, Float64, b, n, b)
        Tm = borrow!(arn, Float64, b, b, b)
        G = borrow!(arn, Float64, b, b, b)
        Wl = borrow!(arn, Float64, b, n, b)
        Wr = borrow!(arn, Float64, m, b, _offway_ld(m, Float64))
        tq = borrow!(arn, Float64, b)
        tqL = borrow!(arn, Float64, b)
        tp = borrow!(arn, Float64, b)
        i = 1
        @inbounds while i <= n
            kb = min(b, n - i + 1)
            rows = m - i + 1
            # ── QR on the column panel A[i:m, i:i+kb-1] ────────────────────────────────────────────
            geqrf!(view(A, i:m, i:(i + kb - 1)), view(tq, 1:kb))
            jc = i + kb
            if jc <= n
                kq = min(kb, rows)
                Vv = view(Vq, 1:rows, 1:kq)                       # stage V: unit diagonal, zeros above
                for c in 1:kq, r in 1:rows
                    Vv[r, c] = r < c ? 0.0 : (r == c ? 1.0 : A[i + r - 1, i + c - 1])
                end
                # `geqrf!` (real) stores the FAER τ_f = 1/τ_L, with Inf meaning "identity reflector" —
                # documented at qr.jl:445, and deliberate so the convention does not vary by element
                # type. The compact-WY factor below is written in LAPACK's τ_L, so convert here. This
                # is the same conversion the C-ABI `dgeqrf_64_` shim does for `orgqr!`/`ormqr!`.
                for c in 1:kq
                    tf = tq[c]
                    tqL[c] = isfinite(tf) ? 1.0 / tf : 0.0
                end
                _wy_left_T!(Vv, view(tqL, 1:kq), view(A, i:m, jc:n), Tm, G, Wl)
                # ── LQ on the row panel A[i:i+kb-1, jc:n] ─────────────────────────────────────────
                nrow = n - jc + 1
                gelqf!(view(A, i:(i + kb - 1), jc:n), view(tp, 1:min(kb, nrow)))
                kp = min(kb, nrow)
                if i + kb <= m
                    Vw = view(Vp, 1:kp, 1:nrow)                   # reflector ROWS, unit at the start
                    for r in 1:kp, c in 1:nrow
                        Vw[r, c] = c < r ? 0.0 : (c == r ? 1.0 : A[i + r - 1, jc + c - 1])
                    end
                    _wy_right!(Vw, view(tp, 1:kp), view(A, (i + kb):m, jc:n), Tm, G, Wr)
                end
            end
            i += kb
        end
    end
    return A
end

# ── Stage 2: band → bidiagonal by bulge chasing (values path; no rotations accumulated) ─────────────
# Range-limited Givens. The library's `_rot_cols!` (svd.jl) applies over EVERY row, which is right for
# bdsqr's dense accumulators and wrong here: a chase step touches only O(b) rows, and applying over n
# rows would put this back at O(n³) — the whole point of stage 2 is that it is O(n²·b).
@inline function _band_rot_right!(A::AbstractMatrix{Float64}, j1::Int, j2::Int, c::Float64, s::Float64, r1::Int, r2::Int)
    @inbounds for i in r1:r2
        a = A[i, j1]; t = A[i, j2]
        A[i, j1] = muladd(c, a, s * t)
        A[i, j2] = muladd(c, t, -s * a)
    end
    return A
end

@inline function _band_rot_left!(A::AbstractMatrix{Float64}, i1::Int, i2::Int, c::Float64, s::Float64, c1::Int, c2::Int)
    @inbounds for j in c1:c2
        a = A[i1, j]; t = A[i2, j]
        A[i1, j] = muladd(c, a, s * t)
        A[i2, j] = muladd(c, t, -s * a)
    end
    return A
end

# Reduce the upper band in A (bandwidth `b`) to bidiagonal, writing d/e. Orthogonal on both sides, so
# the singular values are preserved exactly — that invariant is the whole contract, and the probe checks
# it against LAPACK's own svdvals rather than against a reconstruction.
#
# The sweep: for each row j, annihilate A[j, j+2 … j+b] from the OUTSIDE in. Zeroing A[j,t] with a right
# rotation on columns (t−1,t) pushes a fill to A[t,t−1] (just under the diagonal); a left rotation on
# rows (t−1,t) removes it and pushes a fill to A[t−1, t+b] (one past the band); a right rotation on
# columns (t+b−1, t+b) removes THAT and pushes the next fill b further down-right. So each annihilation
# costs a chase of ~n/b steps of O(b) work, and the whole sweep is O(n²·b).
#
# A's off-band entries hold stage 1's REFLECTORS, not zeros, and the chase reads them — so the band is
# cleaned first. That is O(n²) against the O(n²·b) chase.
function _band_to_bidiag!(
        A::AbstractMatrix{Float64}, b::Int, d::AbstractVector{Float64}, e::AbstractVector{Float64}
    )
    m, n = size(A)
    @inbounds for j in 1:n, i in 1:m           # keep only i ≤ j ≤ i+b
        (i <= j <= i + b) || (A[i, j] = 0.0)
    end
    if b > 1
        @inbounds for j in 1:(n - 2)
            for t in min(j + b, n):-1:(j + 2)
                g = A[j, t]
                iszero(g) && continue
                c, s, _ = _givens(A[j, t - 1], g)
                _band_rot_right!(A, t - 1, t, c, s, max(1, t - b - 1), min(m, t))
                A[j, t] = 0.0
                bc = t - 1; br = t
                while br <= m
                    gg = A[br, bc]
                    iszero(gg) && break
                    c1, s1, _ = _givens(A[bc, bc], gg)
                    _band_rot_left!(A, bc, br, c1, s1, bc, min(n, br + b))
                    A[br, bc] = 0.0
                    nc = br + b
                    nc > n && break
                    g2 = A[bc, nc]
                    iszero(g2) && break
                    c2, s2, _ = _givens(A[bc, nc - 1], g2)
                    _band_rot_right!(A, nc - 1, nc, c2, s2, max(1, nc - b - 1), min(m, nc))
                    A[bc, nc] = 0.0
                    bc = nc - 1; br = nc
                end
            end
        end
    end
    @inbounds for i in 1:n
        d[i] = A[i, i]
        i < n && (e[i] = A[i, i + 1])
    end
    return d, e
end

# Two-stage bidiagonalization, VALUES path: A (m×n, m ≥ n) → bidiagonal (d, e). A is destroyed and no
# transformation is retained, so this is only for callers that want singular values — `gelsd!`'s
# full-rank fast path and `gesvd`'s values path. Callers needing vectors keep one-stage `gebrd!`.
function gebrd_vals!(
        A::AbstractMatrix{Float64}, d::AbstractVector{Float64}, e::AbstractVector{Float64};
        b::Int = _band_b(size(A, 1), size(A, 2))
    )
    m, n = size(A)
    m >= n || _throw_mge_n(:gebrd_vals!, m, n)
    n == 0 && return d, e
    bb = min(b, max(n - 1, 1))
    _gebrd_band!(A, bb)
    return _band_to_bidiag!(A, bb, d, e)
end
