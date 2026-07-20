# LAPACK SVD (gesvd) — pure Julia, built on PureBLAS blocks. Three layers (ROADMAP):
#   1. gebrd  — two-sided Householder bidiagonalization  A = Q·B·Pᵀ  (B upper-bidiagonal, m≥n).
#   2. bdsqr  — implicit-shift QR on the bidiagonal B (Golub-Kahan), accumulating Givens into U,Vᵀ.
#   3. driver — form Q,Pᵀ from the reflectors and back-transform the bidiagonal singular vectors.
# Float64 path. Householder = standard LAPACK convention (H = I − τ·v·vᵀ, v[1]=1 implicit) so the
# back-transform is self-contained. ponytail: m<n handled by transposing; generic/AD SVD deferred.

# --- Householder generator (LAPACK dlarfg) on a strided segment ---------------------------------
# x = [α; tail]. Returns (β, τ): the reflector H = I − τ·v·vᵀ with v = [1; x[2:]/(α−β)] zeros the
# tail, leaving β at x[1]. On return x[2:] holds the essential v; x[1] is left to the caller.
@inline function _larfg!(x::AbstractVector{T}) where {T <: Real}
    n = length(x)
    @inbounds begin
        α = x[1]
        n == 1 && return α, zero(T)
        ss = zero(T)                                 # SIMD sum-of-squares (avoids O(n²) Base.hypot in gebrd);
        @simd for i in 2:n
            ss = muladd(x[i], x[i], ss)
        end   # fast path — exact for the common regime.
        xnorm = sqrt(ss)
        # Scaled recompute when the fast ss over/underflowed. The underflow case matters: when x is NORMAL
        # but its squares are DENORMAL (|x| ≲ √floatmin), ss < floatmin loses mantissa bits and the
        # reflector loses orthogonality (dlarfg guards this; same principle as req#6 nrm2/lassq). The
        # common path (ss finite, ≥ floatmin) is UNCHANGED — this only reroutes the extreme-scale tails.
        if !isfinite(xnorm) || ss < floatmin(T)
            scale = zero(T)
            for i in 2:n
                scale = max(scale, abs(x[i]))
            end
            scale == zero(T) && return α, zero(T)
            ssum = zero(T)
            for i in 2:n
                t = x[i] / scale; ssum = muladd(t, t, ssum)
            end
            xnorm = scale * sqrt(ssum)
        end
        xnorm == zero(T) && return α, zero(T)
        β = -copysign(hypot(α, xnorm), α)
        τ = (β - α) / β
        s = one(T) / (α - β)
        for i in 2:n
            x[i] *= s
        end
    end
    return β, τ
end

# Apply H = I − τ·v·vᵀ (v[1]≡1, v[2:]=v[2:]) to C (size len×nc) from the LEFT:  C := H·C.
@inline function _house_left!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T <: Real}
    τ == zero(T) && return C
    len, nc = size(C)
    @inbounds for j in 1:nc
        w = C[1, j]
        for i in 2:len
            w = muladd(v[i], C[i, j], w)
        end
        w *= τ
        C[1, j] -= w
        for i in 2:len
            C[i, j] -= v[i] * w
        end
    end
    return C
end

# Apply H = I − τ·v·vᵀ (v[1]≡1) to C (size nr×len) from the RIGHT:  C := C·H.
@inline function _house_right!(C::AbstractMatrix{Float64}, v::AbstractVector{Float64}, τ::Float64)
    τ == 0.0 && return C
    nr, len = size(C)
    @inbounds for i in 1:nr
        w = C[i, 1]
        for j in 2:len
            w = muladd(C[i, j], v[j], w)
        end
        w *= τ
        C[i, 1] -= w
        for j in 2:len
            C[i, j] -= w * v[j]
        end
    end
    return C
end

# --- Stage 1: unblocked bidiagonalization (LAPACK dgebd2), m ≥ n → upper bidiagonal -------------
# Overwrites A: below-diag holds the left reflectors (Q), above-superdiag the right reflectors (Pᵀ).
# d[1:n] = diagonal, e[1:n-1] = superdiagonal of B. tauq/taup the reflector coefficients.
function gebd2!(
        A::AbstractMatrix{Float64}, d::AbstractVector{Float64}, e::AbstractVector{Float64},
        tauq::AbstractVector{Float64}, taup::AbstractVector{Float64}
    )
    m, n = size(A)
    m >= n || throw(ArgumentError("gebd2!: requires m ≥ n (got $m×$n)"))
    @inbounds for i in 1:n
        xq = view(A, i:m, i)                      # left reflector zeros A[i+1:m, i]
        β, τq = _larfg!(xq)
        d[i] = β; tauq[i] = τq
        if i < n
            _house_left!(view(A, i:m, (i + 1):n), xq, τq)   # xq[1] treated as 1
        end
        A[i, i] = β
        if i < n
            xp = view(A, i, (i + 1):n)                # right reflector zeros A[i, i+2:n]
            β2, τp = _larfg!(xp)
            e[i] = β2; taup[i] = τp
            if i < m
                _house_right!(view(A, (i + 1):m, (i + 1):n), xp, τp)
            end
            A[i, i + 1] = β2
        else
            taup[i] = 0.0
        end
    end
    return A
end

# ── Complex bidiagonalization (LAPACK zgebd2): real d,e — τ complex, β real. Left applies conj(τq); the
# right reflector operates on the zlacgv-CONJUGATED row (the phase dance that keeps e real). WIP-debug.
@inline function _larfg!(x::AbstractVector{T}) where {T <: Complex}
    R = real(T); n = length(x)
    @inbounds begin
        α = x[1]
        ss = zero(R); for i in 2:n
            ss += abs2(x[i])
        end
        xnorm = sqrt(ss)
        # Scaled recompute when the naive Σ|xᵢ|² over/underflowed. The underflow case is real: when x is
        # NORMAL but its squares are DENORMAL (|x| ≲ √floatmin), ss < floatmin loses mantissa bits and the
        # reflector goes non-unitary — mirrors the real _larfg! guard (req#6-analogous). This method is shared
        # by the complex SVD path (zgebd2/zgesvd), so the fix reaches there too. Common path UNCHANGED.
        if !isfinite(xnorm) || (ss < floatmin(R) && n > 1)
            scale = zero(R)
            for i in 2:n
                scale = max(scale, abs(x[i]))
            end
            if scale != zero(R)
                ssum = zero(R)
                for i in 2:n
                    t = abs(x[i]) / scale; ssum = muladd(t, t, ssum)
                end
                xnorm = scale * sqrt(ssum)
            end
        end
        # n==1 (empty tail) with imag(α)≠0 still needs the phase rotation (τ≠0, β real) — do NOT early-return
        # on n==1; only the genuinely-trivial α (xnorm=0 AND real α) is τ=0.
        (xnorm == 0 && imag(α) == 0) && return real(α), zero(T)
        β = -copysign(hypot(abs(α), xnorm), real(α))
        τ = T((β - real(α)) / β, -imag(α) / β)
        s = one(T) / (α - β)
        for i in 2:n
            x[i] *= s
        end
    end
    return β, τ
end
@inline function _house_left!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T <: Complex}
    iszero(τ) && return C
    len, nc = size(C)
    @inbounds for j in 1:nc
        w = C[1, j]; for i in 2:len
            w += conj(v[i]) * C[i, j]
        end
        w *= τ; C[1, j] -= w
        for i in 2:len
            C[i, j] -= v[i] * w
        end
    end
    return C
end
@inline function _house_right!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T <: Complex}
    iszero(τ) && return C
    nr, len = size(C)
    @inbounds for i in 1:nr
        w = C[i, 1]; for j in 2:len
            w += C[i, j] * v[j]
        end
        w *= τ; C[i, 1] -= w
        for j in 2:len
            C[i, j] -= w * conj(v[j])
        end
    end
    return C
end
function gebd2!(
        A::AbstractMatrix{T}, d::AbstractVector{R}, e::AbstractVector{R},
        tauq::AbstractVector{T}, taup::AbstractVector{T}
    ) where {T <: Complex, R <: Real}
    m, n = size(A)
    m >= n || throw(ArgumentError("gebd2!: requires m ≥ n (got $m×$n)"))
    @inbounds for i in 1:n
        β, τq = _larfg!(view(A, i:m, i))
        d[i] = β; tauq[i] = τq
        i < n && _house_left!(view(A, i:m, (i + 1):n), view(A, i:m, i), conj(τq))
        A[i, i] = β
        if i < n
            xp = view(A, i, (i + 1):n)
            for j in eachindex(xp)
                xp[j] = conj(xp[j])
            end
            β2, τp = _larfg!(xp); e[i] = β2; taup[i] = τp
            i < m && _house_right!(view(A, (i + 1):m, (i + 1):n), xp, τp)
            for j in eachindex(xp)
                xp[j] = conj(xp[j])
            end
            A[i, i + 1] = β2
        else
            taup[i] = zero(T)
        end
    end
    return A
end

# --- Stage 1b: BLOCKED bidiagonalization (LAPACK dlabrd panel + gemm trailing update), m ≥ n -----
# Reduce the first nb rows/cols of the (mm×nn) submatrix As to bidiagonal form, accumulating the
# matrices X (mm×nb) and Y (nn×nb) that drive the rank-2nb trailing update. Faithful dlabrd port;
# the matrix-vector ops are PureBLAS gemv!. d,e,tauq,taup,X,Y are local (1-based) to this block.
# Direct gemv kernel (skips the public kwarg wrapper's ~200 ns dispatch/char-parse — critical in _labrd's
# thousands of tiny gemv calls). y := α·op(A)·x + β·y with op = ('T' if tr) on A (m×n = size(Av)), unit inc.
@inline _lg!(yv, Av, xv, α::Float64, β::Float64, tr::Bool) =
    _gemv!(tr, false, size(Av, 1), size(Av, 2), α, Av, xv, 1, β, yv, 1)

# arow/tmp are caller-provided scratch (from the SVD workspace): arow ≥ nn (active reflector row), tmp ≥ nb.
function _labrd!(
        As::AbstractMatrix{Float64}, d, e, tauq, taup, X, Y, nb::Int,
        arow::AbstractVector{Float64}, tmp::AbstractVector{Float64}
    )
    mm, nn = size(As)
    @inbounds for i in 1:nb
        if i > 1
            for t in 1:(i - 1)
                tmp[t] = Y[i, t]
            end                       # Y[i,1:i-1] strided → contiguous
            _lg!(view(As, i:mm, i), view(As, i:mm, 1:(i - 1)), view(tmp, 1:(i - 1)), -1.0, 1.0, false)
            _lg!(view(As, i:mm, i), view(X, i:mm, 1:(i - 1)), view(As, 1:(i - 1), i), -1.0, 1.0, false)
        end
        β, τq = _larfg!(view(As, i:mm, i))
        d[i] = β; tauq[i] = τq; As[i, i] = 1.0
        if i < nn
            L = nn - i
            _lg!(view(Y, (i + 1):nn, i), view(As, i:mm, (i + 1):nn), view(As, i:mm, i), 1.0, 0.0, true)
            if i > 1
                _lg!(view(Y, 1:(i - 1), i), view(As, i:mm, 1:(i - 1)), view(As, i:mm, i), 1.0, 0.0, true)
                _lg!(view(Y, (i + 1):nn, i), view(Y, (i + 1):nn, 1:(i - 1)), view(Y, 1:(i - 1), i), -1.0, 1.0, false)
                _lg!(view(Y, 1:(i - 1), i), view(X, i:mm, 1:(i - 1)), view(As, i:mm, i), 1.0, 0.0, true)
                _lg!(view(Y, (i + 1):nn, i), view(As, 1:(i - 1), (i + 1):nn), view(Y, 1:(i - 1), i), -1.0, 1.0, true)
            end
            for r in (i + 1):nn
                Y[r, i] *= τq
            end
            # Update the active row A(i, i+1:nn) entirely in a contiguous buffer.
            for t in 1:L
                arow[t] = As[i, i + t]
            end
            for t in 1:i
                tmp[t] = As[i, t]
            end                        # As[i,1:i] strided → contiguous
            _lg!(view(arow, 1:L), view(Y, (i + 1):nn, 1:i), view(tmp, 1:i), -1.0, 1.0, false)
            if i > 1
                for t in 1:(i - 1)
                    tmp[t] = X[i, t]
                end                   # X[i,1:i-1] strided → contiguous
                _lg!(view(arow, 1:L), view(As, 1:(i - 1), (i + 1):nn), view(tmp, 1:(i - 1)), -1.0, 1.0, true)
            end
            β2, τp = _larfg!(view(arow, 1:L))
            e[i] = β2; taup[i] = τp; arow[1] = 1.0                       # arow now = the reflector v (v[1]=1)
            As[i, i + 1] = 1.0
            for t in 2:L
                As[i, i + t] = arow[t]
            end
            _lg!(view(X, (i + 1):mm, i), view(As, (i + 1):mm, (i + 1):nn), view(arow, 1:L), 1.0, 0.0, false)
            _lg!(view(X, 1:i, i), view(Y, (i + 1):nn, 1:i), view(arow, 1:L), 1.0, 0.0, true)
            _lg!(view(X, (i + 1):mm, i), view(As, (i + 1):mm, 1:i), view(X, 1:i, i), -1.0, 1.0, false)
            if i > 1
                _lg!(view(X, 1:(i - 1), i), view(As, 1:(i - 1), (i + 1):nn), view(arow, 1:L), 1.0, 0.0, false)
                _lg!(view(X, (i + 1):mm, i), view(X, (i + 1):mm, 1:(i - 1)), view(X, 1:(i - 1), i), -1.0, 1.0, false)
            end
            for r in (i + 1):mm
                X[r, i] *= τp
            end
        else
            taup[i] = 0.0
        end
    end
    return As
end

# Complex blocked panel (LAPACK zlabrd): faithful port of the real _labrd! with zlabrd's conjugation
# dance — d,e stay REAL; τ complex; left ops use vᴴ (gemv 'C' where the real path uses 'T'); the active
# row is reduced in the CONJUGATED domain (arow ≡ conj(row)), so the P reflector is built there and the
# essential v is stored back conjugated (matching zgebrd's 'N','C' trailing gemm). mode 0=N,1=T,2=C.
@inline _lgc!(yv, Av, xv, α::T, β::T, mode::Int) where {T <: Complex} =
    _gemv!(mode != 0, mode == 2, size(Av, 1), size(Av, 2), α, Av, xv, 1, β, yv, 1)

function _labrd!(
        As::AbstractMatrix{T}, d::AbstractVector{R}, e::AbstractVector{R}, tauq, taup, X, Y,
        nb::Int, arow::AbstractVector{T}, tmp::AbstractVector{T}
    ) where {T <: Complex, R <: Real}
    mm, nn = size(As)
    o = one(T); z = zero(T)
    @inbounds for i in 1:nb
        if i > 1
            for t in 1:(i - 1)
                tmp[t] = conj(Y[i, t])
            end                 # zlacgv Y(i,1:i-1)
            _lgc!(view(As, i:mm, i), view(As, i:mm, 1:(i - 1)), view(tmp, 1:(i - 1)), -o, o, 0)
            _lgc!(view(As, i:mm, i), view(X, i:mm, 1:(i - 1)), view(As, 1:(i - 1), i), -o, o, 0)
        end
        β, τq = _larfg!(view(As, i:mm, i))
        d[i] = β; tauq[i] = τq; As[i, i] = o
        if i < nn
            L = nn - i
            _lgc!(view(Y, (i + 1):nn, i), view(As, i:mm, (i + 1):nn), view(As, i:mm, i), o, z, 2)   # Aᴴ·v
            if i > 1
                _lgc!(view(Y, 1:(i - 1), i), view(As, i:mm, 1:(i - 1)), view(As, i:mm, i), o, z, 2)
                _lgc!(view(Y, (i + 1):nn, i), view(Y, (i + 1):nn, 1:(i - 1)), view(Y, 1:(i - 1), i), -o, o, 0)
                _lgc!(view(Y, 1:(i - 1), i), view(X, i:mm, 1:(i - 1)), view(As, i:mm, i), o, z, 2)
                _lgc!(view(Y, (i + 1):nn, i), view(As, 1:(i - 1), (i + 1):nn), view(Y, 1:(i - 1), i), -o, o, 2)
            end
            for r in (i + 1):nn
                Y[r, i] *= τq
            end
            # Update the active row in the CONJUGATED domain: arow ← conj(A(i,i+1:nn)).
            for t in 1:L
                arow[t] = conj(As[i, i + t])
            end
            for t in 1:i
                tmp[t] = conj(As[i, t])
            end                  # zlacgv A(i,1:i)
            _lgc!(view(arow, 1:L), view(Y, (i + 1):nn, 1:i), view(tmp, 1:i), -o, o, 0)
            if i > 1
                for t in 1:(i - 1)
                    tmp[t] = conj(X[i, t])
                end             # zlacgv X(i,1:i-1)
                _lgc!(view(arow, 1:L), view(As, 1:(i - 1), (i + 1):nn), view(tmp, 1:(i - 1)), -o, o, 2)
            end
            β2, τp = _larfg!(view(arow, 1:L))
            e[i] = β2; taup[i] = τp; arow[1] = o                        # arow = conj-domain reflector v (v[1]=1)
            As[i, i + 1] = o
            for t in 2:L
                As[i, i + t] = conj(arow[t])
            end               # store the un-conjugated essential v
            _lgc!(view(X, (i + 1):mm, i), view(As, (i + 1):mm, (i + 1):nn), view(arow, 1:L), o, z, 0)
            _lgc!(view(X, 1:i, i), view(Y, (i + 1):nn, 1:i), view(arow, 1:L), o, z, 2)
            _lgc!(view(X, (i + 1):mm, i), view(As, (i + 1):mm, 1:i), view(X, 1:i, i), -o, o, 0)
            if i > 1
                _lgc!(view(X, 1:(i - 1), i), view(As, 1:(i - 1), (i + 1):nn), view(arow, 1:L), o, z, 0)
                _lgc!(view(X, (i + 1):mm, i), view(X, (i + 1):mm, 1:(i - 1)), view(X, 1:(i - 1), i), -o, o, 0)
            end
            for r in (i + 1):mm
                X[r, i] *= τp
            end
        else
            taup[i] = z
        end
    end
    return As
end

const _BRD_NB = 16     # bidiagonalization panel width; measured optimal across n=256–2048 (narrow panel ⇒
# less BLAS-2 work/panel; the rank-16 trailing gemm stays efficient). ponytail: Zen4.
const _BT_NB = 32      # back-transform (compact-WY dlarfb) block: larger than gebrd's — its gemms want
# bigger T blocks (nb=16 there regressed large-n vectors). Decoupled from _BRD_NB.
const _SVD_DC_CROSS = 1     # vectors: bdsqr (QR) only at n≤1 (trivial, no sweep), divide-and-conquer for n≥2.
# CORRECTNESS OVERRIDE (2026-07-19, Fable adversarial review): the bdsqr! QR sweep
# FAILS on near-degenerate singular-value clusters (two σ agreeing to relative spread
# in ~(3e-15,1e-7)) — it either exhausts maxit ("failed to converge", a hard crash on
# any clustered/rank-deficient input — reachable via gelsd/ggsvd, whose inputs have
# clustered/zero σ BY DEFINITION) or, worse, SILENTLY drops the largest σ on graded
# bidiagonals (recon error ~1). The D&C solver (_dc_qr! base) has correct
# Demmel–Kahan-style deflation and is machine-eps on all these inputs (verified n=2..50,
# clustered+graded), so route ALL with-vectors SVD through it. bdsqr! is a simplified
# port; fixing its convergence (proper relative-accuracy tests + direction-dependent
# shift, per reference dbdsqr) restores the perf crossover below — until then, D&C.
# PERF NOTE (pre-fix, boost-locked bench/gesvd_cross.jl): bdsqr won n≤96 (1.13-1.42 vs
# OB gesdd) but that path was numerically UNSOUND; the small-n perf regression from
# using D&C is the correctness-first tradeoff. (Algorithm-intrinsic crossover like
# LAPACK gesdd's SMLSIZ — not a cache-derived block size.)

# ── SVD scratch: one owned workspace per element type (mirrors L3Workspace/workspace.jl) ────────────
# Every internal SVD buffer — bidiag arrays, gebrd/labrd panels, bidiagonal singular-vector blocks, the
# bdsdc D&C staging, the two back-transform accumulators (padded), the compact-WY T/G/W/Y blocks, and the
# m<n transpose staging — lives here as a concrete field, grown on demand and reused across calls. So a
# warm `gesvd!` into caller-provided U/S/Vt allocates NOTHING. gesvd is Float64-ONLY (no s/c/z SVD kernel),
# so unlike L3Workspace there is NO per-type dispatch and NO IdDict fallback: one module-level const,
# reached by a bare field load (unconditionally trim-safe). Single global ⇒ single-thread only (project's
# current mode); MT swaps _svdws() for a per-task owner, nothing else.
mutable struct SVDWorkspace{T}
    d::Vector{T}; e::Vector{T}; tauq::Vector{T}; taup::Vector{T}   # bidiagonal + reflector scalars
    gebrd_X::Matrix{T}; gebrd_Y::Matrix{T}                         # dlabrd panels
    labrd_arow::Vector{T}; labrd_tmp::Vector{T}                    # dlabrd contiguous row/col temps
    Lvec::Matrix{T}; Rvec::Matrix{T}                               # B's left/right singular vectors (N×N)
    dc_diag::Vector{T}; dc_subdiag::Vector{T}; dc_U::Matrix{T}     # bdsdc D&C staging (dc_U is (N+1)²)
    UApad::Matrix{T}; VQ::Matrix{T}                                # U back-transform: padded accumulator + Q reflectors
    Vpad::Matrix{T}; VP::Matrix{T}                                 # V back-transform: padded accumulator + P reflectors
    bt_T::Matrix{T}; bt_G::Matrix{T}; bt_W::Matrix{T}; bt_Yb::Matrix{T}   # compact-WY back-transform blocks
    trbuf::Matrix{T}; Usc::Matrix{T}; Vtsc::Matrix{T}             # m<n transpose staging (Aᵀ, Ū, V̄ᵀ)
    cabi_U::Matrix{T}; cabi_Vt::Matrix{T}                         # C-ABI dgesvd scratch for jobu/jobvt ∈ {'N','O'}
    dqds_Z::Vector{Float64}; dqds_st::_DqdsState                  # dqds (dlasq) values path: 4n qd-array + state
end
function SVDWorkspace{T}() where {T}
    ev() = T[]; em() = Matrix{T}(undef, 0, 0)
    return SVDWorkspace{T}(
        ev(), ev(), ev(), ev(), em(), em(), ev(), ev(), em(), em(),
        ev(), ev(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(),
        Float64[], _DqdsState()
    )
end

const _SVDWS = SVDWorkspace{Float64}()
@inline _svdws() = _SVDWS
# Complex SVD values path: a separate owned workspace. Only the blocked-bidiag panels (gebrd_X/Y,
# labrd_arow/tmp) are ever grown/used here — the singular-VECTOR buffers stay empty (vectors are the
# follow-up). d,e stay real (local to the values entry), so they don't live in this complex workspace.
const _SVDWS_C = SVDWorkspace{ComplexF64}()
const _SVDWS_C32 = SVDWorkspace{ComplexF32}()
@inline _svdws(::Type{ComplexF64}) = _SVDWS_C
@inline _svdws(::Type{ComplexF32}) = _SVDWS_C32

# Grow only the blocked-bidiag panels (complex values path; vectors buffers untouched).
function _svd_grow_bidiag!(ws::SVDWorkspace{T}, M::Int, N::Int) where {T}
    nbb = _BRD_NB
    ws.gebrd_X = _gm(ws.gebrd_X, M, nbb); ws.gebrd_Y = _gm(ws.gebrd_Y, N, nbb)
    ws.labrd_arow = _gv(ws.labrd_arow, max(N, 1)); ws.labrd_tmp = _gv(ws.labrd_tmp, max(N, 1))
    ws.dqds_Z = _gv(ws.dqds_Z, 4 * N + 4)
    return ws
end

@inline _gm(b::Matrix{T}, r::Int, c::Int) where {T} = (size(b, 1) < r || size(b, 2) < c) ? Matrix{T}(undef, r, c) : b
@inline _gv(b::Vector{T}, n::Int) where {T} = length(b) < n ? Vector{T}(undef, n) : b

# Grow every m≥n-path buffer to fit a reduced M×N problem forming `nu` U-columns. Buffers are pure scratch
# (fully re-initialized per call), so growth just reallocates when too small — nothing to preserve.
function _svd_grow!(ws::SVDWorkspace{T}, M::Int, N::Int, nu::Int) where {T}
    nbb = _BRD_NB; nbt = _BT_NB
    ldu = M % 256 == 0 ? M + 8 : M
    ldv = N % 256 == 0 ? N + 8 : N
    ws.d = _gv(ws.d, N); ws.e = _gv(ws.e, max(N, 1)); ws.tauq = _gv(ws.tauq, N); ws.taup = _gv(ws.taup, N)
    ws.gebrd_X = _gm(ws.gebrd_X, M, nbb); ws.gebrd_Y = _gm(ws.gebrd_Y, N, nbb)
    ws.labrd_arow = _gv(ws.labrd_arow, N); ws.labrd_tmp = _gv(ws.labrd_tmp, N)
    ws.Lvec = _gm(ws.Lvec, N, N); ws.Rvec = _gm(ws.Rvec, N, N)
    ws.dc_diag = _gv(ws.dc_diag, N); ws.dc_subdiag = _gv(ws.dc_subdiag, N); ws.dc_U = _gm(ws.dc_U, N + 1, N + 1)
    ws.UApad = _gm(ws.UApad, ldu, nu); ws.VQ = _gm(ws.VQ, M, N)
    ws.Vpad = _gm(ws.Vpad, ldv, N); ws.VP = _gm(ws.VP, N, max(N - 1, 1))
    ws.bt_T = _gm(ws.bt_T, nbt, nbt); ws.bt_G = _gm(ws.bt_G, nbt, nbt)
    ws.bt_W = _gm(ws.bt_W, nbt, nu); ws.bt_Yb = _gm(ws.bt_Yb, nbt, nu)
    ws.dqds_Z = _gv(ws.dqds_Z, 4 * N + 4)
    return ws
end

# Blocked bidiagonalization driver (LAPACK dgebrd): blocked panels via _labrd! + two gemm! trailing
# updates, finishing the tail with the unblocked gebd2!. Requires m ≥ n.
function gebrd!(
        A::AbstractMatrix{Float64}, d::AbstractVector{Float64}, e::AbstractVector{Float64},
        tauq::AbstractVector{Float64}, taup::AbstractVector{Float64}, ws::SVDWorkspace{Float64};
        nb::Int = _BRD_NB
    )
    m, n = size(A)
    m >= n || throw(ArgumentError("gebrd!: requires m ≥ n (got $m×$n)"))
    k = n
    nx = nb
    if k <= nx || nb < 2
        return gebd2!(A, d, e, tauq, taup)
    end
    X = ws.gebrd_X; Y = ws.gebrd_Y
    i = 1
    @inbounds while i <= k - nx
        mm = m - i + 1; nn = n - i + 1
        As = view(A, i:m, i:n)
        di = view(d, i:k); ei = view(e, i:(k - 1)); tqi = view(tauq, i:k); tpi = view(taup, i:k)
        _labrd!(
            As, di, ei, tqi, tpi, view(X, 1:mm, 1:nb), view(Y, 1:nn, 1:nb), nb,
            view(ws.labrd_arow, 1:nn), view(ws.labrd_tmp, 1:nb)
        )
        # trailing update A[i+nb:m, i+nb:n] −= V·Yₜᵀ + Xₜ·Ar
        if i + nb <= k
            tr = view(A, (i + nb):m, (i + nb):n)
            gemm!(tr, view(A, (i + nb):m, i:(i + nb - 1)), view(Y, (nb + 1):nn, 1:nb); transB = 'T', alpha = -1.0, beta = 1.0)
            gemm!(tr, view(X, (nb + 1):mm, 1:nb), view(A, i:(i + nb - 1), (i + nb):n); alpha = -1.0, beta = 1.0)
        end
        for j in i:(i + nb - 1)                      # restore the panel's diagonal/superdiagonal
            A[j, j] = d[j]
            A[j, j + 1] = e[j]
        end
        i += nb
    end
    if i <= k                                  # unblocked tail
        gebd2!(view(A, i:m, i:n), view(d, i:k), view(e, i:(k - 1)), view(tauq, i:k), view(taup, i:k))
    end
    return A
end

# Complex blocked bidiagonalization driver (LAPACK zgebrd): zlabrd panels + two gemm trailing updates
# (the first with transB='C' — the block update is A −= V·Yᴴ − X·Uᴴ), unblocked zgebd2! tail. m ≥ n.
function gebrd!(
        A::AbstractMatrix{T}, d::AbstractVector{R}, e::AbstractVector{R}, tauq::AbstractVector{T},
        taup::AbstractVector{T}, ws::SVDWorkspace{T}; nb::Int = _BRD_NB
    ) where {T <: Complex, R <: Real}
    m, n = size(A)
    m >= n || throw(ArgumentError("gebrd!: requires m ≥ n (got $m×$n)"))
    k = n; nx = nb
    if k <= nx || nb < 2
        return gebd2!(A, d, e, tauq, taup)
    end
    X = ws.gebrd_X; Y = ws.gebrd_Y
    o = one(T)
    i = 1
    @inbounds while i <= k - nx
        mm = m - i + 1; nn = n - i + 1
        As = view(A, i:m, i:n)
        _labrd!(
            As, view(d, i:k), view(e, i:(k - 1)), view(tauq, i:k), view(taup, i:k),
            view(X, 1:mm, 1:nb), view(Y, 1:nn, 1:nb), nb,
            view(ws.labrd_arow, 1:nn), view(ws.labrd_tmp, 1:nb)
        )
        if i + nb <= k
            tr = view(A, (i + nb):m, (i + nb):n)
            gemm!(tr, view(A, (i + nb):m, i:(i + nb - 1)), view(Y, (nb + 1):nn, 1:nb); transB = 'C', alpha = -o, beta = o)
            gemm!(tr, view(X, (nb + 1):mm, 1:nb), view(A, i:(i + nb - 1), (i + nb):n); alpha = -o, beta = o)
        end
        for j in i:(i + nb - 1)
            A[j, j] = d[j]
            A[j, j + 1] = e[j]
        end
        i += nb
    end
    if i <= k
        gebd2!(view(A, i:m, i:n), view(d, i:k), view(e, i:(k - 1)), view(tauq, i:k), view(taup, i:k))
    end
    return A
end

# --- Stage 2: bidiagonal SVD via implicit-shift QR (Golub-Kahan / LAPACK dbdsqr core) -----------
# Smaller singular value of the 2×2 [[f,g],[0,h]] — the Wilkinson shift. Approximate is fine: the
# shift only affects convergence speed, never accuracy (the orthogonal sweeps preserve σ exactly).
@inline function _svd_2x2_smin(f::Float64, g::Float64, h::Float64)
    fa = abs(f); ga = abs(g); ha = abs(h)
    s = fa * fa + ga * ga + ha * ha
    p = fa * ha
    disc = s * s - 4.0 * p * p
    disc = disc < 0.0 ? 0.0 : disc
    smax2 = 0.5 * (s + sqrt(disc))
    smax2 == 0.0 && return 0.0
    return p / sqrt(smax2)
end

# Givens: (c,s,r) with c·f + s·g = r, −s·f + c·g = 0. r ≥ 0 (sign absorbed by later normalization).
# Generic over T<:Real (Float64 codegen unchanged); the T-generic form also drives Float32 _steqr!/_stedc!.
@inline function _givens(f::T, g::T) where {T <: Real}
    r = sqrt(f * f + g * g)          # bdsqr! scales the bidiagonal to O(1) ⇒ no overflow; skip Base.hypot
    r == zero(T) && return one(T), zero(T), zero(T)
    return f / r, g / r, r
end

# M := M·G over columns (j1,j2), G = [[c,−s],[s,c]]: new_col_j1 = c·old_j1 + s·old_j2, etc. The two
# columns are contiguous (column-major) → SIMD over rows; this is bdsqr's hot vector-accumulation loop.
@inline function _rot_cols!(M::AbstractMatrix{Float64}, j1::Int, j2::Int, c::Float64, s::Float64)
    s == 0.0 && return M
    nr = size(M, 1)
    if M isa StridedMatrix && stride(M, 1) == 1
        ld = stride(M, 2)
        GC.@preserve M begin
            p = pointer(M); vc = _CVF(c); vs = _CVF(s); i = 1
            @inbounds while i + _CHOLW - 1 <= nr
                pa = _cvptr(p, i, j1, ld); pb = _cvptr(p, i, j2, ld)
                a = vload(_CVF, pa); b = vload(_CVF, pb)
                vstore(vc * a + vs * b, pa); vstore(vc * b - vs * a, pb); i += _CHOLW
            end
            @inbounds while i <= nr
                a = unsafe_load(p, _clidx(i, j1, ld)); b = unsafe_load(p, _clidx(i, j2, ld))
                unsafe_store!(p, c * a + s * b, _clidx(i, j1, ld)); unsafe_store!(p, c * b - s * a, _clidx(i, j2, ld)); i += 1
            end
        end
    else
        @inbounds for i in 1:nr
            a = M[i, j1]; b = M[i, j2]
            M[i, j1] = c * a + s * b; M[i, j2] = c * b - s * a
        end
    end
    return M
end

# Generic (non-Float64) scalar column rotation — the Float32 _steqr!/_stedc! eigen path (no SIMD
# fast-path). Float64 dispatches to the more-specific SIMD method above; Float32 lands here.
@inline function _rot_cols!(M::AbstractMatrix{T}, j1::Int, j2::Int, c::T, s::T) where {T <: Real}
    s == zero(T) && return M
    @inbounds for i in 1:size(M, 1)
        a = M[i, j1]; b = M[i, j2]
        M[i, j1] = c * a + s * b; M[i, j2] = c * b - s * a
    end
    return M
end

# One implicit-shift QR sweep on the bidiagonal block d[l:u], e[l:u-1]. Chases the bulge downward,
# accumulating the right rotations into V columns and the left rotations into U columns.
# (LAPACK dbdsqr forward recurrence; shift folded as f=(d[l]²−shift²)/d[l], g=e[l].)
function _bdsqr_sweep!(
        d::AbstractVector{Float64}, e::AbstractVector{Float64}, l::Int, u::Int,
        shift::Float64, U, V
    )
    @inbounds begin
        f = shift == 0.0 ? d[l] : (d[l] - shift) * (sign(d[l]) + shift / d[l])
        g = e[l]
        for k in l:(u - 1)
            c, s, r = _givens(f, g)                  # right rotation (cols k,k+1)
            k > l && (e[k - 1] = r)
            f = c * d[k] + s * e[k]
            e[k] = c * e[k] - s * d[k]
            g = s * d[k + 1]
            d[k + 1] = c * d[k + 1]
            !isnothing(V) && _rot_cols!(V, k, k + 1, c, s)
            c, s, r = _givens(f, g)                  # left rotation (rows k,k+1)
            d[k] = r
            f = c * e[k] + s * d[k + 1]
            d[k + 1] = c * d[k + 1] - s * e[k]
            if k < u - 1
                g = s * e[k + 1]
                e[k + 1] = c * e[k + 1]
            end
            e[k] = f
            !isnothing(U) && _rot_cols!(U, k, k + 1, c, s)
        end
    end
    return nothing
end

# Bidiagonal SVD: overwrite d with the singular values (descending, ≥0); accumulate left/right
# rotations into U (cols) and V (cols) if provided, so that B₀ = U·diag(d)·Vᵀ. e is destroyed.
function bdsqr!(d::AbstractVector{Float64}, e::AbstractVector{Float64}, U, V)
    n = length(d)
    n == 0 && return d
    mx = 0.0                                          # scale the bidiagonal to O(1) so _givens is fast+safe
    @inbounds for i in 1:n
        mx = max(mx, abs(d[i]))
    end
    @inbounds for i in 1:(n - 1)
        mx = max(mx, abs(e[i]))
    end
    mx == 0.0 && return d
    minv = 1.0 / mx
    @inbounds for i in 1:n
        d[i] *= minv
    end
    @inbounds for i in 1:(n - 1)
        e[i] *= minv
    end
    tol = 8.0 * eps(Float64)
    m = n
    iter = 0; maxit = 12 * n * n + 100
    while m > 1
        iter += 1
        iter > maxit && error("bdsqr!: failed to converge")
        @inbounds for i in 1:(m - 1)                      # deflate negligible superdiagonals
            if abs(e[i]) <= tol * (abs(d[i]) + abs(d[i + 1]))
                e[i] = 0.0
            end
        end
        if e[m - 1] == 0.0
            m -= 1
            continue
        end
        l = m - 1                                      # top of the bottom nonzero-e block
        @inbounds while l >= 2 && e[l - 1] != 0.0
            l -= 1
        end
        shift = @inbounds _svd_2x2_smin(d[m - 1], e[m - 1], d[m])
        @inbounds (d[l] == 0.0) && (shift = 0.0)       # avoid /0 in the shift fold
        _bdsqr_sweep!(d, e, l, m, shift, U, V)
    end
    @inbounds for i in 1:n
        d[i] *= mx
    end            # unscale the singular values
    # singular values nonnegative
    @inbounds for i in 1:n
        if d[i] < 0.0
            d[i] = -d[i]
            !isnothing(V) && _rot_cols_negate!(V, i)
        end
    end
    _svd_sort!(d, U, V)                                # descending
    return d
end

# ── Complex bidiagonal SVD: LAPACK {c,z}bdsqr (Demmel–Kahan implicit-zero-shift QR). The bidiagonal B
# is REAL (diag d, off-diag e); the Givens rotations are REAL but accumulate into COMPLEX Vt (right
# rotations on ROWS), U (left rotations on COLUMNS), C (left rotations on ROWS). This is the ROBUST
# sweep (correct on clustered/graded σ) — NOT the simplified real bdsqr! above. Control flow mirrors
# Reference-LAPACK dbdsqr.f (tol>0 relative-accuracy path); reuses _lartg/_lasv2/_qz_safmin (qz.jl) +
# _swap_cols! (above). Validated bit-identical σ to OpenBLAS {c,z}bdsqr across clustered/graded/scaled.

# DLAS2: singular values of the 2×2 upper-triangular [f g; 0 h] (overflow/underflow-safe), for the shift.
@inline function _las2(f::R, g::R, h::R) where {R <: Real}
    fa = abs(f); ga = abs(g); ha = abs(h)
    fhmn = min(fa, ha); fhmx = max(fa, ha)
    if fhmn == zero(R)
        ssmin = zero(R)
        if fhmx == zero(R)
            ssmax = ga
        else
            mn = min(fhmx, ga); mx = max(fhmx, ga)
            ssmax = mx * sqrt(one(R) + (mn / mx)^2)
        end
        return ssmin, ssmax
    end
    if ga < fhmx
        as = one(R) + fhmn / fhmx
        at = (fhmx - fhmn) / fhmx
        au = (ga / fhmx)^2
        c = R(2) / (sqrt(as * as + au) + sqrt(at * at + au))
        return fhmn * c, fhmx / c
    end
    au = fhmx / ga
    if au == zero(R)
        return (fhmn * fhmx) / ga, ga
    end
    as = one(R) + fhmn / fhmx
    at = (fhmx - fhmn) / fhmx
    c = one(R) / (sqrt(one(R) + (as * au)^2) + sqrt(one(R) + (at * au)^2))
    ssmin = (fhmn * c) * au
    return ssmin + ssmin, ga / (c + c)
end

# Real Givens on complex ROWS (i1,i2): [c s; −s c] convention (matches _lartg), dlasr 'L'.
@inline function _rot_rows_cx!(M::AbstractMatrix, i1::Int, i2::Int, c::R, s::R) where {R <: Real}
    @inbounds for j in 1:size(M, 2)
        a = M[i1, j]; b = M[i2, j]
        M[i1, j] = c * a + s * b
        M[i2, j] = c * b - s * a
    end
    return M
end
# Real Givens on complex COLUMNS (j1,j2) — dlasr 'R'.
@inline function _rot_cols_cx!(M::AbstractMatrix, j1::Int, j2::Int, c::R, s::R) where {R <: Real}
    @inbounds for i in 1:size(M, 1)
        a = M[i, j1]; b = M[i, j2]
        M[i, j1] = c * a + s * b
        M[i, j2] = c * b - s * a
    end
    return M
end
@inline function _negate_row_cx!(M::AbstractMatrix, i::Int)
    @inbounds for j in 1:size(M, 2)
        M[i, j] = -M[i, j]
    end
    return M
end
@inline function _swap_rows_cx!(M::AbstractMatrix, i1::Int, i2::Int)
    @inbounds for j in 1:size(M, 2)
        M[i1, j], M[i2, j] = M[i2, j], M[i1, j]
    end
    return M
end

# bdsqr!(uplo, d, e, Vt, U, C) — the {c,z}bdsqr_64_ entry. d,e REAL; Vt (n×ncvt), U (nru×n), C (n×ncc)
# COMPLEX, any may be empty (rotations no-op). On exit d holds σ (≥0, DESCENDING), e destroyed; with
# Vt=U=C=I on entry, B₀ = U·Diagonal(d)·Vt. Returns (d, Vt, U, C).
function bdsqr!(
        uplo::AbstractChar, d::AbstractVector{R}, e::AbstractVector{R},
        Vt::AbstractMatrix{Complex{R}}, U::AbstractMatrix{Complex{R}},
        C::AbstractMatrix{Complex{R}}
    ) where {R <: AbstractFloat}
    n = length(d)
    (uplo == 'U' || uplo == 'L') || throw(ArgumentError("bdsqr!: uplo must be 'U' or 'L'"))
    length(e) >= n - 1 || throw(DimensionMismatch("bdsqr!: e must have length >= n-1"))
    !isempty(Vt) && size(Vt, 1) != n && throw(DimensionMismatch("bdsqr!: size(Vt,1) != n"))
    !isempty(U) && size(U, 2) != n && throw(DimensionMismatch("bdsqr!: size(U,2) != n"))
    !isempty(C) && size(C, 1) != n && throw(DimensionMismatch("bdsqr!: size(C,1) != n"))
    n == 0 && return d, Vt, U, C
    if n == 1
        @inbounds if d[1] < zero(R)
            d[1] = -d[1]; _negate_row_cx!(Vt, 1)
        end
        return d, Vt, U, C
    end
    epsv = eps(R) / 2
    unfl = _qz_safmin(R)
    if uplo == 'L'
        @inbounds for i in 1:(n - 1)
            cs, sn, r = _lartg(d[i], e[i])
            d[i] = r; e[i] = sn * d[i + 1]; d[i + 1] = cs * d[i + 1]
            _rot_cols_cx!(U, i, i + 1, cs, sn)
            _rot_rows_cx!(C, i, i + 1, cs, sn)
        end
    end
    tolmul = max(R(10), min(R(100), epsv^(-R(1) / 8)))
    tol = tolmul * epsv
    smax = zero(R)
    @inbounds for i in 1:n
        smax = max(smax, abs(d[i]))
    end
    @inbounds for i in 1:(n - 1)
        smax = max(smax, abs(e[i]))
    end
    smax == zero(R) && return d, Vt, U, C
    sminoa = abs(@inbounds d[1])
    if sminoa != zero(R)
        mu = sminoa
        @inbounds for i in 2:n
            mu = abs(d[i]) * (mu / (mu + abs(e[i - 1])))
            sminoa = min(sminoa, mu)
            sminoa == zero(R) && break
        end
    end
    sminoa /= sqrt(R(n))
    maxitr = 6
    thresh = max(tol * sminoa, R(maxitr * n) * (R(n) * unfl))
    maxit = maxitr * n * n
    iter = 0
    oldll = -1; oldm = -1; idir = 0
    m = n
    while m > 1
        iter > maxit && error("bdsqr!: the QR iteration failed to converge")
        smax_b = abs(@inbounds d[m]); ll = 0; split = false
        @inbounds for lll in 1:(m - 1)
            llc = m - lll; abse = abs(e[llc])
            if abse <= thresh
                ll = llc; split = true; break
            end
            smax_b = max(smax_b, abs(d[llc]), abse)
        end
        if split
            @inbounds e[ll] = zero(R)
            if ll == m - 1
                m -= 1; continue
            end
        end
        ll += 1
        if ll == m - 1
            @inbounds begin
                sigmn, sigmx, sinr, cosr, sinl, cosl = _lasv2(d[m - 1], e[m - 1], d[m])
                d[m - 1] = sigmx; e[m - 1] = zero(R); d[m] = sigmn
            end
            _rot_rows_cx!(Vt, m - 1, m, cosr, sinr)
            _rot_cols_cx!(U, m - 1, m, cosl, sinl)
            _rot_rows_cx!(C, m - 1, m, cosl, sinl)
            m -= 2; continue
        end
        if ll > oldm || m < oldll
            idir = abs(@inbounds d[ll]) >= abs(@inbounds d[m]) ? 1 : 2
        end
        converged = false; sminl = zero(R)
        if idir == 1
            @inbounds if abs(e[m - 1]) <= tol * abs(d[m])
                e[m - 1] = zero(R); converged = true
            else
                mu = abs(d[ll]); sminl = mu
                for lll in ll:(m - 1)
                    if abs(e[lll]) <= tol * mu
                        e[lll] = zero(R); converged = true; break
                    end
                    mu = abs(d[lll + 1]) * (mu / (mu + abs(e[lll]))); sminl = min(sminl, mu)
                end
            end
        else
            @inbounds if abs(e[ll]) <= tol * abs(d[ll])
                e[ll] = zero(R); converged = true
            else
                mu = abs(d[m]); sminl = mu
                for lll in (m - 1):-1:ll
                    if abs(e[lll]) <= tol * mu
                        e[lll] = zero(R); converged = true; break
                    end
                    mu = abs(d[lll]) * (mu / (mu + abs(e[lll]))); sminl = min(sminl, mu)
                end
            end
        end
        converged && continue
        oldll = ll; oldm = m
        shift = zero(R)
        if !(R(n) * tol * (sminl / smax_b) <= max(epsv, R(1) / 100 * tol))
            @inbounds if idir == 1
                sll = abs(d[ll]); shift, _ = _las2(d[m - 1], e[m - 1], d[m])
            else
                sll = abs(d[m]); shift, _ = _las2(d[ll], e[ll], d[ll + 1])
            end
            sll > zero(R) && (shift / sll)^2 < epsv && (shift = zero(R))
        end
        iter += m - ll
        if shift == zero(R)
            if idir == 1
                cs = one(R); oldcs = one(R); sn = zero(R); oldsn = zero(R)
                @inbounds for i in ll:(m - 1)
                    cs, sn, r = _lartg(d[i] * cs, e[i])
                    i > ll && (e[i - 1] = oldsn * r)
                    oldcs, oldsn, dnew = _lartg(oldcs * r, d[i + 1] * sn)
                    d[i] = dnew
                    _rot_rows_cx!(Vt, i, i + 1, cs, sn)
                    _rot_cols_cx!(U, i, i + 1, oldcs, oldsn)
                    _rot_rows_cx!(C, i, i + 1, oldcs, oldsn)
                end
                @inbounds begin
                    h = d[m] * cs; d[m] = h * oldcs; e[m - 1] = h * oldsn
                    abs(e[m - 1]) <= thresh && (e[m - 1] = zero(R))
                end
            else
                cs = one(R); oldcs = one(R); sn = zero(R); oldsn = zero(R)
                @inbounds for i in m:-1:(ll + 1)
                    cs, sn, r = _lartg(d[i] * cs, e[i - 1])
                    i < m && (e[i] = oldsn * r)
                    oldcs, oldsn, dnew = _lartg(oldcs * r, d[i - 1] * sn)
                    d[i] = dnew
                    _rot_rows_cx!(Vt, i - 1, i, oldcs, -oldsn)
                    _rot_cols_cx!(U, i - 1, i, cs, -sn)
                    _rot_rows_cx!(C, i - 1, i, cs, -sn)
                end
                @inbounds begin
                    h = d[ll] * cs; d[ll] = h * oldcs; e[ll] = h * oldsn
                    abs(e[ll]) <= thresh && (e[ll] = zero(R))
                end
            end
        else
            if idir == 1
                @inbounds begin
                    f = (abs(d[ll]) - shift) * (copysign(one(R), d[ll]) + shift / d[ll]); g = e[ll]
                end
                @inbounds for i in ll:(m - 1)
                    cosr, sinr, r = _lartg(f, g)
                    i > ll && (e[i - 1] = r)
                    f = cosr * d[i] + sinr * e[i]; e[i] = cosr * e[i] - sinr * d[i]
                    g = sinr * d[i + 1]; d[i + 1] = cosr * d[i + 1]
                    cosl, sinl, r = _lartg(f, g)
                    d[i] = r; f = cosl * e[i] + sinl * d[i + 1]; d[i + 1] = cosl * d[i + 1] - sinl * e[i]
                    if i < m - 1
                        g = sinl * e[i + 1]; e[i + 1] = cosl * e[i + 1]
                    end
                    _rot_rows_cx!(Vt, i, i + 1, cosr, sinr)
                    _rot_cols_cx!(U, i, i + 1, cosl, sinl)
                    _rot_rows_cx!(C, i, i + 1, cosl, sinl)
                end
                @inbounds begin
                    e[m - 1] = f; abs(e[m - 1]) <= thresh && (e[m - 1] = zero(R))
                end
            else
                @inbounds begin
                    f = (abs(d[m]) - shift) * (copysign(one(R), d[m]) + shift / d[m]); g = e[m - 1]
                end
                @inbounds for i in m:-1:(ll + 1)
                    cosr, sinr, r = _lartg(f, g)
                    i < m && (e[i] = r)
                    f = cosr * d[i] + sinr * e[i - 1]; e[i - 1] = cosr * e[i - 1] - sinr * d[i]
                    g = sinr * d[i - 1]; d[i - 1] = cosr * d[i - 1]
                    cosl, sinl, r = _lartg(f, g)
                    d[i] = r; f = cosl * e[i - 1] + sinl * d[i - 1]; d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1]
                    if i > ll + 1
                        g = sinl * e[i - 2]; e[i - 2] = cosl * e[i - 2]
                    end
                    _rot_rows_cx!(Vt, i - 1, i, cosl, -sinl)
                    _rot_cols_cx!(U, i - 1, i, cosr, -sinr)
                    _rot_rows_cx!(C, i - 1, i, cosr, -sinr)
                end
                @inbounds begin
                    e[ll] = f; abs(e[ll]) <= thresh && (e[ll] = zero(R))
                end
            end
        end
    end
    @inbounds for i in 1:n
        if d[i] < zero(R)
            d[i] = -d[i]; _negate_row_cx!(Vt, i)
        end
    end
    @inbounds for i in 1:(n - 1)
        k = i
        for j in (i + 1):n
            d[j] > d[k] && (k = j)
        end
        if k != i
            d[i], d[k] = d[k], d[i]
            _swap_rows_cx!(Vt, i, k); _swap_cols!(U, i, k); _swap_rows_cx!(C, i, k)
        end
    end
    return d, Vt, U, C
end

@inline function _rot_cols_negate!(M::AbstractMatrix{Float64}, j::Int)
    return @inbounds for i in 1:size(M, 1)
        M[i, j] = -M[i, j]
    end
end

# Sort singular values descending, permuting U and V columns to match (selection sort: n is the
# matrix dim, swaps are O(n) columns each — negligible vs the O(n³) sweeps). ponytail.
function _svd_sort!(d::AbstractVector{Float64}, U, V)
    n = length(d)
    @inbounds for i in 1:(n - 1)
        kmax = i
        for j in (i + 1):n
            d[j] > d[kmax] && (kmax = j)
        end
        if kmax != i
            d[i], d[kmax] = d[kmax], d[i]
            !isnothing(U) && _swap_cols!(U, i, kmax)
            !isnothing(V) && _swap_cols!(V, i, kmax)
        end
    end
    return d
end

@inline function _swap_cols!(M::AbstractMatrix{T}, j1::Int, j2::Int) where {T <: Number}
    return @inbounds for i in 1:size(M, 1)
        M[i, j1], M[i, j2] = M[i, j2], M[i, j1]
    end
end

# --- Stage 3: driver. A = Q·B·Pᵀ (gebrd) and B = Ub·Σ·Vbᵀ (bdsqr) ⟹ A = (Q·Ub)·Σ·(P·Vb)ᵀ ---------
# Build Q (m×n) and P (n×n) from the stored reflectors, then let bdsqr accumulate Ub,Vb into them.
# Returns (U, S, Vt): U is m×n (thin), S length-n descending, Vt is n×n. ponytail: m<n via transpose.

# Form the thin Q (m×n) = H(1)···H(n) from the left reflectors below A's diagonal. Each reflector is
# applied to the trailing columns with H·C = C − τ·v·(vᵀ·C): one gemv! (vᵀC) + one ger! (rank-1).
function _form_Q!(A::AbstractMatrix{Float64}, tauq::AbstractVector{Float64}, m::Int, n::Int)
    Q = zeros(Float64, m, n)
    @inbounds for i in 1:n
        Q[i, i] = 1.0
    end
    w = Vector{Float64}(undef, n)
    @inbounds for i in n:-1:1
        τ = tauq[i]
        τ == 0.0 && continue
        A[i, i] = 1.0                          # make the implicit reflector 1 explicit (A unused after)
        v = view(A, i:m, i); C = view(Q, i:m, 1:n); wv = view(w, 1:n)
        gemv!(wv, C, v; trans = 'T', alpha = 1.0, beta = 0.0)
        ger!(-τ, v, wv, C)
    end
    return Q
end

# Form P (n×n) = G(1)···G(n-1) from the right reflectors above A's superdiagonal.
function _form_P!(A::AbstractMatrix{Float64}, taup::AbstractVector{Float64}, n::Int)
    P = zeros(Float64, n, n)
    @inbounds for i in 1:n
        P[i, i] = 1.0
    end
    w = Vector{Float64}(undef, n); vb = Vector{Float64}(undef, n)
    @inbounds for i in (n - 1):-1:1
        τ = taup[i]
        τ == 0.0 && continue
        len = n - i                                  # reflector lives in row i, cols i+1:n (v[1]=1)
        vb[1] = 1.0
        for t in 2:len
            vb[t] = A[i, i + t]                         # contiguous copy of the strided row reflector
        end
        v = view(vb, 1:len); C = view(P, (i + 1):n, 1:n); wv = view(w, 1:n)
        gemv!(wv, C, v; trans = 'T', alpha = 1.0, beta = 0.0)
        ger!(-τ, v, wv, C)
    end
    return P
end

# Apply Q = H(1)···H(k) (standard reflectors, columns of Vfull, implicit unit diagonal already made
# explicit; roff = row offset of reflector i's support below its index) to C from the left, in place:
# C := Q·C. Blocked compact-WY (dlarft + dlarfb) driven by PureBLAS gemm! — the BLAS-3 back-transform
# that replaces explicit form-Q/P + combine. Vfull is M×k with the reflector vectors as its columns.
# Compact-WY T/G/W/Y blocks come from the SVD workspace (ws.bt_*, grown in _svd_grow!) — the former per-call
# fresh zeros(nb,nb)+3 allocs cost ~32 KB per gesvd (2 calls), dominating tiny-n SVD. T's region is re-zeroed
# per use below (gemm reads the full Tv, so its strict-lower must be 0).
function _apply_reflectors_left!(
        Vfull::AbstractMatrix{Float64}, tau::AbstractVector{Float64},
        C::AbstractMatrix{Float64}, k::Int, nb::Int, roff::Int, ws::SVDWorkspace{Float64}
    )
    M = size(Vfull, 1); nc = size(C, 2)
    (k == 0 || nc == 0) && return C
    T = view(ws.bt_T, 1:nb, 1:nb); G = view(ws.bt_G, 1:nb, 1:nb)
    W = view(ws.bt_W, 1:nb, 1:nc); Yb = view(ws.bt_Yb, 1:nb, 1:nc)
    @inbounds for j in 1:nb, i in 1:nb
        T[i, j] = 0.0
    end   # gemm reads the full Tv (lower must be 0)
    nblk = cld(k, nb)
    @inbounds for b in nblk:-1:1                          # blocks right-to-left (apply H(k)…H(1))
        pc = (b - 1) * nb + 1
        pb = min(nb, k - pc + 1)
        rs = pc + roff
        Vp = view(Vfull, rs:M, pc:(pc + pb - 1))               # (M-rs+1)×pb, unit lower trapezoid
        Cb = view(C, rs:M, 1:nc)
        Gv = view(G, 1:pb, 1:pb); gemm!(Gv, Vp, Vp; transA = 'T', alpha = true, beta = false)  # G = VᵀV
        Tv = view(T, 1:pb, 1:pb)                          # dlarft (forward, columnwise): T upper-tri
        for c in 1:pb
            tc = tau[pc + c - 1]
            Tv[c, c] = tc
            for ii in 1:(c - 1)
                s = 0.0
                for kk in ii:(c - 1)
                    s = muladd(Tv[ii, kk], Gv[kk, c], s)
                end
                Tv[ii, c] = -tc * s
            end
        end
        Wv = view(W, 1:pb, 1:nc); gemm!(Wv, Vp, Cb; transA = 'T', alpha = true, beta = false)   # W = VᵀC
        Yv = view(Yb, 1:pb, 1:nc); gemm!(Yv, Tv, Wv; alpha = true, beta = false)                # Y = T·W
        gemm!(Cb, Vp, Yv; alpha = -1.0, beta = true)                                            # C −= V·Y
    end
    return C
end

# Trim-safe transpose into a caller buffer B (n×m) ← A (m×n). Hand loop (permutedims isn't guaranteed
# trim-clean and the C-ABI SVD path must be juliac-analyzable). O(mn), negligible vs the O(mn·min) SVD.
function _svd_transpose!(B::AbstractMatrix{Float64}, A::AbstractMatrix{Float64})
    m, n = size(A)
    @inbounds for j in 1:n, i in 1:m
        B[j, i] = A[i, j]
    end
    return B
end
# Allocating variant (used only by the concrete-return C-ABI wrappers, which may allocate their outputs).
_svd_transpose(A::AbstractMatrix{Float64}) = _svd_transpose!(Matrix{Float64}(undef, size(A, 2), size(A, 1)), A)

# Trim-safe CONJUGATE transpose B (n×m) ← Aᴴ (from A m×n). The complex SVD's Vᴴ (and the m<n transpose
# path) need conjugate-transpose, not plain transpose. Real T is a plain transpose (conj is a no-op).
function _svd_ctranspose!(B::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T <: Number}
    m, n = size(A)
    @inbounds for j in 1:n, i in 1:m
        B[j, i] = conj(A[i, j])
    end
    return B
end

# ── in-place SVD core (m ≥ n) ─────────────────────────────────────────────────────────────────────────
# Writes into caller U (m×nu), S (length n), Vt (n×n). ALL scratch comes from ws (grown once up front) ⇒
# 0-alloc steady state. Destroys A. nu = m if full_u&&m>n (form the orthonormal complement of range(A)),
# else n (=min). full_v is handled one level up (transpose), so here Vt is always n×n.
function _gesvd_core!(
        A::AbstractMatrix{Float64}, U::AbstractMatrix{Float64}, S::AbstractVector{Float64},
        Vt::AbstractMatrix{Float64}, ws::SVDWorkspace{Float64}; full_u::Bool = false
    )
    m, n = size(A)
    nu = (full_u && m > n) ? m : n
    _svd_grow!(ws, m, n, nu)
    d = view(ws.d, 1:n); e = view(ws.e, 1:max(n - 1, 0))
    tauq = view(ws.tauq, 1:n); taup = view(ws.taup, 1:n)
    gebrd!(A, d, e, tauq, taup, ws)
    nb = _BT_NB
    # B's left/right singular vectors (Lvec/Rvec, n×n): bdsqr below the crossover (less D&C overhead at
    # small n, like LAPACK gesdd's QR-below-SMLSIZ), divide-and-conquer above. Both write svals into d.
    Lvec = view(ws.Lvec, 1:n, 1:n); Rvec = view(ws.Rvec, 1:n, 1:n)
    if n <= _SVD_DC_CROSS
        fill!(Lvec, 0.0); fill!(Rvec, 0.0)
        @inbounds for i in 1:n
            Lvec[i, i] = 1.0; Rvec[i, i] = 1.0
        end
        bdsqr!(d, e, Lvec, Rvec)                 # B = Lvec·diag(d)·Rvecᵀ
    else
        bdsdc!(d, e, Lvec, Rvec, ws)             # svals→d; Lvec=Vl (left), Rvec=Ul (right)
    end
    s = d
    # U_A = Q·[Lvec 0; 0 I]. Full-U (m>n): trailing bidiagonal rows are zero ⇒ Ub_full = [Lvec 0; 0 I_{m−n}];
    # the extra unit columns pushed through Q become the orthonormal complement of range(A). ws.UApad's row
    # count is the padded ld (+8 on po2 m) so the transA='T' accumulator's column stride doesn't thrash L1.
    UA = view(ws.UApad, 1:m, 1:nu)
    @inbounds for j in 1:nu, i in 1:m
        UA[i, j] = 0.0
    end
    @inbounds for j in 1:n, i in 1:n
        UA[i, j] = Lvec[i, j]
    end
    @inbounds for j in (n + 1):nu
        UA[j, j] = 1.0
    end   # complement unit columns e_{n+1..m}
    VQ = view(ws.VQ, 1:m, 1:n)
    @inbounds for j in 1:n, i in 1:m
        VQ[i, j] = 0.0
    end
    @inbounds for j in 1:n
        VQ[j, j] = 1.0
        for i in (j + 1):m
            VQ[i, j] = A[i, j]
        end
    end
    _apply_reflectors_left!(VQ, tauq, UA, n, nb, 0, ws)   # Q applied over all nu columns
    # V_A = P·Rvec — apply the right (row) reflectors (k=n-1, support offset by 1).
    Vmat = view(ws.Vpad, 1:n, 1:n)
    @inbounds for j in 1:n, i in 1:n
        Vmat[i, j] = Rvec[i, j]
    end
    if n > 1
        VP = view(ws.VP, 1:n, 1:(n - 1))
        @inbounds for j in 1:(n - 1), i in 1:n
            VP[i, j] = 0.0
        end
        @inbounds for j in 1:(n - 1)
            VP[j + 1, j] = 1.0
            for r in (j + 2):n
                VP[r, j] = A[j, r]
            end
        end
        _apply_reflectors_left!(VP, taup, Vmat, n - 1, nb, 1, ws)
    end
    _svd_sort!(s, UA, Vmat)                     # descending; sorts cols 1:n, complement cols n+1:nu untouched
    @inbounds for i in 1:n
        S[i] = s[i]
    end
    @inbounds for j in 1:nu, i in 1:m
        U[i, j] = UA[i, j]
    end
    _svd_transpose!(Vt, Vmat)                   # Vt = V_Aᵀ (n×n)
    return U, S, Vt
end

# In-place SVD values core (m ≥ n): fill S (length n) with the singular values. 0-alloc steady state.
function _svals_core!(A::AbstractMatrix{Float64}, S::AbstractVector{Float64}, ws::SVDWorkspace{Float64})
    m, n = size(A)
    _svd_grow!(ws, m, n, n)
    d = view(ws.d, 1:n); e = view(ws.e, 1:max(n - 1, 0))
    tauq = view(ws.tauq, 1:n); taup = view(ws.taup, 1:n)
    gebrd!(A, d, e, tauq, taup, ws)
    # dqds (dlasq) for singular VALUES — ~1.7× faster than the QR bdsqr!; QR is the rare-failure fallback.
    _dlasq1!(d, e, ws.dqds_Z, ws.dqds_st) != 0 && bdsqr!(d, e, nothing, nothing)
    @inbounds for i in 1:n
        S[i] = d[i]
    end
    return S
end

# ── in-place entries (caller provides the output buffers; 0-alloc steady state) ─────────────────────────
# Full SVD: writes into U (m×nu / m×m), S (min), Vt (ncv×n / n×n). full_u: U's m×m complement (m>n).
# full_v: Vt's n×n complement (n>m) — realized via the transpose path (full_v on A ≡ full_u on Aᵀ).
# m<n: SVD Aᵀ (tall) = Ū·Σ·V̄ᵀ ⟹ A = V̄·Σ·Ūᵀ ⟹ U(A)=V̄=(V̄ᵀ)ᵀ, Vt(A)=Ūᵀ. All staging (Aᵀ, Ū, V̄ᵀ) from ws.
function gesvd!(
        A::AbstractMatrix{Float64}, U::AbstractMatrix{Float64}, S::AbstractVector{Float64},
        Vt::AbstractMatrix{Float64}; full_u::Bool = false, full_v::Bool = false
    )
    m, n = size(A)
    ws = _svdws()
    if m < n
        ws.trbuf = _gm(ws.trbuf, n, m)
        At = view(ws.trbuf, 1:n, 1:m); _svd_transpose!(At, A)
        nU = (full_v && n > m) ? n : m           # Ū columns (full_v(A) ≡ full_u(Aᵀ)); full_u(A) adds nothing
        ws.Usc = _gm(ws.Usc, n, nU); ws.Vtsc = _gm(ws.Vtsc, m, m)
        Usc = view(ws.Usc, 1:n, 1:nU); Vtsc = view(ws.Vtsc, 1:m, 1:m)
        _gesvd_core!(At, Usc, S, Vtsc, ws; full_u = full_v)
        _svd_transpose!(U, Vtsc)                 # U(A) = (V̄ᵀ)ᵀ  (m×m)
        _svd_transpose!(Vt, Usc)                 # Vt(A) = Ūᵀ  (nU×n)
        return U, S, Vt
    end
    _gesvd_core!(A, U, S, Vt, ws; full_u = full_u)
    return U, S, Vt
end

# In-place singular values: fill S (length min(m,n)). 0-alloc steady state.
function gesvd_vals!(A::AbstractMatrix{Float64}, S::AbstractVector{Float64})
    m, n = size(A)
    ws = _svdws()
    if m < n                                      # σ(A) = σ(Aᵀ)
        ws.trbuf = _gm(ws.trbuf, n, m)
        At = view(ws.trbuf, 1:n, 1:m); _svd_transpose!(At, A)
        return _svals_core!(At, S, ws)
    end
    return _svals_core!(A, S, ws)
end

# Full SVD of A (Float64), convenience allocating form. want_vectors=false → (S,); true → (U, S, Vᵀ),
# economy (U m×min, Vᵀ min×n). Allocates the outputs, then calls the 0-alloc in-place entry.
function gesvd!(A::AbstractMatrix{Float64}; want_vectors::Bool = true)
    m, n = size(A); mn = min(m, n)
    if !want_vectors
        S = Vector{Float64}(undef, mn)
        gesvd_vals!(A, S)
        return (S,)
    end
    U = Matrix{Float64}(undef, m, mn)
    S = Vector{Float64}(undef, mn)
    Vt = Matrix{Float64}(undef, mn, n)
    gesvd!(A, U, S, Vt)
    return U, S, Vt
end

# Complex SVD singular VALUES (zgesvd, want_vectors=false): complex bidiagonalization (gebd2!, real d,e) →
# the REUSED real-bidiagonal stage 2 (bdsqr! on real d,e). σ(A)=σ(Aᵀ), so m<n transposes to tall. Vectors
# (the complex back-transform of Q,P onto the real bidiagonal vectors) are the follow-up. ponytail: local
# d,e,τ scratch (values path only); the SVDWorkspace real/complex split lands with the vectors path.
function gesvd_vals!(A::AbstractMatrix{T}, S::AbstractVector{<:Real}) where {T <: BlasComplex}
    m, n = size(A)
    if m < n                                                  # tall via transpose (σ preserved). Explicit copy
        At = Matrix{T}(undef, n, m)                            # into a concrete Matrix — permutedims on a PtrMatrix
        @inbounds for j in 1:n, i in 1:m
            At[j, i] = A[i, j]
        end   # yields a non-trim-safe PermutedDimsArray path.
        return gesvd_vals!(At, S)
    end
    ws = _svdws(T)
    _svd_grow_bidiag!(ws, m, n)
    # Bidiagonal d,e in Float64 unconditionally: bdsqr! is the Float64 implicit-QR core, and the real
    # bidiagonal of a ComplexF32 A is fine to refine in double (σ then rounded back into S's eltype).
    d = zeros(Float64, n); e = zeros(Float64, max(n - 1, 0)); tauq = zeros(T, n); taup = zeros(T, n)
    gebrd!(A, d, e, tauq, taup, ws)                          # blocked zgebrd (zlabrd panels + gemm trailing)
    # dqds (dlasq) singular VALUES on the real bidiagonal; QR bdsqr! is the rare-failure fallback.
    _dlasq1!(d, e, ws.dqds_Z, ws.dqds_st) != 0 && bdsqr!(d, e, nothing, nothing)
    @inbounds for i in 1:n
        S[i] = d[i]
    end
    return S
end
# ── Complex singular-VECTOR back-transform (unblocked, mirrors _unmtr! in eigen.jl) ─────────────────────
# Q = H_1·H_2·⋯·H_n from the LEFT (tauq) reflectors: apply C := Q·C. H_i = I − τq_i·v_i·v_iᴴ acts on rows
# i:m, v_i[1]≡1 (at row i), v_i[2:] = A[i+1:m, i] (stored un-conjugated — the LEFT reflectors get no zlacgv
# dance). ALL n reflectors applied (i=n has a length-1 v with a nonzero phase τ — no real-code τ=0 shortcut,
# per the eigen lesson). trans='N' (form Q, not Qᴴ) ⇒ τ un-conjugated.
function _zapply_Q_left!(
        A::AbstractMatrix{T}, tauq::AbstractVector{T}, C::AbstractMatrix{T},
        m::Int, n::Int
    ) where {T <: Complex}
    v = Vector{T}(undef, m)
    @inbounds for i in n:-1:1
        len = m - i + 1
        v[1] = one(T)
        for r in 2:len
            v[r] = A[i + r - 1, i]
        end
        _house_left!(view(C, i:m, :), view(v, 1:len), tauq[i])
    end
    return C
end

# P = R_1·R_2·⋯·R_{n-1} from the LEFT (taup) reflectors: apply C := P·C. From A = Q·B·Pᴴ, P = R_1…R_{n-1}
# with R_i = I − τp_i·w_i·w_iᴴ acting on the col-index space i+1:n (rows i+1:n of C), w_i[1]≡1 (col i+1),
# w_i[2:] = conj(A[i, i+2:n]). The CONJUGATION on w (vs the real _form_P!, which uses A[i,i+t] directly) is
# the P-side subtlety: zgebd2 reduces the CONJUGATED row (zlacgv), so the reflector applied from the right
# during factorization is w_i = conj(stored essential); forming P from the left re-uses that same w_i. τ is
# un-conjugated (form P, not Pᴴ). Derivation cross-checked against the known-correct Q side + verified
# numerically (reconstruction ‖A−UΣVᴴ‖ at machine eps across scaled/rank-deficient cases). i=n−1 has a
# length-1 w (phase-only) — applied, no τ=0 shortcut.
function _zapply_P_left!(
        A::AbstractMatrix{T}, taup::AbstractVector{T}, C::AbstractMatrix{T},
        n::Int
    ) where {T <: Complex}
    n <= 1 && return C
    v = Vector{T}(undef, n)
    @inbounds for i in (n - 1):-1:1
        len = n - i
        v[1] = one(T)
        for t in 2:len
            v[t] = conj(A[i, i + t])
        end
        _house_left!(view(C, (i + 1):n, :), view(v, 1:len), taup[i])
    end
    return C
end

# ── Complex in-place SVD core (m ≥ n), mirrors _gesvd_core! (the _heev! pattern for SVD) ────────────────
# gebrd! (complex; real d,e + complex tauq/taup) → REAL bidiagonal SVD (bdsqr!/bdsdc! into real Lvec/Rvec)
# → realify into complex U/V staging → apply complex Q (tauq) / P (taup) back-transforms → Vᴴ = conjugate
# transpose. B is REAL, so its singular-vector blocks are REAL scratch (allocated here — perf/0-alloc is the
# follow-up). nu = m when full_u&&m>n (orthonormal complement of range(A)), else n.
function _zgesvd_core!(
        A::AbstractMatrix{T}, U::AbstractMatrix{T}, S::AbstractVector{R},
        Vt::AbstractMatrix{T}, ws::SVDWorkspace{T}; full_u::Bool = false
    ) where {T <: Complex, R <: Real}
    m, n = size(A)
    nu = (full_u && m > n) ? m : n
    _svd_grow_bidiag!(ws, m, n)
    d = zeros(Float64, n); e = zeros(Float64, max(n - 1, 0))
    tauq = Vector{T}(undef, n); taup = Vector{T}(undef, n)
    gebrd!(A, d, e, tauq, taup, ws)                     # complex bidiagonalization (real d,e; complex τ)
    # REAL bidiagonal SVD: B = Lvec·diag(d)·Rvecᵀ (bdsqr below the crossover, D&C above — mirror the real
    # core). Lvec/Rvec are REAL (B is real bidiagonal); the D&C reuses the Float64 global workspace.
    Lvec = Matrix{Float64}(undef, n, n); Rvec = Matrix{Float64}(undef, n, n)
    if n <= _SVD_DC_CROSS
        fill!(Lvec, 0.0); fill!(Rvec, 0.0)
        @inbounds for i in 1:n
            Lvec[i, i] = 1.0; Rvec[i, i] = 1.0
        end
        bdsqr!(d, e, Lvec, Rvec)
    else
        rws = _svdws()                                  # grow the Float64 D&C staging on the real workspace
        rws.dc_diag = _gv(rws.dc_diag, n); rws.dc_subdiag = _gv(rws.dc_subdiag, n)
        rws.dc_U = _gm(rws.dc_U, n + 1, n + 1)
        bdsdc!(d, e, Lvec, Rvec, rws)
    end
    s = d
    # U_A = Q·[Lvec 0; 0 I]: realify Lvec into the complex staging, push the n+1:nu unit columns through Q
    # (they become the orthonormal complement of range(A) when full_u & m>n).
    UA = Matrix{T}(undef, m, nu)
    fill!(UA, zero(T))
    @inbounds for j in 1:n, i in 1:n
        UA[i, j] = T(Lvec[i, j])
    end
    @inbounds for j in (n + 1):nu
        UA[j, j] = one(T)
    end
    _zapply_Q_left!(A, tauq, UA, m, n)
    # V_A = P·Rvec (realify Rvec, apply the right reflectors).
    VA = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        VA[i, j] = T(Rvec[i, j])
    end
    _zapply_P_left!(A, taup, VA, n)
    _svd_sort!(s, UA, VA)                                # descending; complement cols n+1:nu untouched
    @inbounds for i in 1:n
        S[i] = R(s[i])
    end
    @inbounds for j in 1:nu, i in 1:m
        U[i, j] = UA[i, j]
    end
    _svd_ctranspose!(Vt, VA)                             # Vt = V_Aᴴ (CONJUGATE transpose — n×n)
    return U, S, Vt
end

# Complex full SVD (in-place; caller provides U/S/Vt). m<n via the CONJUGATE transpose: SVD(Aᴴ)=Ū·Σ·V̄ᴴ ⟹
# A=V̄·Σ·Ūᴴ ⟹ U(A)=V̄=(V̄ᴴ)ᴴ, Vᴴ(A)=Ūᴴ (σ preserved). full_u/full_v mirror the real path (full_v(A)≡full_u(Aᴴ)).
function gesvd!(
        A::AbstractMatrix{T}, U::AbstractMatrix{T}, S::AbstractVector{<:Real},
        Vt::AbstractMatrix{T}; full_u::Bool = false, full_v::Bool = false
    ) where {T <: Complex}
    m, n = size(A)
    ws = _svdws(T)
    if m < n
        At = Matrix{T}(undef, n, m); _svd_ctranspose!(At, A)     # At = Aᴴ (tall, n>m)
        nU = (full_v && n > m) ? n : m
        Usc = Matrix{T}(undef, n, nU); Vtsc = Matrix{T}(undef, m, m)
        _zgesvd_core!(At, Usc, S, Vtsc, ws; full_u = full_v)
        _svd_ctranspose!(U, Vtsc)                                # U(A) = V̄ = (V̄ᴴ)ᴴ  (m×m)
        _svd_ctranspose!(Vt, Usc)                                # Vt(A) = Ūᴴ  (nU×n)
        return U, S, Vt
    end
    _zgesvd_core!(A, U, S, Vt, ws; full_u = full_u)
    return U, S, Vt
end

# Complex full SVD (zgesvd), convenience allocating form. want_vectors=false → (S,); true → (U, S, Vᴴ),
# economy (U m×min, Vᴴ min×n). Mirrors the Float64 convenience entry.
function gesvd!(A::AbstractMatrix{T}; want_vectors::Bool = true) where {T <: BlasComplex}
    m, n = size(A); mn = min(m, n)
    if !want_vectors
        S = Vector{real(T)}(undef, mn)
        gesvd_vals!(A, S)                                        # destructive (mirrors the real path)
        return (S,)
    end
    U = Matrix{T}(undef, m, mn)
    S = Vector{real(T)}(undef, mn)
    Vt = Matrix{T}(undef, mn, n)
    gesvd!(A, U, S, Vt)
    return U, S, Vt
end
