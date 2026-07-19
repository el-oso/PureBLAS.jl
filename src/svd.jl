# LAPACK SVD (gesvd) — pure Julia, built on PureBLAS blocks. Three layers (ROADMAP):
#   1. gebrd  — two-sided Householder bidiagonalization  A = Q·B·Pᵀ  (B upper-bidiagonal, m≥n).
#   2. bdsqr  — implicit-shift QR on the bidiagonal B (Golub-Kahan), accumulating Givens into U,Vᵀ.
#   3. driver — form Q,Pᵀ from the reflectors and back-transform the bidiagonal singular vectors.
# Float64 path. Householder = standard LAPACK convention (H = I − τ·v·vᵀ, v[1]=1 implicit) so the
# back-transform is self-contained. ponytail: m<n handled by transposing; generic/AD SVD deferred.

# --- Householder generator (LAPACK dlarfg) on a strided segment ---------------------------------
# x = [α; tail]. Returns (β, τ): the reflector H = I − τ·v·vᵀ with v = [1; x[2:]/(α−β)] zeros the
# tail, leaving β at x[1]. On return x[2:] holds the essential v; x[1] is left to the caller.
@inline function _larfg!(x::AbstractVector{T}) where {T<:Real}
    n = length(x)
    @inbounds begin
        α = x[1]
        n == 1 && return α, zero(T)
        ss = zero(T)                                 # SIMD sum-of-squares (avoids O(n²) Base.hypot in gebrd);
        @simd for i in 2:n; ss = muladd(x[i], x[i], ss); end   # fast path — exact for the common regime.
        xnorm = sqrt(ss)
        # Scaled recompute when the fast ss over/underflowed. The underflow case matters: when x is NORMAL
        # but its squares are DENORMAL (|x| ≲ √floatmin), ss < floatmin loses mantissa bits and the
        # reflector loses orthogonality (dlarfg guards this; same principle as req#6 nrm2/lassq). The
        # common path (ss finite, ≥ floatmin) is UNCHANGED — this only reroutes the extreme-scale tails.
        if !isfinite(xnorm) || ss < floatmin(T)
            scale = zero(T)
            for i in 2:n; scale = max(scale, abs(x[i])); end
            scale == zero(T) && return α, zero(T)
            ssum = zero(T)
            for i in 2:n; t = x[i] / scale; ssum = muladd(t, t, ssum); end
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
@inline function _house_left!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T<:Real}
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
function gebd2!(A::AbstractMatrix{Float64}, d::AbstractVector{Float64}, e::AbstractVector{Float64},
        tauq::AbstractVector{Float64}, taup::AbstractVector{Float64})
    m, n = size(A)
    m >= n || throw(ArgumentError("gebd2!: requires m ≥ n (got $m×$n)"))
    @inbounds for i in 1:n
        xq = view(A, i:m, i)                      # left reflector zeros A[i+1:m, i]
        β, τq = _larfg!(xq)
        d[i] = β; tauq[i] = τq
        if i < n
            _house_left!(view(A, i:m, i+1:n), xq, τq)   # xq[1] treated as 1
        end
        A[i, i] = β
        if i < n
            xp = view(A, i, i+1:n)                # right reflector zeros A[i, i+2:n]
            β2, τp = _larfg!(xp)
            e[i] = β2; taup[i] = τp
            if i < m
                _house_right!(view(A, i+1:m, i+1:n), xp, τp)
            end
            A[i, i+1] = β2
        else
            taup[i] = 0.0
        end
    end
    return A
end

# ── Complex bidiagonalization (LAPACK zgebd2): real d,e — τ complex, β real. Left applies conj(τq); the
# right reflector operates on the zlacgv-CONJUGATED row (the phase dance that keeps e real). WIP-debug.
@inline function _larfg!(x::AbstractVector{T}) where {T<:Complex}
    R = real(T); n = length(x)
    @inbounds begin
        α = x[1]
        ss = zero(R); for i in 2:n; ss += abs2(x[i]); end
        xnorm = sqrt(ss)
        # Scaled recompute when the naive Σ|xᵢ|² over/underflowed. The underflow case is real: when x is
        # NORMAL but its squares are DENORMAL (|x| ≲ √floatmin), ss < floatmin loses mantissa bits and the
        # reflector goes non-unitary — mirrors the real _larfg! guard (req#6-analogous). This method is shared
        # by the complex SVD path (zgebd2/zgesvd), so the fix reaches there too. Common path UNCHANGED.
        if !isfinite(xnorm) || (ss < floatmin(R) && n > 1)
            scale = zero(R)
            for i in 2:n; scale = max(scale, abs(x[i])); end
            if scale != zero(R)
                ssum = zero(R)
                for i in 2:n; t = abs(x[i]) / scale; ssum = muladd(t, t, ssum); end
                xnorm = scale * sqrt(ssum)
            end
        end
        # n==1 (empty tail) with imag(α)≠0 still needs the phase rotation (τ≠0, β real) — do NOT early-return
        # on n==1; only the genuinely-trivial α (xnorm=0 AND real α) is τ=0.
        (xnorm == 0 && imag(α) == 0) && return real(α), zero(T)
        β = -copysign(hypot(abs(α), xnorm), real(α))
        τ = T((β - real(α)) / β, -imag(α) / β)
        s = one(T) / (α - β)
        for i in 2:n; x[i] *= s; end
    end
    return β, τ
end
@inline function _house_left!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T<:Complex}
    iszero(τ) && return C
    len, nc = size(C)
    @inbounds for j in 1:nc
        w = C[1, j]; for i in 2:len; w += conj(v[i]) * C[i, j]; end
        w *= τ; C[1, j] -= w
        for i in 2:len; C[i, j] -= v[i] * w; end
    end
    return C
end
@inline function _house_right!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T<:Complex}
    iszero(τ) && return C
    nr, len = size(C)
    @inbounds for i in 1:nr
        w = C[i, 1]; for j in 2:len; w += C[i, j] * v[j]; end
        w *= τ; C[i, 1] -= w
        for j in 2:len; C[i, j] -= w * conj(v[j]); end
    end
    return C
end
function gebd2!(A::AbstractMatrix{T}, d::AbstractVector{R}, e::AbstractVector{R},
        tauq::AbstractVector{T}, taup::AbstractVector{T}) where {T<:Complex, R<:Real}
    m, n = size(A)
    m >= n || throw(ArgumentError("gebd2!: requires m ≥ n (got $m×$n)"))
    @inbounds for i in 1:n
        β, τq = _larfg!(view(A, i:m, i))
        d[i] = β; tauq[i] = τq
        i < n && _house_left!(view(A, i:m, i+1:n), view(A, i:m, i), conj(τq))
        A[i, i] = β
        if i < n
            xp = view(A, i, i+1:n)
            for j in eachindex(xp); xp[j] = conj(xp[j]); end
            β2, τp = _larfg!(xp); e[i] = β2; taup[i] = τp
            i < m && _house_right!(view(A, i+1:m, i+1:n), xp, τp)
            for j in eachindex(xp); xp[j] = conj(xp[j]); end
            A[i, i+1] = β2
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
function _labrd!(As::AbstractMatrix{Float64}, d, e, tauq, taup, X, Y, nb::Int,
        arow::AbstractVector{Float64}, tmp::AbstractVector{Float64})
    mm, nn = size(As)
    @inbounds for i in 1:nb
        if i > 1
            for t in 1:i-1; tmp[t] = Y[i, t]; end                       # Y[i,1:i-1] strided → contiguous
            _lg!(view(As, i:mm, i), view(As, i:mm, 1:i-1), view(tmp, 1:i-1), -1.0, 1.0, false)
            _lg!(view(As, i:mm, i), view(X, i:mm, 1:i-1), view(As, 1:i-1, i), -1.0, 1.0, false)
        end
        β, τq = _larfg!(view(As, i:mm, i))
        d[i] = β; tauq[i] = τq; As[i, i] = 1.0
        if i < nn
            L = nn - i
            _lg!(view(Y, i+1:nn, i), view(As, i:mm, i+1:nn), view(As, i:mm, i), 1.0, 0.0, true)
            if i > 1
                _lg!(view(Y, 1:i-1, i), view(As, i:mm, 1:i-1), view(As, i:mm, i), 1.0, 0.0, true)
                _lg!(view(Y, i+1:nn, i), view(Y, i+1:nn, 1:i-1), view(Y, 1:i-1, i), -1.0, 1.0, false)
                _lg!(view(Y, 1:i-1, i), view(X, i:mm, 1:i-1), view(As, i:mm, i), 1.0, 0.0, true)
                _lg!(view(Y, i+1:nn, i), view(As, 1:i-1, i+1:nn), view(Y, 1:i-1, i), -1.0, 1.0, true)
            end
            for r in i+1:nn
                Y[r, i] *= τq
            end
            # Update the active row A(i, i+1:nn) entirely in a contiguous buffer.
            for t in 1:L; arow[t] = As[i, i+t]; end
            for t in 1:i; tmp[t] = As[i, t]; end                        # As[i,1:i] strided → contiguous
            _lg!(view(arow, 1:L), view(Y, i+1:nn, 1:i), view(tmp, 1:i), -1.0, 1.0, false)
            if i > 1
                for t in 1:i-1; tmp[t] = X[i, t]; end                   # X[i,1:i-1] strided → contiguous
                _lg!(view(arow, 1:L), view(As, 1:i-1, i+1:nn), view(tmp, 1:i-1), -1.0, 1.0, true)
            end
            β2, τp = _larfg!(view(arow, 1:L))
            e[i] = β2; taup[i] = τp; arow[1] = 1.0                       # arow now = the reflector v (v[1]=1)
            As[i, i+1] = 1.0
            for t in 2:L; As[i, i+t] = arow[t]; end
            _lg!(view(X, i+1:mm, i), view(As, i+1:mm, i+1:nn), view(arow, 1:L), 1.0, 0.0, false)
            _lg!(view(X, 1:i, i), view(Y, i+1:nn, 1:i), view(arow, 1:L), 1.0, 0.0, true)
            _lg!(view(X, i+1:mm, i), view(As, i+1:mm, 1:i), view(X, 1:i, i), -1.0, 1.0, false)
            if i > 1
                _lg!(view(X, 1:i-1, i), view(As, 1:i-1, i+1:nn), view(arow, 1:L), 1.0, 0.0, false)
                _lg!(view(X, i+1:mm, i), view(X, i+1:mm, 1:i-1), view(X, 1:i-1, i), -1.0, 1.0, false)
            end
            for r in i+1:mm
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
@inline _lgc!(yv, Av, xv, α::T, β::T, mode::Int) where {T<:Complex} =
    _gemv!(mode != 0, mode == 2, size(Av, 1), size(Av, 2), α, Av, xv, 1, β, yv, 1)

function _labrd!(As::AbstractMatrix{T}, d::AbstractVector{R}, e::AbstractVector{R}, tauq, taup, X, Y,
        nb::Int, arow::AbstractVector{T}, tmp::AbstractVector{T}) where {T<:Complex, R<:Real}
    mm, nn = size(As)
    o = one(T); z = zero(T)
    @inbounds for i in 1:nb
        if i > 1
            for t in 1:i-1; tmp[t] = conj(Y[i, t]); end                 # zlacgv Y(i,1:i-1)
            _lgc!(view(As, i:mm, i), view(As, i:mm, 1:i-1), view(tmp, 1:i-1), -o, o, 0)
            _lgc!(view(As, i:mm, i), view(X, i:mm, 1:i-1), view(As, 1:i-1, i), -o, o, 0)
        end
        β, τq = _larfg!(view(As, i:mm, i))
        d[i] = β; tauq[i] = τq; As[i, i] = o
        if i < nn
            L = nn - i
            _lgc!(view(Y, i+1:nn, i), view(As, i:mm, i+1:nn), view(As, i:mm, i), o, z, 2)   # Aᴴ·v
            if i > 1
                _lgc!(view(Y, 1:i-1, i), view(As, i:mm, 1:i-1), view(As, i:mm, i), o, z, 2)
                _lgc!(view(Y, i+1:nn, i), view(Y, i+1:nn, 1:i-1), view(Y, 1:i-1, i), -o, o, 0)
                _lgc!(view(Y, 1:i-1, i), view(X, i:mm, 1:i-1), view(As, i:mm, i), o, z, 2)
                _lgc!(view(Y, i+1:nn, i), view(As, 1:i-1, i+1:nn), view(Y, 1:i-1, i), -o, o, 2)
            end
            for r in i+1:nn; Y[r, i] *= τq; end
            # Update the active row in the CONJUGATED domain: arow ← conj(A(i,i+1:nn)).
            for t in 1:L; arow[t] = conj(As[i, i+t]); end
            for t in 1:i; tmp[t] = conj(As[i, t]); end                  # zlacgv A(i,1:i)
            _lgc!(view(arow, 1:L), view(Y, i+1:nn, 1:i), view(tmp, 1:i), -o, o, 0)
            if i > 1
                for t in 1:i-1; tmp[t] = conj(X[i, t]); end             # zlacgv X(i,1:i-1)
                _lgc!(view(arow, 1:L), view(As, 1:i-1, i+1:nn), view(tmp, 1:i-1), -o, o, 2)
            end
            β2, τp = _larfg!(view(arow, 1:L))
            e[i] = β2; taup[i] = τp; arow[1] = o                        # arow = conj-domain reflector v (v[1]=1)
            As[i, i+1] = o
            for t in 2:L; As[i, i+t] = conj(arow[t]); end               # store the un-conjugated essential v
            _lgc!(view(X, i+1:mm, i), view(As, i+1:mm, i+1:nn), view(arow, 1:L), o, z, 0)
            _lgc!(view(X, 1:i, i), view(Y, i+1:nn, 1:i), view(arow, 1:L), o, z, 2)
            _lgc!(view(X, i+1:mm, i), view(As, i+1:mm, 1:i), view(X, 1:i, i), -o, o, 0)
            if i > 1
                _lgc!(view(X, 1:i-1, i), view(As, 1:i-1, i+1:nn), view(arow, 1:L), o, z, 0)
                _lgc!(view(X, i+1:mm, i), view(X, i+1:mm, 1:i-1), view(X, 1:i-1, i), -o, o, 0)
            end
            for r in i+1:mm; X[r, i] *= τp; end
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
const _SVD_DC_CROSS = 96    # vectors: bdsqr (QR) at/below this n, divide-and-conquer above. Measured boost-
                            # locked (bench/gesvd_cross.jl, PB/OB, want_vectors): within the bdsqr regime the
                            # ratio DEGRADES as n grows (n=48 1.42 → n=128 1.04 vs OB's gesdd) — bdsqr's QR-
                            # iteration loses ground to OB's D&C. bdsqr still wins n≤96 (1.13-1.42 > D&C's
                            # 1.10-1.22), but D&C wins n≥112 (1.09-1.15 > bdsqr's 1.04). The old 128 kept
                            # n=112/128 on the degrading bdsqr tail → a visible KINK at n=128 (all 3 boxes
                            # ~1.0-1.11). Crossing at 96 puts n≥112 on D&C: n=128 1.035→1.146, curve smooth.
                            # (Algorithm-intrinsic QR-vs-D&C crossover, like LAPACK gesdd's SMLSIZ — not a
                            # cache-derived block size; fleet-validated single value.)

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
    SVDWorkspace{T}(ev(), ev(), ev(), ev(), em(), em(), ev(), ev(), em(), em(),
        ev(), ev(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(), em(),
        Float64[], _DqdsState())
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
function gebrd!(A::AbstractMatrix{Float64}, d::AbstractVector{Float64}, e::AbstractVector{Float64},
        tauq::AbstractVector{Float64}, taup::AbstractVector{Float64}, ws::SVDWorkspace{Float64};
        nb::Int = _BRD_NB)
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
        di = view(d, i:k); ei = view(e, i:k-1); tqi = view(tauq, i:k); tpi = view(taup, i:k)
        _labrd!(As, di, ei, tqi, tpi, view(X, 1:mm, 1:nb), view(Y, 1:nn, 1:nb), nb,
            view(ws.labrd_arow, 1:nn), view(ws.labrd_tmp, 1:nb))
        # trailing update A[i+nb:m, i+nb:n] −= V·Yₜᵀ + Xₜ·Ar
        if i + nb <= k
            tr = view(A, i+nb:m, i+nb:n)
            gemm!(tr, view(A, i+nb:m, i:i+nb-1), view(Y, nb+1:nn, 1:nb); transB = 'T', alpha = -1.0, beta = 1.0)
            gemm!(tr, view(X, nb+1:mm, 1:nb), view(A, i:i+nb-1, i+nb:n); alpha = -1.0, beta = 1.0)
        end
        for j in i:i+nb-1                      # restore the panel's diagonal/superdiagonal
            A[j, j] = d[j]
            A[j, j+1] = e[j]
        end
        i += nb
    end
    if i <= k                                  # unblocked tail
        gebd2!(view(A, i:m, i:n), view(d, i:k), view(e, i:k-1), view(tauq, i:k), view(taup, i:k))
    end
    return A
end

# Complex blocked bidiagonalization driver (LAPACK zgebrd): zlabrd panels + two gemm trailing updates
# (the first with transB='C' — the block update is A −= V·Yᴴ − X·Uᴴ), unblocked zgebd2! tail. m ≥ n.
function gebrd!(A::AbstractMatrix{T}, d::AbstractVector{R}, e::AbstractVector{R}, tauq::AbstractVector{T},
        taup::AbstractVector{T}, ws::SVDWorkspace{T}; nb::Int = _BRD_NB) where {T<:Complex, R<:Real}
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
        _labrd!(As, view(d, i:k), view(e, i:k-1), view(tauq, i:k), view(taup, i:k),
            view(X, 1:mm, 1:nb), view(Y, 1:nn, 1:nb), nb,
            view(ws.labrd_arow, 1:nn), view(ws.labrd_tmp, 1:nb))
        if i + nb <= k
            tr = view(A, i+nb:m, i+nb:n)
            gemm!(tr, view(A, i+nb:m, i:i+nb-1), view(Y, nb+1:nn, 1:nb); transB = 'C', alpha = -o, beta = o)
            gemm!(tr, view(X, nb+1:mm, 1:nb), view(A, i:i+nb-1, i+nb:n); alpha = -o, beta = o)
        end
        for j in i:i+nb-1
            A[j, j] = d[j]
            A[j, j+1] = e[j]
        end
        i += nb
    end
    if i <= k
        gebd2!(view(A, i:m, i:n), view(d, i:k), view(e, i:k-1), view(tauq, i:k), view(taup, i:k))
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
@inline function _givens(f::T, g::T) where {T<:Real}
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
@inline function _rot_cols!(M::AbstractMatrix{T}, j1::Int, j2::Int, c::T, s::T) where {T<:Real}
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
function _bdsqr_sweep!(d::AbstractVector{Float64}, e::AbstractVector{Float64}, l::Int, u::Int,
        shift::Float64, U, V)
    @inbounds begin
        f = shift == 0.0 ? d[l] : (d[l] - shift) * (sign(d[l]) + shift / d[l])
        g = e[l]
        for k in l:u-1
            c, s, r = _givens(f, g)                  # right rotation (cols k,k+1)
            k > l && (e[k-1] = r)
            f      = c * d[k]   + s * e[k]
            e[k]   = c * e[k]   - s * d[k]
            g      = s * d[k+1]
            d[k+1] = c * d[k+1]
            !isnothing(V) && _rot_cols!(V, k, k+1, c, s)
            c, s, r = _givens(f, g)                  # left rotation (rows k,k+1)
            d[k]   = r
            f      = c * e[k]   + s * d[k+1]
            d[k+1] = c * d[k+1] - s * e[k]
            if k < u - 1
                g      = s * e[k+1]
                e[k+1] = c * e[k+1]
            end
            e[k] = f
            !isnothing(U) && _rot_cols!(U, k, k+1, c, s)
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
    @inbounds for i in 1:n; mx = max(mx, abs(d[i])); end
    @inbounds for i in 1:n-1; mx = max(mx, abs(e[i])); end
    mx == 0.0 && return d
    minv = 1.0 / mx
    @inbounds for i in 1:n; d[i] *= minv; end
    @inbounds for i in 1:n-1; e[i] *= minv; end
    tol = 8.0 * eps(Float64)
    m = n
    iter = 0; maxit = 12 * n * n + 100
    while m > 1
        iter += 1
        iter > maxit && error("bdsqr!: failed to converge")
        @inbounds for i in 1:m-1                      # deflate negligible superdiagonals
            if abs(e[i]) <= tol * (abs(d[i]) + abs(d[i+1]))
                e[i] = 0.0
            end
        end
        if e[m-1] == 0.0
            m -= 1
            continue
        end
        l = m - 1                                      # top of the bottom nonzero-e block
        @inbounds while l >= 2 && e[l-1] != 0.0
            l -= 1
        end
        shift = @inbounds _svd_2x2_smin(d[m-1], e[m-1], d[m])
        @inbounds (d[l] == 0.0) && (shift = 0.0)       # avoid /0 in the shift fold
        _bdsqr_sweep!(d, e, l, m, shift, U, V)
    end
    @inbounds for i in 1:n; d[i] *= mx; end            # unscale the singular values
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

@inline function _rot_cols_negate!(M::AbstractMatrix{Float64}, j::Int)
    @inbounds for i in 1:size(M, 1)
        M[i, j] = -M[i, j]
    end
end

# Sort singular values descending, permuting U and V columns to match (selection sort: n is the
# matrix dim, swaps are O(n) columns each — negligible vs the O(n³) sweeps). ponytail.
function _svd_sort!(d::AbstractVector{Float64}, U, V)
    n = length(d)
    @inbounds for i in 1:n-1
        kmax = i
        for j in i+1:n
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

@inline function _swap_cols!(M::AbstractMatrix{T}, j1::Int, j2::Int) where {T<:Real}
    @inbounds for i in 1:size(M, 1)
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
    @inbounds for i in n-1:-1:1
        τ = taup[i]
        τ == 0.0 && continue
        len = n - i                                  # reflector lives in row i, cols i+1:n (v[1]=1)
        vb[1] = 1.0
        for t in 2:len
            vb[t] = A[i, i+t]                         # contiguous copy of the strided row reflector
        end
        v = view(vb, 1:len); C = view(P, i+1:n, 1:n); wv = view(w, 1:n)
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
function _apply_reflectors_left!(Vfull::AbstractMatrix{Float64}, tau::AbstractVector{Float64},
        C::AbstractMatrix{Float64}, k::Int, nb::Int, roff::Int, ws::SVDWorkspace{Float64})
    M = size(Vfull, 1); nc = size(C, 2)
    (k == 0 || nc == 0) && return C
    T = view(ws.bt_T, 1:nb, 1:nb); G = view(ws.bt_G, 1:nb, 1:nb)
    W = view(ws.bt_W, 1:nb, 1:nc); Yb = view(ws.bt_Yb, 1:nb, 1:nc)
    @inbounds for j in 1:nb, i in 1:nb; T[i, j] = 0.0; end   # gemm reads the full Tv (lower must be 0)
    nblk = cld(k, nb)
    @inbounds for b in nblk:-1:1                          # blocks right-to-left (apply H(k)…H(1))
        pc = (b - 1) * nb + 1
        pb = min(nb, k - pc + 1)
        rs = pc + roff
        Vp = view(Vfull, rs:M, pc:pc+pb-1)               # (M-rs+1)×pb, unit lower trapezoid
        Cb = view(C, rs:M, 1:nc)
        Gv = view(G, 1:pb, 1:pb); gemm!(Gv, Vp, Vp; transA = 'T', alpha = true, beta = false)  # G = VᵀV
        Tv = view(T, 1:pb, 1:pb)                          # dlarft (forward, columnwise): T upper-tri
        for c in 1:pb
            tc = tau[pc+c-1]
            Tv[c, c] = tc
            for ii in 1:c-1
                s = 0.0
                for kk in ii:c-1; s = muladd(Tv[ii, kk], Gv[kk, c], s); end
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

# ── in-place SVD core (m ≥ n) ─────────────────────────────────────────────────────────────────────────
# Writes into caller U (m×nu), S (length n), Vt (n×n). ALL scratch comes from ws (grown once up front) ⇒
# 0-alloc steady state. Destroys A. nu = m if full_u&&m>n (form the orthonormal complement of range(A)),
# else n (=min). full_v is handled one level up (transpose), so here Vt is always n×n.
function _gesvd_core!(A::AbstractMatrix{Float64}, U::AbstractMatrix{Float64}, S::AbstractVector{Float64},
        Vt::AbstractMatrix{Float64}, ws::SVDWorkspace{Float64}; full_u::Bool = false)
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
        @inbounds for i in 1:n; Lvec[i, i] = 1.0; Rvec[i, i] = 1.0; end
        bdsqr!(d, e, Lvec, Rvec)                 # B = Lvec·diag(d)·Rvecᵀ
    else
        bdsdc!(d, e, Lvec, Rvec, ws)             # svals→d; Lvec=Vl (left), Rvec=Ul (right)
    end
    s = d
    # U_A = Q·[Lvec 0; 0 I]. Full-U (m>n): trailing bidiagonal rows are zero ⇒ Ub_full = [Lvec 0; 0 I_{m−n}];
    # the extra unit columns pushed through Q become the orthonormal complement of range(A). ws.UApad's row
    # count is the padded ld (+8 on po2 m) so the transA='T' accumulator's column stride doesn't thrash L1.
    UA = view(ws.UApad, 1:m, 1:nu)
    @inbounds for j in 1:nu, i in 1:m; UA[i, j] = 0.0; end
    @inbounds for j in 1:n, i in 1:n; UA[i, j] = Lvec[i, j]; end
    @inbounds for j in n+1:nu; UA[j, j] = 1.0; end   # complement unit columns e_{n+1..m}
    VQ = view(ws.VQ, 1:m, 1:n)
    @inbounds for j in 1:n, i in 1:m; VQ[i, j] = 0.0; end
    @inbounds for j in 1:n
        VQ[j, j] = 1.0
        for i in j+1:m; VQ[i, j] = A[i, j]; end
    end
    _apply_reflectors_left!(VQ, tauq, UA, n, nb, 0, ws)   # Q applied over all nu columns
    # V_A = P·Rvec — apply the right (row) reflectors (k=n-1, support offset by 1).
    Vmat = view(ws.Vpad, 1:n, 1:n)
    @inbounds for j in 1:n, i in 1:n; Vmat[i, j] = Rvec[i, j]; end
    if n > 1
        VP = view(ws.VP, 1:n, 1:n-1)
        @inbounds for j in 1:n-1, i in 1:n; VP[i, j] = 0.0; end
        @inbounds for j in 1:n-1
            VP[j+1, j] = 1.0
            for r in j+2:n; VP[r, j] = A[j, r]; end
        end
        _apply_reflectors_left!(VP, taup, Vmat, n - 1, nb, 1, ws)
    end
    _svd_sort!(s, UA, Vmat)                     # descending; sorts cols 1:n, complement cols n+1:nu untouched
    @inbounds for i in 1:n; S[i] = s[i]; end
    @inbounds for j in 1:nu, i in 1:m; U[i, j] = UA[i, j]; end
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
    @inbounds for i in 1:n; S[i] = d[i]; end
    return S
end

# ── in-place entries (caller provides the output buffers; 0-alloc steady state) ─────────────────────────
# Full SVD: writes into U (m×nu / m×m), S (min), Vt (ncv×n / n×n). full_u: U's m×m complement (m>n).
# full_v: Vt's n×n complement (n>m) — realized via the transpose path (full_v on A ≡ full_u on Aᵀ).
# m<n: SVD Aᵀ (tall) = Ū·Σ·V̄ᵀ ⟹ A = V̄·Σ·Ūᵀ ⟹ U(A)=V̄=(V̄ᵀ)ᵀ, Vt(A)=Ūᵀ. All staging (Aᵀ, Ū, V̄ᵀ) from ws.
function gesvd!(A::AbstractMatrix{Float64}, U::AbstractMatrix{Float64}, S::AbstractVector{Float64},
        Vt::AbstractMatrix{Float64}; full_u::Bool = false, full_v::Bool = false)
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
function gesvd_vals!(A::AbstractMatrix{T}, S::AbstractVector{<:Real}) where {T<:BlasComplex}
    m, n = size(A)
    m >= n || return gesvd_vals!(permutedims(A), S)          # tall via transpose (σ preserved)
    ws = _svdws(T)
    _svd_grow_bidiag!(ws, m, n)
    # Bidiagonal d,e in Float64 unconditionally: bdsqr! is the Float64 implicit-QR core, and the real
    # bidiagonal of a ComplexF32 A is fine to refine in double (σ then rounded back into S's eltype).
    d = zeros(Float64, n); e = zeros(Float64, max(n - 1, 0)); tauq = zeros(T, n); taup = zeros(T, n)
    gebrd!(A, d, e, tauq, taup, ws)                          # blocked zgebrd (zlabrd panels + gemm trailing)
    # dqds (dlasq) singular VALUES on the real bidiagonal; QR bdsqr! is the rare-failure fallback.
    _dlasq1!(d, e, ws.dqds_Z, ws.dqds_st) != 0 && bdsqr!(d, e, nothing, nothing)
    @inbounds for i in 1:n; S[i] = d[i]; end
    return S
end
function gesvd!(A::AbstractMatrix{T}; want_vectors::Bool = true) where {T<:BlasComplex}
    m, n = size(A); mn = min(m, n)
    want_vectors && throw(ArgumentError("complex gesvd! with singular vectors not yet implemented — " *
        "use want_vectors=false for singular values (the vectors' back-transform is the follow-up)"))
    S = Vector{real(T)}(undef, mn)
    gesvd_vals!(A, S)                                         # destructive (mirrors the real want_vectors=false path)
    return (S,)
end
