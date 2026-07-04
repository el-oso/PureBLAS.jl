# LAPACK SVD (gesvd) — pure Julia, built on PureBLAS blocks. Three layers (ROADMAP):
#   1. gebrd  — two-sided Householder bidiagonalization  A = Q·B·Pᵀ  (B upper-bidiagonal, m≥n).
#   2. bdsqr  — implicit-shift QR on the bidiagonal B (Golub-Kahan), accumulating Givens into U,Vᵀ.
#   3. driver — form Q,Pᵀ from the reflectors and back-transform the bidiagonal singular vectors.
# Float64 path. Householder = standard LAPACK convention (H = I − τ·v·vᵀ, v[1]=1 implicit) so the
# back-transform is self-contained. ponytail: m<n handled by transposing; generic/AD SVD deferred.

# --- Householder generator (LAPACK dlarfg) on a strided segment ---------------------------------
# x = [α; tail]. Returns (β, τ): the reflector H = I − τ·v·vᵀ with v = [1; x[2:]/(α−β)] zeros the
# tail, leaving β at x[1]. On return x[2:] holds the essential v; x[1] is left to the caller.
@inline function _larfg!(x::AbstractVector{Float64})
    n = length(x)
    @inbounds begin
        α = x[1]
        n == 1 && return α, 0.0
        ss = 0.0                                     # SIMD sum-of-squares (avoids O(n²) Base.hypot in gebrd);
        @simd for i in 2:n; ss = muladd(x[i], x[i], ss); end   # fall back to scaled hypot on overflow only
        xnorm = sqrt(ss)
        if !isfinite(xnorm)
            xnorm = 0.0
            for i in 2:n; xnorm = hypot(xnorm, x[i]); end
        end
        xnorm == 0.0 && return α, 0.0
        β = -copysign(hypot(α, xnorm), α)
        τ = (β - α) / β
        s = 1.0 / (α - β)
        for i in 2:n
            x[i] *= s
        end
    end
    return β, τ
end

# Apply H = I − τ·v·vᵀ (v[1]≡1, v[2:]=v[2:]) to C (size len×nc) from the LEFT:  C := H·C.
@inline function _house_left!(C::AbstractMatrix{Float64}, v::AbstractVector{Float64}, τ::Float64)
    τ == 0.0 && return C
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

# --- Stage 1b: BLOCKED bidiagonalization (LAPACK dlabrd panel + gemm trailing update), m ≥ n -----
# Reduce the first nb rows/cols of the (mm×nn) submatrix As to bidiagonal form, accumulating the
# matrices X (mm×nb) and Y (nn×nb) that drive the rank-2nb trailing update. Faithful dlabrd port;
# the matrix-vector ops are PureBLAS gemv!. d,e,tauq,taup,X,Y are local (1-based) to this block.
# Direct gemv kernel (skips the public kwarg wrapper's ~200 ns dispatch/char-parse — critical in _labrd's
# thousands of tiny gemv calls). y := α·op(A)·x + β·y with op = ('T' if tr) on A (m×n = size(Av)), unit inc.
@inline _lg!(yv, Av, xv, α::Float64, β::Float64, tr::Bool) =
    _gemv!(tr, false, size(Av, 1), size(Av, 2), α, Av, xv, 1, β, yv, 1)

function _labrd!(As::AbstractMatrix{Float64}, d, e, tauq, taup, X, Y, nb::Int)
    mm, nn = size(As)
    arow = Vector{Float64}(undef, nn)      # contiguous copy of the active reflector row (kills strided gemv)
    tmp = Vector{Float64}(undef, nb)       # contiguous copy of strided row-vectors used as x
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

const _BRD_NB = 16     # bidiagonalization panel width; measured optimal across n=256–2048 (narrow panel ⇒
                       # less BLAS-2 work/panel; the rank-16 trailing gemm stays efficient). ponytail: Zen4.
const _BT_NB = 32      # back-transform (compact-WY dlarfb) block: larger than gebrd's — its gemms want
                       # bigger T blocks (nb=16 there regressed large-n vectors). Decoupled from _BRD_NB.
const _SVD_DC_CROSS = 128   # vectors: bdsqr (QR) at/below this n, divide-and-conquer above (retuned down from 144 once the CAP=0 secular cut made bdsdc merges cheap enough to win at 136-144).
                            # wins ≤~144 (less D&C fixed overhead at tiny n), D&C wins ≥160 (O(n³) rotations).

# Blocked bidiagonalization driver (LAPACK dgebrd): blocked panels via _labrd! + two gemm! trailing
# updates, finishing the tail with the unblocked gebd2!. Requires m ≥ n.
function gebrd!(A::AbstractMatrix{Float64}, d::AbstractVector{Float64}, e::AbstractVector{Float64},
        tauq::AbstractVector{Float64}, taup::AbstractVector{Float64}; nb::Int = _BRD_NB)
    m, n = size(A)
    m >= n || throw(ArgumentError("gebrd!: requires m ≥ n (got $m×$n)"))
    k = n
    nx = nb
    if k <= nx || nb < 2
        return gebd2!(A, d, e, tauq, taup)
    end
    X = Matrix{Float64}(undef, m, nb); Y = Matrix{Float64}(undef, n, nb)
    i = 1
    @inbounds while i <= k - nx
        mm = m - i + 1; nn = n - i + 1
        As = view(A, i:m, i:n)
        di = view(d, i:k); ei = view(e, i:k-1); tqi = view(tauq, i:k); tpi = view(taup, i:k)
        _labrd!(As, di, ei, tqi, tpi, view(X, 1:mm, 1:nb), view(Y, 1:nn, 1:nb), nb)
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
@inline function _givens(f::Float64, g::Float64)
    r = sqrt(f * f + g * g)          # bdsqr! scales the bidiagonal to O(1) ⇒ no overflow; skip Base.hypot
    r == 0.0 && return 1.0, 0.0, 0.0
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

@inline function _swap_cols!(M::AbstractMatrix{Float64}, j1::Int, j2::Int)
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
# Cached T/G/W/Y workspace — fresh zeros(nb,nb)+3 allocs per call cost ~32 KB per gesvd (2 calls),
# dominating tiny-n SVD. Regrown on nc; T's needed region is re-zeroed per use below.
const _BT_WS = Ref{NTuple{4, Matrix{Float64}}}((zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0)))
@inline function _bt_ws(nb::Int, nc::Int)
    T, G, W, Yb = _BT_WS[]
    if size(T, 1) < nb || size(W, 2) < nc
        T = zeros(Float64, nb, nb); G = Matrix{Float64}(undef, nb, nb)
        W = Matrix{Float64}(undef, nb, nc); Yb = Matrix{Float64}(undef, nb, nc)
        _BT_WS[] = (T, G, W, Yb)
    end
    return T, G, W, Yb
end
function _apply_reflectors_left!(Vfull::AbstractMatrix{Float64}, tau::AbstractVector{Float64},
        C::AbstractMatrix{Float64}, k::Int, nb::Int, roff::Int)
    M = size(Vfull, 1); nc = size(C, 2)
    (k == 0 || nc == 0) && return C
    T, G, W, Yb = _bt_ws(nb, nc)
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

# Trim-safe transpose. permutedims's generic machinery is not guaranteed trim-clean, and the C-ABI SVD
# path must be fully juliac-analyzable, so we use a hand-written loop. Allocates an n×m Matrix; O(mn),
# negligible vs the O(mn·min) SVD.
function _svd_transpose(A::AbstractMatrix{Float64})
    m, n = size(A)
    B = Matrix{Float64}(undef, n, m)
    @inbounds for j in 1:n, i in 1:m
        B[j, i] = A[i, j]
    end
    return B
end

# Singular values only — concrete `Vector{Float64}` return (the trim-safe C-ABI values path).
function _gesvd_vals!(A::AbstractMatrix{Float64})
    m, n = size(A)
    m < n && return _gesvd_vals!(_svd_transpose(A))   # σ(A) = σ(Aᵀ)
    k = n
    d = Vector{Float64}(undef, k)
    e = Vector{Float64}(undef, max(k - 1, 0))
    tauq = Vector{Float64}(undef, k); taup = Vector{Float64}(undef, k)
    gebrd!(A, d, e, tauq, taup)
    bdsqr!(d, e, nothing, nothing)
    return d
end

# Full SVD — concrete `Tuple{Matrix,Vector,Matrix}` return (U, S, Vᵀ), the trim-safe C-ABI vectors path.
#   full_u=true, m>n: form the FULL m×m U (economy columns + the orthonormal complement of range(A)).
#   full_v=true, n>m: form the FULL n×n Vᵀ (via the transpose path — full_v on A ≡ full_u on Aᵀ).
# Otherwise economy: U is m×min, Vᵀ is min×n. All returns are dense Matrix/Vector (no views, no Union),
# so the C-ABI copyto! sees a concrete source and the whole graph passes juliac --trim=safe.
function _gesvd_full!(A::AbstractMatrix{Float64}; full_u::Bool = false,
        full_v::Bool = false)::Tuple{Matrix{Float64}, Vector{Float64}, Matrix{Float64}}
    m, n = size(A)
    if m < n                                    # work on Aᵀ: A = Ṽ Σ Ũᵀ (full_u/full_v swap under transpose)
        Ut, S, Vtt = _gesvd_full!(_svd_transpose(A); full_u = full_v, full_v = full_u)
        return _svd_transpose(Vtt), S, _svd_transpose(Ut)
    end
    k = n
    d = Vector{Float64}(undef, k)
    e = Vector{Float64}(undef, max(k - 1, 0))
    tauq = Vector{Float64}(undef, k); taup = Vector{Float64}(undef, k)
    gebrd!(A, d, e, tauq, taup)
    nb = _BT_NB
    # B's left/right singular vectors (Lvec/Rvec, n×n): bdsqr below the crossover (less D&C overhead at
    # small n, like LAPACK gesdd's QR-below-SMLSIZ), divide-and-conquer above.
    if n <= _SVD_DC_CROSS
        Lvec = zeros(Float64, n, n); Rvec = zeros(Float64, n, n)
        @inbounds for i in 1:n; Lvec[i, i] = 1.0; Rvec[i, i] = 1.0; end
        bdsqr!(d, e, Lvec, Rvec)                 # B = Lvec·diag(d)·Rvecᵀ
        s = d
    else
        s, Rvec, Lvec = bdsdc!(d, e)             # bdsdc → (s, Ul, Vl); B left=Vl=Lvec, right=Ul=Rvec
    end
    # U_A = Q·[Lvec 0; 0 I] — apply the left (column) reflectors to the embedded bidiagonal left vectors.
    # Full-U (m>n): the trailing bidiagonal rows are zero, so Ub_full = [Lvec 0; 0 I_{m−n}]; the extra
    # unit columns pushed through Q become the orthonormal complement of range(A). The accumulator C in the
    # back-transform's VᵀC gemm is UA/Vmat here — pad its leading dim (+8) so its column stride isn't a
    # power-of-2 multiple that thrashes one L1 set in the unpacked transA='T' kernel.
    nu = (full_u && m > n) ? m : n              # #U columns to form (m for full, else economy min=n)
    ldu = m % 256 == 0 ? m + 8 : m              # pad only when the po2 leading dim aliases (gcd(m/8,64)≥32)
    UApad = zeros(Float64, ldu, nu); UA = view(UApad, 1:m, 1:nu)
    @inbounds for j in 1:n, i in 1:n; UA[i, j] = Lvec[i, j]; end
    @inbounds for j in n+1:nu; UA[j, j] = 1.0; end   # complement unit columns e_{n+1..m}
    VQ = zeros(Float64, m, n)
    @inbounds for j in 1:n
        VQ[j, j] = 1.0
        for i in j+1:m; VQ[i, j] = A[i, j]; end
    end
    _apply_reflectors_left!(VQ, tauq, UA, n, nb, 0)   # Q applied over all nu columns
    # V_A = P·Rvec — apply the right (row) reflectors (k=n-1, support offset by 1).
    ldv = n % 256 == 0 ? n + 8 : n
    Vpad = zeros(Float64, ldv, n); Vmat = view(Vpad, 1:n, 1:n)
    @inbounds for j in 1:n, i in 1:n; Vmat[i, j] = Rvec[i, j]; end
    if n > 1
        VP = zeros(Float64, n, n - 1)
        @inbounds for j in 1:n-1
            VP[j+1, j] = 1.0
            for r in j+2:n; VP[r, j] = A[j, r]; end
        end
        _apply_reflectors_left!(VP, taup, Vmat, n - 1, nb, 1)
    end
    _svd_sort!(s, UA, Vmat)                     # descending; sorts cols 1:n, complement cols n+1:nu untouched
    Uout = Matrix{Float64}(undef, m, nu)        # materialize the padded view → concrete Matrix for the C-ABI
    @inbounds for j in 1:nu, i in 1:m; Uout[i, j] = UA[i, j]; end
    return Uout, s, _svd_transpose(Vmat)        # Vt = V_Aᵀ
end

# Full SVD of A (Float64). want_vectors=false returns (S,) only (singular values). Thin Mode-2 wrapper:
# the C-ABI (cabi_lapack.jl) calls _gesvd_vals!/_gesvd_full! directly so no Union crosses the trim boundary.
function gesvd!(A::AbstractMatrix{Float64}; want_vectors::Bool = true)
    want_vectors && return _gesvd_full!(A)
    return (_gesvd_vals!(A),)
end
