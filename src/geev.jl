# LAPACK nonsymmetric-eigen DRIVERS (`eigen(A)`/`eigvals(A)`/`schur(A)`): compose the validated
# reduction/Schur/eigenvector kernels into the general eigensolver, mirroring Reference-LAPACK
# dgeev/zgeev (eigenvalues + right eigenvectors) and dgees/zgees (Schur form + Schur vectors).
#
#   geev pipeline: gebal (balance) → gehrd (Hessenberg) → orghr (Q) → hseqr (Schur T + vectors)
#                  → trevc (right eigenvectors of the BALANCED matrix) → gebak (undo balance) → normalize.
#   gees pipeline: gebal(job='P', permute-only so Z stays orthogonal) → gehrd → orghr → hseqr → gebak('P').
#
# `gebak!` (LAPACK dgebak/zgebak) is the inverse of gebal's row/col transforms applied to the eigen/Schur
# vectors — implemented here (it had no home yet). The final eigenvector normalization copies dgeev/zgeev
# exactly (unit 2-norm, largest-magnitude component made real; the real-packed conjugate-pair (re,im)
# two-column layout for real A). Generic over T<:Number, scalar loops, trim-safe. Left eigenvectors
# (jobvl='V') route to trevc side='L' which is a documented follow-up — jobvl='V' is rejected here.

# ── DGEBAK / ZGEBAK — undo gebal's permutation + diagonal scaling on the eigen/Schur vectors ──────────
# `job` matches the gebal! job used ('N'/'P'/'S'/'B'); `side='R'` (right vectors). `scale` is gebal!'s
# output: permutation indices in scale[1:ilo-1] & scale[ihi+1:n], diagonal scale factors in scale[ilo:ihi].
# Transcribed from dgebak.f: backward scaling (rows ilo:ihi) then backward permutation (reverse order).
function gebak!(job::AbstractChar, side::AbstractChar, ilo::Integer, ihi::Integer,
        scale::AbstractVector{<:Real}, V::AbstractMatrix{T}) where {T<:Number}
    (job === 'N' || job === 'P' || job === 'S' || job === 'B') ||
        throw(ArgumentError("gebak!: job must be one of N/P/S/B"))
    (side === 'R' || side === 'L') || throw(ArgumentError("gebak!: side must be 'R' or 'L'"))
    n, m = size(V)
    m == 0 && return V
    job === 'N' && return V
    lo = Int(ilo); hi = Int(ihi)
    rightv = side === 'R'
    # backward balance (scaling) — skipped when ilo == ihi (no scaled block)
    if (job === 'S' || job === 'B') && lo != hi
        @inbounds for i in lo:hi
            s = rightv ? scale[i] : inv(scale[i])
            for j in 1:m
                V[i, j] *= s
            end
        end
    end
    # backward permutation (applies to both sides identically — the swaps are orthogonal)
    if job === 'P' || job === 'B'
        @inbounds for ii in 1:n
            i = ii
            (i >= lo && i <= hi) && continue
            i < lo && (i = lo - ii)
            k = round(Int, scale[i])
            k == i && continue
            for j in 1:m
                V[i, j], V[k, j] = V[k, j], V[i, j]
            end
        end
    end
    return V
end

# Zero gehrd's reflector storage (A[i+2:ihi, i]) so the matrix is a CLEAN upper-Hessenberg for hseqr.
# hseqr/dlahqr only clears a 2-wide sub-band of trash; reflectors of length > 2 (columns whose reduction
# spans ≥ 4 rows) leave garbage below that band which corrupts the Schur form/vectors. Q must already be
# formed (orghr reads these reflectors) — here A's own copy is scrubbed after that.
@inline function _geev_clear_hess!(A::AbstractMatrix{T}, ilo::Int, ihi::Int) where {T<:Number}
    @inbounds for i in ilo:(ihi - 1)
        for r in (i + 2):ihi
            A[r, i] = zero(T)
        end
    end
    return A
end

# 1-norm (max absolute column sum) of the balanced matrix — DLANGE('1'), used for geevx's ABNRM output.
@inline function _geev_lange1(A::AbstractMatrix{T}) where {T<:Number}
    R = real(T); m = size(A, 1); n = size(A, 2); best = zero(R)
    @inbounds for j in 1:n
        s = zero(R)
        for i in 1:m
            s += abs(A[i, j])
        end
        s > best && (best = s)
    end
    return best
end

# lassq-style Euclidean norm of column j (rows 1:n) — DNRM2/DZNRM2, overflow/underflow safe (req#6).
@inline function _geev_colnrm2(V::AbstractMatrix{T}, j::Int, n::Int) where {T<:Number}
    R = real(T); scl = zero(R); ssq = one(R); nz = false
    @inbounds for k in 1:n
        x = V[k, j]
        if !iszero(x)
            nz = true; ax = abs(x)
            if scl < ax
                ssq = one(R) + ssq * (scl / ax)^2; scl = ax
            else
                ssq += (ax / scl)^2
            end
        end
    end
    return nz ? scl * sqrt(ssq) : zero(R)
end

# dgeev final normalization (REAL, real-packed VR): unit 2-norm + largest component made real. A real
# eigenvalue owns one real column; a complex-conjugate pair owns columns (i,i+1) as (re,im) with
# v = VR[:,i] ± i·VR[:,i+1] — normalize the pair jointly and rotate so its largest |v| entry is real.
function _geev_normalize_real!(VR::AbstractMatrix{R}, wi::AbstractVector{R}, n::Int) where {R<:Real}
    ZERO = zero(R); ONE = one(R)
    i = 1
    @inbounds while i <= n
        if iszero(wi[i])
            s = _geev_colnrm2(VR, i, n)
            if !iszero(s)
                g = ONE / s
                for k in 1:n; VR[k, i] *= g; end
            end
            i += 1
        else
            # complex-conjugate pair at columns i, i+1 (wi[i] > 0 first by hseqr ordering)
            s = hypot(_geev_colnrm2(VR, i, n), _geev_colnrm2(VR, i + 1, n))
            if !iszero(s)
                g = ONE / s
                for k in 1:n; VR[k, i] *= g; VR[k, i + 1] *= g; end
            end
            kmax = 1; best = VR[1, i]^2 + VR[1, i + 1]^2
            for k in 2:n
                v = VR[k, i]^2 + VR[k, i + 1]^2
                v > best && (best = v; kmax = k)
            end
            f = VR[kmax, i]; g2 = VR[kmax, i + 1]; r = hypot(f, g2)
            if !iszero(r)
                cs = f / r; sn = g2 / r
                for k in 1:n
                    t = VR[k, i]; u = VR[k, i + 1]
                    VR[k, i] = cs * t + sn * u
                    VR[k, i + 1] = cs * u - sn * t
                end
            end
            VR[kmax, i + 1] = ZERO
            i += 2
        end
    end
    return VR
end

# zgeev final normalization (COMPLEX): unit 2-norm, then multiply by conj/|·| of the largest component
# so that component becomes real positive.
function _geev_normalize_cmplx!(VR::AbstractMatrix{C}, n::Int) where {C<:Complex}
    R = real(C)
    @inbounds for i in 1:n
        s = _geev_colnrm2(VR, i, n)
        if !iszero(s)
            g = one(R) / s
            for k in 1:n; VR[k, i] *= g; end
        end
        kmax = 1; best = abs2(VR[1, i])
        for k in 2:n
            v = abs2(VR[k, i]); v > best && (best = v; kmax = k)
        end
        d = VR[kmax, i]; ad = abs(d)
        if !iszero(ad)
            t = conj(d) / ad
            for k in 1:n; VR[k, i] *= t; end
            VR[kmax, i] = Complex(real(VR[kmax, i]), zero(R))
        end
    end
    return VR
end

# ── geev core (REAL) — returns (WR, WI, VL, VR, ilo, ihi, scale, abnrm). A is overwritten (Schur form). ──
function _geev_run!(balanc::Char, jobvl::Char, jobvr::Char, A::AbstractMatrix{T}) where {T<:Real}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("geev!: A must be square"))
    jobvl === 'N' || throw(ArgumentError("geev!: left eigenvectors (jobvl='V') not implemented"))
    (jobvr === 'N' || jobvr === 'V') || throw(ArgumentError("geev!: jobvr must be 'N' or 'V'"))
    wantvr = jobvr === 'V'
    wr = zeros(T, n); wi = zeros(T, n)
    VL = Matrix{T}(undef, n, 0)
    VR = Matrix{T}(undef, n, wantvr ? n : 0)
    n == 0 && return wr, wi, VL, VR, 1, 0, T[], zero(T)
    ilo, ihi, scale = gebal!(A; job = balanc)
    abnrm = _geev_lange1(A)
    tau = zeros(T, max(n - 1, 0))
    gehrd!(A, ilo, ihi, tau)
    w = Vector{Complex{T}}(undef, n)
    if wantvr
        @inbounds for j in 1:n, i in 1:n; VR[i, j] = A[i, j]; end   # reflectors → form Q
        orghr!(VR, ilo, ihi, tau)
        _geev_clear_hess!(A, ilo, ihi)                             # scrub reflectors → clean Hessenberg
        hseqr!('S', 'V', A, ilo, ihi, w, VR)                       # Schur T in A; VR := Schur vectors
        trevc!('R', 'B', A, VL, VR)                                # VR := right eigenvectors (balanced)
        gebak!(balanc, 'R', ilo, ihi, scale, VR)                   # undo balancing
        @inbounds for i in 1:n; wr[i] = real(w[i]); wi[i] = imag(w[i]); end
        _geev_normalize_real!(VR, wi, n)
    else
        _geev_clear_hess!(A, ilo, ihi)
        Zdummy = Matrix{T}(undef, 0, 0)
        hseqr!('E', 'N', A, ilo, ihi, w, Zdummy)
        @inbounds for i in 1:n; wr[i] = real(w[i]); wi[i] = imag(w[i]); end
    end
    return wr, wi, VL, VR, ilo, ihi, scale, abnrm
end

# ── geev core (COMPLEX) — returns (W, VL, VR, ilo, ihi, scale, abnrm). ──────────────────────────────────
function _geev_run!(balanc::Char, jobvl::Char, jobvr::Char, A::AbstractMatrix{T}) where {T<:Complex}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("geev!: A must be square"))
    jobvl === 'N' || throw(ArgumentError("geev!: left eigenvectors (jobvl='V') not implemented"))
    (jobvr === 'N' || jobvr === 'V') || throw(ArgumentError("geev!: jobvr must be 'N' or 'V'"))
    R = real(T)
    wantvr = jobvr === 'V'
    w = zeros(T, n)
    VL = Matrix{T}(undef, n, 0)
    VR = Matrix{T}(undef, n, wantvr ? n : 0)
    n == 0 && return w, VL, VR, 1, 0, R[], zero(R)
    ilo, ihi, scale = gebal!(A; job = balanc)
    abnrm = _geev_lange1(A)
    tau = zeros(T, max(n - 1, 0))
    gehrd!(A, ilo, ihi, tau)
    if wantvr
        @inbounds for j in 1:n, i in 1:n; VR[i, j] = A[i, j]; end
        orghr!(VR, ilo, ihi, tau)
        _geev_clear_hess!(A, ilo, ihi)
        hseqr!('S', 'V', A, ilo, ihi, w, VR)
        trevc!('R', 'B', A, VL, VR)
        gebak!(balanc, 'R', ilo, ihi, scale, VR)
        _geev_normalize_cmplx!(VR, n)
    else
        _geev_clear_hess!(A, ilo, ihi)
        Zdummy = Matrix{T}(undef, 0, 0)
        hseqr!('E', 'N', A, ilo, ihi, w, Zdummy)
    end
    return w, VL, VR, ilo, ihi, scale, abnrm
end

# ── gees core — Schur form (A overwritten) + Schur vectors. Permute-only balance keeps Z orthogonal. ────
function _gees_run!(jobvs::Char, A::AbstractMatrix{T}) where {T<:Real}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("gees!: A must be square"))
    (jobvs === 'N' || jobvs === 'V') || throw(ArgumentError("gees!: jobvs must be 'N' or 'V'"))
    wantvs = jobvs === 'V'
    wr = zeros(T, n); wi = zeros(T, n)
    VS = Matrix{T}(undef, n, wantvs ? n : 0)
    n == 0 && return wr, wi, VS
    ilo, ihi, scale = gebal!(A; job = 'P')
    tau = zeros(T, max(n - 1, 0))
    gehrd!(A, ilo, ihi, tau)
    w = Vector{Complex{T}}(undef, n)
    if wantvs
        @inbounds for j in 1:n, i in 1:n; VS[i, j] = A[i, j]; end
        orghr!(VS, ilo, ihi, tau)
        _geev_clear_hess!(A, ilo, ihi)
        hseqr!('S', 'V', A, ilo, ihi, w, VS)
        gebak!('P', 'R', ilo, ihi, scale, VS)
    else
        _geev_clear_hess!(A, ilo, ihi)
        Zdummy = Matrix{T}(undef, 0, 0)
        hseqr!('S', 'N', A, ilo, ihi, w, Zdummy)
    end
    @inbounds for i in 1:n; wr[i] = real(w[i]); wi[i] = imag(w[i]); end
    return wr, wi, VS
end

function _gees_run!(jobvs::Char, A::AbstractMatrix{T}) where {T<:Complex}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("gees!: A must be square"))
    (jobvs === 'N' || jobvs === 'V') || throw(ArgumentError("gees!: jobvs must be 'N' or 'V'"))
    wantvs = jobvs === 'V'
    w = zeros(T, n)
    VS = Matrix{T}(undef, n, wantvs ? n : 0)
    n == 0 && return w, VS
    ilo, ihi, scale = gebal!(A; job = 'P')
    tau = zeros(T, max(n - 1, 0))
    gehrd!(A, ilo, ihi, tau)
    if wantvs
        @inbounds for j in 1:n, i in 1:n; VS[i, j] = A[i, j]; end
        orghr!(VS, ilo, ihi, tau)
        _geev_clear_hess!(A, ilo, ihi)
        hseqr!('S', 'V', A, ilo, ihi, w, VS)
        gebak!('P', 'R', ilo, ihi, scale, VS)
    else
        _geev_clear_hess!(A, ilo, ihi)
        Zdummy = Matrix{T}(undef, 0, 0)
        hseqr!('S', 'N', A, ilo, ihi, w, Zdummy)
    end
    return w, VS
end

"""
    geev!(jobvl, jobvr, A) -> (WR, WI, VL, VR)   [real]
    geev!(jobvl, jobvr, A) -> (W, VL, VR)         [complex]

Eigenvalues and (optionally) right eigenvectors of a general square `A` (LAPACK dgeev/zgeev; balances
with job='B'). `jobvr='V'` computes right eigenvectors into `VR`, `'N'` skips them. `jobvl='V'` (left
eigenvectors) is not implemented (throws). For real `A`, a complex-conjugate eigenvalue pair occupies two
consecutive `VR` columns as (real, imag) parts (LAPACK real convention). `A` is overwritten.
"""
function geev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{T}) where {T<:Real}
    wr, wi, VL, VR = _geev_run!('B', Char(jobvl), Char(jobvr), A)
    return wr, wi, VL, VR
end
function geev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{T}) where {T<:Complex}
    w, VL, VR = _geev_run!('B', Char(jobvl), Char(jobvr), A)
    return w, VL, VR
end

"""
    gees!(jobvs, A) -> (T, Z, w)

Schur decomposition of a general square `A` (LAPACK dgees/zgees): `A` is overwritten with the
(quasi-)upper-triangular Schur form `T`, `Z` (returned) holds the Schur vectors when `jobvs='V'`
(`A₀ = Z·T·Zᴴ`), and `w` the eigenvalues (complex). `jobvs='N'` computes `T`/`w` only.
"""
function gees!(jobvs::AbstractChar, A::AbstractMatrix{T}) where {T<:Real}
    wr, wi, VS = _gees_run!(Char(jobvs), A)
    return A, VS, complex.(wr, wi)
end
function gees!(jobvs::AbstractChar, A::AbstractMatrix{T}) where {T<:Complex}
    w, VS = _gees_run!(Char(jobvs), A)
    return A, VS, w
end
