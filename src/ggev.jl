# LAPACK GENERALIZED nonsymmetric-eigen DRIVERS (`eigen(A,B)`/`eigvals(A,B)`/`schur(A,B)`): compose the
# validated QZ kernels (qz.jl gghrd!/hgeqz!, tgevc_gen.jl tgevc!) into the generalized eigensolver,
# mirroring Reference-LAPACK dggev/zggev (eigenvalues + right eigenvectors) and dgges/zgges (generalized
# Schur form + Schur vectors).
#
#   ggev pipeline: QR of B (geqrf!) → apply Qᴴ to A → reduce (A,R) to Hessenberg-triangular (gghrd!,
#                  accumulate Z) → QZ iteration (hgeqz!, Schur form + Z) → right eigenvectors (tgevc! 'B',
#                  back-transform with Z) → normalize (dggev max-|component| scaling).
#   gges pipeline: QR of B → apply Qᴴ to A → form Q_B into VSL → gghrd! (compq='V' into VSL, compz='I'
#                  into VSR) → hgeqz! ('S', accumulate both) → (S,P,α,β,VSL,VSR).
#
# The QR-of-B is a LEFT transform (Q_Bᴴ common to A and B), so it leaves the RIGHT eigenvectors of the
# pencil INVARIANT — for ggev (right vectors only) Q_B is NOT applied to the eigenvectors; only Z (from
# gghrd/hgeqz, applied inside tgevc howmny='B') back-transforms them. For gges the Schur left vectors DO
# carry Q_B (VSL = Q_B·Q_qz). Balancing (dggbal) is skipped (ilo=1, ihi=n) — correctness-first; the QZ
# kernels are norm-scaled internally. jobvl='V' (left eigenvectors) is unsupported (tgevc side='L' is a
# follow-up). Generic over T<:Number, scalar loops, trim-safe.

# geqrf!'s τ convention differs by type: real Float64 stores τ_stored = 1/τ_LAPACK (faer), complex stores
# τ_LAPACK directly (zlarfg). Convert to LAPACK convention (H_i = I − τ·v·vᴴ) for the explicit-Q build.
@inline _ggev_tauL(t::Real) = (isfinite(t) && !iszero(t)) ? one(t) / t : zero(t)
@inline _ggev_tauL(t::Complex) = t

# Form the explicit orthogonal/unitary factor Q = H_1·H_2·…·H_n (n×n) from geqrf!'s reflectors (essential
# part v_i in B[i+1:n, i], v_i[i]=1) and the LAPACK-convention τ. Correctness-first O(n³) accumulation.
function _ggev_formQ!(Q::AbstractMatrix{T}, B::AbstractMatrix{T}, tauL::AbstractVector{T}, n::Int) where {T<:Number}
    fill!(Q, zero(T))
    @inbounds for i in 1:n; Q[i, i] = one(T); end
    @inbounds for i in n:-1:1
        τ = tauL[i]
        iszero(τ) && continue
        for c in 1:n                                       # H_i·Q[:,c] = Q[:,c] − τ·(vᴴ Q[:,c])·v
            s = Q[i, c]
            for r in i+1:n; s += conj(B[r, i]) * Q[r, c]; end
            s *= τ
            Q[i, c] -= s
            for r in i+1:n; Q[r, c] -= s * B[r, i]; end
        end
    end
    return Q
end

# QR of B, apply Qᴴ to A, and return Q_B (n×n) + R (in the upper triangle of B; strict-lower still holds
# reflectors, which gghrd! zeros). ct = conjugate-transpose char.
function _ggev_qrB!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, n::Int, ct::Char) where {T<:Number}
    tau = zeros(T, n)
    geqrf!(B, tau)
    tauL = similar(tau)
    @inbounds for i in 1:n; tauL[i] = _ggev_tauL(tau[i]); end
    Q = Matrix{T}(undef, n, n)
    _ggev_formQ!(Q, B, tauL, n)
    tmp = Matrix{T}(undef, n, n)
    gemm!(tmp, Q, A; transA = ct, transB = 'N', alpha = one(T), beta = zero(T))   # A := Qᴴ·A
    copyto!(A, tmp)
    return Q
end

# dggev eigenvector normalization (REAL, real-packed VR): scale each eigenvector so its largest-magnitude
# component (|·| for real λ; |re|+|im| jointly for a conjugate pair) is 1. alphai<0 marks the 2nd of a pair.
function _ggev_normalize_real!(VR::AbstractMatrix{R}, alphai::AbstractVector{R}, n::Int) where {R<:Real}
    ZERO = zero(R); smlnum = floatmin(R)
    jc = 1
    @inbounds while jc <= n
        if alphai[jc] < ZERO
            jc += 1; continue
        end
        temp = ZERO
        if iszero(alphai[jc])
            for jr in 1:n; temp = max(temp, abs(VR[jr, jc])); end
            if temp >= smlnum
                t = one(R) / temp
                for jr in 1:n; VR[jr, jc] *= t; end
            end
            jc += 1
        else
            for jr in 1:n; temp = max(temp, abs(VR[jr, jc]) + abs(VR[jr, jc+1])); end
            if temp >= smlnum
                t = one(R) / temp
                for jr in 1:n; VR[jr, jc] *= t; VR[jr, jc+1] *= t; end
            end
            jc += 2
        end
    end
    return VR
end

# zggev eigenvector normalization (COMPLEX): scale each column so its largest |component| is 1.
function _ggev_normalize_cmplx!(VR::AbstractMatrix{C}, n::Int) where {C<:Complex}
    R = real(C); smlnum = floatmin(R)
    @inbounds for jc in 1:n
        temp = zero(R)
        for jr in 1:n; temp = max(temp, abs(real(VR[jr, jc])) + abs(imag(VR[jr, jc]))); end
        if temp >= smlnum
            t = one(R) / temp
            for jr in 1:n; VR[jr, jc] *= t; end
        end
    end
    return VR
end

# ── ggev core (REAL): returns (alphar, alphai, beta, VR). A/B overwritten (Schur form). ────────────────
function _ggev_run!(jobvl::Char, jobvr::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Real}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("ggev!: A and B must be square and the same size"))
    jobvl === 'N' || throw(ArgumentError("ggev!: left eigenvectors (jobvl='V') not implemented"))
    (jobvr === 'N' || jobvr === 'V') || throw(ArgumentError("ggev!: jobvr must be 'N' or 'V'"))
    wantvr = jobvr === 'V'
    alphar = zeros(T, n); alphai = zeros(T, n); beta = zeros(T, n)
    VR = Matrix{T}(undef, n, wantvr ? n : 0)
    n == 0 && return alphar, alphai, beta, VR
    alphaC = Vector{Complex{T}}(undef, n)
    Qd = Matrix{T}(undef, 0, 0)
    _ggev_qrB!(A, B, n, 'T')
    if wantvr
        gghrd!('N', 'I', A, B, Qd, VR)
        hgeqz!('S', 'N', 'V', A, B, alphaC, beta, Qd, VR)
        VLd = Matrix{T}(undef, 0, 0)
        tgevc!('R', 'B', A, B, VLd, VR)
        @inbounds for i in 1:n; alphar[i] = real(alphaC[i]); alphai[i] = imag(alphaC[i]); end
        _ggev_normalize_real!(VR, alphai, n)
    else
        gghrd!('N', 'N', A, B, Qd, Qd)
        hgeqz!('E', 'N', 'N', A, B, alphaC, beta, Qd, Qd)
        @inbounds for i in 1:n; alphar[i] = real(alphaC[i]); alphai[i] = imag(alphaC[i]); end
    end
    return alphar, alphai, beta, VR
end

# ── ggev core (COMPLEX): returns (alpha, beta, VR). ───────────────────────────────────────────────────
function _ggev_run!(jobvl::Char, jobvr::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("ggev!: A and B must be square and the same size"))
    jobvl === 'N' || throw(ArgumentError("ggev!: left eigenvectors (jobvl='V') not implemented"))
    (jobvr === 'N' || jobvr === 'V') || throw(ArgumentError("ggev!: jobvr must be 'N' or 'V'"))
    wantvr = jobvr === 'V'
    alpha = zeros(T, n); beta = zeros(T, n)
    VR = Matrix{T}(undef, n, wantvr ? n : 0)
    n == 0 && return alpha, beta, VR
    Qd = Matrix{T}(undef, 0, 0)
    _ggev_qrB!(A, B, n, 'C')
    if wantvr
        gghrd!('N', 'I', A, B, Qd, VR)
        hgeqz!('S', 'N', 'V', A, B, alpha, beta, Qd, VR)
        VLd = Matrix{T}(undef, 0, 0)
        tgevc!('R', 'B', A, B, VLd, VR)
        _ggev_normalize_cmplx!(VR, n)
    else
        gghrd!('N', 'N', A, B, Qd, Qd)
        hgeqz!('E', 'N', 'N', A, B, alpha, beta, Qd, Qd)
    end
    return alpha, beta, VR
end

# ── gges core (jobvsl='V', jobvsr='V'): Schur form S,P (A,B overwritten) + Schur vectors VSL,VSR. ──────
function _gges_run!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Real}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("gges!: A and B must be square and the same size"))
    alphar = zeros(T, n); alphai = zeros(T, n); beta = zeros(T, n)
    VSL = Matrix{T}(undef, n, n); VSR = Matrix{T}(undef, n, n)
    n == 0 && return A, B, Complex{T}[], beta, VSL, VSR
    alphaC = Vector{Complex{T}}(undef, n)
    Qb = _ggev_qrB!(A, B, n, 'T')
    copyto!(VSL, Qb)                                       # VSL starts as Q_B, gghrd/hgeqz post-multiply Q
    gghrd!('V', 'I', A, B, VSL, VSR)
    hgeqz!('S', 'V', 'V', A, B, alphaC, beta, VSL, VSR)
    return A, B, alphaC, beta, VSL, VSR
end

function _gges_run!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("gges!: A and B must be square and the same size"))
    alpha = zeros(T, n); beta = zeros(T, n)
    VSL = Matrix{T}(undef, n, n); VSR = Matrix{T}(undef, n, n)
    n == 0 && return A, B, alpha, beta, VSL, VSR
    Qb = _ggev_qrB!(A, B, n, 'C')
    copyto!(VSL, Qb)
    gghrd!('V', 'I', A, B, VSL, VSR)
    hgeqz!('S', 'V', 'V', A, B, alpha, beta, VSL, VSR)
    return A, B, alpha, beta, VSL, VSR
end

"""
    ggev!(jobvl, jobvr, A, B) -> (alphar, alphai, beta, vl, vr)   [real]
    ggev!(jobvl, jobvr, A, B) -> (alpha, beta, vl, vr)             [complex]

Generalized eigenvalues and (optionally) right eigenvectors of the pencil `(A,B)` — `A·x = λ·B·x` with
`λ = α/β` (LAPACK dggev/zggev). `jobvr='V'` computes right eigenvectors into `vr`, `'N'` skips them.
`jobvl='V'` (left eigenvectors) is not implemented (throws). For real `A,B`, a complex-conjugate
eigenvalue pair occupies two consecutive `vr` columns as (real, imag) parts (LAPACK real convention).
`A` and `B` are overwritten. `vl` is always empty (left vectors unsupported).
"""
function ggev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Real}
    alphar, alphai, beta, VR = _ggev_run!(Char(jobvl), Char(jobvr), A, B)
    return alphar, alphai, beta, Matrix{T}(undef, size(A, 1), 0), VR
end
function ggev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex}
    alpha, beta, VR = _ggev_run!(Char(jobvl), Char(jobvr), A, B)
    return alpha, beta, Matrix{T}(undef, size(A, 1), 0), VR
end

"""
    gges!(jobvsl, jobvsr, A, B) -> (A, B, alpha, beta, vsl, vsr)

Generalized Schur decomposition of the pencil `(A,B)` (LAPACK dgges/zgges): `A`→`S`, `B`→`P` (generalized
Schur form, `A₀ = vsl·S·vsrᴴ`, `B₀ = vsl·P·vsrᴴ`), `alpha`/`beta` the generalized eigenvalues (`λ=α/β`,
`alpha` complex), and the Schur vectors `vsl`/`vsr`. Only `jobvsl=jobvsr='V'` is supported (what `schur(A,B)`
requests); other values fall back to computing both sets of vectors. `A` and `B` are overwritten.
"""
function gges!(jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Real}
    S, P, alphaC, beta, VSL, VSR = _gges_run!(A, B)
    return S, P, alphaC, beta, VSL, VSR
end
function gges!(jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex}
    S, P, alpha, beta, VSL, VSR = _gges_run!(A, B)
    return S, P, alpha, beta, VSL, VSR
end
