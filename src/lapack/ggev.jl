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
function _ggev_formQ!(Q::AbstractMatrix{T}, B::AbstractMatrix{T}, tauL::AbstractVector{T}, n::Int) where {T <: Number}
    fill!(Q, zero(T))
    @inbounds for i in 1:n
        Q[i, i] = one(T)
    end
    @inbounds for i in n:-1:1
        τ = tauL[i]
        iszero(τ) && continue
        for c in 1:n                                       # H_i·Q[:,c] = Q[:,c] − τ·(vᴴ Q[:,c])·v
            s = Q[i, c]
            for r in (i + 1):n
                s += conj(B[r, i]) * Q[r, c]
            end
            s *= τ
            Q[i, c] -= s
            for r in (i + 1):n
                Q[r, c] -= s * B[r, i]
            end
        end
    end
    return Q
end

# QR of B, apply Qᴴ to A, and leave R in the upper triangle of B (strict-lower still holds reflectors,
# which gghrd! zeros). ct = conjugate-transpose char. `Qout`, when given, RECEIVES the explicit Q_B
# (gges!'s VSL); ggev! passes nothing, since the QR is a LEFT transform and leaves the right eigenvectors
# invariant. That out-parameter replaces the `return Q` the field-owned form had: under the arena Q is a
# borrow that dies with the scope, so the copy has to happen INSIDE it.
function _ggev_qrB!(
        A::AbstractMatrix{T}, B::AbstractMatrix{T}, n::Int, ct::Char,
        Qout::Union{Nothing, AbstractMatrix{T}} = nothing
    ) where {T <: Number}
    # Arena borrows (arena.jl), not owned fields. Q and tmp stay SEPARATE borrows: the gemm! reads Q while
    # it writes tmp. No fill! on any of the four: geqrf! writes tau[1:min(m,n)] = tau[1:n] for the square B
    # this path always passes (qr.jl:467/480 → qr_unblocked!, which assigns tau[i] for every i in 1:k),
    # tauL is fully written by the loop below, `_ggev_formQ!` opens with its own fill!(Q, 0), and the gemm!
    # runs beta=0 over all of tmp. Exact `ld` (no `_odd_ld`): the fields these replace were plain
    # `_wsgrow` buffers with no anti-aliasing stride.
    # ESCAPE AUDIT: nothing outlives the scope. `tau` goes only to geqrf!, which writes it and returns.
    # `tauL`/`Q` go to `_ggev_formQ!` (above in this file — a scalar accumulation that writes Q, returns
    # it, and stores nothing anywhere). `Q`/`tmp` go to gemm! (blas3/gemm.jl), which packs into its own
    # workspace and retains neither operand. Both copyto!s run INSIDE the block, so the only things
    # crossing the closing `end` are the caller's own `A` and `Qout`.
    @scope arn begin
        tau = borrow!(arn, T, n)
        tauL = borrow!(arn, T, n)
        Q = borrow!(arn, T, n, n)
        tmp = borrow!(arn, T, n, n)
        geqrf!(B, tau)
        @inbounds for i in 1:n
            tauL[i] = _ggev_tauL(tau[i])
        end
        _ggev_formQ!(Q, B, tauL, n)
        gemm!(tmp, Q, A; transA = ct, transB = 'N', alpha = one(T), beta = zero(T))   # A := Qᴴ·A
        copyto!(A, tmp)
        isnothing(Qout) || copyto!(Qout, Q)
    end
    return nothing
end

# dggev eigenvector normalization (REAL, real-packed VR): scale each eigenvector so its largest-magnitude
# component (|·| for real λ; |re|+|im| jointly for a conjugate pair) is 1. alphai<0 marks the 2nd of a pair.
function _ggev_normalize_real!(VR::AbstractMatrix{R}, alphai::AbstractVector{R}, n::Int) where {R <: Real}
    ZERO = zero(R); smlnum = floatmin(R)
    jc = 1
    @inbounds while jc <= n
        if alphai[jc] < ZERO
            jc += 1; continue
        end
        temp = ZERO
        if iszero(alphai[jc])
            for jr in 1:n
                temp = max(temp, abs(VR[jr, jc]))
            end
            if temp >= smlnum
                t = one(R) / temp
                for jr in 1:n
                    VR[jr, jc] *= t
                end
            end
            jc += 1
        else
            for jr in 1:n
                temp = max(temp, abs(VR[jr, jc]) + abs(VR[jr, jc + 1]))
            end
            if temp >= smlnum
                t = one(R) / temp
                for jr in 1:n
                    VR[jr, jc] *= t; VR[jr, jc + 1] *= t
                end
            end
            jc += 2
        end
    end
    return VR
end

# zggev eigenvector normalization (COMPLEX): scale each column so its largest |component| is 1.
function _ggev_normalize_cmplx!(VR::AbstractMatrix{C}, n::Int) where {C <: Complex}
    R = real(C); smlnum = floatmin(R)
    @inbounds for jc in 1:n
        temp = zero(R)
        for jr in 1:n
            temp = max(temp, abs(real(VR[jr, jc])) + abs(imag(VR[jr, jc])))
        end
        if temp >= smlnum
            t = one(R) / temp
            for jr in 1:n
                VR[jr, jc] *= t
            end
        end
    end
    return VR
end

# ── ggev core (REAL), IN-PLACE: every output is caller-provided. A/B overwritten (Schur form). ─────────
# `VR` doubles as the (unused) Q slot of gghrd!/hgeqz!: both take compq='N', under which `_qz_init_qz!`
# is a no-op and every Q write is guarded by `ilq`, so the argument is never touched. That removes the
# three empty placeholder matrices the allocating form used to build, with no workspace field.
function _ggev_core!(
        jobvl::Char, jobvr::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T},
        alphar::AbstractVector{T}, alphai::AbstractVector{T}, beta::AbstractVector{T},
        VL::AbstractMatrix{T}, VR::AbstractMatrix{T}
    ) where {T <: Real}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("ggev!: A and B must be square and the same size"))
    jobvl === 'N' || throw(ArgumentError("ggev!: left eigenvectors (jobvl='V') not implemented"))
    (jobvr === 'N' || jobvr === 'V') || throw(ArgumentError("ggev!: jobvr must be 'N' or 'V'"))
    wantvr = jobvr === 'V'
    n == 0 && return nothing
    # `alphaC` is the COMPLEX eigenvalue staging a REAL pencil still needs (hgeqz! reports α complex for
    # both types), so the borrow is `Complex{T}` — the arena's answer to the field that had to live on
    # `_l3ws(Complex{T})`. No fill!: hgeqz! assigns alpha[i] for every i in 1:n on every exit, before the
    # split below reads it back.
    # ESCAPE AUDIT: `alphaC` reaches only hgeqz! (qz.jl), which writes it and returns; gghrd!/tgevc! never
    # see it and nothing stores it. The scope has to span the whole pipeline because hgeqz! is what fills
    # alphaC and the alphar/alphai split is what empties it — both are inside, so no handle crosses the
    # closing `end`. `_ggev_qrB!` opens its own nested scope, which is exactly what nesting is for.
    @scope arn begin
        alphaC = borrow!(arn, Complex{T}, n)
        _ggev_qrB!(A, B, n, 'T')
        if wantvr
            gghrd!('N', 'I', A, B, VR, VR)
            hgeqz!('S', 'N', 'V', A, B, alphaC, beta, VR, VR)
            tgevc!('R', 'B', A, B, VL, VR)
        else
            gghrd!('N', 'N', A, B, VR, VR)
            hgeqz!('E', 'N', 'N', A, B, alphaC, beta, VR, VR)
        end
        @inbounds for i in 1:n
            alphar[i] = real(alphaC[i]); alphai[i] = imag(alphaC[i])
        end
    end
    wantvr && _ggev_normalize_real!(VR, alphai, n)
    return nothing
end

# Allocating form kept verbatim in behaviour and return value — `src/cabi/cabi_lapack.jl:1894` calls it.
function _ggev_run!(jobvl::Char, jobvr::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Real}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("ggev!: A and B must be square and the same size"))
    jobvl === 'N' || throw(ArgumentError("ggev!: left eigenvectors (jobvl='V') not implemented"))
    (jobvr === 'N' || jobvr === 'V') || throw(ArgumentError("ggev!: jobvr must be 'N' or 'V'"))
    alphar = zeros(T, n); alphai = zeros(T, n); beta = zeros(T, n)
    VR = Matrix{T}(undef, n, jobvr === 'V' ? n : 0)
    n == 0 && return alphar, alphai, beta, VR
    _ggev_core!(jobvl, jobvr, A, B, alphar, alphai, beta, VR, VR)
    return alphar, alphai, beta, VR
end

# ── ggev core (COMPLEX), IN-PLACE. Same VR-as-dummy-Q argument as the real path. ───────────────────────
function _ggev_core!(
        jobvl::Char, jobvr::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T},
        alpha::AbstractVector{T}, beta::AbstractVector{T},
        VL::AbstractMatrix{T}, VR::AbstractMatrix{T}
    ) where {T <: Complex}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("ggev!: A and B must be square and the same size"))
    jobvl === 'N' || throw(ArgumentError("ggev!: left eigenvectors (jobvl='V') not implemented"))
    (jobvr === 'N' || jobvr === 'V') || throw(ArgumentError("ggev!: jobvr must be 'N' or 'V'"))
    wantvr = jobvr === 'V'
    n == 0 && return nothing
    _ggev_qrB!(A, B, n, 'C')
    if wantvr
        gghrd!('N', 'I', A, B, VR, VR)
        hgeqz!('S', 'N', 'V', A, B, alpha, beta, VR, VR)
        tgevc!('R', 'B', A, B, VL, VR)
        _ggev_normalize_cmplx!(VR, n)
    else
        gghrd!('N', 'N', A, B, VR, VR)
        hgeqz!('E', 'N', 'N', A, B, alpha, beta, VR, VR)
    end
    return nothing
end

function _ggev_run!(jobvl::Char, jobvr::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Complex}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("ggev!: A and B must be square and the same size"))
    jobvl === 'N' || throw(ArgumentError("ggev!: left eigenvectors (jobvl='V') not implemented"))
    (jobvr === 'N' || jobvr === 'V') || throw(ArgumentError("ggev!: jobvr must be 'N' or 'V'"))
    alpha = zeros(T, n); beta = zeros(T, n)
    VR = Matrix{T}(undef, n, jobvr === 'V' ? n : 0)
    n == 0 && return alpha, beta, VR
    _ggev_core!(jobvl, jobvr, A, B, alpha, beta, VR, VR)
    return alpha, beta, VR
end

# ── gges core (jobvsl='V', jobvsr='V'), IN-PLACE: Schur form S,P (A,B overwritten) + VSL,VSR. ──────────
function _gges_core!(
        A::AbstractMatrix{T}, B::AbstractMatrix{T}, alpha::AbstractVector{<:Complex},
        beta::AbstractVector{T}, VSL::AbstractMatrix{T}, VSR::AbstractMatrix{T}
    ) where {T <: Real}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("gges!: A and B must be square and the same size"))
    n == 0 && return nothing
    _ggev_qrB!(A, B, n, 'T', VSL)                          # VSL starts as Q_B, gghrd/hgeqz post-multiply Q
    gghrd!('V', 'I', A, B, VSL, VSR)
    hgeqz!('S', 'V', 'V', A, B, alpha, beta, VSL, VSR)
    return nothing
end

function _gges_core!(
        A::AbstractMatrix{T}, B::AbstractMatrix{T}, alpha::AbstractVector{T},
        beta::AbstractVector{T}, VSL::AbstractMatrix{T}, VSR::AbstractMatrix{T}
    ) where {T <: Complex}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("gges!: A and B must be square and the same size"))
    n == 0 && return nothing
    _ggev_qrB!(A, B, n, 'C', VSL)                          # VSL starts as Q_B (see the real method above)
    gghrd!('V', 'I', A, B, VSL, VSR)
    hgeqz!('S', 'V', 'V', A, B, alpha, beta, VSL, VSR)
    return nothing
end

# Allocating forms kept verbatim — `src/cabi/cabi_lapack.jl:1968/1990` call these.
function _gges_run!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Real}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("gges!: A and B must be square and the same size"))
    beta = zeros(T, n)
    VSL = Matrix{T}(undef, n, n); VSR = Matrix{T}(undef, n, n)
    n == 0 && return A, B, Complex{T}[], beta, VSL, VSR
    alphaC = Vector{Complex{T}}(undef, n)
    _gges_core!(A, B, alphaC, beta, VSL, VSR)
    return A, B, alphaC, beta, VSL, VSR
end

function _gges_run!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Complex}
    n = size(A, 1)
    size(A, 2) == n && size(B, 1) == n && size(B, 2) == n ||
        throw(DimensionMismatch("gges!: A and B must be square and the same size"))
    alpha = zeros(T, n); beta = zeros(T, n)
    VSL = Matrix{T}(undef, n, n); VSR = Matrix{T}(undef, n, n)
    n == 0 && return A, B, alpha, beta, VSL, VSR
    _gges_core!(A, B, alpha, beta, VSL, VSR)
    return A, B, alpha, beta, VSL, VSR
end

# Lesson 9 — `vr`/`vsl`/`vsr` are WRITTEN by gghrd!/hgeqz!/tgevc! while A and B are still being READ and
# transformed, so an input passed as an output buffer is silent garbage rather than the netlib
# overwrite-in-place a caller might expect. `vl === vr` / `vsl === vsr` are equally fatal (gghrd!
# accumulates the LEFT rotations into VSL and the RIGHT ones into VSR simultaneously).
@inline function _ggev_check_alias(A, B, VL, VR, f::String, nl::String, nr::String)
    (Base.mightalias(VR, A) || Base.mightalias(VR, B)) && throw(ArgumentError("$f: $nr must not alias A or B"))
    (Base.mightalias(VL, A) || Base.mightalias(VL, B)) && throw(ArgumentError("$f: $nl must not alias A or B"))
    Base.mightalias(VL, VR) && throw(ArgumentError("$f: $nl and $nr must be distinct"))
    return nothing
end

"""
    ggev!(jobvl, jobvr, A, B) -> (alphar, alphai, beta, vl, vr)   [real]
    ggev!(jobvl, jobvr, A, B) -> (alpha, beta, vl, vr)             [complex]
    ggev!(jobvl, jobvr, A, B, alphar, alphai, beta, vl, vr)        [real, in-place]
    ggev!(jobvl, jobvr, A, B, alpha, beta, vl, vr)                 [complex, in-place]

Generalized eigenvalues and (optionally) right eigenvectors of the pencil `(A,B)` — `A·x = λ·B·x` with
`λ = α/β` (LAPACK dggev/zggev). `jobvr='V'` computes right eigenvectors into `vr`, `'N'` skips them.
`jobvl='V'` (left eigenvectors) is not implemented (throws). For real `A,B`, a complex-conjugate
eigenvalue pair occupies two consecutive `vr` columns as (real, imag) parts (LAPACK real convention).
`A` and `B` are overwritten. `vl` is always empty (left vectors unsupported).

The trailing-buffer form writes into the caller's `alphar`/`alphai`/`beta`/`vr` (`alpha`/`beta`/`vr` for
complex) and allocates nothing; `vl` is accepted and ignored. It throws if `vl`/`vr` alias `A`, `B` or
each other.
"""
function ggev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Real}
    alphar, alphai, beta, VR = _ggev_run!(Char(jobvl), Char(jobvr), A, B)
    return alphar, alphai, beta, Matrix{T}(undef, size(A, 1), 0), VR
end
function ggev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Complex}
    alpha, beta, VR = _ggev_run!(Char(jobvl), Char(jobvr), A, B)
    return alpha, beta, Matrix{T}(undef, size(A, 1), 0), VR
end

function ggev!(
        jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T},
        alphar::AbstractVector{T}, alphai::AbstractVector{T}, beta::AbstractVector{T},
        vl::AbstractMatrix{T}, vr::AbstractMatrix{T}
    ) where {T <: Real}
    _ggev_check_alias(A, B, vl, vr, "ggev!", "vl", "vr")
    n = size(A, 1)
    (length(alphar) == n && length(alphai) == n && length(beta) == n) ||
        throw(DimensionMismatch("ggev!: alphar, alphai, beta must have length n"))
    _ggev_core!(Char(jobvl), Char(jobvr), A, B, alphar, alphai, beta, vl, vr)
    return alphar, alphai, beta, vl, vr
end
function ggev!(
        jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T},
        alpha::AbstractVector{T}, beta::AbstractVector{T},
        vl::AbstractMatrix{T}, vr::AbstractMatrix{T}
    ) where {T <: Complex}
    _ggev_check_alias(A, B, vl, vr, "ggev!", "vl", "vr")
    n = size(A, 1)
    (length(alpha) == n && length(beta) == n) ||
        throw(DimensionMismatch("ggev!: alpha and beta must have length n"))
    Base.mightalias(alpha, beta) && throw(ArgumentError("ggev!: alpha and beta must be distinct"))
    _ggev_core!(Char(jobvl), Char(jobvr), A, B, alpha, beta, vl, vr)
    return alpha, beta, vl, vr
end

"""
    gges!(jobvsl, jobvsr, A, B) -> (A, B, alpha, beta, vsl, vsr)
    gges!(jobvsl, jobvsr, A, B, alpha, beta, vsl, vsr)   [in-place]

Generalized Schur decomposition of the pencil `(A,B)` (LAPACK dgges/zgges): `A`→`S`, `B`→`P` (generalized
Schur form, `A₀ = vsl·S·vsrᴴ`, `B₀ = vsl·P·vsrᴴ`), `alpha`/`beta` the generalized eigenvalues (`λ=α/β`,
`alpha` complex), and the Schur vectors `vsl`/`vsr`. Only `jobvsl=jobvsr='V'` is supported (what `schur(A,B)`
requests); other values fall back to computing both sets of vectors. `A` and `B` are overwritten.

The trailing-buffer form writes into the caller's `alpha`/`beta`/`vsl`/`vsr` and allocates nothing. It
throws if `vsl`/`vsr` alias `A`, `B` or each other.
"""
function gges!(jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Real}
    S, P, alphaC, beta, VSL, VSR = _gges_run!(A, B)
    return S, P, alphaC, beta, VSL, VSR
end
function gges!(jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Complex}
    S, P, alpha, beta, VSL, VSR = _gges_run!(A, B)
    return S, P, alpha, beta, VSL, VSR
end

function gges!(
        jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T},
        alpha::AbstractVector{<:Complex}, beta::AbstractVector{T},
        vsl::AbstractMatrix{T}, vsr::AbstractMatrix{T}
    ) where {T <: Real}
    _ggev_check_alias(A, B, vsl, vsr, "gges!", "vsl", "vsr")
    n = size(A, 1)
    (length(alpha) == n && length(beta) == n) ||
        throw(DimensionMismatch("gges!: alpha and beta must have length n"))
    _gges_core!(A, B, alpha, beta, vsl, vsr)
    return A, B, alpha, beta, vsl, vsr
end
function gges!(
        jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T},
        alpha::AbstractVector{T}, beta::AbstractVector{T},
        vsl::AbstractMatrix{T}, vsr::AbstractMatrix{T}
    ) where {T <: Complex}
    _ggev_check_alias(A, B, vsl, vsr, "gges!", "vsl", "vsr")
    n = size(A, 1)
    (length(alpha) == n && length(beta) == n) ||
        throw(DimensionMismatch("gges!: alpha and beta must have length n"))
    Base.mightalias(alpha, beta) && throw(ArgumentError("gges!: alpha and beta must be distinct"))
    _gges_core!(A, B, alpha, beta, vsl, vsr)
    return A, B, alpha, beta, vsl, vsr
end
