# Rank-deficient least squares via the SINGULAR VALUE DECOMPOSITION (LAPACK gelsd). Solves
# min‖A·X − B‖₂ for a possibly rank-deficient A, returning the MINIMUM-NORM solution
# X = V·Σ⁺·Uᴴ·B, where singular values ≤ rcond·σ_max are treated as zero (their reciprocals dropped).
# Composed from PureBLAS's own economy SVD (`gesvd!`, svd.jl — gebrd bidiagonalization + bdsqr/bdsdc)
# and `gemm!` (gemm.jl) for the two back-projections. Returns the singular values too.
# Generic over Float64/ComplexF32/ComplexF64 (native gesvd! kernels); Float32-real is computed in
# Float64 (see below). Mirrors dgelsd/zgelsd: bidiagonalize → SVD → rcond-threshold → solve.

# Default rank-cut when the caller passes rcond ∉ (0,1). PureBLAS's gelsd composes a FULL SVD and then
# thresholds σᵢ ≤ rcond·σ₁ (vs LAPACK dlalsd's D&C-integrated deflation, which collapses null σ's to
# ~machine-zero). For an exactly rank-deficient A the compose-SVD path leaves the null σ's at a FLOOR of
# ~a-few·eps·σ₁ — above LAPACK's eps·σ₁ cut (measured up to 3.1e-16·σ₁ > eps) — so a fixed eps·σ₁ threshold
# splits the null cluster, keeps a ~3e-16 σ, and divides by it → a ‖x‖~1e14 garbage solution (Fable
# adversarial review). Scale the default cut with the problem size — `min(m,n)·eps·σ₁`, exactly Julia's
# own `pinv` rtol convention — which clears the O(√n)·eps SVD null floor while retaining every genuine
# singular value (the spectrum shows a >10-order gap between the smallest true σ and the null cluster).
@inline _gelsd_eps(::Type{R}, mn::Int) where {R <: Real} = R(max(mn, 1)) * eps(R)

# ── Fast full-rank path: solve the BIDIAGONAL, never form the singular vectors ─────────────────────────
# gelsd needs the SVD only to DETECT rank. For a full-rank A the minimum-norm solution IS the ordinary
# least-squares solution, and `gebrd!` already hands us A = Q·B·Pᵀ with B upper bidiagonal — so
# min‖A·x − b‖ becomes min‖B·y − Qᵀb‖ with x = P·y, and that bidiagonal solve is O(n) back-substitution.
# Both singular-vector sets drop out; the SVD shrinks to a values-only dqds on the bidiagonal.
#
# WHY (kb `gelsd-forms-vectors-lapack-does-not.md`): PB's gelsd at n=1000 was 412.5 ms vs LAPACK's
# 209.0. `gebrd` itself is at PARITY (148.2 vs 152.0) — the whole gap is that we compose a full economy
# SVD (`gesvd!` = 89.8% of PB's time) where dgelsd's `dlalsd` applies its D&C directly to the RHS and
# never forms U or V. Rather than port dlasda/dlalsa, drop the vectors entirely. Measured pricing of
# gebrd + values-only dqds against the reference (bench/probes/gelsd_valsonly.jl, Zen4):
#     n=256   9.511 ms ref vs 5.338 → 1.78      n=512  62.272 vs 41.311 → 1.51
#     n=1000  211.593 ms ref vs 179.410 → 1.18   (the O(n²) applies below are inside the noise)
#
# WHEN IT DECLINES. Rank deficiency is the reason gelsd exists, and at the rank cut the two paths'
# null-σ floors differ by a few ulp — so a bare `rank == n` test would let a ~1e-16 pivot into the
# back-substitution and return a ‖x‖~1e14 answer where the SVD path returns the min-norm solution. The
# guard therefore demands a MARGIN: σ_min > √eps·σ₁ (κ₂ < 6.7e7). That also bounds the solve: for an
# upper bidiagonal B, ‖B⁻¹‖ ≥ 1/|dᵢ| for every i, hence |dᵢ| ≥ σ_min > 0 — no tiny pivot is reachable.
# Anything below the margin falls back to the composed SVD, which is what the SVD is FOR.
#
# Returns the rank (== n) when it solved, 0 when it declined and the caller must take the SVD path (B is
# untouched in that case; only `s` has been written, with the same singular values the SVD path produces).
_gelsd_fast!(::AbstractMatrix, ::AbstractMatrix, ::Real, ::AbstractVector) = 0

function _gelsd_fast!(
        A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64}, rcond::Real,
        s::AbstractVector{Float64}
    )
    m, n = size(A)
    (m >= n && n >= 1) || return 0                # m<n is min-norm-underdetermined: not a back-substitution
    ws = _svdws(Float64)
    _svd_grow_bidiag!(ws, m, n)
    ws.d = _gv(ws.d, n); ws.e = _gv(ws.e, max(n, 1))
    ws.tauq = _gv(ws.tauq, n); ws.taup = _gv(ws.taup, n)
    # `VQ` is the SVD path's Q-reflector panel; here it holds gebrd's whole output, whose lower triangle
    # IS that panel. A `Matrix` (not an arena borrow) on purpose: `_house_left!` takes its SIMD path only
    # for a `_dense1` v, and a column view of a `PtrMatrix` is not a `StridedVector` (ptrmat.jl) — it
    # would have silently walked scalar over the O(m·n) reflector applies.
    ws.VQ = _gm(ws.VQ, m, n)
    ne = max(n - 1, 0)
    Ac = view(ws.VQ, 1:m, 1:n)
    d = view(ws.d, 1:n); e = view(ws.e, 1:ne)
    tauq = view(ws.tauq, 1:n); taup = view(ws.taup, 1:n)
    @inbounds for j in 1:n, i in 1:m
        Ac[i, j] = A[i, j]
    end
    gebrd!(Ac, d, e, tauq, taup, ws)              # A = Q·B·Pᵀ, B upper bidiagonal (d, e)
    @scope arn begin
        # Singular VALUES on a COPY — dqds destroys its (d,e), and (d,e) is the operator we solve with.
        # The copy of d lands directly in the caller's `s`, which is where the values belong anyway.
        ecp = borrow!(arn, Float64, max(ne, 1))
        @inbounds for i in 1:n
            s[i] = d[i]
        end
        @inbounds for i in 1:ne
            ecp[i] = e[i]
        end
        sv = view(s, 1:n); ev = view(ecp, 1:ne)
        _dlasq1!(sv, ev, ws.dqds_Z, ws.dqds_st) != 0 && bdsqr!(sv, ev, nothing, nothing)
        rcnd = (rcond <= 0 || rcond >= 1) ? _gelsd_eps(Float64, n) : Float64(rcond)
        (sv[n] > rcnd * sv[1] && sv[n] > sqrt(eps(Float64)) * sv[1]) || return 0
    end
    nrhs = size(B, 2)
    # c := Qᵀ·b.  Q = H₁⋯H_n ⇒ Qᵀ = H_n⋯H₁, so apply in FORWARD order. H_i acts on rows i:m with
    # v_i[1] ≡ 1 (`_house_left!` ignores v[1]; the diagonal of Ac holds d, not the unit).
    # NOTE for the C-ABI arity (cabi_lapack.jl): there `B` is a `PtrMatrix`, and a SubArray of one is not
    # a `StridedMatrix`, so `_strided1` is false and this walks the scalar arm — O(m·n·nrhs), ~1% of the
    # routine at n=1000, nrhs=1. Widening `_strided1` to SubArrays-of-PtrMatrix touches every kernel, so
    # it is not done here; revisit if a wide-nrhs C-ABI caller ever makes it matter.
    @inbounds for i in 1:n
        _house_left!(view(B, i:m, 1:nrhs), view(Ac, i:m, i), tauq[i])
    end
    # y := B⁻¹·c.  Rows n+1:m of c are the least-squares residual and drop out of the solve.
    @inbounds for jc in 1:nrhs
        B[n, jc] /= d[n]
        for i in (n - 1):-1:1
            B[i, jc] = (B[i, jc] - e[i] * B[i + 1, jc]) / d[i]
        end
    end
    # x := P·y.  P = R₁⋯R_{n−1} ⇒ apply in REVERSE order. R_i acts on rows i+1:n with w_i[1] ≡ 1 (at
    # column i+1) and w_i[2:] = Ac[i, i+2:n] — a ROW slice, hence the scalar `_house_left!` arm; it is
    # O(n²) total against the O(m·n²) bidiagonalization.
    @inbounds for i in (n - 1):-1:1
        _house_left!(view(B, (i + 1):n, 1:nrhs), view(Ac, i, (i + 1):n), taup[i])
    end
    return n
end

# Solve min‖A·X − B‖₂ (A m×n). B is size ≥ max(m,n) × nrhs (LAPACK ldb): input rows 1:m hold b,
# output rows 1:n hold X. rcond thresholds the singular values (∉(0,1) ⇒ machine precision, per
# dlalsd). Overwrites A and B. Returns (B, rank, s) with s the descending singular values (length
# min(m,n)).
#
# IN-PLACE FORM (caller-owned `s` last, mirroring getrf!(A, ipiv)): `s` needs length ≥ min(m,n) and
# holds the descending singular values on exit — EXCEPT when min(m,n)==0, where the routine returns
# immediately and leaves `s` untouched. `s` must not alias `A` or `B`: gesvd! fills `s` before the
# `Uᴴ·b` gemm! reads B, so an aliased `s` would destroy the right-hand side (and `A` is copied to
# workspace first, so that half is only a latent hazard — both are rejected).
function gelsd!(
        A::AbstractMatrix{T}, B::AbstractMatrix{T}, rcond::Real,
        s::AbstractVector{<:Real}
    ) where {T <: BlasFloat}
    m, n = size(A); mn = min(m, n); R = real(T); nrhs = size(B, 2)
    size(B, 1) >= max(m, n) || _throw_brows_mn(:gelsd!, size(B, 1), max(m, n))
    length(s) >= mn || throw(ArgumentError("gelsd!: length(s) must be ≥ min(m,n)"))
    (Base.mightalias(s, A) || Base.mightalias(s, B)) &&
        throw(ArgumentError("gelsd!: `s` must not alias `A` or `B`"))
    mn == 0 && return B, 0, s
    # Full-rank, well-conditioned, real, m ≥ n ⇒ solve the bidiagonal directly and skip both singular
    # vector sets (see `_gelsd_fast!`). It returns 0 — having touched only `s` — when it declines.
    rkf = _gelsd_fast!(A, B, rcond, s)
    rkf > 0 && return B, rkf, s
    sv = view(s, 1:mn)
    # economy SVD  A = U·diag(s)·Vᴴ  (U m×mn, s descending, Vt = Vᴴ mn×n). All four are borrowed at the
    # exact shape with the default (exact) ld — same criterion as gels!: `_wsgrow` already gives ld =
    # row count at every new high-water mark, so exact ld reproduces the gate's stride and removes the
    # dependence on call history. None of the four escapes; `s`/`B` are the caller's.
    @scope arn begin
        U = borrow!(arn, T, m, mn)
        Vt = borrow!(arn, T, mn, n)
        C = borrow!(arn, T, mn, nrhs)
        Ac = borrow!(arn, T, m, n)
        @inbounds for j in 1:n, i in 1:m       # gesvd! destroys its A; A itself stays the caller's
            Ac[i, j] = A[i, j]
        end
        gesvd!(Ac, U, sv, Vt)
        # effective rank: σ_i ≤ tol treated as zero (dlalsd: rcond∉(0,1) ⇒ rounding unit)
        rcnd = (rcond <= 0 || rcond >= 1) ? _gelsd_eps(R, mn) : R(rcond)
        tol = rcnd * sv[1]
        rank = 0
        @inbounds for i in 1:mn
            sv[i] > tol && (rank += 1)
        end
        # c := Uᴴ·b  (mn × nrhs)
        gemm!(C, U, view(B, 1:m, :); transA = 'C', alpha = one(T), beta = zero(T))
        # apply Σ⁺ (drop the reciprocals of the thresholded-to-zero singular values)
        @inbounds for jc in 1:nrhs, i in 1:mn
            C[i, jc] = sv[i] > tol ? C[i, jc] / sv[i] : zero(T)
        end
        # X := V·c = (Vᴴ)ᴴ·c  (n × nrhs) → B(1:n)   (C already holds Uᴴb, so overwriting B(1:m) is safe)
        gemm!(view(B, 1:n, :), Vt, C; transA = 'C', alpha = one(T), beta = zero(T))
        return B, rank, s
    end
end

# Allocating convenience form — identical behaviour and return value to the pre-workspace gelsd!.
function gelsd!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, rcond::Real) where {T <: BlasFloat}
    s = Vector{real(T)}(undef, min(size(A, 1), size(A, 2)))
    return gelsd!(A, B, rcond, s)
end

# Float32-real path: PureBLAS's gesvd! has no Float32-real kernel (svd.jl covers Float64 + complex).
# Compute in Float64 — MORE accurate than sgelsd but the same min-norm LS solution to Float32 tolerance.
# ponytail: promote-to-Float64; add a native Float32 SVD kernel if Float32 gelsd perf ever matters.
# The Float64 staging is BORROWED as Float64 out of the same arena the Float32 entry uses — byte ranges
# are disjoint by construction, so the old "different owner object from the Float32 workspace" argument
# (a `_l3ws(Float64)` reached UP from a Float32 entry) is retired rather than restated. The A and B
# staging share a row count, so `mb = max(m,n)` (B's ldb-mandated rows) sizes both and the A staging is
# then narrowed to its own m×n leading block. The nested Float64 `gelsd!` below opens its OWN scope
# inside this one — perfectly nested, so its borrows sit above these and rewind first.
function gelsd!(A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32}, rcond::Real, s::AbstractVector{<:Real})
    m, n = size(A); mn = min(m, n); nrhs = size(B, 2); mb = max(m, n)
    size(B, 1) >= mb || _throw_brows_mn(:gelsd!, size(B, 1), mb)
    length(s) >= mn || throw(ArgumentError("gelsd!: length(s) must be ≥ min(m,n)"))
    (Base.mightalias(s, A) || Base.mightalias(s, B)) &&
        throw(ArgumentError("gelsd!: `s` must not alias `A` or `B`"))
    mn == 0 && return B, 0, s
    @scope arn begin
        Ad0 = borrow!(arn, Float64, mb, n)
        Bd = borrow!(arn, Float64, mb, nrhs)
        sd = borrow!(arn, Float64, mn)
        Ad = view(Ad0, 1:m, 1:n)
        @inbounds for j in 1:n, i in 1:m
            Ad[i, j] = Float64(A[i, j])
        end
        @inbounds for j in 1:nrhs, i in 1:mb
            Bd[i, j] = Float64(B[i, j])
        end
        _, rank, _ = gelsd!(Ad, Bd, rcond, sd)
        @inbounds for j in 1:nrhs, i in 1:mb
            B[i, j] = Float32(Bd[i, j])
        end
        @inbounds for i in 1:mn
            s[i] = Float32(sd[i])
        end
        return B, rank, s
    end
end

function gelsd!(A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32}, rcond::Real)
    s = Vector{Float32}(undef, min(size(A, 1), size(A, 2)))
    return gelsd!(A, B, rcond, s)
end
