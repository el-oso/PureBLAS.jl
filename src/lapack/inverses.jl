# LAPACK EXPLICIT INVERSES — trtri / potri / getri, native (Mode-2) entry points.
#
# Same story as `solves.jl`, one step later. These existed ONLY inside the `@ccallable` shims in
# cabi_lapack.jl, with the same two consequences: no AD-traceable Mode-2 API, and — because
# bench/plots.jl compares `PureBLAS.foo!` against `LinearAlgebra.LAPACK.foo!` — no way to GATE them.
# All three are LBT-forwarded into user code by `cabi_forward.jl`, so they ship; they were simply never
# measured. A user's PureOSQP run found `potri` at 0.67x (322 us vs OpenBLAS 216 us at n=200) that no
# red cell could report, because the op had no cell to be red.
#
# THE SHIMS CALL THESE, so there is exactly one implementation of each.
#
# ── Why these were slow, and it was the ALGORITHM, not the kernels ────────────────────────────────
# Both `trtri` and `potri` used to solve against a DENSE n×n identity:
#     trtri:  one trsm  vs I   ->  n³      (LAPACK's blocked trtri: n³/3)
#     potri:  two trsms vs I   ->  2n³     (LAPACK: trtri + lauum   = 2n³/3)
# i.e. 3x the necessary work. potri losing at only 1.49x the time while doing 3x the flops says the
# underlying kernels were already ~2x more efficient per flop; the algorithm was the whole gap.
#
# `_trtri!` (blas3/level3.jl) is the blocked recursive triangular inverse that trsm already uses
# internally — n³/3, because it never touches the zero triangle. It was one file away the whole time.
# This is the "wire the fastest path" rule: a C-ABI shim must route to the FASTEST correct kernel, and
# these two declined to.

# ── _trtri_ip!: IN-PLACE blocked triangular inverse (dtrtri.f / dtrti2.f) ─────────────────────────
# `_trtri!` in blas3/level3.jl is deliberately OUT-of-place: trsm calls it and needs its input intact.
# For the standalone trtri/potri entries that is the wrong trade, and the gate shows it on both ends —
# full-arms Zen3, PB/OpenBLAS, with the out-of-place version:
#
#   n        8     32     50    100    128    256    512   1000   1024   2048
#   trtri  0.66   1.10   1.35   1.65   1.85   1.90   1.11   0.76   0.81   0.62
#   potri  0.70   0.95   1.08   1.36   1.48   1.61   1.05   0.84   0.89   0.73
#
# One cause, two symptoms: an n×n scratch. At n=8 the allocation IS the runtime. At n≥1000 it doubles
# the working set and writes n² where n²/2 is needed (the scratch is filled including its zero
# triangle), so the memory traffic — not the flops — sets the ceiling. The flop count was already
# LAPACK's by then, which is why the middle of the range is 1.6–1.9x and the ends still lose.
#
# Blocked recursion, LAPACK's own partitioning. Lower L = [L11 0; L21 L22]:
#     L⁻¹ = [L11⁻¹ 0; −L22⁻¹·L21·L11⁻¹, L22⁻¹]
# so invert both diagonal blocks in place, then update L21 with two trmm's against the ALREADY-inverted
# blocks. Upper is the mirror. No scratch at any level.
@inline function _trtri_ip_base!(A::AbstractMatrix{T}, up::Bool, unit::Bool) where {T}
    n = size(A, 1)
    if up
        # Ascending i: writing A[i,j] never clobbers an A[k,j] with k > i that a later row still needs.
        @inbounds for j in 1:n
            ajj = unit ? -one(T) : (A[j, j] = inv(A[j, j]); -A[j, j])
            for i in 1:(j - 1)
                s = zero(T)
                for k in i:(j - 1)
                    s += ((unit && i == k) ? one(T) : A[i, k]) * A[k, j]
                end
                A[i, j] = ajj * s
            end
        end
    else
        # Descending i, for the same reason mirrored.
        @inbounds for j in n:-1:1
            ajj = unit ? -one(T) : (A[j, j] = inv(A[j, j]); -A[j, j])
            for i in n:-1:(j + 1)
                s = zero(T)
                for k in (j + 1):i
                    s += ((unit && i == k) ? one(T) : A[i, k]) * A[k, j]
                end
                A[i, j] = ajj * s
            end
        end
    end
    return A
end

function _trtri_ip!(A::AbstractMatrix{T}, up::Bool, unit::Bool) where {T}
    n = size(A, 1)
    n == 0 && return A
    n <= _trtri_base() && return _trtri_ip_base!(A, up, unit)
    h = n ÷ 2
    A11 = view(A, 1:h, 1:h)
    A22 = view(A, (h + 1):n, (h + 1):n)
    dg = unit ? 'U' : 'N'
    _trtri_ip!(A11, up, unit)
    _trtri_ip!(A22, up, unit)
    if up
        A12 = view(A, 1:h, (h + 1):n)
        trmm!(A12, A11; side = 'L', uplo = 'U', transA = 'N', diag = dg, alpha = -one(T))
        trmm!(A12, A22; side = 'R', uplo = 'U', transA = 'N', diag = dg, alpha = one(T))
    else
        A21 = view(A, (h + 1):n, 1:h)
        trmm!(A21, A22; side = 'L', uplo = 'L', transA = 'N', diag = dg, alpha = -one(T))
        trmm!(A21, A11; side = 'R', uplo = 'L', transA = 'N', diag = dg, alpha = one(T))
    end
    return A
end

# ── trtri: in-place inverse of a triangular A (dtrtri.f) ──────────────────────────────────────────
# Overwrites the `uplo` triangle of A with A⁻¹. `diag='U'` treats the diagonal as unit and leaves it.
# Routes to `_trtri_ip!` — no scratch. See the table above it for why the out-of-place `_trtri!`, which
# trsm needs for its own reasons, is the wrong kernel to reach for here.
function trtri!(A::AbstractMatrix{T}; uplo::AbstractChar = 'L', diag::AbstractChar = 'N') where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("trtri!: A must be square"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("trtri!: uplo must be 'L' or 'U'"))
    (diag == 'N' || diag == 'U') || throw(ArgumentError("trtri!: diag must be 'N' or 'U'"))
    n == 0 && return A
    return _trtri_ip!(A, uplo == 'U', diag == 'U')
end

# ── lauum: in-place triangular product (dlauum.f) ────────────────────────────────────────────────
# Lower: M := Mᴴ·M.  Upper: M := M·Mᴴ.  Only the `uplo` triangle is read and written.
#
# WHY THIS EXISTS RATHER THAN JUST CALLING syrk. The product of a triangular matrix with its own
# adjoint is n³/3, but syrk on the same operand treats it as DENSE and pays n³ — the zero triangle is
# multiplied anyway. Measured: potri built on a dense syrk gates at n=256 (1.33) and then FALLS OVER
# at n=1024 (0.59 vs OpenBLAS), because at large n the flop ratio is the whole story — 4n³/3 against
# LAPACK's 2n³/3 is 2x the work and ≈0.5x the ratio, which is what the number says. Small n hides it
# behind per-call overhead, which is exactly how a shortcut like that survives a spot check.
#
# The recursion, for lower M = [M11 0; M21 M22] and R = MᴴM:
#     R11 = M11ᴴM11 + M21ᴴM21     (recurse, then syrk accumulating with beta=1)
#     R21 = M22ᴴM21               (trmm, side='L')
#     R22 = M22ᴴM22               (recurse)
# The ORDER matters: R11's syrk must read M21 before the trmm overwrites it. Upper is the mirror with
# M = [M11 M12; 0 M22], R = MMᴴ, trmm on the right.
#
# The base does one dense syrk on a b×b block through a copy (syrk cannot alias C and A) into OWNED
# scratch (`_lauum_tmp`), so the routine allocates nothing. The copy is safe against a reused buffer:
# `copyto!` writes all n² entries before anything is read, and the triangle-zeroing only overwrites.
# The recursion is strictly sequential (M11 subtree completes, then M22), so no two base cases are
# ever live at once and the single shared buffer cannot be clobbered mid-use. That block's
# 3x waste is bounded by 2b²/n² of the total — 0.2% at b=16, n=1024 — so the base stays small. It
# reuses `_trtri_base()`: the same recursion-base criterion for the same triangular shape, and that
# knob is already documented FLAT across 8/16/32/64 on all three µarchs.
function _lauum!(M::AbstractMatrix{T}, up::Bool) where {T}
    n = size(M, 1)
    n == 0 && return M
    if n <= _trtri_base()
        Tmp = _lauum_tmp(T, n)          # owned scratch, reused across calls — no allocation
        copyto!(Tmp, M)                 # writes all n² before any read; the zeroing below overwrites
        if up                                   # zero the strict lower so the dense product is exact
            @inbounds for j in 1:n, i in (j + 1):n
                Tmp[i, j] = zero(T)
            end
        else
            @inbounds for j in 1:n, i in 1:(j - 1)
                Tmp[i, j] = zero(T)
            end
        end
        if T <: Complex
            herk!(M, Tmp; uplo = up ? 'U' : 'L', trans = up ? 'N' : 'C', alpha = true, beta = false)
        else
            syrk!(M, Tmp; uplo = up ? 'U' : 'L', trans = up ? 'N' : 'T',
                  alpha = one(T), beta = zero(T))
        end
        return M
    end
    ct = T <: Complex ? 'C' : 'T'
    h = n ÷ 2
    M11 = view(M, 1:h, 1:h)
    M22 = view(M, (h + 1):n, (h + 1):n)
    if up
        M12 = view(M, 1:h, (h + 1):n)
        _lauum!(M11, true)                                        # R11 = M11·M11ᴴ
        if T <: Complex
            herk!(M11, M12; uplo = 'U', trans = 'N', alpha = true, beta = true)   # += M12·M12ᴴ
        else
            syrk!(M11, M12; uplo = 'U', trans = 'N', alpha = one(T), beta = one(T))
        end
        trmm!(M12, M22; side = 'R', uplo = 'U', transA = ct, alpha = true)        # R12 = M12·M22ᴴ
        _lauum!(M22, true)
    else
        M21 = view(M, (h + 1):n, 1:h)
        _lauum!(M11, false)                                       # R11 = M11ᴴ·M11
        if T <: Complex
            herk!(M11, M21; uplo = 'L', trans = 'C', alpha = true, beta = true)   # += M21ᴴ·M21
        else
            syrk!(M11, M21; uplo = 'L', trans = 'T', alpha = one(T), beta = one(T))
        end
        trmm!(M21, M22; side = 'L', uplo = 'L', transA = ct, alpha = true)        # R21 = M22ᴴ·M21
        _lauum!(M22, false)
    end
    return M
end

# ── potri: A⁻¹ from the Cholesky factor already in A (dpotri.f) ───────────────────────────────────
# Overwrites the `uplo` triangle of A with A⁻¹ (which is Hermitian, so LAPACK stores one triangle).
#
# M := the factor's inverse (`_trtri!`). Then A = L·Lᴴ ⇒ A⁻¹ = L⁻ᴴ·L⁻¹ = Mᴴ·M, a rank-k update — so
# syrk (real) / herk (complex), both of which gate. Upper stores A = Uᴴ·U ⇒ A⁻¹ = U⁻¹·U⁻ᴴ = M·Mᴴ,
# hence trans='N' there.
#
# A dense syrk here was tried first and MEASURED INADEQUATE: it gates at n=256 (1.33) and collapses at
# n=1024 (0.59), because treating the triangular M as dense costs n³ instead of n³/3 and at large n the
# flop ratio is the whole story. `_lauum!` above does the triangular product properly, which brings the
# total to trtri + lauum = 2n³/3 — LAPACK's own count.
function potri!(A::AbstractMatrix{T}; uplo::AbstractChar = 'L') where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potri!: A must be square"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("potri!: uplo must be 'L' or 'U'"))
    n == 0 && return A
    up = uplo == 'U'
    trtri!(A; uplo = uplo, diag = 'N')     # A := M, the factor's inverse (in place, n³/3)
    _lauum!(A, up)                         # A := MᴴM (lower) / MMᴴ (upper), n³/3
    return A
end

# ── getri: A⁻¹ from the LU factors in A (dgetri.f) ────────────────────────────────────────────────
# The third shim with the same defect as its two siblings: it built a dense n×n identity, applied the
# row pivots to it, and ran two trsm solves — 2n³, against LAPACK's 4n³/3 — plus the same n×n scratch
# that set the ceiling for trtri/potri at both ends of the size range.
#
# LAPACK's algorithm, and the one used here:
#   1. inv(U) in place                                     n³/3   (`trtri!`, now scratch-free)
#   2. solve inv(A)·L = inv(U), block column at a time      n³
#   3. undo the COLUMN pivots (inverse order)
# Step 2 walks block columns right-to-left: stash the strict-lower part of the block (that is L), zero
# it in A, subtract the already-computed columns to the right via one gemm, then a right-side unit-lower
# trsm against the block's own diagonal. The workspace is n×nb, not n×n — nb from `_lu_nb`, the same
# validated LU panel width, reused exactly as `_pstrf_nb` reuses it — and it is OWNED (`_getri_work`),
# so getri! allocates nothing. Safe against a reused buffer: each block column writes rows j:n of its
# jb columns (the jb×jb diagonal zeroing plus the strict-lower stash) and reads only rows j:n; rows
# 1:j−1 are never read, so no stale value from a previous call can be consumed.
function getri!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("getri!: A must be square"))
    length(ipiv) >= n || throw(DimensionMismatch("getri!: ipiv shorter than n"))
    n == 0 && return A
    trtri!(A; uplo = 'U', diag = 'N')                       # 1. inv(U), in place
    nb = max(1, min(_lu_nb(n), n))
    W = _getri_work(T, n, nb)                               # owned scratch, reused — no allocation
    j = ((n - 1) ÷ nb) * nb + 1                             # start of the LAST block column
    while j >= 1
        jb = min(nb, n - j + 1)
        # Stash L's strict-lower block and zero it in A. The jb×jb diagonal region is zeroed first: the
        # unit-diagonal and strict-upper entries are never read by a correct trsm, but leaving them as
        # uninitialised memory is a trap for any kernel that touches a full tile.
        @inbounds for c in 1:jb, i in j:(j + jb - 1)
            W[i, c] = zero(T)
        end
        @inbounds for jj in j:(j + jb - 1)
            c = jj - j + 1
            for i in (jj + 1):n
                W[i, c] = A[i, jj]
                A[i, jj] = zero(T)
            end
        end
        if j + jb <= n
            gemm!(view(A, 1:n, j:(j + jb - 1)), view(A, 1:n, (j + jb):n), view(W, (j + jb):n, 1:jb);
                  alpha = -one(T), beta = one(T))
        end
        trsm!(view(A, 1:n, j:(j + jb - 1)), view(W, j:(j + jb - 1), 1:jb);
              side = 'R', uplo = 'L', transA = 'N', diag = 'U', alpha = one(T))
        j -= nb
    end
    @inbounds for jj in (n - 1):-1:1                        # 3. undo the column pivots
        jp = Int(ipiv[jj])
        if jp != jj
            for i in 1:n
                A[i, jj], A[i, jp] = A[i, jp], A[i, jj]
            end
        end
    end
    return A
end
