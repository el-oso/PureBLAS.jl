# Bunch-Kaufman factorization of a symmetric-indefinite / Hermitian matrix.
#
#   sytrf!  A = L·D·Lᵀ  (uplo='L')  or  U·D·Uᵀ  (uplo='U')   — symmetric (real OR complex-symmetric)
#   hetrf!  A = L·D·Lᴴ  (uplo='L')  or  U·D·Uᴴ  (uplo='U')   — Hermitian (complex; real ⇒ same as sytrf!)
#
# D is block-diagonal with 1×1 and 2×2 pivots. `ipiv` follows the LAPACK convention:
#   ipiv[k] > 0                         : 1×1 pivot, rows/cols k and ipiv[k] were interchanged.
#   ipiv[k] = ipiv[k-1] < 0 (uplo='L')  : 2×2 pivot in (k,k+1); rows/cols k+1 and -ipiv[k] interchanged.
#   ipiv[k] = ipiv[k+1] < 0 (uplo='U')  : 2×2 pivot in (k-1,k); rows/cols k-1 and -ipiv[k] interchanged.
#
# The pivot decision copies LAPACK's `dsytf2`/`zhetf2`/`zsytf2` EXACTLY: the α=(1+√17)/8 threshold and
# the 1×1-vs-2×2 selection (ABSAKK ≥ α·COLMAX → 1×1; else the ROWMAX branch). This is the UNBLOCKED
# Bunch-Kaufman (partial pivoting, LAPACK default `dsytrf`'s base case). Generic scalar loops keep the
# path AD-traceable and trim-safe over any T<:Number; the rank-1/rank-2 trailing updates are written as
# explicit triangular loops (the LAPACK `dsyr`/`dsyr2`/`dger` calls).
#
# `sytrs!`/`hetrs!` solve A·X=B from the factors (mirrors `dsytrs`/`zhetrs`/`zsytrs`).
#
# ponytail: scalar-generic, not SIMD-tuned — correctness-first standalone kernel; the trailing update is
# the syr/ger hot spot to vectorize later when assembled behind the L2 fast paths.

# LAPACK CABS1 / IDAMAX-IZAMAX magnitude: |re|+|im| (== abs on reals). Used for all off-diagonal
# comparisons and the column/row index-max, matching idamax (real) and izamax (complex).
@inline _bk_cabs1(x::Real) = abs(x)
@inline _bk_cabs1(x::Complex) = abs(real(x)) + abs(imag(x))

const _BK_ALPHA = (1 + sqrt(17.0)) / 8   # (1+√17)/8 ≈ 0.6403882, the Bunch-Kaufman pivot threshold

# ---------------------------------------------------------------------------------------------------
# Factorization
# ---------------------------------------------------------------------------------------------------

# Index of the max |·| (CABS1) entry among A[rows, col]; returns the row index. rows must be non-empty.
@inline function _bk_colmax(A, rows::UnitRange{Int}, col::Int)
    imax = first(rows); cmax = _bk_cabs1(A[imax, col])
    @inbounds for i in rows
        v = _bk_cabs1(A[i, col])
        if v > cmax
            cmax = v; imax = i
        end
    end
    return imax, cmax
end
# Index of the max |·| entry among A[row, cols] (a row scan); returns the col index.
@inline function _bk_rowmax(A, row::Int, cols::UnitRange{Int})
    jmax = first(cols); rmax = _bk_cabs1(A[row, jmax])
    @inbounds for j in cols
        v = _bk_cabs1(A[row, j])
        if v > rmax
            rmax = v; jmax = j
        end
    end
    return jmax, rmax
end

# uplo='L': A = L·D·Lᵀ (herm=false) or L·D·Lᴴ (herm=true). Reads/writes only the lower triangle.
function _sytf2_lower!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, herm::Bool) where {T}
    n = size(A, 1)
    Tr = real(T)
    alpha = Tr(_BK_ALPHA)
    info = 0
    k = 1
    @inbounds while k <= n
        kstep = 1
        absakk = herm ? abs(real(A[k, k])) : _bk_cabs1(A[k, k])
        if k < n
            imax, colmax = _bk_colmax(A, (k + 1):n, k)
        else
            imax = k; colmax = zero(Tr)
        end
        kp = k
        if max(absakk, colmax) == 0
            info == 0 && (info = k)
            kp = k
            herm && (A[k, k] = real(A[k, k]))
        else
            if absakk >= alpha * colmax
                kp = k                                   # 1×1, no interchange
            else
                # ROWMAX: largest off-diagonal in row imax (cols k..imax-1 and rows imax+1..n of col imax)
                _, rowmax = _bk_rowmax(A, imax, k:(imax - 1))
                if imax < n
                    _, rm2 = _bk_colmax(A, (imax + 1):n, imax)
                    rowmax = max(rowmax, rm2)
                end
                if absakk >= alpha * colmax * (colmax / rowmax)
                    kp = k                               # 1×1 pivot
                elseif (herm ? abs(real(A[imax, imax])) : _bk_cabs1(A[imax, imax])) >= alpha * rowmax
                    kp = imax                            # 1×1 pivot, interchange k <-> imax
                else
                    kp = imax; kstep = 2                 # 2×2 pivot
                end
            end
            kk = k + kstep - 1
            if kp != kk
                # interchange rows/cols kk and kp in the trailing submatrix A(k:n,k:n)
                if kp < n
                    for i in (kp + 1):n
                        t = A[i, kk]; A[i, kk] = A[i, kp]; A[i, kp] = t
                    end
                end
                # the crossing strip A(kk+1:kp-1, kk) <-> A(kp, kk+1:kp-1) (conj for Hermitian)
                for j in (kk + 1):(kp - 1)
                    if herm
                        t = conj(A[j, kk]); A[j, kk] = conj(A[kp, j]); A[kp, j] = t
                    else
                        t = A[j, kk]; A[j, kk] = A[kp, j]; A[kp, j] = t
                    end
                end
                herm && (A[kp, kk] = conj(A[kp, kk]))
                if herm
                    r1 = real(A[kk, kk]); A[kk, kk] = real(A[kp, kp]); A[kp, kp] = r1
                else
                    t = A[kk, kk]; A[kk, kk] = A[kp, kp]; A[kp, kp] = t
                end
                if kstep == 2
                    t = A[k + 1, k]; A[k + 1, k] = A[kp, k]; A[kp, k] = t
                end
            end
            if herm
                # D is Hermitian, so its diagonal is real BY DEFINITION — any imaginary part is pure
                # roundoff carried in by the trailing updates. This used to be an `elseif` on the
                # no-interchange path only, and the interchange path realifies A[kk,kk]/A[kp,kp],
                # which for a 2×2 pivot is A[k+1,k+1] and NEVER A[k,k] (kk = k+kstep-1). So a 2×2
                # pivot *with* an interchange left A[k,k] complex: measured ~5e-8 relative residue
                # against LAPACK's exact zero. Unconditional covers all four combinations.
                A[k, k] = real(A[k, k])
                kstep == 2 && (A[k + 1, k + 1] = real(A[k + 1, k + 1]))
            end

            if kstep == 1
                if k < n
                    if herm
                        r1 = one(Tr) / real(A[k, k])
                        for j in (k + 1):n               # rank-1 downdate of lower A(k+1:n,k+1:n)
                            wj = r1 * conj(A[j, k])
                            @simd ivdep for i in j:n
                                A[i, j] -= A[i, k] * wj
                            end
                            A[j, k] *= r1                # scale column below diagonal
                        end
                        A[k, k] = real(A[k, k])
                    else
                        d11 = one(T) / A[k, k]
                        for j in (k + 1):n
                            wj = d11 * A[j, k]
                            @simd ivdep for i in j:n
                                A[i, j] -= A[i, k] * wj
                            end
                            A[j, k] *= d11
                        end
                    end
                end
            else                                          # 2×2 pivot, columns k and k+1
                if k < n - 1
                    if herm
                        d = hypot(real(A[k + 1, k]), imag(A[k + 1, k]))
                        d11 = real(A[k + 1, k + 1]) / d
                        d22 = real(A[k, k]) / d
                        tt = one(Tr) / (d11 * d22 - one(Tr))
                        d21 = A[k + 1, k] / d
                        dm = tt / d
                        for j in (k + 2):n
                            wk = dm * (d11 * A[j, k] - d21 * A[j, k + 1])
                            wkp1 = dm * (d22 * A[j, k + 1] - conj(d21) * A[j, k])
                            @simd ivdep for i in j:n
                                A[i, j] -= A[i, k] * conj(wk) + A[i, k + 1] * conj(wkp1)
                            end
                            A[j, k] = wk; A[j, k + 1] = wkp1
                            A[j, j] = real(A[j, j])
                        end
                    else
                        d21 = A[k + 1, k]
                        d11 = A[k + 1, k + 1] / d21
                        d22 = A[k, k] / d21
                        tt = one(T) / (d11 * d22 - one(T))
                        d21 = tt / d21
                        for j in (k + 2):n
                            wk = d21 * (d11 * A[j, k] - A[j, k + 1])
                            wkp1 = d21 * (d22 * A[j, k + 1] - A[j, k])
                            @simd ivdep for i in j:n
                                A[i, j] -= A[i, k] * wk + A[i, k + 1] * wkp1
                            end
                            A[j, k] = wk; A[j, k + 1] = wkp1
                        end
                    end
                end
            end
        end
        if kstep == 1
            ipiv[k] = kp
        else
            ipiv[k] = -kp; ipiv[k + 1] = -kp
        end
        k += kstep
    end
    return info
end

# uplo='U': A = U·D·Uᵀ (herm=false) or U·D·Uᴴ (herm=true). Reads/writes only the upper triangle.
function _sytf2_upper!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, herm::Bool) where {T}
    n = size(A, 1)
    Tr = real(T)
    alpha = Tr(_BK_ALPHA)
    info = 0
    k = n
    @inbounds while k >= 1
        kstep = 1
        absakk = herm ? abs(real(A[k, k])) : _bk_cabs1(A[k, k])
        if k > 1
            imax, colmax = _bk_colmax(A, 1:(k - 1), k)
        else
            imax = k; colmax = zero(Tr)
        end
        kp = k
        if max(absakk, colmax) == 0
            info == 0 && (info = k)
            kp = k
            herm && (A[k, k] = real(A[k, k]))
        else
            if absakk >= alpha * colmax
                kp = k
            else
                _, rowmax = _bk_rowmax(A, imax, (imax + 1):k)
                if imax > 1
                    _, rm2 = _bk_colmax(A, 1:(imax - 1), imax)
                    rowmax = max(rowmax, rm2)
                end
                if absakk >= alpha * colmax * (colmax / rowmax)
                    kp = k
                elseif (herm ? abs(real(A[imax, imax])) : _bk_cabs1(A[imax, imax])) >= alpha * rowmax
                    kp = imax
                else
                    kp = imax; kstep = 2
                end
            end
            kk = k - kstep + 1
            if kp != kk
                if kp > 1
                    for i in 1:(kp - 1)
                        t = A[i, kk]; A[i, kk] = A[i, kp]; A[i, kp] = t
                    end
                end
                for j in (kp + 1):(kk - 1)
                    if herm
                        t = conj(A[j, kk]); A[j, kk] = conj(A[kp, j]); A[kp, j] = t
                    else
                        t = A[j, kk]; A[j, kk] = A[kp, j]; A[kp, j] = t
                    end
                end
                herm && (A[kp, kk] = conj(A[kp, kk]))
                if herm
                    r1 = real(A[kk, kk]); A[kk, kk] = real(A[kp, kp]); A[kp, kp] = r1
                else
                    t = A[kk, kk]; A[kk, kk] = A[kp, kp]; A[kp, kp] = t
                end
                if kstep == 2
                    t = A[k - 1, k]; A[k - 1, k] = A[kp, k]; A[kp, k] = t
                end
            end
            if herm                                       # see _sytf2_lower!: kk = k-kstep+1 here,
                A[k, k] = real(A[k, k])                   # so a 2×2 interchange missed A[k,k] too
                kstep == 2 && (A[k - 1, k - 1] = real(A[k - 1, k - 1]))
            end

            if kstep == 1
                if k > 1
                    if herm
                        r1 = one(Tr) / real(A[k, k])
                        for j in 1:(k - 1)                # rank-1 downdate on ORIGINAL column k
                            wj = r1 * conj(A[j, k])
                            @simd ivdep for i in 1:j
                                A[i, j] -= A[i, k] * wj
                            end
                        end
                        for j in 1:(k - 1)
                            A[j, k] *= r1
                        end   # then scale column (DSCAL after DSYR)
                        A[k, k] = real(A[k, k])
                    else
                        d11 = one(T) / A[k, k]
                        for j in 1:(k - 1)
                            wj = d11 * A[j, k]
                            @simd ivdep for i in 1:j
                                A[i, j] -= A[i, k] * wj
                            end
                        end
                        for j in 1:(k - 1)
                            A[j, k] *= d11
                        end
                    end
                end
            else                                          # 2×2 pivot, columns k and k-1
                if k > 2
                    if herm
                        d = hypot(real(A[k - 1, k]), imag(A[k - 1, k]))
                        d11 = real(A[k, k]) / d
                        d22 = real(A[k - 1, k - 1]) / d
                        tt = one(Tr) / (d11 * d22 - one(Tr))
                        d12 = A[k - 1, k] / d
                        dm = tt / d
                        for j in (k - 2):-1:1
                            wkm1 = dm * (d11 * A[j, k - 1] - conj(d12) * A[j, k])
                            wk = dm * (d22 * A[j, k] - d12 * A[j, k - 1])
                            for i in j:-1:1
                                A[i, j] -= A[i, k] * conj(wk) + A[i, k - 1] * conj(wkm1)
                            end
                            A[j, k] = wk; A[j, k - 1] = wkm1
                            A[j, j] = real(A[j, j])
                        end
                    else
                        d12 = A[k - 1, k]
                        d22 = A[k - 1, k - 1] / d12
                        d11 = A[k, k] / d12
                        tt = one(T) / (d11 * d22 - one(T))
                        d12 = tt / d12
                        for j in (k - 2):-1:1
                            wkm1 = d12 * (d11 * A[j, k - 1] - A[j, k])
                            wk = d12 * (d22 * A[j, k] - A[j, k - 1])
                            for i in j:-1:1
                                A[i, j] -= A[i, k] * wk + A[i, k - 1] * wkm1
                            end
                            A[j, k] = wk; A[j, k - 1] = wkm1
                        end
                    end
                end
            end
        end
        if kstep == 1
            ipiv[k] = kp
        else
            ipiv[k] = -kp; ipiv[k - 1] = -kp
        end
        k -= kstep
    end
    return info
end

# ═══ blocked Bunch-Kaufman (dlasyf) ═══════════════════════════════════════════════════════════════
# `_sytf2_*` updates the whole trailing submatrix after EVERY pivot — a rank-1/rank-2 BLAS-2 sweep.
# `dlasyf` updates only the current column, into a panel W that accumulates L21·D, and defers the
# trailing update to one rank-kb BLAS-3 pass. That is the entire difference, and it is why the
# unblocked kernel measures ~0.5 vs OpenBLAS and degrades monotonically with n.
#
# THE PANEL BOUND. `dlasyf` factors kb ≤ nb columns, not exactly nb, and the bound is on W's COLUMN
# budget, not on rows: the pivot search may and routinely does reach outside the panel, because every
# W column is full height and every interchange swaps full rows of A(:,1:kk) and W. One step consumes
# at most two W columns — k, and k+1 in BOTH the 2×2 case and the 1×1-with-interchange case (which
# materialises column imax into W[:,k+1] and then copies it back). Guarding `k ≤ nb-1` bounds the
# highest column touched at exactly nb, so kb ∈ {nb-1, nb}. A 2×2 that would START at k = nb is never
# attempted — the guard fires and the next panel takes it. That is precisely "a 2×2 pivot cannot
# straddle the panel boundary".
#
# TRAILING UPDATE. Reference master uses one DGEMMTR (gemm into a triangle); PureBLAS has no gemmt,
# so this uses the OLDER dlasyf structure, which is equivalent and maps onto kernels that exist: per
# nb-wide block column, `syr2k!` for the jb×jb diagonal triangle and `gemm!` for the rectangle below.
# A22 -= L21·D·L21ᵀ = L21·Wᵀ, and L21·Wᵀ is symmetric, so ½(L21·Wᵀ + W·L21ᵀ) = L21·Wᵀ exactly — that
# is the syr2k. It costs 2× flops, but ONLY on the diagonal blocks (≈ nb/m of the work, ~3% at
# n=4096), versus running that fraction at BLAS-2 speed; the rectangle stays 1×-flop peak gemm.
# Isolated in `_lasyf_diag!` so reverting to LAPACK's per-column gemv loop is one method body.
function _lasyf_lower!(
        A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
        W::AbstractMatrix{T}, xs::AbstractVector{T}, nb::Int
    ) where {T}
    n = size(A, 1)
    Tr = real(T)
    alpha = Tr(_BK_ALPHA)
    info = 0
    k = 1
    @inbounds while !(k >= nb && nb < n) && k <= n
        # W[k:n,k] = A[k:n,k] - A[k:n,1:k-1]·W[k,1:k-1]ᵀ. The W row is strided by ldw, and
        # `_l2_simd_ok` needs incx==1, so gather it contiguously first (the pstrf.jl:276-282 idiom).
        for i in k:n
            W[i, k] = A[i, k]
        end
        if k > 1
            for t in 1:(k - 1)
                xs[t] = W[k, t]
            end
            _gemv!(false, false, n - k + 1, k - 1, -one(T),
                view(A, k:n, 1:(k - 1)), xs, 1, one(T), view(W, k:n, k), 1)
        end

        kstep = 1
        absakk = _bk_cabs1(W[k, k])
        if k < n
            imax, colmax = _bk_colmax(W, (k + 1):n, k)
        else
            imax = k; colmax = zero(Tr)
        end

        if max(absakk, colmax) == zero(Tr)
            info == 0 && (info = k)
            kp = k
        else
            if absakk >= alpha * colmax
                kp = k
            else
                # Materialise column imax into W[:,k+1]; the ROWMAX search then runs against a COLUMN
                # of W rather than a row of A — which is why `_bk_rowmax`'s strided row walk vanishes
                # from the blocked path entirely.
                for i in k:(imax - 1)
                    W[i, k + 1] = A[imax, i]              # transposed strip (symmetric)
                end
                for i in imax:n
                    W[i, k + 1] = A[i, imax]
                end
                if k > 1
                    for t in 1:(k - 1)
                        xs[t] = W[imax, t]
                    end
                    _gemv!(false, false, n - k + 1, k - 1, -one(T),
                        view(A, k:n, 1:(k - 1)), xs, 1, one(T), view(W, k:n, k + 1), 1)
                end
                _, rowmax = _bk_colmax(W, k:(imax - 1), k + 1)
                if imax < n
                    _, r2 = _bk_colmax(W, (imax + 1):n, k + 1)
                    rowmax = max(rowmax, r2)
                end
                if absakk >= alpha * colmax * (colmax / rowmax)
                    kp = k
                elseif _bk_cabs1(W[imax, k + 1]) >= alpha * rowmax
                    kp = imax
                    for i in k:n                          # updated imax column becomes the pivot col
                        W[i, k] = W[i, k + 1]
                    end
                else
                    kp = imax; kstep = 2
                end
            end

            kk = k + kstep - 1
            if kp != kk
                # A directed COPY, not a swap: kk is LEAVING the trailing set this step, so the only
                # surviving effect is "row/col kp now holds what kk held". This is where _sytf2_lower!
                # and dlasyf structurally diverge.
                A[kp, kp] = A[kk, kk]
                for j in (kk + 1):(kp - 1)
                    A[kp, j] = A[j, kk]                   # crossing strip, transposed
                end
                for i in (kp + 1):n
                    A[i, kp] = A[i, kk]
                end
                if k > 1                                  # swap rows kk/kp over A's first k-1 columns
                    for j in 1:(k - 1)
                        t = A[kk, j]; A[kk, j] = A[kp, j]; A[kp, j] = t
                    end
                end
                for j in 1:kk                             # and over W's first kk columns
                    t = W[kk, j]; W[kk, j] = W[kp, j]; W[kp, j] = t
                end
            end

            if kstep == 1
                for i in k:n
                    A[i, k] = W[i, k]
                end
                if k < n
                    r1 = one(T) / A[k, k]
                    for i in (k + 1):n
                        A[i, k] *= r1
                    end
                end
            else
                if k < n - 1
                    d21 = W[k + 1, k]
                    d11 = W[k + 1, k + 1] / d21
                    d22 = W[k, k] / d21
                    tt = one(T) / (d11 * d22 - one(T))
                    d21 = tt / d21
                    for j in (k + 2):n
                        A[j, k] = d21 * (d11 * W[j, k] - W[j, k + 1])
                        A[j, k + 1] = d21 * (d22 * W[j, k + 1] - W[j, k])
                    end
                end
                A[k, k] = W[k, k]
                A[k + 1, k] = W[k + 1, k]
                A[k + 1, k + 1] = W[k + 1, k + 1]
            end
        end

        if kstep == 1
            ipiv[k] = kp
        else
            ipiv[k] = -kp; ipiv[k + 1] = -kp
        end
        k += kstep
    end

    kb = k - 1
    if kb > 0
        # Trailing update over nb-wide block columns (see the header on why not gemmt).
        j = k
        while j <= n
            jb = min(nb, n - j + 1)
            _lasyf_diag!(view(A, j:(j + jb - 1), j:(j + jb - 1)),
                view(A, j:(j + jb - 1), 1:kb), view(W, j:(j + jb - 1), 1:kb), false)
            if j + jb <= n
                # `_gemm_core!`, not the kwarg `gemm!`: the latter gates on `C isa StridedMatrix`,
                # which PtrMatrix is NOT, so a Mode-1 / .so caller would silently take the generic
                # kernel (banded_chol.jl:517-521 documents the same trap).
                _gemm_core!(view(A, (j + jb):n, j:(j + jb - 1)),
                    view(A, (j + jb):n, 1:kb), view(W, j:(j + jb - 1), 1:kb),
                    -one(T), one(T), false, true, false, false)
            end
            j += jb
        end

        # Put L21 in standard form: partially undo the row interchanges in columns 1:k-1, backwards.
        # The panel swapped FULL rows so the trailing BLAS-3 sees a consistent block; `_sytrs_*`
        # expects dsytf2's convention, where column p carries no permutation from steps after p.
        # Skip this and the factor is internally consistent but wrong against LAPACK.
        j = kb
        while j >= 1
            jj = j
            jp = ipiv[j]
            if jp < 0
                jp = -jp; j -= 1
            end
            j -= 1
            if jp != jj && j >= 1
                for c in 1:j
                    t = A[jp, c]; A[jp, c] = A[jj, c]; A[jj, c] = t
                end
            end
            j <= 1 && break
        end
    end
    return kb, info
end

# A22 -= L·Wᵀ on the `uplo` triangle only. L·Wᵀ = L·D·Lᵀ is symmetric, so ½(L·Wᵀ + W·Lᵀ) is exactly
# it — one existing BLAS-3 call in place of LAPACK's jb per-column gemv loop, at 2× flops on the
# diagonal blocks only. Val-dispatched so the symmetric/Hermitian split is a compile-time choice
# (the pstrf.jl:74-79 pattern); `her2k!` additionally zeroes the imaginary diagonal for free, which
# is what zlahef's A(JJ,JJ)=DBLE(...) bracketing exists to do.
@inline function _lasyf_diag!(C, L, Wv, herm::Bool)
    T = eltype(C)
    syr2k!(C, L, Wv; uplo = 'L', trans = 'N', alpha = -one(T) / 2, beta = one(T))
    return C
end

# Blocked driver (dsytrf, lower). THREE things bite here, all of which produce a factor that looks
# plausible and passes a closed-loop residual test:
#  (1) `kb`, NOT `nb`, advances k — the panel stops short whenever a 2×2 would straddle the boundary.
#  (2) The ipiv LOCAL→GLOBAL remap is SIGN-AWARE: positives get +k-1, negatives -k+1. A naive
#      `+= k-1` corrupts every 2×2 block outside the first panel.
#  (3) info is offset by k-1 (the sub-problem is the TRAILING block here; the upper driver's is the
#      LEADING one and needs neither remap nor offset).
# `_sytf2_lower!` remains the tail (k > n-nb), the whole path for non-BlasFloat T, and the small-n
# path — the last falls out structurally from nb >= n with no extra threshold.
function _sytrf_blocked_lower!(
        A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, nb::Int
    ) where {T}
    n = size(A, 1)
    W, xs = _sytrf_work(T, n, nb)
    info = 0
    k = 1
    @inbounds while k <= n
        if k <= n - nb
            kb, iinfo = _lasyf_lower!(view(A, k:n, k:n), view(ipiv, k:n), W, xs, nb)
        else
            iinfo = _sytf2_lower!(view(A, k:n, k:n), view(ipiv, k:n), false)
            kb = n - k + 1
        end
        info == 0 && iinfo > 0 && (info = iinfo + k - 1)
        for j in k:(k + kb - 1)
            ipiv[j] = ipiv[j] > 0 ? ipiv[j] + k - 1 : ipiv[j] - k + 1
        end
        k += kb
    end
    return info
end

# PDM tier: MEASURE. Panel/port-balance crossover, same class as `_pbtrf_nb` and `_gbtrf_nb`; anchor
# is `_LU_NB` for the reason `_pstrf_nb` also reuses it (a pivoted BLAS-2 panel amortized against a
# rank-nb BLAS-3 trailing update is the physically identical shape), bracket widened downward because
# dlasyf's panel does up to TWICE getrf's BLAS-2 work per column — the ROWMAX branch fires a second
# full-height gemv — so its optimum sits below the dense anchor. Measured Zen4 F64 PB/OB, indefinite:
#     n     nb=8   16    24    32    48    64    96
#     128   1.29  1.58  1.62  1.46  1.26  1.28  1.05
#     256   1.32  1.67  1.55  1.47  1.34  1.18  0.97
#     512   1.10  1.46  1.54  1.41  1.28  1.13  0.94
#     1024  0.83  1.13  1.21  1.23  1.17  1.09  0.95
#     2048  0.68  0.90  1.01  1.08  1.09  1.06  0.99
# The optimum DRIFTS with n (≈√n: 16/24/32/48 at n=256/512/1024/2048), so a single constant cannot be
# optimal everywhere. It is measured at a LARGE probe size on purpose: the ratio is tightest at the
# big end (1.08-1.09 at n=2048 vs 1.5-1.7 at mid n), so a knob tuned there gates everywhere, while
# one tuned at mid n would land on 24 and leave n=2048 at 1.01. Promoting to an n-scaled formula is
# the identified refinement; the data above is the record to derive it from.
const _SYTRF_NB_PREF = @load_preference("sytrf_nb", nothing)
@static if isnothing(_SYTRF_NB_PREF)
    function _measure_sytrf_nb(::Type{T})::Int where {T}
        nb0 = _LU_NB
        Base.generating_output() && return nb0
        try
            nloc = 16 * nb0                                # 768: large enough that the big-n regime
            Am = Matrix{T}(undef, nloc, nloc)               # governs, small enough to stay ~100 ms
            ipv = Vector{Int}(undef, nloc)
            refill! = () -> begin                           # indefinite symmetric, deterministic
                @inbounds for j in 1:nloc, i in 1:nloc
                    Am[i, j] = T(((i * 7 + j * 13) % 101) - 50) / T(50)
                end
                @inbounds for j in 1:nloc, i in (j + 1):nloc
                    Am[i, j] = Am[j, i]
                end
            end
            best = nb0; tbest = typemax(UInt64)
            for c in (nb0 ÷ 6, nb0 ÷ 4, nb0 ÷ 3, nb0 >> 1, (3 * nb0) >> 2, nb0)
                nb = c
                (nb < 1 || nb >= nloc) && continue
                refill!(); _sytrf_blocked_lower!(Am, ipv, nb)        # untimed warmup
                t = typemax(UInt64)
                for _ in 1:3
                    refill!(); s = time_ns()
                    _sytrf_blocked_lower!(Am, ipv, nb)
                    t = min(t, time_ns() - s)
                end
                t < tbest && (tbest = t; best = nb)
            end
            return best
        catch
            return nb0
        end
    end
    const _SYTRF_NB_F64 = Base.OncePerProcess{Int}(() -> _measure_sytrf_nb(Float64))
    const _SYTRF_NB_F32 = Base.OncePerProcess{Int}(() -> _measure_sytrf_nb(Float32))
    @inline _sytrf_nb(::Type{Float64}) = _SYTRF_NB_F64()
    @inline _sytrf_nb(::Type{Float32}) = _SYTRF_NB_F32()
    @inline _sytrf_nb(::Type{T}) where {T} = _LU_NB
else
    @inline _sytrf_nb(::Type{T}) where {T} = _SYTRF_NB_PREF::Int
end

# Dispatch. Gated on `_strided1`, NOT `A isa StridedMatrix`: cabi_lapack.jl calls sytrf!/hetrf! with
# a PtrMatrix, so a StridedMatrix gate would leave Mode 1 and the .so on the unblocked kernel — the
# port would be invisible exactly where LBT users live. Only the REAL LOWER path is blocked so far;
# upper and Hermitian keep `_sytf2_*` until their panels are ported and validated the same way.
@inline function _bk_factor_lower(A::AbstractMatrix{T}, ipiv, herm::Bool) where {T}
    n = size(A, 1)
    if T <: BlasReal && !herm && _strided1(A)
        nb = _sytrf_nb(T)
        nb > 1 && nb < n && return _sytrf_blocked_lower!(A, ipiv, nb)
    end
    return _sytf2_lower!(A, ipiv, herm)
end

"""
    sytrf!(A, ipiv; uplo='L') -> info

Bunch-Kaufman factorization of a symmetric (or complex-symmetric) matrix A = L·D·Lᵀ / U·D·Uᵀ,
in place in the `uplo` triangle. `ipiv` (length n) receives the LAPACK pivot encoding. Returns
`info` (0, or the index of the first zero pivot block). Generic over T<:Number.
"""
function sytrf!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("sytrf!: A must be square"))
    length(ipiv) == n || throw(DimensionMismatch("sytrf!: length(ipiv) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("sytrf!: uplo must be 'L' or 'U'"))
    return uplo == 'L' ? _bk_factor_lower(A, ipiv, false) : _sytf2_upper!(A, ipiv, false)
end

"""
    hetrf!(A, ipiv; uplo='L') -> info

Bunch-Kaufman factorization of a Hermitian matrix A = L·D·Lᴴ / U·D·Uᴴ. For real `eltype(A)` this is
identical to `sytrf!` (real symmetric). Otherwise the Hermitian variant (real diagonal, conjugated
off-diagonals) is used, mirroring LAPACK `zhetf2`.
"""
function hetrf!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("hetrf!: A must be square"))
    length(ipiv) == n || throw(DimensionMismatch("hetrf!: length(ipiv) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("hetrf!: uplo must be 'L' or 'U'"))
    herm = eltype(A) <: Complex
    return uplo == 'L' ? _bk_factor_lower(A, ipiv, herm) : _sytf2_upper!(A, ipiv, herm)
end

# ---------------------------------------------------------------------------------------------------
# Solve  A·X = B  from the factors (dsytrs / zhetrs / zsytrs)
# ---------------------------------------------------------------------------------------------------

@inline _bk_swap_rows!(B, r1::Int, r2::Int) = @inbounds for j in axes(B, 2)
    t = B[r1, j]; B[r1, j] = B[r2, j]; B[r2, j] = t
end

function _sytrs_lower!(A, ipiv, B, herm::Bool)
    n = size(A, 1); nrhs = size(B, 2)
    @inbounds begin
        # First solve L·D·X = B (forward, k = 1..n)
        k = 1
        while k <= n
            if ipiv[k] > 0                               # 1×1
                kp = ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                # `bkj` is HOISTED out of the i-loop deliberately. Written as `B[i,j] -= A[i,k]*B[k,j]`
                # the compiler must re-load B[k,j] on every iteration: it cannot prove the store to
                # B[i,j] does not alias it (i > k always, but that is not visible to alias analysis).
                # That unforwarded store-to-load is a known ~10 cyc/elem hazard in this codebase's
                # generic kernels, and it is on the hot path of every LDLᵀ solve.
                for j in 1:nrhs
                    bkj = B[k, j]
                    for i in (k + 1):n
                        B[i, j] -= A[i, k] * bkj
                    end
                end
                dk = herm ? real(A[k, k]) : A[k, k]
                for j in 1:nrhs
                    B[k, j] /= dk
                end
                k += 1
            else                                         # 2×2, rows (k,k+1)
                kp = -ipiv[k]
                kp != k + 1 && _bk_swap_rows!(B, k + 1, kp)
                for j in 1:nrhs                          # same store-to-load hoist as the 1×1 case
                    bkj = B[k, j]; bk1j = B[k + 1, j]
                    for i in (k + 2):n
                        B[i, j] -= A[i, k] * bkj + A[i, k + 1] * bk1j
                    end
                end
                akm1k = A[k + 1, k]
                akm1 = A[k, k] / (herm ? conj(akm1k) : akm1k)
                ak = A[k + 1, k + 1] / akm1k
                denom = akm1 * ak - one(eltype(A))
                for j in 1:nrhs
                    bkm1 = B[k, j] / (herm ? conj(akm1k) : akm1k)
                    bk = B[k + 1, j] / akm1k
                    B[k, j] = (ak * bkm1 - bk) / denom
                    B[k + 1, j] = (akm1 * bk - bkm1) / denom
                end
                k += 2
            end
        end
        # Then solve Lᵀ·X = B (or Lᴴ) (backward, k = n..1)
        k = n
        while k >= 1
            if ipiv[k] > 0                               # 1×1
                # The `herm ? conj(x) : x` ternary is hoisted OUT of the i-loop (hand loop-unswitch).
                # `herm` is a plain Bool argument, not a type parameter, so left inside it is a
                # runtime branch per element that blocks vectorisation of an otherwise clean dot.
                for j in 1:nrhs
                    s = zero(eltype(B))
                    if herm
                        @simd for i in (k + 1):n
                            s += conj(A[i, k]) * B[i, j]
                        end
                    else
                        @simd for i in (k + 1):n
                            s += A[i, k] * B[i, j]
                        end
                    end
                    B[k, j] -= s
                end
                kp = ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                k -= 1
            else                                         # 2×2, rows (k-1,k)
                for j in 1:nrhs
                    s1 = zero(eltype(B)); s2 = zero(eltype(B))
                    if herm
                        @simd for i in (k + 1):n
                            s1 += conj(A[i, k]) * B[i, j]
                            s2 += conj(A[i, k - 1]) * B[i, j]
                        end
                    else
                        @simd for i in (k + 1):n
                            s1 += A[i, k] * B[i, j]
                            s2 += A[i, k - 1] * B[i, j]
                        end
                    end
                    B[k, j] -= s1; B[k - 1, j] -= s2
                end
                kp = -ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                k -= 2
            end
        end
    end
    return B
end

function _sytrs_upper!(A, ipiv, B, herm::Bool)
    n = size(A, 1); nrhs = size(B, 2)
    @inbounds begin
        # First solve U·D·X = B (backward, k = n..1)
        k = n
        while k >= 1
            if ipiv[k] > 0                               # 1×1
                kp = ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                for j in 1:nrhs                        # bkj hoisted: see the lower path's note
                    bkj = B[k, j]
                    for i in 1:(k - 1)
                        B[i, j] -= A[i, k] * bkj
                    end
                end
                dk = herm ? real(A[k, k]) : A[k, k]
                for j in 1:nrhs
                    B[k, j] /= dk
                end
                k -= 1
            else                                         # 2×2, rows (k-1,k)
                kp = -ipiv[k]
                kp != k - 1 && _bk_swap_rows!(B, k - 1, kp)
                for j in 1:nrhs
                    bkj = B[k, j]; bkm1j = B[k - 1, j]
                    for i in 1:(k - 2)
                        B[i, j] -= A[i, k] * bkj + A[i, k - 1] * bkm1j
                    end
                end
                akm1k = A[k - 1, k]
                akm1 = A[k - 1, k - 1] / akm1k
                ak = A[k, k] / (herm ? conj(akm1k) : akm1k)
                denom = akm1 * ak - one(eltype(A))
                for j in 1:nrhs
                    bkm1 = B[k - 1, j] / akm1k
                    bk = B[k, j] / (herm ? conj(akm1k) : akm1k)
                    B[k - 1, j] = (ak * bkm1 - bk) / denom
                    B[k, j] = (akm1 * bk - bkm1) / denom
                end
                k -= 2
            end
        end
        # Then solve Uᵀ·X = B (or Uᴴ) (forward, k = 1..n)
        k = 1
        while k <= n
            if ipiv[k] > 0                               # 1×1
                for j in 1:nrhs
                    s = zero(eltype(B))
                    if herm
                        @simd for i in 1:(k - 1)
                            s += conj(A[i, k]) * B[i, j]
                        end
                    else
                        @simd for i in 1:(k - 1)
                            s += A[i, k] * B[i, j]
                        end
                    end
                    B[k, j] -= s
                end
                kp = ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                k += 1
            else                                         # 2×2, rows (k,k+1)
                for j in 1:nrhs
                    s1 = zero(eltype(B)); s2 = zero(eltype(B))
                    if herm
                        @simd for i in 1:(k - 1)
                            s1 += conj(A[i, k]) * B[i, j]
                            s2 += conj(A[i, k + 1]) * B[i, j]
                        end
                    else
                        @simd for i in 1:(k - 1)
                            s1 += A[i, k] * B[i, j]
                            s2 += A[i, k + 1] * B[i, j]
                        end
                    end
                    B[k, j] -= s1; B[k + 1, j] -= s2
                end
                kp = -ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                k += 2
            end
        end
    end
    return B
end

"""
    sytrs!(A, ipiv, B; uplo='L') -> B

Solve A·X = B in place (B overwritten by X) using the symmetric Bunch-Kaufman factors from `sytrf!`.
"""
function sytrs!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}, B::AbstractVecOrMat; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("sytrs!: A must be square"))
    size(B, 1) == n || throw(DimensionMismatch("sytrs!: size(B,1) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("sytrs!: uplo must be 'L' or 'U'"))
    Bm = B isa AbstractVector ? reshape(B, n, 1) : B
    uplo == 'L' ? _sytrs_lower!(A, ipiv, Bm, false) : _sytrs_upper!(A, ipiv, Bm, false)
    return B
end

"""
    hetrs!(A, ipiv, B; uplo='L') -> B

Solve A·X = B in place using the Hermitian Bunch-Kaufman factors from `hetrf!` (real ⇒ same as `sytrs!`).
"""
function hetrs!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}, B::AbstractVecOrMat; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("hetrs!: A must be square"))
    size(B, 1) == n || throw(DimensionMismatch("hetrs!: size(B,1) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("hetrs!: uplo must be 'L' or 'U'"))
    herm = eltype(A) <: Complex
    Bm = B isa AbstractVector ? reshape(B, n, 1) : B
    uplo == 'L' ? _sytrs_lower!(A, ipiv, Bm, herm) : _sytrs_upper!(A, ipiv, Bm, herm)
    return B
end
