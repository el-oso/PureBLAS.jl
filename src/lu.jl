# LAPACK LU (getrf) — partial pivoting, blocked right-looking. FROM SCRATCH (BlazingPorts has no LU
# source — only bench data), but the same recipe as the faer ports: a simple panel kernel + PureBLAS's
# gated trsm!/gemm! for the trailing. A = P·L·U (L unit-lower, U upper; ipiv[i] = global row swapped to
# position i, LAPACK convention). Float64. ponytail: generic/AD LU deferred (pivoting is data-dependent).

const _LU_NB = 48       # blocked panel width (measured optimum on Zen4: small nb trims panel/trsm
# overhead but shrinks the rank-nb trailing gemm — balance ~48). ponytail: revisit with the large-n gap.

# Unblocked panel LU with partial pivoting (LAPACK dgetf2) on an mp×pb panel whose rows are global
# (offset roff). Fills ipiv[ioff+1 : ioff+pb] with GLOBAL 1-based pivot rows. Returns the first
# zero-pivot global column (0 if none). Column-indexed so the inner loops auto-vectorize (columns
# contiguous). Pivoting is a correctness boundary — do not simplify.
function _getf2!(A, mp::Int, pb::Int, roff::Int, ipiv, ioff::Int)
    info = 0
    @inbounds for jl in 1:pb
        piv = jl; pmax = abs(A[jl, jl])                  # partial pivot: max |·| in column jl, rows jl:mp
        for il in (jl + 1):mp
            a = abs(A[il, jl]); a > pmax && (pmax = a; piv = il)
        end
        ipiv[ioff + jl] = roff + piv
        if A[piv, jl] != 0.0
            if piv != jl                                  # swap rows jl ↔ piv across the panel
                for jc in 1:pb
                    A[jl, jc], A[piv, jc] = A[piv, jc], A[jl, jc]
                end
            end
            d = 1.0 / A[jl, jl]
            for il in (jl + 1):mp; A[il, jl] *= d; end     # scale column below the diagonal
        elseif info == 0
            info = roff + jl
        end
        for jc in (jl + 1):pb                             # rank-1 update of the panel trailing
            ajc = A[jl, jc]
            for il in (jl + 1):mp; A[il, jc] -= A[il, jl] * ajc; end
        end
    end
    return info
end

# Apply row interchanges ipiv[k1:k2] to columns j1:j2 (LAPACK dlaswp), in sequence. Column-OUTER /
# pivot-inner: each (contiguous) column gets all its swaps while resident → the matrix is streamed once,
# not once per pivot (the row-outer order was the whole gap — strided, ~pb× the memory traffic).
function _laswp!(A, ipiv, k1::Int, k2::Int, j1::Int, j2::Int)
    j1 > j2 && return
    @inbounds for j in j1:j2
        for i in k1:k2
            ip = ipiv[i]
            if ip != i
                A[i, j], A[ip, j] = A[ip, j], A[i, j]
            end
        end
    end
end

# Blocked right-looking LU (LAPACK dgetrf's algorithm — the reference, faster here than a recursive LU
# which over-decomposes into many small gemm! calls). Factor each nb-panel (getf2), swap the rest of the
# rows (laswp), solve the row panel (trsm, L11⁻¹·A12), downdate the trailing (gemm, A22 −= L21·U12).
# The cheap unblocked panel + ONE big rank-nb trailing gemm per step is the win. Returns (A, ipiv, info).
function getrf!(A::StridedMatrix{Float64}, ipiv::Vector{Int}; nb::Int = _LU_NB)
    m, n = size(A); k = min(m, n)
    k == 0 && return A, ipiv, 0
    length(ipiv) >= k || throw(DimensionMismatch("getrf!: length(ipiv) < min(size(A))"))
    nb = clamp(nb, 1, k)
    info = 0; pc = 1
    @inbounds while pc <= k
        pb = min(nb, k - pc + 1); mp = m - pc + 1
        pinfo = _getf2!(view(A, pc:m, pc:pc+pb-1), mp, pb, pc - 1, ipiv, pc - 1)
        (info == 0 && pinfo != 0) && (info = pinfo)
        _laswp!(A, ipiv, pc, pc + pb - 1, 1, pc - 1)                 # swap the already-done left columns
        jt0 = pc + pb
        if jt0 <= n
            _laswp!(A, ipiv, pc, pc + pb - 1, jt0, n)                # swap the trailing columns
            trsm!(view(A, pc:pc+pb-1, jt0:n), view(A, pc:pc+pb-1, pc:pc+pb-1);
                  side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = true)   # U12 = L11⁻¹ A12
            if pc + pb <= m
                gemm!(view(A, pc+pb:m, jt0:n), view(A, pc+pb:m, pc:pc+pb-1), view(A, pc:pc+pb-1, jt0:n);
                      alpha = -1, beta = true)                       # A22 −= L21 U12
            end
        end
        pc += pb
    end
    return A, ipiv, info
end

# Convenience: allocate ipiv, return (A overwritten with L\U, ipiv, info).
function getrf!(A::StridedMatrix{Float64})
    ipiv = Vector{Int}(undef, min(size(A)...))
    return getrf!(A, ipiv)
end
