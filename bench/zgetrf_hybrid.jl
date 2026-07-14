# Hybrid getrf core: swap individual phases (panel/trsm/gemm) to OB and see which swap makes getrf gate.
# Isolates the IN-CONTEXT laggard (isolated sub-op ratios miss cache overlap).
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
import LinearAlgebra.LAPACK as LA
using PureBLAS: _getf2!, _laswp!, _clu_nb
BLAS.set_num_threads(1); const SINK = Ref(0.0); const T = ComplexF64
rep3(n) = clamp(30_000_000 ÷ (n^3), 1, 300)

# phase flags: pnl/trs/gem ∈ (:pb,:ob)
function core_h(A, ipiv, nb, pnl, trs, gem)
    m, n = size(A); k = min(m, n); nb = clamp(nb, 1, k); info = 0; pc = 1
    @inbounds while pc <= k
        pb = min(nb, k-pc+1); mp = m-pc+1
        pv = view(A, pc:m, pc:pc+pb-1)
        if pnl === :ob
            sub = A[pc:m, pc:pc+pb-1]; ip = Vector{Int}(undef, pb)
            LA.getrf!(sub, ip); A[pc:m, pc:pc+pb-1] = sub
            for t in 1:pb; ipiv[pc-1+t] = pc-1 + ip[t]; end
        else
            _getf2!(pv, mp, pb, pc-1, ipiv, pc-1)
        end
        jt0 = pc+pb
        if jt0 <= n
            _laswp!(A, ipiv, pc, pc+pb-1, jt0, n)
            L11 = view(A, pc:pc+pb-1, pc:pc+pb-1); U12 = view(A, pc:pc+pb-1, jt0:n)
            if trs === :ob; B.trsm!('L','L','N','U', one(T), L11, U12)
            else; PureBLAS.trsm!(U12, L11; side='L',uplo='L',transA='N',diag='U',alpha=true); end
            if pc+pb <= m
                A21 = view(A, pc+pb:m, pc:pc+pb-1); A22 = view(A, pc+pb:m, jt0:n)
                if gem === :ob; B.gemm!('N','N', T(-1), A21, U12, T(1), A22)
                else; PureBLAS.gemm!(A22, A21, U12; alpha=-1, beta=true); end
            end
        end
        pc += pb
    end
    pc = 1
    @inbounds while pc <= k
        pb = min(nb, k-pc+1); jt0 = pc+pb
        jt0 <= k && _laswp!(A, ipiv, jt0, k, pc, pc+pb-1)
        pc += pb
    end
    A, ipiv, info
end

function ratio(n, pnl, trs, gem, rounds)
    reps = rep3(n); nb = _clu_nb(n, T); rs = Float64[]
    for _ in 1:rounds
        base = randn(T, n, n)
        obb = [copy(base) for _ in 1:reps]; pbb = [copy(base) for _ in 1:reps]
        obp = [Vector{Int}(undef,n) for _ in 1:reps]; pbp = [Vector{Int}(undef,n) for _ in 1:reps]
        t0=time_ns(); for r in 1:reps; LA.getrf!(obb[r], obp[r]); end; t1=time_ns()
        for r in 1:reps; core_h(pbb[r], pbp[r], nb, pnl, trs, gem); end; t2=time_ns()
        SINK[]+=real(obb[1][1])+real(pbb[1][1]); push!(rs,(t1-t0)/(t2-t1))
    end
    median(rs)
end

for n in (256, 1024)
    @printf("n=%-5d  allPB=%.3f  OBpanel=%.3f  OBtrsm=%.3f  OBgemm=%.3f  allOB=%.3f\n", n,
        ratio(n,:pb,:pb,:pb,13), ratio(n,:ob,:pb,:pb,13), ratio(n,:pb,:ob,:pb,13),
        ratio(n,:pb,:pb,:ob,13), ratio(n,:ob,:ob,:ob,13)); flush(stdout)
end
