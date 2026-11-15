# The SIMD kernels finish a ragged length with masked vector loads and stores, never a scalar loop. These
# items run every kernel over the lengths and panel shapes that exercise the ragged edge and compare against
# a scalar reference: bitwise for elementwise kernels, packs and the trsm tiles, to a BigFloat tolerance for
# reductions, and by first-index semantics for iamax.

@testitem "vector tails: elementwise kernels match the scalar fma reference" begin
    using PureBLAS, Random
    const P = PureBLAS
    rng = Random.MersenneTwister(1)
    W = P._vwidth(Float64)
    specials = (NaN, Inf, -Inf, -0.0, 0.0, 1.0e-310, 1.0e300)
    for n in vcat(0:(6W + 3), [255, 256, 257, 1003, 5003]), a in (1.7, 0.0, Inf)
        x = randn(rng, n); y = randn(rng, n)
        for i in 1:7:n
            x[i] = specials[mod1(i ÷ 7 + 1, 7)]
        end
        refy = fma.(a, x, y); refx = a .* x
        for nroute in (-1, 10_000_000)
            y1 = copy(y); P._axpy_simd!(n, a, x, y1, 0, nroute)
            @test reinterpret(UInt64, y1) == reinterpret(UInt64, refy)
            x1 = copy(x); P._scal_simd!(n, a, x1, nroute)
            @test reinterpret(UInt64, x1) == reinterpret(UInt64, refx)
        end
    end
end

@testitem "vector tails: reductions stay accurate and iamax keeps its first index" begin
    using PureBLAS, Random
    const P = PureBLAS
    rng = Random.MersenneTwister(2)
    W = P._vwidth(Float64)
    for n in (W - 1, W + 1, 2W + 3, 61, 100, 1003, 20_003)
        x = randn(rng, n); y = randn(rng, n)
        scale = sum(abs.(big.(x) .* big.(y)))
        @test abs(P._dot_simd(n, x, y, Float64) - sum(big.(x) .* big.(y))) <= 8 * eps() * scale
        @test abs(P._asum_simd(n, x, Float64) - sum(abs.(big.(x)))) <= 8 * eps() * sum(abs.(big.(x)))
        @test abs(P._sumsq_simd(n, x, Float64) - sum(big.(x) .^ 2)) <= 8 * eps() * sum(big.(x) .^ 2)
    end
    for n in vcat((4W):(8W + 3), [255, 1003, 5003]), pat in 1:3
        x = pat == 1 ? randn(rng, n) : pat == 2 ? Float64.(rand(rng, -3:3, n)) : fill(1.0, n)
        pat == 1 && (x[n] = 10.0)                       # the maximum sits in the ragged tail
        mx = maximum(abs, x)
        want = findfirst(v -> abs(v) == mx, x)          # first index on ties
        GC.@preserve x begin
            @test P._iamax_simd!(n, pointer(x)) == want
            for NB in (2, 4, 8)
                seed = abs(x[1])
                @test (b = P._iamax_tree!(Val(NB), n, pointer(x), seed); iszero(b) ? 1 : b) == want
                @test (b = P._iamax_thresh!(Val(NB), n, pointer(x), seed); iszero(b) ? 1 : b) == want
            end
        end
    end
end

@testitem "vector tails: trsm tiles match the row-by-row substitution bitwise" begin
    using PureBLAS, Random
    const P = PureBLAS
    rng = Random.MersenneTwister(3)
    W = P._vwidth(Float64)
    sent = reinterpret(Float64, 0x7ff8dead0000beef)
    # Left side: solve L·X = B column by column with the scalar recurrence.
    function ref_left!(B, L, k, n, up, unit)
        for c in 1:n
            for i in (up ? (k:-1:1) : (1:k))
                s = B[i, c]
                for j in (up ? ((i + 1):k) : (1:(i - 1)))
                    s = muladd(-L[i, j], B[j, c], s)
                end
                unit || (s *= inv(L[i, i]))
                B[i, c] = s
            end
        end
        return B
    end
    for k in (1, 3, 7, 8, 9, 17, 33, 70), n in (1, 4, 5, 9, 13), up in (false, true), unit in (false, true)
        L = fill(NaN, k + 2, k)
        for j in 1:k, i in 1:k
            (up ? i <= j : i >= j) && (L[i, j] = i == j ? 2.0 + rand(rng) : 0.3 * randn(rng))
        end
        B0 = fill(sent, k + 3, n); B0[1:k, :] .= randn(rng, k, n)
        B = copy(B0); GC.@preserve L B P._trsm_tile_L_f64!(up, unit, pointer(L), k + 2, pointer(B), k + 3, k, n)
        R = ref_left!(copy(B0), L, k, n, up, unit)
        # A backward solve downdates a W-row block against the rows below it before the rows inside it, so
        # its summation order differs from the plain one once there is more than one block.
        if !up || k <= W
            @test reinterpret(UInt64, B) == reinterpret(UInt64, R)
        else
            @test B[1:k, :] ≈ R[1:k, :] rtol = 1.0e-9
            @test all(reinterpret(UInt64, B[(k + 1):end, :]) .== reinterpret(UInt64, sent))   # padding untouched
        end
    end
    # Right side: solve X·op(A) = B, one row at a time.
    function ref_right!(B, A, m, k, up, tr, unit)
        cf(j, l) = tr ? A[j, l] : A[l, j]
        asc = (up != tr)
        for i in 1:m
            for j in (asc ? (1:k) : (k:-1:1))
                s = B[i, j]
                for l in (asc ? (1:(j - 1)) : ((j + 1):k))
                    s = muladd(-cf(j, l), B[i, l], s)
                end
                unit || (s *= inv(cf(j, j)))
                B[i, j] = s
            end
        end
        return B
    end
    for m in (1, 2, 7, 8, 9, 17, 23), k in (1, 3, 4, 5, 9, 16), up in (false, true), tr in (false, true), unit in (false, true)
        A = fill(NaN, k + 1, k)
        for j in 1:k, i in 1:k
            (up ? i <= j : i >= j) && (A[i, j] = i == j ? 2.0 + rand(rng) : 0.3 * randn(rng))
        end
        B0 = fill(sent, m + 2, k); B0[1:m, :] .= randn(rng, m, k)
        B = copy(B0); GC.@preserve A B P._trsm_tile_R_f64!(up, tr, unit, pointer(A), k + 1, pointer(B), m + 2, m, k)
        R = ref_right!(copy(B0), A, m, k, up, tr, unit)
        # A descending solve downdates a block of columns against the columns after it before the columns
        # inside it, so its summation order differs from the plain one once there is a full block.
        if (up != tr) || k < 4
            @test reinterpret(UInt64, B) == reinterpret(UInt64, R)
        else
            @test B[1:m, :] ≈ R[1:m, :] rtol = 1.0e-9
            @test all(reinterpret(UInt64, B[(m + 1):end, :]) .== reinterpret(UInt64, sent))   # padding untouched
        end
    end
end

@testitem "vector tails: packs match the elementwise reference bitwise" begin
    using PureBLAS, Random
    const P = PureBLAS
    rng = Random.MersenneTwister(4)
    W = P._vwidth(Float64); mr = P._MR * W; NR = P._NR
    sent = reinterpret(Float64, 0x7ff8dead0000beef)
    bits(v) = reinterpret(UInt64, v)
    for mce in (1, W - 1, W + 1, mr - 1, mr + 1, 2mr + 3, 61), kce in (1, 3, W, W + 1, 17), alpha in (1.5, Inf)
        A = randn(rng, mce + 3, kce + 2); AT = randn(rng, kce + 2, mce + 3)
        len = cld(mce, mr) * mr * kce
        want = fill(sent, len + 5); wantT = copy(want); wantU = copy(want)
        for pi in 0:(cld(mce, mr) - 1), p in 0:(kce - 1), r in 0:(mr - 1)
            lr = pi * mr + r; idx = pi * mr * kce + p * mr + r + 1
            want[idx] = lr < mce ? alpha * A[2 + lr, 1 + p] : 0.0
            wantT[idx] = lr < mce ? alpha * AT[1 + p, 2 + lr] : 0.0
        end
        Ac = randn(rng, ComplexF64, mce + 3, kce + 2); cmr = P._CMR * W
        clen = cld(mce, cmr) * cmr * kce; wr = fill(sent, clen + 5); wi = copy(wr)
        for pi in 0:(cld(mce, cmr) - 1), p in 0:(kce - 1), r in 0:(cmr - 1)
            lr = pi * cmr + r; idx = pi * cmr * kce + p * cmr + r + 1
            v = lr < mce ? Ac[2 + lr, 1 + p] : 0.0im
            wr[idx] = real(v); wi[idx] = imag(v)
        end
        Rp = fill(sent, clen + 5); Ip = copy(Rp); P._pack_A_cmplx_simd!(Rp, Ip, Ac, 1, 0, mce, kce, cmr)
        @test bits(Rp) == bits(wr) && bits(Ip) == bits(wi)
        Ap = fill(sent, len + 5); P._pack_A_simd!(Ap, A, 1, 0, mce, kce, alpha, mr)
        @test bits(Ap) == bits(want)
        Ap = fill(sent, len + 5); P._pack_A_simd_T!(Ap, AT, 1, 0, mce, kce, alpha, mr)
        @test bits(Ap) == bits(wantT)
    end
    for nce in (1, NR - 1, NR + 1, 2NR + 3, 30), kce in (1, 3, W, W + 1, 17), tB in (false, true), nr in unique((NR, NR + 2, 3))
        B = tB ? randn(rng, nce + 2, kce + 1) : randn(rng, kce + 1, nce + 2)
        len = cld(nce, nr) * nr * kce
        want = fill(sent, len + 5)
        for ji in 0:(cld(nce, nr) - 1), p in 0:(kce - 1), c in 0:(nr - 1)
            lc = ji * nr + c; idx = ji * nr * kce + p * nr + c + 1
            want[idx] = lc < nce ? (tB ? B[1 + lc, 1 + p] : B[1 + p, 1 + lc]) : 0.0
        end
        Bc = tB ? randn(rng, ComplexF64, nce + 2, kce + 1) : randn(rng, ComplexF64, kce + 1, nce + 2)
        wr = fill(sent, len + 5); wi = copy(wr)
        for ji in 0:(cld(nce, nr) - 1), p in 0:(kce - 1), c in 0:(nr - 1)
            lc = ji * nr + c; idx = ji * nr * kce + p * nr + c + 1
            v = lc < nce ? (tB ? Bc[1 + lc, 1 + p] : Bc[1 + p, 1 + lc]) : 0.0im
            wr[idx] = real(v); wi[idx] = imag(v)
        end
        Rp = fill(sent, len + 5); Ip = copy(Rp); P._pack_B_cmplx!(Rp, Ip, Bc, 0, 0, kce, nce, tB, nr)
        @test bits(Rp) == bits(wr) && bits(Ip) == bits(wi)
        Bp = fill(sent, len + 5); P._pack_B_dyn!(Bp, B, 0, 0, kce, nce, tB, nr, 0)
        @test bits(Bp) == bits(want)
        if nr == NR
            Bp = fill(sent, len + 5); P._pack_B_val!(Bp, B, 0, 0, kce, nce, tB, Val(NR), 0)
            @test bits(Bp) == bits(want)
        end
    end
end
