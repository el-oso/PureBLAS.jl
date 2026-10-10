# `trmm!` carries a tiny real fast path that answers a call directly instead of going through
# `_trmm!` → `_trmm_left!`/`_trmm_right!`, and it is only correct to skip that chain if it picks the
# same base the chain would. The two are separate expressions over separate constants, so nothing but
# a test holds them together: when they disagree, the intercepted call takes a different route, sums
# in a different order, and returns different bits from the same arguments.
#
# The invariant is therefore bit equality, not approximate agreement. Both routes are accurate — the
# divergence that matters reads ~1e-14 — so a tolerance would pass while the mirror was broken.
#
# ⚠ THIS ITEM CANNOT FAIL WHERE THE TWO BASES ARE THE SAME INTEGER. `_TRMM_BASE_R` is `_L3_NB` off
# SME and `_TRMM_BASE` is `_L3_NB` everywhere, so the k band in which a mirror can diverge is empty
# on a machine without the coprocessor and every combination below agrees trivially. That is a
# property of the constants, not of the routing, and it is why a side-R mirror drift went unseen: no
# x86 box can observe this class of defect, by measurement or by test. The band is checked and
# reported rather than assumed, so a vacuous run says so.
@testitem "trmm tiny path mirrors the wrapper chain bit for bit" tags = [:checks] begin
    using PureBLAS
    P = PureBLAS
    lo, hi = P._TRMM_BASE_R, P._TRMM_BASE
    @test lo <= hi                      # the tiny path's cut never exceeds the chain's
    if lo == hi
        @info "trmm mirror band is empty (both bases are $lo) — this item cannot diverge here" lo hi
    end
    # Operands are deterministic so a failure reproduces exactly: a mirror defect is a routing fact,
    # and a seed would make the report depend on the draw.
    gen(T, r, c) = T <: Complex ?
        [T(sinpi((i + 3j) / 97) + 2, cospi((2i + j) / 53)) for i in 1:r, j in 1:c] :
        [T(sinpi((i + 3j) / 97) + 2) for i in 1:r, j in 1:c]
    m = 24
    ks = sort(unique([8, lo, lo + 1, (lo + hi) ÷ 2, hi - 1, hi, hi + 1, 2hi]))
    diverged = Tuple{DataType, Int, Char, String}[]
    for T in (Float64, Float32, ComplexF64), k in ks, side in ('L', 'R'),
            up in ('U', 'L'), ta in ('N', 'T'), dg in ('N', 'U')
        k >= 1 || continue
        A = gen(T, k, k)
        B0 = side == 'L' ? gen(T, k, m) : gen(T, m, k)
        pub = copy(B0)
        P.trmm!(pub, A; side = side, uplo = up, transA = ta, diag = dg, alpha = one(T))
        chain = copy(B0)
        up_ = up == 'U'; tr_ = ta != 'N'; cj_ = ta == 'C'; unit_ = dg == 'U'
        side == 'L' ? P._trmm_left!(up_, tr_, cj_, unit_, A, chain) :
            P._trmm_right!(up_, tr_, cj_, unit_, A, chain)
        pub == chain || push!(diverged, (T, k, side, up * ta * dg))
    end
    # Named rather than counted: the failure says which (eltype, k, side, flags) lost the mirror,
    # which is the whole diagnosis — a base that disagrees with its callee shows up as one contiguous
    # k band on one side.
    @test isempty(diverged)
    isempty(diverged) || @info "trmm mirror divergences" first(diverged, 12) total = length(diverged)
end
