# REGIME: correctness only, no timing. The packed complex trmm drivers have NEVER been compiled at
# W=8 (`mr = _CMR*W` is 16 there vs 4 on AVX2), so the tile geometry is untested. Sweep every
# side x uplo x trans x conj x unit and a k-range straddling the pack floor (_fh_ctrmm_pack_min()=48),
# comparing against a dense reference. k below the floor is a CONTROL: packed cannot engage there.
using PureBLAS, LinearAlgebra, Random
const P = PureBLAS
Random.seed!(20260830)

# Dense reference: explicitly zero the off-triangle, apply unit diagonal, then op() and multiply.
function refmul(side, up, tr, cj, unit, A, B)
    T = copy(A)
    for i in axes(T, 1), j in axes(T, 2)
        if up ? (i > j) : (i < j)
            T[i, j] = 0
        end
    end
    if unit
        for i in axes(T, 1)
            T[i, i] = 1
        end
    end
    op = tr ? (cj ? adjoint(T) : transpose(T)) : T
    return side == 'L' ? op * B : B * op
end

P._FKR_ctrmm_pack[] = 1
println("packed forced ON: _fh_ctrmm_pack() = ", P._fh_ctrmm_pack(),
        "  W = ", P._vwidth(Float64), "  pack_min = ", P._fh_ctrmm_pack_min())
P._fh_ctrmm_pack() || error("force hook is dead — the A/B would be a null result")

worst = 0.0
nfail = 0
ncase = 0
for side in ('L', 'R'), up in (true, false), tr in (true, false), cj in (true, false), unit in (true, false)
    for k in (8, 15, 16, 17, 32, 47, 48, 49, 64, 65, 96, 128)
        nrhs = 7
        A = randn(ComplexF64, k, k) + 3.0I
        B = side == 'L' ? randn(ComplexF64, k, nrhs) : randn(ComplexF64, nrhs, k)
        got = copy(B)
        ta = tr ? (cj ? 'C' : 'T') : 'N'
        try
            P.trmm!(got, A; side = side, uplo = up ? 'U' : 'L', transA = ta,
                    diag = unit ? 'U' : 'N', alpha = one(ComplexF64))
        catch e
            println("  THREW side=$side up=$up ta=$ta unit=$unit k=$k : ", sprint(showerror, e))
            global nfail += 1
            continue
        end
        exp_ = refmul(side, up, tr, cj, unit, A, B)
        err = maximum(abs, got .- exp_) / max(1e-300, maximum(abs, exp_))
        global ncase += 1
        if err > 1e-12
            println("  MISMATCH side=$side up=$up ta=$ta unit=$unit k=$k rel=$err")
            global nfail += 1
        end
        global worst = max(worst, err)
    end
end
P._FKR_ctrmm_pack[] = -1
println("cases=$ncase  failures=$nfail  worst_rel_err=$worst")
println(nfail == 0 ? "PACKED-AT-W8 CORRECTNESS: PASS" : "PACKED-AT-W8 CORRECTNESS: FAIL")
