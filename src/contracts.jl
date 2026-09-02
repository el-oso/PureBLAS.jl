# Compile-time interface contract for BLAS Level-1 backends (TypeContracts.jl).
#
# A *backend* is the swappable unit of the Pure ecosystem: the M1 `SIMDBackend` (backend.jl), a
# future reference/GPU backend, etc. Each must provide the same Level-1 operation set so callers
# (the native API in native.jl, and downstream packages) depend on the interface, not the struct.
# `@contract` checks method existence + inferred return types at PRECOMPILE time and is eliminated
# by the trimmer — zero runtime cost (mirrors PureFFT's `AbstractFFTPlan` contract). Implementing
# methods carry explicit concrete return-type annotations so inference matches the contract.

using TypeContracts
using StrictMode  # @strict_contract / @verify_strict — the performance layer over the method surface

"""
    AbstractBLAS1

Supertype of all PureBLAS Level-1 backends. Concrete backends must satisfy the [`@contract`](@ref)
below over high-level `AbstractVector` arguments.
"""
abstract type AbstractBLAS1 end

function axpy! end      # y .+= a .* x
function scal! end      # x .*= a
function blascopy! end  # y .= x
function swap! end      # x ⇄ y
function dot end        # conjugated inner product (conj(x)·y)
function dotu end       # unconjugated inner product (x·y)
function nrm2 end       # Euclidean norm
function asum end       # Σ|xᵢ| (complex: Σ|Re|+|Im|)
function iamax end      # argmax|xᵢ|

# Level-1 is a *strict* contract: implementations must satisfy not just the method surface
# (TypeContracts) but StrictMode's performance guarantees — type-stable and allocation-free. The
# bandwidth-bound L1 kernels are where a stray allocation or type instability is most costly, so
# they carry the hardest guarantee. Verified by `@verify_strict SIMDBackend` (backend.jl).
@strict_contract AbstractBLAS1 begin
    axpy!(::Self, ::AbstractVector, ::Number, ::AbstractVector)::AbstractVector => "y := y + a·x"
    scal!(::Self, ::Number, ::AbstractVector)::AbstractVector => "x := a·x"
    blascopy!(::Self, ::AbstractVector, ::AbstractVector)::AbstractVector => "y := x"
    swap!(::Self, ::AbstractVector, ::AbstractVector)::Nothing => "exchange the contents of x and y"
    dot(::Self, ::AbstractVector, ::AbstractVector)::Number => "conjugated inner product conj(x)·y"
    dotu(::Self, ::AbstractVector, ::AbstractVector)::Number => "unconjugated inner product x·y"
    nrm2(::Self, ::AbstractVector)::Real => "Euclidean norm ‖x‖₂ via scaled accumulation (overflow/underflow safe)"
    asum(::Self, ::AbstractVector)::Real => "Σ|xᵢ| (complex: Σ|Re| + |Im|)"
    iamax(::Self, ::AbstractVector)::Integer => "index of the first element of largest magnitude"
end

"""
    AbstractBLAS2 <: AbstractBLAS1

Supertype of Level-2 backends (matrix-vector); an L2 backend is also an L1 backend. Concrete
backends provide `gemv!` and `ger!` in addition to the Level-1 ops.
"""
abstract type AbstractBLAS2 <: AbstractBLAS1 end

function gemv! end  # y := β·y + α·op(A)·x
function ger! end   # A := α·x·yᵀ + A  (geru / gerc)
function symv! end  # y := α·A·x + β·y, A symmetric
function hemv! end  # y := α·A·x + β·y, A Hermitian
function trmv! end  # x := op(A)·x, A triangular
function trsv! end  # x := op(A)⁻¹·x, A triangular (solve)
# Packed storage (AP::AbstractVector) and band storage (AB::AbstractMatrix) L2 variants.
function spmv! end  # y := α·A·x + β·y, A symmetric packed
function hpmv! end  # y := α·A·x + β·y, A Hermitian packed
function tpmv! end  # x := op(A)·x, A triangular packed
function tpsv! end  # x := op(A)⁻¹·x, A triangular packed (solve)
function gbmv! end  # y := α·op(A)·x + β·y, A general banded
function sbmv! end  # y := α·A·x + β·y, A symmetric banded
function hbmv! end  # y := α·A·x + β·y, A Hermitian banded
function tbmv! end  # x := op(A)·x, A triangular banded
function tbsv! end  # x := op(A)⁻¹·x, A triangular banded (solve)
function spr! end   # A := α·x·xᵀ + A, A symmetric packed (rank-1)
function spr2! end  # A := α·x·yᵀ + α·y·xᵀ + A, A symmetric packed (rank-2)
function hpr! end   # A := α·x·xᴴ + A, A Hermitian packed (rank-1, α real)
function hpr2! end  # A := α·x·yᴴ + ᾱ·y·xᴴ + A, A Hermitian packed (rank-2)

# Level-2 is also a *strict* contract: every matrix-vector kernel is type-stable and allocation-free
# — dense (gemv/ger/symv/hemv/trmv/trsv), packed (spmv/hpmv/tpmv/tpsv/spr/spr2/hpr/hpr2), and banded
# (gbmv/sbmv/hbmv/tbmv/tbsv), all verified by `@verify_strict SIMDBackend` (verify.jl).
@strict_contract AbstractBLAS2 begin
    gemv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector => "y := β·y + α·op(A)·x"
    ger!(::Self, ::Number, ::AbstractVector, ::AbstractVector, ::AbstractMatrix)::AbstractMatrix => "A := α·x·yᵀ + A (geru / gerc)"
    symv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector => "y := α·A·x + β·y, A symmetric"
    hemv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector => "y := α·A·x + β·y, A Hermitian"
    trmv!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractVector => "x := op(A)·x, A triangular"
    trsv!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractVector => "x := op(A)⁻¹·x, A triangular (solve)"
    spmv!(::Self, ::AbstractVector, ::AbstractVector, ::AbstractVector)::AbstractVector => "y := α·A·x + β·y, A symmetric in packed storage"
    hpmv!(::Self, ::AbstractVector, ::AbstractVector, ::AbstractVector)::AbstractVector => "y := α·A·x + β·y, A Hermitian in packed storage"
    tpmv!(::Self, ::AbstractVector, ::AbstractVector)::AbstractVector => "x := op(A)·x, A triangular packed"
    tpsv!(::Self, ::AbstractVector, ::AbstractVector)::AbstractVector => "x := op(A)⁻¹·x, A triangular packed (solve)"
    gbmv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector, ::Integer, ::Integer, ::Integer)::AbstractVector => "y := α·op(A)·x + β·y, A general banded"
    sbmv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector => "y := α·A·x + β·y, A symmetric banded"
    hbmv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector => "y := α·A·x + β·y, A Hermitian banded"
    tbmv!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractVector => "x := op(A)·x, A triangular banded"
    tbsv!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractVector => "x := op(A)⁻¹·x, A triangular banded (solve)"
    spr!(::Self, ::Number, ::AbstractVector, ::AbstractVector)::AbstractVector => "A := α·x·xᵀ + A, symmetric packed rank-1"
    spr2!(::Self, ::Number, ::AbstractVector, ::AbstractVector, ::AbstractVector)::AbstractVector => "A := α·x·yᵀ + α·y·xᵀ + A, symmetric packed rank-2"
    hpr!(::Self, ::Number, ::AbstractVector, ::AbstractVector)::AbstractVector => "A := α·x·xᴴ + A, Hermitian packed rank-1 (α real)"
    hpr2!(::Self, ::Number, ::AbstractVector, ::AbstractVector, ::AbstractVector)::AbstractVector => "A := α·x·yᴴ + ᾱ·y·xᴴ + A, Hermitian packed rank-2"
end

"""
    AbstractBLAS3 <: AbstractBLAS2

Supertype of Level-3 backends (matrix-matrix); an L3 backend is also an L2 (and L1) backend. Concrete
backends provide the matrix-matrix set below in addition to the Level-1/2 ops. The `@contract` is the
single discoverable spec of what a swappable L3 backend must implement — a second backend (reference,
GPU, …) that omits any op fails the precompile-time check. The high-level `op!(::Backend, …)` methods
dispatch on the backend; the bare `op!(C, A, B; …)` entry points (gemm.jl / level3.jl) are the
default-backend fast paths (kept backend-free so the hot L3 kernels take no extra dispatch).
"""
abstract type AbstractBLAS3 <: AbstractBLAS2 end

function gemm! end   # C := β·C + α·op(A)·op(B)
function symm! end   # C := β·C + α·A·B / α·B·A,  A symmetric
function hemm! end   # C := β·C + α·A·B / α·B·A,  A Hermitian
function syrk! end   # C := β·C + α·op(A)·op(A)ᵀ, C symmetric
function herk! end   # C := β·C + α·op(A)·op(A)ᴴ, C Hermitian (α,β real)
function syr2k! end  # C := β·C + α·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ), C symmetric
function her2k! end  # C := β·C + α·op(A)·op(B)ᴴ + ᾱ·op(B)·op(A)ᴴ, C Hermitian
function trmm! end   # B := α·op(A)·B / α·B·op(A),      A triangular
function trsm! end   # B := α·op(A)⁻¹·B / α·B·op(A)⁻¹,  A triangular (solve)

# Level-3 is a strict contract: every matrix-matrix op is type-stable and allocation-free, verified by
# `@verify_strict SIMDBackend` (verify.jl). It ALSO carries a TRIM-COMPATIBILITY guarantee — the C-ABI
# entry points that export these ops (cabi.jl: s/d/c/z `gemm_64_` …) must compile under juliac --trim for
# the Mode-1 LBT `.so`, a hard requirement. The trim-critical path is the complex unpacked gemm kernel
# (`_gemm_cmplx_unpacked!`): it is asserted `@assert_trim_compatible` in the strict-verify pass — heuristic
# in :fast (dev, verify.jl) and juliac's authoritative verify_typeinf_trim in :full (tests, the strictmode
# dogfood, TrimCheck loaded). trim_tests.jl keeps the exhaustive ccallable-rooted belt (strict isn't perfect
# yet; StrictMode 0.3.9 logs a one-time caveat on the fast/full heuristic-vs-authoritative trim gap, issue #13).
# The rank-k/hemm family's divide-and-conquer drivers were
# refactored to carry integer offsets into the original arrays (not fresh sub-block SubArrays, which
# are non-isbits and heap-box when passed to the non-inlined recursive call — the sub-block views are
# also built per concrete type, never as a Union, so they stay stack-allocated). All nine now gate 0-alloc.
@strict_contract AbstractBLAS3 begin
    gemm!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix => "C := β·C + α·op(A)·op(B)"
    symm!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix => "C := β·C + α·A·B or α·B·A, A symmetric"
    hemm!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix => "C := β·C + α·A·B or α·B·A, A Hermitian"
    syrk!(::Self, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix => "C := β·C + α·op(A)·op(A)ᵀ, C symmetric"
    herk!(::Self, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix => "C := β·C + α·op(A)·op(A)ᴴ, C Hermitian (α, β real)"
    syr2k!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix => "C := β·C + α·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ), C symmetric"
    her2k!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix => "C := β·C + α·op(A)·op(B)ᴴ + ᾱ·op(B)·op(A)ᴴ, C Hermitian"
    trmm!(::Self, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix => "B := α·op(A)·B or α·B·op(A), A triangular"
    trsm!(::Self, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix => "B := α·op(A)⁻¹·B or α·B·op(A)⁻¹, A triangular (solve)"
end

"""
    AbstractLAPACK <: AbstractBLAS3

Supertype of LAPACK backends (dense factorizations); a LAPACK backend is also an L3/L2/L1 backend
(it builds the factorizations on the gated Level-3 kernels). Concrete backends provide the
factorization set below. As with the BLAS levels this `@contract` is the single discoverable spec
of a swappable LAPACK backend — a second backend (reference, GPU, …) that omits any factorization
fails the precompile-time check. The high-level `fac!(::Backend, …)` methods dispatch on the
backend; the bare `fac!(A; …)` entry points (lapack.jl / lu.jl / qr.jl / svd.jl) are the
default-backend implementations. Factorizations that return multiple outputs (pivots, τ, U/S/Vᵀ)
are typed `::Tuple`; potrf! overwrites-and-returns its matrix.
"""
abstract type AbstractLAPACK <: AbstractBLAS3 end

function potrf! end  # Cholesky:  A = L·Lᴴ (or Uᴴ·U); overwrites the stored triangle
function getrf! end  # LU w/ partial pivoting: P·A = L·U → (A, ipiv, info)
function geqrf! end  # QR (Householder): A = Q·R → (A, tau)
function gesvd! end  # SVD: A = U·Σ·Vᵀ → (U, S, Vᵀ)

# LAPACK is a strict contract — ALL FOUR factorizations are type-stable and allocation-free, verified by
# `@verify_strict SIMDBackend` (verify.jl). potrf! factors through its own pointer-based
# `_syrk_lower_f64!`/trsm kernels (lapack.jl); getrf!/geqrf!/gesvd! reach 0-alloc through their IN-PLACE
# forms — getrf!(A,ipiv)/geqrf!(A,τ)/gesvd!(A,U,S,Vᵀ) take caller-provided output buffers, and gesvd's
# bidiagonalization scratch comes from a cached Float64 SVDWorkspace (svd.jl). The convenience forms that
# allocate pivots/τ/U/S/Vᵀ are unchanged; the allocation is the inherent output, and the C-ABI + strict
# paths use the in-place forms.
# (The contract itself is declared once, below, after the solve/inverse/condition names are introduced.)

# ── The solve / inverse / condition-estimate surface, built on the four factorizations above.
#
# These live in `AbstractLAPACK` itself rather than a separate sub-interface. A split was tried and
# reverted: it would have been a second interface with the same single implementation, and it works
# against what `AbstractLAPACK`'s docstring already promises — "the single discoverable spec of a
# swappable LAPACK backend" is less true, not more, when the spec is spread over two types. If a
# second backend ever makes the 28-routine bar too high, split it THEN, against a real requirement.
#
# Every member is IN-PLACE and so carries the strict guarantee: type-stable AND allocation-free. That
# is not decoration — each name ends in `!`, which in this package is a promise, and nine of them were
# silently breaking it (measured at n=8: gecon!/trcon!/pocon! 352 B, potri!/getri! 592 B, pstrf! 640 B,
# trrfs! 1056 B, gbtrf! 128 B, plus gesvx! allocating X/ferr/berr). The scratch behind them is owned
# per element type by `L3Workspace` (workspace.jl); the two whose OUTPUTS were allocated internally
# gained the in-place forms `gbtrf!(kl,ku,m,AB,ipiv)` and `gesvx!(…,B,X,ferr,berr)` that netlib's
# IPIV/X/FERR/BERR arguments always implied. The allocating convenience forms are retained and are
# deliberately NOT contract members — a contract satisfiable by a method that allocates means nothing.

function potrs! end   # Cholesky solve:            A·X = B given the factor
function potri! end   # Cholesky inverse:          A⁻¹ from the factor (trtri + lauum)
function pptrf! end   # packed Cholesky factor
function pptrs! end   # packed Cholesky solve
function pbtrf! end   # band Cholesky factor
function pbtrs! end   # band Cholesky solve
function pstrf! end   # pivoted Cholesky (rank-revealing)
function pocon! end   # posdef reciprocal condition estimate
function getrs! end   # LU solve
function getri! end   # LU inverse
function trtrs! end   # triangular solve
function trtri! end   # triangular inverse
function gecon! end   # general reciprocal condition estimate
function trcon! end   # triangular reciprocal condition estimate
function trrfs! end   # triangular solve error bounds (ferr/berr)
function gbtrf! end   # general banded LU factor
function gbtrs! end   # general banded LU solve
function gtsv! end    # tridiagonal solve (factor + solve)
function gttrf! end   # tridiagonal LU factor
function gttrs! end   # tridiagonal LU solve
function pttrf! end   # SPD tridiagonal LDLᴴ factor
function pttrs! end   # SPD tridiagonal solve
function ptsv! end    # SPD tridiagonal solve (factor + solve)
function gesvx! end   # expert general driver (equilibrate + factor + solve + error bounds)

# ── QR / LQ / QL / RZ family: factorizations, Q-generators, Q-appliers ─────────────────────────────
# 14 members, not 25 names. Nine of the public names — ungqr!/unglq!/ungql!/ungrq! and
# unmqr!/unmlq!/unmql!/unmrq!/unmrz! — are `const un*! = or*!` aliases (qr.jl:590, lq.jl:133/:180,
# qlrq.jl:120/:156/:230/:272, gelsy.jl:209). They are the SAME function object, so one backend method
# on the `or*!` name IS the `un*!` method; contracting both would also be impossible, because
# declaring `function ungqr! end` here (contracts.jl is included at PureBLAS.jl:20, long before
# qr.jl:27) would make the later `const ungqr! = orgqr!` a redefinition error.
# orgtr!/ungtr! ARE included, but only via their IN-PLACE forms `orgtr!(uplo, A, tau, Q)`. Their
# 3-arg convenience forms allocate Q (netlib DORGTR overwrites A instead; PureBLAS returns a fresh Q),
# and a contract satisfiable by an allocating method would be worth nothing.
function gelqf! end   # LQ factorization: A = L·Q
function geqlf! end   # QL factorization: A = Q·L
function gerqf! end   # RQ factorization: A = R·Q
function geqp3! end   # column-pivoted QR (rank-revealing): A·P = Q·R
function tzrzf! end   # trapezoidal RZ: [R 0]·Z = [T 0]
function orgqr! end   # form Q from geqrf reflectors (ungqr! is the same object)
function orglq! end   # form Q from gelqf reflectors
function orgql! end   # form Q from geqlf reflectors
function orgrq! end   # form Q from gerqf reflectors
function ormqr! end   # apply geqrf Q to C (unmqr! is the same object)
function ormlq! end   # apply gelqf Q to C
function ormql! end   # apply geqlf Q to C
function ormrq! end   # apply gerqf Q to C
function ormrz! end   # apply tzrzf Z to C
function orgtr! end   # form Q from sytrd reflectors (in-place form)
function ungtr! end   # form Q from hetrd reflectors (in-place form)

@strict_contract AbstractLAPACK begin
    potrf!(::Self, ::AbstractMatrix)::AbstractMatrix => "Cholesky factor A = L·Lᴴ (or Uᴴ·U); overwrites the stored triangle"
    getrf!(::Self, ::AbstractMatrix)::Tuple => "LU with partial pivoting, P·A = L·U → (A, ipiv, info)"
    geqrf!(::Self, ::AbstractMatrix)::Tuple => "Householder QR, A = Q·R → (A, τ)"
    gesvd!(::Self, ::AbstractMatrix)::Tuple => "SVD, A = U·Σ·Vᵀ → (U, S, Vᵀ)"
    potrs!(::Self, ::AbstractMatrix, ::AbstractVecOrMat)::AbstractVecOrMat => "solve A·X = B from a Cholesky factor"
    potri!(::Self, ::AbstractMatrix)::AbstractMatrix => "explicit inverse from a Cholesky factor (trtri then lauum)"
    pptrf!(::Self, ::AbstractVector)::AbstractVector => "Cholesky factor in packed triangular storage"
    pptrs!(::Self, ::AbstractVector, ::AbstractVecOrMat)::AbstractVecOrMat => "solve from a packed Cholesky factor"
    pbtrf!(::Self, ::AbstractMatrix)::AbstractMatrix => "Cholesky factor of a symmetric/Hermitian band matrix"
    pbtrs!(::Self, ::AbstractMatrix, ::AbstractVecOrMat)::AbstractVecOrMat => "solve from a band Cholesky factor"
    pstrf!(::Self, ::AbstractMatrix, ::AbstractVector, ::Real)::Tuple => "pivoted rank-revealing Cholesky → (A, piv, rank, info)"
    pocon!(::Self, ::Real, ::AbstractMatrix)::Real => "reciprocal condition number of a posdef matrix, from its factor"
    getrs!(::Self, ::AbstractMatrix, ::AbstractVector, ::AbstractVecOrMat)::AbstractVecOrMat => "solve A·X = B from LU factors and pivots"
    getri!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractMatrix => "explicit inverse from LU factors and pivots"
    trtrs!(::Self, ::AbstractMatrix, ::AbstractVecOrMat)::AbstractVecOrMat => "triangular solve op(A)·X = B"
    trtri!(::Self, ::AbstractMatrix)::AbstractMatrix => "explicit inverse of a triangular matrix, in place"
    gecon!(::Self, ::Real, ::AbstractMatrix, ::AbstractVector)::Real => "reciprocal condition number of a general matrix, from its LU"
    trcon!(::Self, ::AbstractMatrix)::Real => "reciprocal condition number of a triangular matrix"
    trrfs!(::Self, ::AbstractChar, ::AbstractChar, ::AbstractChar, ::AbstractMatrix, ::AbstractVecOrMat, ::AbstractVecOrMat, ::AbstractVector, ::AbstractVector)::Tuple => "forward/backward error bounds for a triangular solve → (ferr, berr)"
    gbtrf!(::Self, ::Integer, ::Integer, ::Integer, ::AbstractMatrix, ::AbstractVector)::Tuple => "banded LU with partial pivoting; ipiv is caller-provided → (AB, ipiv, info)"
    gbtrs!(::Self, ::AbstractChar, ::Integer, ::Integer, ::Integer, ::AbstractMatrix, ::AbstractVector, ::AbstractVecOrMat)::AbstractVecOrMat => "solve from banded LU factors"
    gtsv!(::Self, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVecOrMat)::AbstractVecOrMat => "tridiagonal solve — factor and solve in one pass"
    gttrf!(::Self, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector)::Tuple => "tridiagonal LU with partial pivoting"
    gttrs!(::Self, ::AbstractChar, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVecOrMat)::AbstractVecOrMat => "solve from tridiagonal LU factors"
    pttrf!(::Self, ::AbstractVector, ::AbstractVector)::Tuple => "LDLᴴ factor of an SPD tridiagonal matrix"
    pttrs!(::Self, ::AbstractVector, ::AbstractVector, ::AbstractVecOrMat)::AbstractVecOrMat => "solve from an SPD tridiagonal LDLᴴ factor"
    ptsv!(::Self, ::AbstractVector, ::AbstractVector, ::AbstractVecOrMat)::Tuple => "SPD tridiagonal solve — factor and solve in one pass"
    gesvx!(::Self, ::Char, ::Char, ::AbstractMatrix, ::AbstractMatrix, ::AbstractVector, ::Char, ::AbstractVector, ::AbstractVector, ::AbstractMatrix, ::AbstractMatrix, ::AbstractVector, ::AbstractVector)::Tuple => "expert driver: equilibrate, factor, solve, refine, bound the error"
    gelqf!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractMatrix => "LQ factorization A = L·Q, reflectors in A and tau"
    geqlf!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractMatrix => "QL factorization A = Q·L"
    gerqf!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractMatrix => "RQ factorization A = R·Q"
    geqp3!(::Self, ::AbstractMatrix, ::AbstractVector, ::AbstractVector)::Tuple => "column-pivoted QR, A·P = Q·R → (A, jpvt, tau)"
    tzrzf!(::Self, ::AbstractMatrix, ::AbstractVector)::Tuple => "trapezoidal RZ reduction → (A, tau)"
    orgqr!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractMatrix => "form Q in place from geqrf reflectors"
    orglq!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractMatrix => "form Q in place from gelqf reflectors"
    orgql!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractMatrix => "form Q in place from geqlf reflectors"
    orgrq!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractMatrix => "form Q in place from gerqf reflectors"
    ormqr!(::Self, ::Char, ::Char, ::AbstractMatrix, ::AbstractVector, ::AbstractMatrix)::AbstractMatrix => "apply geqrf Q to C (side, trans)"
    ormlq!(::Self, ::Char, ::Char, ::AbstractMatrix, ::AbstractVector, ::AbstractMatrix)::AbstractMatrix => "apply gelqf Q to C (side, trans)"
    ormql!(::Self, ::Char, ::Char, ::AbstractMatrix, ::AbstractVector, ::AbstractMatrix)::AbstractMatrix => "apply geqlf Q to C (side, trans)"
    ormrq!(::Self, ::Char, ::Char, ::AbstractMatrix, ::AbstractVector, ::AbstractMatrix)::AbstractMatrix => "apply gerqf Q to C (side, trans)"
    ormrz!(::Self, ::Char, ::Char, ::AbstractMatrix, ::AbstractVector, ::AbstractMatrix)::AbstractMatrix => "apply tzrzf Z to C (side, trans)"
    orgtr!(::Self, ::Char, ::AbstractMatrix, ::AbstractVector, ::AbstractMatrix)::AbstractMatrix => "form Q from sytrd reflectors into a caller-supplied Q"
    ungtr!(::Self, ::Char, ::AbstractMatrix, ::AbstractVector, ::AbstractMatrix)::AbstractMatrix => "form Q from hetrd reflectors into a caller-supplied Q"
end
