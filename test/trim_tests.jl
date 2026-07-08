# Deep trim-safety check on the @ccallable C-ABI entry points — these are exactly what
# juliac --trim compiles into libpureblas.so, so they must contain no dynamic dispatch / runtime
# reflection. TrimCheck @validate runs the same reachability analysis as juliac. Mirrors PureFFT's
# "TrimCheck trim-safety" testitem (signatures given by argument TYPES, not values).

@testitem "TrimCheck trim-safety (C-ABI entry points)" begin
    using TrimCheck
    @validate(
        init = begin
            using PureBLAS
        end,
        # void-returning real + complex
        PureBLAS.daxpy_64_(Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64}),
        PureBLAS.zaxpy_64_(Ptr{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{Int64}, Ptr{ComplexF64}, Ptr{Int64}),
        PureBLAS.dscal_64_(Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}),
        PureBLAS.dcopy_64_(Ptr{Int64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64}),
        # value-returning
        PureBLAS.ddot_64_(Ptr{Int64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64}),
        PureBLAS.dnrm2_64_(Ptr{Int64}, Ptr{Float64}, Ptr{Int64}),
        PureBLAS.dznrm2_64_(Ptr{Int64}, Ptr{ComplexF64}, Ptr{Int64}),
        PureBLAS.dzasum_64_(Ptr{Int64}, Ptr{ComplexF64}, Ptr{Int64}),
        PureBLAS.idamax_64_(Ptr{Int64}, Ptr{Float64}, Ptr{Int64}),
        # Level-3 GEMM: char args (2× Ptr{UInt8}) + trailing Fortran string-length Clongs. Exercises the
        # full blocked/unpacked/complex-split-pack path + the L3 workspace scratch (all const-dispatched
        # for s/d/c/z, so the workspace IdDict fallback is never reached).
        PureBLAS.dgemm_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Float64},
            Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Clong, Clong),
        PureBLAS.sgemm_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Float32},
            Ptr{Float32}, Ptr{Int64}, Ptr{Float32}, Ptr{Int64}, Ptr{Float32}, Ptr{Float32}, Ptr{Int64}, Clong, Clong),
        PureBLAS.zgemm_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF64},
            Ptr{ComplexF64}, Ptr{Int64}, Ptr{ComplexF64}, Ptr{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{Int64}, Clong, Clong),
        PureBLAS.cgemm_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF32},
            Ptr{ComplexF32}, Ptr{Int64}, Ptr{ComplexF32}, Ptr{Int64}, Ptr{ComplexF32}, Ptr{ComplexF32}, Ptr{Int64}, Clong, Clong),
        # Complex rank-k (herk/syrk): the small-n trans='N' path now routes through the unpacked triangular
        # kernel (`_ctri_unpacked!` → `_uker_cmplx!` with the TRI-store Val + runtime d0/upper). Same union-
        # split-prone graph as cgemm — validate the ccallable roots so a trim regression in the tri sweep or
        # the resolver chain REDs here, not in a benchmark.
        PureBLAS.zsyrk_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64},
            Ptr{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{Int64}, Clong, Clong),
        PureBLAS.zherk_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Ptr{ComplexF64},
            Ptr{Int64}, Ptr{Float64}, Ptr{ComplexF64}, Ptr{Int64}, Clong, Clong),
        PureBLAS.csyrk_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF32}, Ptr{ComplexF32},
            Ptr{Int64}, Ptr{ComplexF32}, Ptr{ComplexF32}, Ptr{Int64}, Clong, Clong),
        PureBLAS.cherk_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Float32}, Ptr{ComplexF32},
            Ptr{Int64}, Ptr{Float32}, Ptr{ComplexF32}, Ptr{Int64}, Clong, Clong),
        # Complex rank-2k (her2k/syr2k): small-n trans='N' routes through `_ctri2_unpacked!` (two unpacked-tri
        # products sharing the same `_ctri_core!`/`_uker_cmplx!` graph as herk/syrk above).
        PureBLAS.zsyr2k_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64},
            Ptr{Int64}, Ptr{ComplexF64}, Ptr{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{Int64}, Clong, Clong),
        PureBLAS.zher2k_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64},
            Ptr{Int64}, Ptr{ComplexF64}, Ptr{Int64}, Ptr{Float64}, Ptr{ComplexF64}, Ptr{Int64}, Clong, Clong),
        # Complex trsm (all sides/trans): side='R' transA='C' now has a direct SIMD base `_trsm_cmplx_dRC!`
        # (the zpotrf-lower recursion path). Validate the ccallable roots the whole trsm dispatch tree.
        PureBLAS.ztrsm_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64},
            Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{Int64}, Ptr{ComplexF64}, Ptr{Int64}, Clong, Clong, Clong, Clong),
        PureBLAS.ctrsm_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64},
            Ptr{ComplexF32}, Ptr{ComplexF32}, Ptr{Int64}, Ptr{ComplexF32}, Ptr{Int64}, Clong, Clong, Clong, Clong),
        # LAPACK gesvd: in-place gesvd!(A,U,S,Vᵀ) into caller PtrMatrix buffers + full jobu/jobvt/'O' coverage.
        # Exercises gebrd + bdsqr + bdsdc divide-and-conquer + the compact-WY back-transform, all trim-clean.
        PureBLAS.dgesvd_64_(Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64},
            Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64}, Ptr{Int64}, Clong, Clong),
    )
end
