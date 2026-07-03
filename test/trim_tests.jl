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
    )
end
