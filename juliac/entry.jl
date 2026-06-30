# juliac entry point for the trimmed shared library: loading PureBLAS defines all the
# `Base.@ccallable` Fortran-ABI symbols (cabi.jl), which `--compile-ccallable` then exports.
using PureBLAS
