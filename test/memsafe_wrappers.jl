# Positional wrappers for test/memsafe_verify.jl. The native API takes `trans`/`conj` as keywords,
# but `memsafe_report` forwards only positional arguments, so each keyword variant is a named
# function. They live in their own self-contained file because the probe's subprocess `include`s the
# file that defines the probed function and looks the function up by name.
using PureBLAS
import PureBLAS as P

gbmvN!(y, AB, x, m, kl, ku) = P.gbmv!(y, AB, x, m, kl, ku; trans = 'N')
gbmvT!(y, AB, x, m, kl, ku) = P.gbmv!(y, AB, x, m, kl, ku; trans = 'T')
gemvN!(y, A, x) = P.gemv!(y, A, x; trans = 'N')
gemvT!(y, A, x) = P.gemv!(y, A, x; trans = 'T')
geru!(x, y, A) = P.ger!(one(eltype(A)), x, y, A; conj = false)
gerc!(x, y, A) = P.ger!(one(eltype(A)), x, y, A; conj = true)
