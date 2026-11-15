# Positive controls for memsafe in the axpy shape: raw pointers, one element past the end.
function ctl_store_past!(n::Int, a::Float64, x::Vector{Float64}, y::Vector{Float64})
    px, py = pointer(x), pointer(y)
    GC.@preserve x y for i in 1:n
        unsafe_store!(py, unsafe_load(py, i) + a * unsafe_load(px, i), i + 1)
    end
    return y
end
function ctl_load_past!(n::Int, a::Float64, x::Vector{Float64}, y::Vector{Float64})
    px, py = pointer(x), pointer(y)
    GC.@preserve x y for i in 1:n
        y[i] += a * unsafe_load(px, i + 1)
    end
    return y
end
