"""
Pose estimation optimization using nonlinear least squares.

This module implements pose estimation by minimizing reprojection errors
using SimpleNonlinearSolve.jl and integrating with ProbabilisticParameterEstimators
noise models.
"""

"""
    inv(U::UpperTriangular{T, <:SMatrix}) where T

Custom inverse for upper triangular static matrices using back-substitution.
Preserves SMatrix type instead of converting to Matrix.
"""
function LinearAlgebra.inv(U::UpperTriangular{T,<:SMatrix{N,N}}) where {T,N}
    A = parent(U)

    # Build columns as SVectors, then construct SMatrix from tuple
    cols = ntuple(N) do j
        # Standard basis vector for column j
        b = SVector{N}(i == j ? one(T) : zero(T) for i in 1:N)

        # Back-substitution for column j
        x = MVector{N,T}(undef)
        for i in N:-1:1
            s = b[i]
            for k in i+1:N
                s -= A[i, k] * x[k]
            end
            x[i] = s / A[i, i]
        end

        SVector{N}(x)
    end

    # Construct matrix from column vectors
    return hcat(cols...)
end

"""
    inv(L::LowerTriangular{T, <:SMatrix}) where T

Custom inverse for lower triangular static matrices using forward-substitution.
Preserves SMatrix type instead of converting to Matrix.
"""
function LinearAlgebra.inv(L::LowerTriangular{T,<:SMatrix{N,N}}) where {T,N}
    A = parent(L)

    # Build columns as SVectors, then construct SMatrix from tuple
    cols = ntuple(N) do j
        # Standard basis vector for column j
        b = SVector{N}(i == j ? one(T) : zero(T) for i in 1:N)

        # Forward-substitution for column j
        x = MVector{N,T}(undef)
        for i in 1:N
            s = b[i]
            for k in 1:i-1
                s -= A[i, k] * x[k]
            end
            x[i] = s / A[i, i]
        end

        SVector{N}(x)
    end

    # Construct matrix from column vectors
    return hcat(cols...)
end
