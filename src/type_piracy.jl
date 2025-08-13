import LinearSolve.init_cacheval
import LinearSolve: OperatorAssumptions
import StaticArrays: MMatrix
#= We commit type piracy here. The reason is that LinearSolve.jl doesn't overload `init_cacheval`
# for `MMatrix` and instead falls back to an `AbstractArray`, which in the end leads to a
# regular `Matrix` in the cache, rather than an `MMatrix`.
# This in turn stops us from stripping BLAS from the library.
# Including this line should allow us to use the upstream LinearSolve.jl again,
# as opposed to my fork: github.com/RomeoV/LinearSolve.jl/tree/rv/allow-mmatrix-cache
=#
function init_cacheval(alg::CholeskyFactorization, A::MMatrix{S1, S2}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {S1, S2}
    cholesky(A)
end
