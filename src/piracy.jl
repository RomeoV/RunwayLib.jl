"See #107"
function NonlinearSolveBase.fix_incompatible_linsolve_arguments(A::Symmetric{T,<:Matrix{T}}, b, u::SArray) where {T}
    (Core.Compiler.return_type(\, Tuple{typeof(A),typeof(b)}) <: typeof(u)) && return u
    return MArray(u)
end
