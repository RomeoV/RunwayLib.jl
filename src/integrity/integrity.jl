using SparseArrays
using LinearAlgebra
using Distributions
using Rotations: RotZYX, params
using StaticArrays
using DifferentiationInterface
using ADTypes

function compute_H(cam_pos::WorldPoint, cam_rot::RotZYX, pts::AbstractVector{<:WorldPoint}, camconf=CAMERA_CONFIG_OFFSET)
    backend = AutoForwardDiff()
    H_pos = reduce(vcat,
        [jacobian(cam_pos_ -> project(WorldPoint(cam_pos_) * m, cam_rot, pt, camconf) |> SVector .|> _ustrip(px),
            backend,
            cam_pos |> SVector .|> _ustrip(m))
         for pt in pts]
    )
    H_rot = reduce(vcat,
        [# `params(::RotZYX)` gives yaw, pitch roll, which is the same order it goes in again (by design)
            jacobian(cam_rot_ -> project(cam_pos, RotZYX(cam_rot_...), pt, camconf) |> SVector .|> _ustrip(px),
                backend,
                params(cam_rot))
            for pt in pts]
    )
    H = [H_pos H_rot]
end

function compute_residual(
    cam_pos::WorldPoint, cam_rot::RotZYX,
    world_pts::AbstractVector{<:WorldPoint},
    observed_pts::AbstractVector{<:ProjectionPoint},
    camconf=CAMERA_CONFIG_OFFSET
)
    @assert length(world_pts) == length(observed_pts) "Number of world points must equal number of observations"

    # Compute predicted projections
    predicted_pts = [project(cam_pos, cam_rot, pt, camconf)
                     for pt in world_pts]

    # Compute Jacobian matrix
    H = compute_H(cam_pos, cam_rot, world_pts, camconf)

    # Compute observation residuals (observed - predicted)
    delta_zs = (observed_pts .- predicted_pts) |> _reduce(vcat)
    r = (I - H * pinv(H)) * delta_zs
    return r
end

compute_integrity_statistic(
    cam_pos::WorldPoint, cam_rot::RotZYX,
    world_pts::AbstractVector{<:WorldPoint},
    observed_pts::AbstractVector{<:ProjectionPoint},
    noise_cov::NoiseModel,
    camconfig=CAMERA_CONFIG_OFFSET
) = compute_integrity_statistic(cam_pos, cam_rot, world_pts, observed_pts, Matrix(covmatrix(noise_cov)), camconfig)


"""
    compute_integrity_statistic(
        cam_pos::WorldPoint, cam_rot::RotZYX,
        world_pts::AbstractVector{<:WorldPoint},
        observed_pts::AbstractVector{<:ProjectionPoint},
        noise_cov::Union{<:AbstractMatrix, <:NoiseModel},
        camconfig=CAMERA_CONFIG_OFFSET
    )

Run the integrity check described in [valentin2025predictive](@cite).
We can use this for runtime assurance to judge whether the measurements and uncertainties are consistent with the parameters of the problem.

# Returns
`NamedTuple` containing
- `stat`: The RAIM-adaptation statistic;
- `p_value`: p-value of Null-hypothesis. If this drops below, say, 5% then we can "reject", i.e., have a failure;
- `dofs`: degrees of freedom (for Χ² distribution); and
- some other information.

# See also
[`WorldPoint`](@ref), [`RotZYX`](@extref Rotations :jl:type:`Rotations.RotZYX`), [`ProjectionPoint`](@ref), [`NoiseModel`](@extref ProbabilisticParameterEstimators.NoiseModel).
"""
function compute_integrity_statistic(
    cam_pos::WorldPoint, cam_rot::RotZYX,
    world_pts::AbstractVector{<:WorldPoint},
    observed_pts::AbstractVector{<:ProjectionPoint},
    noise_cov::AbstractMatrix,
    camconfig=CAMERA_CONFIG_OFFSET
)

    n_pts = length(world_pts)
    n_observations = 2 * n_pts  # Each point gives x,y observation
    n_parameters = 6  # 6-DOF pose: [x, y, z, yaw, pitch, roll]

    @assert n_observations > n_parameters "Need more observations than parameters for integrity monitoring"

    dofs = n_observations - n_parameters

    # Compute residual vector
    r_raw = compute_residual(cam_pos, cam_rot, world_pts, observed_pts, camconfig)
    residual_norm = sqrt(sum(abs2, r_raw))

    # Whiten residuals using noise covariance
    # For χ² test: r_whitened = L^(-T) * r where LL^T = Σ
    L, U = cholesky(noise_cov)
    r_whitened = U' \ r_raw / px .|> _ustrip(NoUnits)

    # Compute chi-squared test statistic
    stat = dot(r_whitened, r_whitened)

    # Compute p-value using chi-squared distribution
    p_value = ccdf(Chisq(dofs), stat)  # P(X > chi_squared) = 1 - cdf(chi_squared)

    return (; stat, p_value, dofs, residual_norm)
end

"""
    compute_worst_case_fault_direction_and_slope_3dof(...)

Compute worst-case fault direction and failure mode slope for 3-DOF (position only).

# Arguments
- `alpha_idx::Int`: Parameter index (1=x, 2=y, 3=z)
- `fault_indices::AbstractVector{Int}`: Measurement indices in fault subset
- `H::AbstractMatrix`: Jacobian matrix (should have 3 columns for position)
- `normalize::Bool=true`: Whether to normalize fault direction

# Returns
- `f_i`: Worst-case fault direction vector
- `slope_g`: Failure mode slope

References (Eq. 32-33, Joerger et al. 2014):
Worst-Case Fault Direction (f_i): f_i = (A) (Aᵀ P A)⁻¹ (Aᵀ s₀)
Worst-Case Failure Mode Slope (g): slope_g² = (s₀ᵀ A) (Aᵀ P A)⁻¹ (Aᵀ s₀)
"""
function compute_worst_case_fault_direction_and_slope_3dof(
    alpha_idx::Int,
    fault_indices::AbstractVector{Int}, 
    H::AbstractMatrix; 
    normalize::Bool=true,
)

    @assert 1 <= alpha_idx <= 3 "alpha_idx must be 1, 2, or 3 for 3-DOF"
    @assert size(H, 2) == 3 "H must have exactly 3 columns for 3-DOF"

    return _compute_worst_case_fault_direction_and_slope(
        alpha_idx, fault_indices, H; normalize=normalize
    )
end


"""
    compute_worst_case_fault_direction_and_slope_6dof(...)

Compute worst-case fault direction and failure mode slope for 6-DOF (position and rotation).

# Arguments
- `alpha_idx::Int`: Parameter index (1=x, 2=y, 3=z, 4=yaw, 5=pitch, 6=roll)
- `fault_indices::AbstractVector{Int}`: Measurement indices in fault subset
- `H::AbstractMatrix`: Jacobian matrix (should have 6 columns for position and rotation)
- `normalize::Bool=true`: Whether to normalize fault direction

# Returns
- `f_i`: Worst-case fault direction vector
- `slope_g`: Failure mode slope

References (Eq. 32-33, Joerger et al. 2014):
Worst-Case Fault Direction (f_i): f_i = (A) (Aᵀ P A)⁻¹ (Aᵀ s₀)
Worst-Case Failure Mode Slope (g): slope_g² = (s₀ᵀ A) (Aᵀ P A)⁻¹ (Aᵀ s₀)
"""
function compute_worst_case_fault_direction_and_slope_6dof(
    alpha_idx::Int, 
    fault_indices::AbstractVector{Int}, 
    H::AbstractMatrix; 
    normalize::Bool=true,
)

    @assert 1 <= alpha_idx <= 6 "alpha_idx must be 1-6 for 6-DOF"
    @assert size(H, 2) == 6 "H must have exactly 6 columns for 6-DOF"

    return _compute_worst_case_fault_direction_and_slope(
        alpha_idx, fault_indices, H; normalize=normalize
    )
end


"""
    _compute_worst_case_fault_direction_and_slope(alpha_idx, fault_indices, H, normalize)

Internal implementation for worst-case fault direction computation.
"""
function _compute_worst_case_fault_direction_and_slope(
    alpha_idx::Int,
    fault_indices::AbstractVector{Int},
    H::AbstractMatrix;
    normalize::Bool=true,
)

    # Define extraction vector s₀ for the state of interest (alpha)
    ndof = size(H, 2)
    α = zeros(ndof)
    α[alpha_idx] = 1

    # Compute S₀ = (HᵀH)⁻¹Hᵀ
    S_0 = pinv(H)
    s_0 = S_0' * α

    # Define Parity Projection Matrix, P = I - H(HᵀH)⁻¹Hᵀ = I - H S₀
    proj_parity = I - H * S_0
    
    # Define Fault Selection Matrix A_i
    n_measurements = size(H, 1)
    n_faults = length(fault_indices)
    A_i = sparse(collect(fault_indices), 1:n_faults, ones(n_faults), n_measurements, n_faults) # |> Matrix

    # Compute the central term, (Aᵀ (I - H S₀) A)⁻¹
    # This measures how "visible" faults in this subspace are to the parity check
    visibility_matrix = A_i' * proj_parity * A_i

    # Compute m_Xi = Aᵀ s₀
    m_Xi = A_i' * s_0

    # Compute worst-case fault direction f_i and normalize
    f_i = A_i * (visibility_matrix \ m_Xi)
    normalize && LinearAlgebra.normalize!(f_i)

    # Compute Slope Squared (Eq 32)
    slope_g_squared = m_Xi' * (visibility_matrix \ m_Xi)
    @assert slope_g_squared >= 0 "Computed negative slope squared, numerical issue?"
    slope_g = sqrt(slope_g_squared)

    return f_i, slope_g
end

