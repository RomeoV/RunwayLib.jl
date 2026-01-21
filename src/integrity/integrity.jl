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

function compute_H(cam_pos::WorldPoint, cam_rot::RotZYX, pf::PointFeatures)
    compute_H(cam_pos, cam_rot, pf.runway_corners, pf.camconfig)
end

function compute_H(cam_pos::WorldPoint, cam_rot::RotZYX, lf::LineFeatures)
    backend = AutoForwardDiff()

    function line_residual(cam_pos_, cam_rot_, endpoints, obs_line)
        p1 = project(cam_pos_, cam_rot_, endpoints[1], lf.camconfig)
        p2 = project(cam_pos_, cam_rot_, endpoints[2], lf.camconfig)
        pred_line = getline(p1, p2)
        comparelines(pred_line, obs_line)
    end

    H_pos = reduce(vcat,
        [jacobian(
            cam_pos_ -> line_residual(WorldPoint(cam_pos_) * m, cam_rot, endpoints, obs_line),
            backend,
            cam_pos |> SVector .|> _ustrip(m))
         for (endpoints, obs_line) in zip(lf.world_line_endpoints, lf.observed_lines)]
    )

    H_rot = reduce(vcat,
        [jacobian(
            cam_rot_ -> line_residual(cam_pos, RotZYX(cam_rot_...), endpoints, obs_line),
            backend,
            params(cam_rot))
         for (endpoints, obs_line) in zip(lf.world_line_endpoints, lf.observed_lines)]
    )

    [H_pos H_rot]
end

function compute_H(cam_pos::WorldPoint, cam_rot::RotZYX, pf::PointFeatures, lf::LineFeatures)
    H_points = compute_H(cam_pos, cam_rot, pf)
    H_lines = compute_H(cam_pos, cam_rot, lf)
    vcat(H_points, H_lines)
end

function compute_parity_residual(
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

function compute_parity_residual(cam_pos::WorldPoint, cam_rot::RotZYX, pf::PointFeatures)
    predicted_pts = [project(cam_pos, cam_rot, pt, pf.camconfig) for pt in pf.runway_corners]
    H = compute_H(cam_pos, cam_rot, pf)
    delta_zs = (pf.observed_corners .- predicted_pts) |> _reduce(vcat)
    (I - H * pinv(H)) * delta_zs
end

function compute_parity_residual(cam_pos::WorldPoint, cam_rot::RotZYX, lf::LineFeatures)
    # Project line endpoints and compute predicted lines
    predicted_lines = [
        let p1 = project(cam_pos, cam_rot, endpoints[1], lf.camconfig),
            p2 = project(cam_pos, cam_rot, endpoints[2], lf.camconfig)
            getline(p1, p2)
        end
        for endpoints in lf.world_line_endpoints
    ]

    H = compute_H(cam_pos, cam_rot, lf)
    # comparelines returns unitless residuals (r in px, cos/sin unitless)
    delta_zs = reduce(vcat, [comparelines(pred, obs) for (pred, obs) in zip(predicted_lines, lf.observed_lines)])
    (I - H * pinv(H)) * delta_zs
end

function compute_parity_residual(cam_pos::WorldPoint, cam_rot::RotZYX, pf::PointFeatures, lf::LineFeatures)
    r_points = compute_parity_residual(cam_pos, cam_rot, pf)
    r_lines = compute_parity_residual(cam_pos, cam_rot, lf)
    vcat(r_points, r_lines)
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

    # Compute parity residual vector
    r_raw = compute_parity_residual(cam_pos, cam_rot, world_pts, observed_pts, camconfig)
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
    compute_integrity_statistic(cam_pos, cam_rot, pf::PointFeatures, lf::LineFeatures; n_parameters=6)

Compute integrity statistic for combined point and line features.

# Arguments
- `cam_pos`: Camera position (WorldPoint)
- `cam_rot`: Camera rotation (RotZYX)
- `pf`: Point features with observations and covariance
- `lf`: Line features with observations and covariance
- `n_parameters`: Number of pose parameters (default 6 for full 6-DOF)

# Returns
Same as the base `compute_integrity_statistic` function.
"""
function compute_integrity_statistic(
    cam_pos::WorldPoint, cam_rot::RotZYX,
    pf::PointFeatures, lf::LineFeatures;
    n_parameters::Int=6
)
    n_point_obs = 2 * length(pf.runway_corners)
    n_line_obs = 3 * length(lf.world_line_endpoints)
    n_observations = n_point_obs + n_line_obs

    @assert n_observations > n_parameters "Need more observations than parameters for integrity monitoring"

    dofs = n_observations - n_parameters

    # Compute parity residuals separately for correct unit handling
    r_points_raw = compute_parity_residual(cam_pos, cam_rot, pf)
    r_lines_raw = compute_parity_residual(cam_pos, cam_rot, lf)

    # Whiten each set of residuals with its own Linv (from feature structs)
    r_points_whitened = (pf.Linv * r_points_raw) ./ px .|> _ustrip(NoUnits)
    r_lines_whitened = lf.Linv * r_lines_raw .|> _ustrip(NoUnits)
    r_whitened = vcat(r_points_whitened, r_lines_whitened)

    # Compute residual norm (after stripping units for consistency)
    residual_norm = sqrt(sum(abs2, r_points_whitened) + sum(abs2, r_lines_whitened))

    # Compute chi-squared test statistic
    stat = dot(r_whitened, r_whitened)

    # Compute p-value using chi-squared distribution
    p_value = ccdf(Chisq(dofs), stat)

    return (; stat, p_value, dofs, residual_norm)
end

# Convenience: points-only with PointFeatures
function compute_integrity_statistic(
    cam_pos::WorldPoint, cam_rot::RotZYX,
    pf::PointFeatures;
    n_parameters::Int=6
)
    n_observations = 2 * length(pf.runway_corners)
    @assert n_observations > n_parameters "Need more observations than parameters for integrity monitoring"

    dofs = n_observations - n_parameters

    r_raw = compute_parity_residual(cam_pos, cam_rot, pf)
    residual_norm = sqrt(sum(abs2, r_raw))

    L, U = cholesky(pf.cov)
    r_whitened = U' \ r_raw / px .|> _ustrip(NoUnits)

    stat = dot(r_whitened, r_whitened)
    p_value = ccdf(Chisq(dofs), stat)

    return (; stat, p_value, dofs, residual_norm)
end

"""
    compute_worst_case_fault_direction_and_slope(
        alpha_idx::Int,
        fault_indices::AbstractVector{Int},
        H::AbstractMatrix,
    )

Computes the worst-case fault direction and corresponding failure mode slope
for a selected pose parameter and fault subset.

# Arguments
- `alpha_idx::Int`: Monitored parameter index
  - 1 = along-track position
  - 2 = cross-track position
  - 3 = height above runway
  - 4 = yaw
  - 5 = pitch
  - 6 = roll
- `fault_indices::AbstractVector{Int}`: Indices of measurements in fault subset
- `H::AbstractMatrix`: Jacobian matrix (ndof columns)
- `noise_cov::AbstractMatrix`: Measurement noise covariance matrix

# Returns
- `f_dir`: Worst-case fault direction (normalized vector)
- `g_slope`: Failure mode slope (quantifies sensitivity to faults in this direction)

!!! note
    The Jacobian `H` must have the correct number of columns for the degrees of
    freedom (3 for position-only, 6 for full pose estimation).
"""
function compute_worst_case_fault_direction_and_slope(
    alpha_idx::Int,
    fault_indices::AbstractVector{Int},
    H::AbstractMatrix,
    noise_cov::AbstractMatrix,
)

    @assert 1 <= alpha_idx <= size(H, 2) "alpha_idx must be 1-ndof"
    @assert all(1 .<= fault_indices .<= size(H, 1)) "fault_indices must in `1:size(H, 1)`"

    # Define extraction vector s₀ for the state of interest (alpha)
    ndof = size(H, 2)
    α = SVector(ntuple(i -> i == alpha_idx ? 1.0 : 0.0, Val(ndof)))

    # Compute S₀ = (HᵀH)⁻¹Hᵀ
    S_0 = pinv(H)
    s_0 = S_0' * α

    # Whiten residuals using noise covariance
    # For χ² test: r_whitened = L^(-T) * r where LL^T = Σ
    L, _ = cholesky(noise_cov)
    Linv = inv(L)

    # Define Parity Projection Matrix, P = I - H(HᵀH)⁻¹Hᵀ = I - H S₀
    proj_parity = I - H * S_0
    proj_parity_Linv = proj_parity * Linv

    # Define Fault Selection Matrix A_i
    n_measurements = size(H, 1)
    n_faults = length(fault_indices)
    A_i = (
        sparse(collect(fault_indices), 1:n_faults, ones(n_faults), n_measurements, n_faults)
    ) |> SMatrix{n_measurements, n_faults}

    # Compute the central term, (Aᵀ (I - H S₀) A)⁻¹
    # This measures how "visible" faults in this subspace are to the parity check
    visibility_matrix = A_i' * proj_parity_Linv * proj_parity_Linv' * A_i

    # Compute m_Xi = Aᵀ s₀
    m_Xi = A_i' * s_0

    # Compute worst-case fault direction f_dir and normalize
    f_dir = A_i * (visibility_matrix \ m_Xi) |> normalize

    # Compute Slope Squared (Eq 32)
    g_slope_squared = m_Xi' * (visibility_matrix \ m_Xi)
    @assert g_slope_squared >= 0 "Computed negative slope squared, numerical issue?"
    g_slope = sqrt(g_slope_squared)

    return f_dir, g_slope
end

