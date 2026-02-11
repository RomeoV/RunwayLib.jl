# using BracketingNonlinearSolve
using LinearAlgebra
using Distributions
using Rotations: RotZYX, params
using StaticArrays
using DifferentiationInterface
using ADTypes

# =============================================================================
# compute_H: Jacobian of measurement function w.r.t. pose
# =============================================================================

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
    isempty(lf.world_line_endpoints) && return SMatrix{0,6,Float64}()

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

# =============================================================================
# compute_parity_residual: Raw parity residual (projected onto null space of H)
# =============================================================================

function compute_parity_residual(cam_pos::WorldPoint, cam_rot::RotZYX, pf::PointFeatures)
    predicted_pts = [project(cam_pos, cam_rot, pt, pf.camconfig) for pt in pf.runway_corners]
    H = compute_H(cam_pos, cam_rot, pf)
    delta_zs = SVector.(pf.observed_corners .- predicted_pts) |> _reduce(vcat)
    # Parity projection: (I - H pinv(H)) δz = δz - H ((H'H) \ (H'δz))
    # Using LU instead of SVD-based pinv for AD compatibility (works with Dual numbers)
    delta_zs - H * ((H' * H) \ (H' * delta_zs))
end

function compute_parity_residual(cam_pos::WorldPoint, cam_rot::RotZYX, lf::LineFeatures)
    isempty(lf.world_line_endpoints) && return SVector{0,Float64}()

    predicted_lines = [
        let p1 = project(cam_pos, cam_rot, endpoints[1], lf.camconfig),
            p2 = project(cam_pos, cam_rot, endpoints[2], lf.camconfig)
            getline(p1, p2)
        end
        for endpoints in lf.world_line_endpoints
    ]

    H = compute_H(cam_pos, cam_rot, lf)
    delta_zs = reduce(vcat, [comparelines(pred, obs) for (pred, obs) in zip(predicted_lines, lf.observed_lines)])
    delta_zs - H * ((H' * H) \ (H' * delta_zs))
end

# =============================================================================
# compute_whitened_parity_residual: Whitened residuals for chi-squared test
# =============================================================================

function compute_whitened_parity_residual(cam_pos::WorldPoint, cam_rot::RotZYX, pf::PointFeatures)
    r_raw = compute_parity_residual(cam_pos, cam_rot, pf)
    (pf.Linv * r_raw) ./ px .|> _ustrip(NoUnits)
end

function compute_whitened_parity_residual(cam_pos::WorldPoint, cam_rot::RotZYX, lf::LineFeatures)
    isempty(lf.world_line_endpoints) && return SVector{0,Float64}()
    r_raw = compute_parity_residual(cam_pos, cam_rot, lf)
    lf.Linv * r_raw .|> _ustrip(NoUnits)
end

# =============================================================================
# compute_integrity_statistic: Main entry point
# =============================================================================

"""
    compute_integrity_statistic(cam_pos, cam_rot, pf::PointFeatures, lf::LineFeatures=NO_LINES; n_parameters=6)

Run the integrity check described in [valentin2025predictive](@cite).
We can use this for runtime assurance to judge whether the measurements and uncertainties
are consistent with the parameters of the problem.

# Arguments
- `cam_pos`: Camera position (WorldPoint)
- `cam_rot`: Camera rotation (RotZYX)
- `pf`: Point features with observations and covariance
- `lf`: Line features with observations and covariance (default: NO_LINES)
- `n_parameters`: Number of pose parameters (default 6 for full 6-DOF)

# Returns
`NamedTuple` containing
- `stat`: The RAIM-adaptation statistic;
- `p_value`: p-value of Null-hypothesis. If this drops below, say, 5% then we can "reject", i.e., have a failure;
- `dofs`: degrees of freedom (for Χ² distribution); and
- `residual_norm`: norm of the whitened residual vector.

# See also
[`WorldPoint`](@ref), [`RotZYX`](@extref Rotations :jl:type:`Rotations.RotZYX`), [`PointFeatures`](@ref), [`LineFeatures`](@ref).
"""
function compute_integrity_statistic(
    cam_pos::WorldPoint, cam_rot::RotZYX,
    pf::PointFeatures, lf::LineFeatures=NO_LINES;
    n_parameters::Int=6
)
    n_point_obs = 2 * length(pf.runway_corners)
    n_line_obs = 3 * length(lf.world_line_endpoints)
    n_observations = n_point_obs + n_line_obs

    @assert n_observations > n_parameters "Need more observations than parameters for integrity monitoring"

    dofs = n_observations - n_parameters

    # Compute whitened parity residuals
    r_points = compute_whitened_parity_residual(cam_pos, cam_rot, pf)
    r_lines = compute_whitened_parity_residual(cam_pos, cam_rot, lf)
    r_whitened = vcat(r_points, r_lines)

    residual_norm = norm(r_whitened)
    stat = dot(r_whitened, r_whitened)
    p_value = ccdf(Chisq(dofs), stat)

    return (; stat, p_value, dofs, residual_norm)
end

# Backward-compatible API: raw vectors → PointFeatures
function compute_integrity_statistic(
    cam_pos::WorldPoint, cam_rot::RotZYX,
    world_pts::AbstractVector{<:WorldPoint},
    observed_pts::AbstractVector{<:ProjectionPoint},
    noise_cov::AbstractMatrix,
    camconfig=CAMERA_CONFIG_OFFSET
)
    pf = PointFeatures(world_pts, observed_pts, camconfig, noise_cov)
    compute_integrity_statistic(cam_pos, cam_rot, pf, NO_LINES)
end

# NoiseModel convenience wrapper
compute_integrity_statistic(
    cam_pos::WorldPoint, cam_rot::RotZYX,
    world_pts::AbstractVector{<:WorldPoint},
    observed_pts::AbstractVector{<:ProjectionPoint},
    noise_cov::NoiseModel,
    camconfig=CAMERA_CONFIG_OFFSET
) = compute_integrity_statistic(cam_pos, cam_rot, world_pts, observed_pts, Matrix(covmatrix(noise_cov)), camconfig)
