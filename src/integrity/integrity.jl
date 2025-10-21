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
) = compute_integrity_statistic(cam_pos, cam_rot, world_pts, observed_pts, covmatrix(noise_cov), camconfig)


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
