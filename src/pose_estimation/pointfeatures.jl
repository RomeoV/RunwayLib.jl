"""
    PointFeatures{T, T′, T′′, RC, OC, CC, M, M′}

Point feature observations and noise model for pose estimation.

# Fields
- `runway_corners`: Vector of WorldPoints defining runway corner positions in world space
- `observed_corners`: Vector of ProjectionPoints from image observations
- `camconfig`: Camera configuration
- `cov`: Covariance matrix for observation errors
- `Linv`: Inverted lower triangular part of covariance
"""
struct PointFeatures{
    T,T′,T′′,
    RC<:AbstractVector{WorldPoint{T}},
    OC<:AbstractVector{ProjectionPoint{T′,:offset}},
    CC<:AbstractCameraConfig{:offset},
    M<:AbstractMatrix{T′′},
    M′<:AbstractMatrix{T′′}
}
    runway_corners::RC
    observed_corners::OC
    camconfig::CC
    cov::M
    Linv::M′
end
function PointFeatures(
    runway_corners::AbstractVector{<:WorldPoint{<:WithDims(m)}},
    observed_corners::AbstractVector{<:ProjectionPoint{<:WithDims(px)}},
    camconfig=CAMERA_CONFIG_OFFSET,
    noisemodel::NoiseModel=_defaultnoisemodel_points(runway_corners)
)
    n = length(runway_corners)
    cov = covmatrix(noisemodel) |> (runway_corners isa StaticArray ? SMatrix{2n,2n} : Matrix)
    return PointFeatures(runway_corners, observed_corners, camconfig, cov)
end
function PointFeatures(
    runway_corners::AbstractVector{<:WorldPoint{<:WithDims(m)}},
    observed_corners::AbstractVector{<:ProjectionPoint{<:WithDims(px)}},
    camconfig, cov::AbstractMatrix)
    U = cholesky(cov).U
    Linv = inv(U')  # Preserve static array type if input is static
    return PointFeatures(runway_corners, observed_corners, camconfig, cov, Linv)
end

"""
    pose_optimization_objective_points(cam_pos, cam_rot, point_features)

Compute weighted point feature residuals.

# Arguments
- `cam_pos`: Camera position (WorldPoint)
- `cam_rot`: Camera rotation (Rotation)
- `point_features`: PointFeatures struct

# Returns
- Weighted reprojection error vector
"""
@stable function pose_optimization_objective_points(
    cam_pos::WorldPoint,
    cam_rot::Rotation,
    point_features::PointFeatures
)
    projected_corners = [project(cam_pos, cam_rot, corner, point_features.camconfig)
                         for corner in point_features.runway_corners]

    corner_errors_vec = _flatten(projected_corners .- point_features.observed_corners)
    Linv = point_features.Linv
    # `Linv` has units 1/px, but adding it directly to `Linv` has some issue, so we just divide here.
    weighted_errors = (Linv * corner_errors_vec) ./ px

    return ustrip.(NoUnits, weighted_errors)
end
