"""
    PointFeatures{T, T′, T′′, S, RC, OC, CC, M, M′}

Point feature observations and noise model for pose estimation.
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
function PointFeatures(runway_corners, observed_corners, camconfig=CAMERA_CONFIG_OFFSET, noisemodel::NoiseModel=_defaultnoisemodel_points(runway_corners))
    cov = covmatrix(noisemodel) |> Matrix
    return PointFeatures(runway_corners, observed_corners, camconfig, cov)
end
function PointFeatures(runway_corners, observed_corners, camconfig, cov::AbstractMatrix)
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
function pose_optimization_objective_points(
    cam_pos::WorldPoint,
    cam_rot::Rotation,
    point_features::PointFeatures
)
    # Project runway corners to image coordinates
    # WARNING: Don't remove this `let` statement without checking JET tests for type inference.
    # For some reason it's necessary for type inference to work.
    projected_corners = let cam_pos = cam_pos
        [
            project(cam_pos, cam_rot, corner, point_features.camconfig)
            for corner in point_features.runway_corners
        ]
    end

    # Compute reprojection errors and convert to SVector for proper vcat behavior
    corner_errors = [
        SVector(proj - obs)
        for (proj, obs) in zip(projected_corners, point_features.observed_corners)
    ]

    # Flatten corner errors and apply weighting
    corner_errors_vec = reduce(vcat, corner_errors)
    Linv = point_features.Linv
    # `Linv` has units 1/px, but adding it directly to `Linv` has some issue, so we just divide here.
    weighted_errors = (Linv * corner_errors_vec) ./ px

    return ustrip.(NoUnits, weighted_errors)
end
