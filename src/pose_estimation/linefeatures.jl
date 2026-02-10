"""
    LineFeatures{T, T′′, S, WL, OL, CC, M, M′}

Line feature observations and noise model for pose estimation.

# Fields
- `world_line_endpoints`: Vector of pairs of WorldPoints defining lines in world space
- `observed_lines`: Vector of Line objects (r, theta) from observations
- `camconfig`: Camera configuration
- `cov`: Covariance matrix for observation errors
- `Linv`: Inverted lower triangular part of covariance
"""
struct LineFeatures{
    T,T′,T′′,
    WL<:AbstractVector{<:Tuple{WorldPoint{T},WorldPoint{T}}},
    OL<:AbstractVector{Line{T′,T′′}},
    CC<:AbstractCameraConfig{:offset},
    M<:AbstractMatrix,
    M′<:AbstractMatrix
}
    world_line_endpoints::WL
    observed_lines::OL
    camconfig::CC
    cov::M
    Linv::M′
end
function LineFeatures(
    world_line_endpoints::AbstractVector{<:NTuple{2,<:WorldPoint{<:WithDims(m)}}},
    observed_lines::AbstractVector{<:Line{<:WithDims(px),<:WithDims(rad)}},
    camconfig=CAMERA_CONFIG_OFFSET,
    noisemodel::NoiseModel=_defaultnoisemodel_lines(world_line_endpoints)
)
    n = length(world_line_endpoints)
    cov = covmatrix(noisemodel) |> (world_line_endpoints isa StaticArray ? SMatrix{3n,3n} : Matrix)
    return LineFeatures(world_line_endpoints, observed_lines, camconfig, cov)
end
function LineFeatures(
    world_line_endpoints::AbstractVector{<:NTuple{2,<:WorldPoint{<:WithDims(m)}}},
    observed_lines::AbstractVector{<:Line{<:WithDims(px),<:WithDims(rad)}},
    camconfig,
    cov::AbstractMatrix
)
    U = cholesky(cov).U
    Linv = inv(U')  # Preserve static array type if input is static
    return LineFeatures(world_line_endpoints, observed_lines, camconfig, cov, Linv)
end

"""
    pose_optimization_objective_lines(cam_pos, cam_rot, line_features)

Compute weighted line feature residuals.

# Arguments
- `cam_pos`: Camera position (WorldPoint)
- `cam_rot`: Camera rotation (Rotation)
- `line_features`: LineFeatures struct

# Returns
- Weighted line error vector (empty if no lines)
"""
# Fast path for empty line features (NO_LINES) — trivially type-stable
@stable function pose_optimization_objective_lines(
    ::WorldPoint, ::Rotation,
    ::LineFeatures{<:Any,<:Any,<:Any,<:StaticVector{0},<:StaticVector{0}}
)
    return SVector{0,Float64}()
end

@stable function pose_optimization_objective_lines(
    cam_pos::WorldPoint,
    cam_rot::Rotation,
    line_features::LineFeatures
)
    projected_lines = [let
        p1 = project(cam_pos, cam_rot, endpoints[1], line_features.camconfig)
        p2 = project(cam_pos, cam_rot, endpoints[2], line_features.camconfig)
        getline(p1, p2)
    end for endpoints in line_features.world_line_endpoints]

    line_errors = [comparelines(pl, ol)
                   for (pl, ol) in zip(projected_lines, line_features.observed_lines)]

    line_errors_vec = _flatten(line_errors)
    Linv = line_features.Linv
    weighted_errors = Linv * line_errors_vec

    return ustrip.(NoUnits, weighted_errors)
end

@stable function comparelines(l1::Line, l2::Line)
    (r1, theta1) = let
        (; r, theta) = l1
        ustrip(px, r), ustrip(rad, theta)
    end
    (r2, theta2) = let
        (; r, theta) = l2
        ustrip(px, r), ustrip(rad, theta)
    end
    resid(r1, theta1, r2, theta2) = SA[
        r1 - r2
        cos(theta1) - cos(theta2)
        sin(theta1) - sin(theta2)
    ]
    resid1 = resid(r1, theta1, r2, theta2)
    resid2 = resid(r1, theta1, -r2, theta2 + pi)
    return (norm(resid1) < norm(resid2) ? resid1 : resid2)
end

# Fully concrete empty line features for when no lines are used
const NO_LINES = let
    T = typeof(1.0m)
    RT = typeof(1.0px)
    TT = typeof(1.0rad)
    LineFeatures(
        SVector{0,Tuple{WorldPoint{T},WorldPoint{T}}}(),
        SVector{0,Line{RT,TT}}(),
        CAMERA_CONFIG_OFFSET,
        SMatrix{0,0,Float64}(),
        SMatrix{0,0,Float64}()
    )
end
