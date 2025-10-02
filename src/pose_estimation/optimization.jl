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
function LinearAlgebra.inv(U::UpperTriangular{T, <:SMatrix{N,N}}) where {T,N}
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
                s -= A[i,k] * x[k]
            end
            x[i] = s / A[i,i]
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
function LinearAlgebra.inv(L::LowerTriangular{T, <:SMatrix{N,N}}) where {T,N}
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
                s -= A[i,k] * x[k]
            end
            x[i] = s / A[i,i]
        end

        SVector{N}(x)
    end

    # Construct matrix from column vectors
    return hcat(cols...)
end

abstract type AbstractPoseOptimizationParams end

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
function PointFeatures(runway_corners, observed_corners, camconfig, noisemodel::NoiseModel)
    cov = covmatrix(noisemodel) |> Matrix
    return PointFeatures(runway_corners, observed_corners, camconfig, cov)
end
function PointFeatures(runway_corners, observed_corners, camconfig, cov::AbstractMatrix)
    U = cholesky(cov).U
    Linv = inv(U')  # Preserve static array type if input is static
    return PointFeatures(runway_corners, observed_corners, camconfig, cov, Linv)
end

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
function LineFeatures(world_line_endpoints, observed_lines, camconfig, noisemodel::NoiseModel)
    cov = covmatrix(noisemodel) |> Matrix
    return LineFeatures(world_line_endpoints, observed_lines, camconfig, cov)
end
function LineFeatures(world_line_endpoints, observed_lines, camconfig, cov::AbstractMatrix)
    U = cholesky(cov).U
    Linv = inv(U')  # Preserve static array type if input is static
    return LineFeatures(world_line_endpoints, observed_lines, camconfig, cov, Linv)
end

# Fully concrete empty line features for when no lines are used
const NO_LINES = let
    T = typeof(1.0m)
    RT = typeof(1.0px)
    TT = typeof(1.0rad)
    LineFeatures(
        Tuple{WorldPoint{T},WorldPoint{T}}[],
        Line{RT,TT}[],
        CAMERA_CONFIG_OFFSET,
        Matrix{Float64}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0)
    )
end

"""
    PoseOptimizationParams6DOF{PF, LF}

Parameters for 6-DOF pose optimization (position + attitude).
"""
struct PoseOptimizationParams6DOF{
    PF<:PointFeatures,
    LF<:LineFeatures
} <: AbstractPoseOptimizationParams
    point_features::PF
    line_features::LF
end

"""
    PoseOptimizationParams3DOF{A, PF, LF}

Parameters for 3-DOF pose optimization (position only with known attitude).
"""
struct PoseOptimizationParams3DOF{
    A<:Rotation{3},
    PF<:PointFeatures,
    LF<:LineFeatures
} <: AbstractPoseOptimizationParams
    point_features::PF
    line_features::LF
    known_attitude::A
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
    Linv = point_features.Linv / 1px
    weighted_errors = Linv * corner_errors_vec

    return ustrip.(NoUnits, weighted_errors)
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
function pose_optimization_objective_lines(
    cam_pos::WorldPoint,
    cam_rot::Rotation,
    line_features::LineFeatures
)
    # Project line endpoints to image coordinates and compute line parameters
    projected_lines = [
        # Project both endpoints of each line
        let p1 = project(cam_pos, cam_rot, endpoints[1], line_features.camconfig),
            p2 = project(cam_pos, cam_rot, endpoints[2], line_features.camconfig)
            # Convert projected endpoints to line representation (r, theta)
            getline(p1, p2)
        end
        for endpoints in line_features.world_line_endpoints
    ]

    # Compute line errors
    line_errors = [
        comparelines(lpred, lobs)
        for (lpred, lobs) in zip(projected_lines, line_features.observed_lines)
    ]

    # Flatten line errors and apply weighting
    line_errors_vec = reduce(vcat, line_errors; init=SVector{0,Float64}())
    Linv = line_features.Linv
    weighted_errors = Linv * line_errors_vec

    return ustrip.(NoUnits, weighted_errors)
end

"""
    pose_optimization_objective(pose_params, ps)

Unified optimization function for pose estimation.

# Arguments
- `pose_params`: Vector of pose parameters
    - `[x, y, z, roll, pitch, yaw]` for 6-DOF
    - `[x, y, z]` for 3-DOF
- `ps`: `PoseOptimizationParams6DOF` or `PoseOptimizationParams3DOF`

# Returns
- Weighted reprojection error vector combining point and line features
"""
function pose_optimization_objective(
    optvar::AbstractVector{T},
    ps::AbstractPoseOptimizationParams
) where {T<:Real}
    # Extract camera position from optimization variables
    cam_pos = WorldPoint(optvar[1:3]m)

    # Determine camera rotation via pattern matching
    cam_rot = @match ps begin
        ps::PoseOptimizationParams6DOF => RotZYX(
            roll=optvar[4]rad, pitch=optvar[5]rad, yaw=optvar[6]rad
        )
        ps::PoseOptimizationParams3DOF => ps.known_attitude
    end

    # Compute point feature residuals
    point_residuals = pose_optimization_objective_points(cam_pos, cam_rot, ps.point_features)

    # Compute line feature residuals
    line_residuals = pose_optimization_objective_lines(cam_pos, cam_rot, ps.line_features)

    # Combine residuals
    return vcat(point_residuals, line_residuals)
end

function comparelines(l1::Line, l2::Line)
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



function setup_for_precompile()
    runway_corners = [
        WorldPoint(1000.0m, -50.0m, 0.0m),
        WorldPoint(1000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, -50.0m, 0.0m),
    ]
    true_pos = WorldPoint(-1300.0m, 0.0m, 80.0m)
    true_rot = RotZYX(roll=0.03, pitch=0.04, yaw=0.05)
    projections = [project(true_pos, true_rot, corner, CAMERA_CONFIG_OFFSET) for corner in runway_corners]
    return (; runway_corners, projections, true_pos, true_rot)
end


const _defaultnoisemodel(pts) =
    let
        distributions = [SA[Normal(0.0, 2.0), Normal(0.0, 2.0)] for _ in pts] |> Array
        UncorrGaussianNoiseModel(reduce(vcat, distributions))
    end

const AD = AutoForwardDiff(; chunksize=1)
const POSEOPTFN = NonlinearFunction{false,FullSpecialize}(pose_optimization_objective)
const ALG = LevenbergMarquardt(; autodiff=AD, linsolve=CholeskyFactorization(),
    disable_geodesic=Val(true))

"Camera configuration type for precompilation"
const CAMCONF4COMP = CAMERA_CONFIG_OFFSET
const S4COMP = :offset

const CACHE_6DOF = let
    (; runway_corners, projections, true_pos, true_rot) = setup_for_precompile()
    noise_model = _defaultnoisemodel(projections)
    point_features = PointFeatures(runway_corners, projections, CAMERA_CONFIG_OFFSET, noise_model)
    ps = PoseOptimizationParams6DOF(point_features, NO_LINES)
    prob = NonlinearLeastSquaresProblem{false}(POSEOPTFN, rand(6), ps)
    T = Float64
    # sqrt of the default
    reltol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    abstol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    init(prob, ALG; reltol, abstol)
end

const CACHE_3DOF = let
    (; runway_corners, projections, true_pos, true_rot) = setup_for_precompile()
    noise_model = _defaultnoisemodel(projections)
    point_features = PointFeatures(runway_corners, projections, CAMERA_CONFIG_OFFSET, noise_model)
    ps = PoseOptimizationParams3DOF(point_features, NO_LINES, true_rot)
    prob = NonlinearLeastSquaresProblem{false}(POSEOPTFN, rand(3), ps)
    T = Float64
    # sqrt of the default
    reltol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    abstol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    init(prob, ALG; reltol, abstol)
end

function estimatepose6dof(
    runway_corners::AbstractVector{<:WorldPoint},
    observed_corners::AbstractVector{<:ProjectionPoint{T,:offset}},
    camconfig::AbstractCameraConfig{:offset}=CAMERA_CONFIG_OFFSET,
    noise_model::N=_defaultnoisemodel(observed_corners);
    initial_guess_pos::AbstractVector{<:Length}=SA[-1000.0, 0.0, 100.0]m,
    initial_guess_rot::AbstractVector{<:DimensionlessQuantity}=SA[0.0, 0.0, 0.0]rad,
    optimization_config=DEFAULT_OPTIMIZATION_CONFIG
) where {T,N}
    u₀ = [
        initial_guess_pos .|> _ustrip(m);
        initial_guess_rot .|> _ustrip(rad)
    ] |> Array

    # for precompile we need the correct types
    observed_corners = [
        convertcamconf(CAMCONF4COMP, camconfig, proj)
        for proj in observed_corners
    ]
    point_features = PointFeatures(
        runway_corners |> Vector, observed_corners |> Vector,
        CameraConfig{S4COMP}(camconfig), noise_model
    )
    ps = PoseOptimizationParams6DOF(point_features, NO_LINES)

    # Get or create cache for this problem size
    cache = CACHE_6DOF
    reinit!(cache, u₀; p=ps)
    solve!(cache)
    sol = (; u=cache.u, retcode=cache.retcode)

    !successful_retcode(sol.retcode) && throw(OptimizationFailedError(sol.retcode, sol))
    pos = WorldPoint(sol.u[1:3]m)
    rot = RotZYX(roll=sol.u[4]rad, pitch=sol.u[5]rad, yaw=sol.u[6]rad)
    return (; pos, rot)
end

function estimatepose3dof(
    runway_corners::AbstractVector{<:WorldPoint},
    observed_corners::AbstractVector{<:ProjectionPoint{T,:offset}},
    known_attitude::RotZYX,
    camconfig::AbstractCameraConfig{:offset}=CAMERA_CONFIG_OFFSET,
    noise_model::N=_defaultnoisemodel(observed_corners);
    initial_guess_pos::AbstractVector{<:Length}=SA[-1000.0, 0.0, 100.0]m,
    optimization_config=DEFAULT_OPTIMIZATION_CONFIG
) where {T,N}

    u₀ = initial_guess_pos .|> _ustrip(m) |> Array

    # for precompile we need the correct types
    observed_corners = [
        convertcamconf(CAMCONF4COMP, camconfig, proj)
        for proj in observed_corners
    ]
    point_features = PointFeatures(
        runway_corners |> Vector, observed_corners |> Vector,
        CameraConfig{S4COMP}(camconfig), noise_model
    )
    ps = PoseOptimizationParams3DOF(point_features, NO_LINES, known_attitude)

    # Get or create cache for this problem size
    cache = CACHE_3DOF
    reinit!(cache, u₀; p=ps)
    solve!(cache)
    sol = (; u=cache.u, retcode=cache.retcode)

    !successful_retcode(sol.retcode) && throw(OptimizationFailedError(sol.retcode, sol))
    pos = WorldPoint(sol.u[1:3]m)
    rot = known_attitude
    return (; pos, rot)
end
