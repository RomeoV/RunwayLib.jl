"""
Pose estimation optimization using nonlinear least squares.

This module implements pose estimation by minimizing reprojection errors
using SimpleNonlinearSolve.jl and integrating with ProbabilisticParameterEstimators
noise models.
"""


abstract type AbstractPoseOptimizationParams end

"""
    PoseOptimizationParams6DOF{T, T′, S, RC, OC, M}

Parameters for 6-DOF pose optimization (position + attitude).
"""
struct PoseOptimizationParams6DOF{
        T, T′, T′′, S,
        RC <: AbstractVector{WorldPoint{T}},
        OC <: AbstractVector{ProjectionPoint{T′, S}},
        M <: AbstractMatrix{T′′},
    } <: AbstractPoseOptimizationParams
    runway_corners::RC
    observed_corners::OC
    camconfig::CameraConfig{S}
    Linv::M
end
function PoseOptimizationParams6DOF(runway_corners, observed_corners, camconfig, noisemodel::NoiseModel)
    cov = covmatrix(noisemodel)
    U = cholesky(cov).U
    Linv = inv(U')
    return PoseOptimizationParams6DOF(runway_corners, observed_corners, camconfig, Linv)
end

"""
    PoseOptimizationParams3DOF{T, T′, S, A, RC, OC, M}

Parameters for 3-DOF pose optimization (position only with known attitude).
"""
struct PoseOptimizationParams3DOF{
        T, T′, T′′, S,
        A <: Rotation{3},
        RC <: AbstractVector{WorldPoint{T}},
        OC <: AbstractVector{ProjectionPoint{T′, S}},
        M <: AbstractMatrix{T′′},
    } <: AbstractPoseOptimizationParams
    runway_corners::RC
    observed_corners::OC
    camconfig::CameraConfig{S}
    Linv::M
    known_attitude::A
end
function PoseOptimizationParams3DOF(runway_corners, observed_corners, camconfig, noisemodel::NoiseModel, known_attitude)
    Linv = inv(cholesky(covmatrix(noisemodel)).U')
    return PoseOptimizationParams3DOF(runway_corners, observed_corners, camconfig, Linv, known_attitude)
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
- Weighted reprojection error vector
"""
function pose_optimization_objective(
        optvar::AbstractVector{T},
        ps::AbstractPoseOptimizationParams
    ) where {T <: Real}
    # Extract camera position from optimization variables
    cam_pos = WorldPoint(optvar[1:3]m)

    # Determine camera rotation via pattern matching
    cam_rot = @match ps begin
        ps::PoseOptimizationParams6DOF => RotZYX(
            roll = optvar[4]rad, pitch = optvar[5]rad, yaw = optvar[6]rad
        )
        ps::PoseOptimizationParams3DOF => ps.known_attitude
    end

    # Project runway corners to image coordinates
    # WARNING: Don't remove this `let` statement without checking JET tests for type inference.
    # For some reason it's necessary for type inference to work.
    projected_corners = let cam_pos = cam_pos
        [
            project(cam_pos, cam_rot, corner, ps.camconfig)
                for corner in ps.runway_corners
        ]
    end

    # Compute reprojection errors
    error_vectors = [
        # we change the type here from a strongly typed "ProjectionPoint"
        # to a more weakly typed vector because we are about to concatenate them
        vec(proj - obs)
            for (proj, obs) in zip(projected_corners, ps.observed_corners)
    ]
    errors = reduce(vcat, error_vectors)

    Linv = ps.Linv / 1px
    weighted_errors = Linv * errors

    return ustrip.(NoUnits, weighted_errors)
end

function setup_for_precompile()
    runway_corners = [
        WorldPoint(1000.0m, -50.0m, 0.0m),
        WorldPoint(1000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, -50.0m, 0.0m),
    ]
    true_pos = WorldPoint(-1300.0m, 0.0m, 80.0m)
    true_rot = RotZYX(roll = 0.03, pitch = 0.04, yaw = 0.05)
    projections = [project(true_pos, true_rot, corner, CAMERA_CONFIG_OFFSET) for corner in runway_corners]
    return (; runway_corners, projections, true_pos, true_rot)
end


const _defaultnoisemodel(pts) = let
    distributions = [SA[Normal(0.0, 2.0), Normal(0.0, 2.0)] for _ in pts]
    UncorrGaussianNoiseModel(reduce(vcat, distributions))
end

const POSEOPTFN = NonlinearFunction{false, SciMLBase.FullSpecialize}(pose_optimization_objective)
const AD = AutoForwardDiff(; chunksize = 1)
const ALG = LevenbergMarquardt(; autodiff = AD, linsolve = CholeskyFactorization())

"Camera configuration type for precompilation"
const CAMCONF4COMP = CAMERA_CONFIG_OFFSET

# Const caches removed - now created per function call to support variable keypoint numbers

function estimatepose6dof(
        runway_corners::AbstractVector{<:WorldPoint},
        observed_corners::AbstractVector{<:ProjectionPoint{T, S}},
        config::CameraConfig{S};
        noise_model = _defaultnoisemodel(observed_corners),
        initial_guess_pos::AbstractVector{<:Length} = SA[-1000.0, 0.0, 100.0]m,
        initial_guess_rot::AbstractVector{<:DimensionlessQuantity} = SA[0.0, 0.0, 0.0]rad,
        optimization_config = DEFAULT_OPTIMIZATION_CONFIG
    ) where {T, S}
    u₀ = [
        initial_guess_pos .|> _ustrip(m);
        initial_guess_rot .|> _ustrip(rad)
    ] |> Array

    # for precompile we need the correct types
    observed_corners = [
        convertcamconf(CAMCONF4COMP, config, proj)
            for proj in observed_corners
    ]
    ps = PoseOptimizationParams6DOF(
        runway_corners, observed_corners,
        CAMCONF4COMP, noise_model
    )

    # Create problem and cache for this specific problem size
    prob = NonlinearLeastSquaresProblem{false}(POSEOPTFN, u₀, ps)
    cache = init(prob, ALG)
    solve!(cache)
    sol = (; u = cache.u, retcode = cache.retcode)

    !successful_retcode(sol.retcode) && throw(OptimizationFailedError(sol.retcode, sol))
    pos = WorldPoint(sol.u[1:3]m)
    rot = RotZYX(roll = sol.u[4]rad, pitch = sol.u[5]rad, yaw = sol.u[6]rad)
    return (; pos, rot)
end

function estimatepose3dof(
        runway_corners::AbstractVector{<:WorldPoint},
        observed_corners::AbstractVector{<:ProjectionPoint{T, S}},
        known_attitude::RotZYX,
        config::CameraConfig{S};
        noise_model = _defaultnoisemodel(observed_corners),
        initial_guess_pos::AbstractVector{<:Length} = SA[-1000.0, 0.0, 100.0]m,
        optimization_config = DEFAULT_OPTIMIZATION_CONFIG
    ) where {T, S}

    u₀ = initial_guess_pos .|> _ustrip(m) |> Array

    # for precompile we need the correct types
    observed_corners = [
        convertcamconf(CAMCONF4COMP, config, proj)
            for proj in observed_corners
    ]
    ps = PoseOptimizationParams3DOF(
        runway_corners, observed_corners,
        CAMCONF4COMP, noise_model, known_attitude
    )

    # Create problem and cache for this specific problem size
    prob = NonlinearLeastSquaresProblem{false}(POSEOPTFN, u₀, ps)
    cache = init(prob, ALG)
    solve!(cache)
    sol = (; u = cache.u, retcode = cache.retcode)

    !successful_retcode(sol.retcode) && throw(OptimizationFailedError(sol.retcode, sol))
    pos = WorldPoint(sol.u[1:3]m)
    rot = known_attitude
    return (; pos, rot)
end
