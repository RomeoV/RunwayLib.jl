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
        T, T′, T′′,
        RC <: AbstractVector{WorldPoint{T}},
        OC <: AbstractVector{ProjectionPoint{T′, :offset}},
        CC <: CameraMatrix{:offset},
        M <: AbstractMatrix{T′′},
        M′<: AbstractMatrix{T′′}
    } <: AbstractPoseOptimizationParams
    runway_corners::RC
    observed_corners::OC
    camconfig::CC
    cov::M
    Linv::M′
end
function PoseOptimizationParams6DOF(runway_corners, observed_corners, camconfig, noisemodel::NoiseModel)
    cov = covmatrix(noisemodel) |> Matrix
    return PoseOptimizationParams6DOF(runway_corners, observed_corners, camconfig, cov)
end
function PoseOptimizationParams6DOF(runway_corners, observed_corners, camconfig, cov::Matrix)
    U = cholesky(cov).U
    Linv = Matrix(inv(U'))  # Ensure dense matrix for consistent performance
    return PoseOptimizationParams6DOF(runway_corners, observed_corners, camconfig, cov, Linv)
end

"""
    PoseOptimizationParams3DOF{T, T′, S, A, RC, OC, M}

Parameters for 3-DOF pose optimization (position only with known attitude).
"""
struct PoseOptimizationParams3DOF{
        T, T′, T′′,
        A <: Rotation{3},
        RC <: AbstractVector{WorldPoint{T}},
        OC <: AbstractVector{ProjectionPoint{T′, :offset}},
        CC <: CameraMatrix{:offset},
        M <: AbstractMatrix{T′′},
        M′<: AbstractMatrix{T′′}
    } <: AbstractPoseOptimizationParams
    runway_corners::RC
    observed_corners::OC
    camconfig::CC
    cov::M
    Linv::M′
    known_attitude::A
end
function PoseOptimizationParams3DOF(runway_corners, observed_corners, camconfig, noisemodel::NoiseModel, known_attitude)
    cov = covmatrix(noisemodel) |> Matrix
    return PoseOptimizationParams3DOF(runway_corners, observed_corners, camconfig, cov, known_attitude)
end
function PoseOptimizationParams3DOF(runway_corners, observed_corners, camconfig, cov::Matrix, known_attitude)
    U = cholesky(cov).U
    Linv = Matrix(inv(U'))  # Ensure dense matrix for consistent performance
    return PoseOptimizationParams3DOF(runway_corners, observed_corners, camconfig, cov, Linv, known_attitude)
end

"From optimization space into regular space."
optvar2nominal(x, ps::PoseOptimizationParams3DOF) = [-exp(x[1]); x[2]; exp(x[3])]
# we need `reduce(vcat, ...)` here instead of [... ; ...] for type inference...
optvar2nominal(x, ps::PoseOptimizationParams6DOF) = reduce(vcat, [
    [-exp(x[1]); x[2]; exp(x[3])];
    RotZYX(RodriguesParam(x[4], x[5], x[6])) |> Rotations.params |> Array
])
"From regular space into optimization space."
nominal2optvar(x, ps::PoseOptimizationParams3DOF) = [log(-x[1]); x[2]; log(x[3])]
nominal2optvar(x, ps::PoseOptimizationParams6DOF) = reduce(vcat, [
    [log(-x[1]); x[2]; log(x[3])];
    RodriguesParam(RotZYX(x[4], x[5], x[6])) |> Rotations.params |> Array
])

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
    optvar = optvar2nominal(optvar, ps)
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
        (proj - obs)
        for (proj, obs) in zip(projected_corners, ps.observed_corners)
    ]
    errors = reduce(vcat, error_vectors)

    Linv = ps.Linv / 1px
    weighted_errors = Linv * errors

    return ustrip.(NoUnits, weighted_errors)
end

function setup_for_precompile(camconfig::CameraMatrix{:offset})
    runway_corners = [
        WorldPoint(1000.0m, -50.0m, 0.0m),
        WorldPoint(1000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, -50.0m, 0.0m),
    ]
    true_pos = WorldPoint(-1300.0m, 0.0m, 80.0m)
    true_rot = RotZYX(roll = 0.03, pitch = 0.04, yaw = 0.05)
    projections = [project(true_pos, true_rot, corner, camconfig) for corner in runway_corners]
    return (; runway_corners, projections, true_pos, true_rot)
end


const _defaultnoisemodel(pts) = let
    distributions = [SA[Normal(0.0, 2.0), Normal(0.0, 2.0)] for _ in pts] |> Array
    UncorrGaussianNoiseModel(reduce(vcat, distributions))
end

const AD = AutoForwardDiff(; chunksize = 1)
const POSEOPTFN = NonlinearFunction{false,FullSpecialize}(pose_optimization_objective)
const ALG = LevenbergMarquardt(; autodiff=AD, linsolve=CholeskyFactorization(),
    disable_geodesic=Val(true))

# Default CameraMatrix constant for cache creation
const CAMERA_MATRIX_OFFSET = CameraMatrix(CAMERA_CONFIG_OFFSET)


# Type-stable cache creation using OncePerTask (Julia 1.12+)
# Parameterized cache creation functions


# Simple cache definitions - CameraMatrix :offset only
const CACHE_6DOF = let
    (; runway_corners, projections, true_pos, true_rot) = setup_for_precompile(CAMERA_MATRIX_OFFSET)
    noise_model = _defaultnoisemodel(projections)
    ps = PoseOptimizationParams6DOF(
        runway_corners, projections,
        CAMERA_MATRIX_OFFSET, noise_model
    )
    u = nominal2optvar([-2000.0; 0; 100; 0; 0; 0], ps)
    prob = NonlinearLeastSquaresProblem{false}(POSEOPTFN, u, ps)
    T = Float64
    reltol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    abstol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    init(prob, ALG; reltol, abstol)
end

const CACHE_3DOF = let
    (; runway_corners, projections, true_pos, true_rot) = setup_for_precompile(CAMERA_MATRIX_OFFSET)
    noise_model = _defaultnoisemodel(projections)
    ps = PoseOptimizationParams3DOF(
        runway_corners, projections,
        CAMERA_MATRIX_OFFSET, noise_model,
        true_rot
    )
    u = nominal2optvar([-2000.0; 0; 100], ps)
    prob = NonlinearLeastSquaresProblem{false}(POSEOPTFN, u, ps)
    T = Float64
    reltol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    abstol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    init(prob, ALG; reltol, abstol)
end

function estimatepose6dof(
        runway_corners::AbstractVector{<:WorldPoint},
        observed_corners::AbstractVector{<:ProjectionPoint{T, :offset}},
        camconfig::CameraMatrix{:offset} = CameraMatrix(CAMERA_CONFIG_OFFSET),
        noise_model::N = _defaultnoisemodel(observed_corners);
        initial_guess_pos::AbstractVector{<:Length} = SA[-1000.0, 0.0, 100.0]m,
        initial_guess_rot::AbstractVector{<:DimensionlessQuantity} = SA[0.0, 0.0, 0.0]rad,
        optimization_config = DEFAULT_OPTIMIZATION_CONFIG
    ) where {T, N}
    u₀ = [
        # convert to SVector here in case we still have a WorldPoint
        initial_guess_pos |> SVector{3} .|> _ustrip(m);
        initial_guess_rot |> SVector{3} .|> _ustrip(rad)
    ]

    ps = PoseOptimizationParams6DOF(
        runway_corners |> Vector, observed_corners |> Vector,
        camconfig, noise_model
    )
    cache = CACHE_6DOF
    reinit!(cache, nominal2optvar(u₀, ps); p=ps)
    solve!(cache)
    sol = (; u=optvar2nominal(cache.u, ps), retcode=cache.retcode)

    !successful_retcode(sol.retcode) && throw(OptimizationFailedError(sol.retcode, sol))
    pos = WorldPoint(sol.u[1:3]m)
    rot = RotZYX(roll = sol.u[4]rad, pitch = sol.u[5]rad, yaw = sol.u[6]rad)
    return (; pos, rot)
end

function estimatepose3dof(
        runway_corners::AbstractVector{<:WorldPoint},
        observed_corners::AbstractVector{<:ProjectionPoint{T, :offset}},
        known_attitude::RotZYX,
        camconfig::CameraMatrix{:offset} = CameraMatrix(CAMERA_CONFIG_OFFSET),
        noise_model::N = _defaultnoisemodel(observed_corners);
        initial_guess_pos::AbstractVector{<:Length} = SA[-1000.0, 0.0, 100.0]m,
        optimization_config = DEFAULT_OPTIMIZATION_CONFIG
    ) where {T, N}

    u₀ = initial_guess_pos .|> _ustrip(m) |> Array

    # Convert coordinates to match cache type and get cache
    observed_corners_converted = [
        convertcamconf(CAMERA_MATRIX_OFFSET, camconfig, proj)
            for proj in observed_corners
    ]
    ps = PoseOptimizationParams3DOF(
        runway_corners |> Vector, observed_corners_converted |> Vector,
        camconfig, noise_model, known_attitude
    )
    
    cache = CACHE_3DOF
    reinit!(cache, nominal2optvar(u₀, ps); p=ps)
    solve!(cache)
    sol = (; u=optvar2nominal(cache.u, ps), retcode=cache.retcode)

    !successful_retcode(sol.retcode) && throw(OptimizationFailedError(sol.retcode, sol))
    pos = WorldPoint(sol.u[1:3]m)
    rot = known_attitude
    return (; pos, rot)
end
