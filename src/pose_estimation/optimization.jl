include("./pointfeatures.jl")
include("./linefeatures.jl")

abstract type AbstractPoseOptimizationParams end

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

optvartransformation(x, ps::PoseOptimizationParams3DOF) = [-exp(x[1]); x[2]; exp(x[3])]
optvartransformation(x, ps::PoseOptimizationParams6DOF) = [-exp(x[1]); x[2]; exp(x[3]); x[4:6]]
revoptvartransformation(x, ps::PoseOptimizationParams3DOF) = [log(-x[1]); x[2]; log(x[3])]
revoptvartransformation(x, ps::PoseOptimizationParams6DOF) = [log(-x[1]); x[2]; log(x[3]); x[4:6]]

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
    optvar = optvar2nominal(optvar, ps)
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

"""
    makecache(u₀, ps::AbstractPoseOptimizationParams)

Create optimization cache.
"""
function makecache(u₀, ps::AbstractPoseOptimizationParams; kwargs...)
    T′ = Float64
    poseoptfn = NonlinearFunction{false,FullSpecialize}(pose_optimization_objective)
    prob = NonlinearLeastSquaresProblem{false}(poseoptfn, u₀, ps)
    reltol = real(oneunit(T′)) * (eps(real(one(T′))))^(2 // 5)
    abstol = real(oneunit(T′)) * (eps(real(one(T′))))^(2 // 5)
    init(prob, ALG; reltol, abstol, kwargs...)
end

# Dispatch that takes PointFeatures and LineFeatures directly
function estimatepose6dof(
    point_features::PointFeatures,
    line_features::LineFeatures=NO_LINES;
    initial_guess_pos::AbstractVector{<:Length}=SA[-1000.0, 0.0, 100.0]m,
    initial_guess_rot::AbstractVector{<:DimensionlessQuantity}=SA[0.0, 0.0, 0.0]rad,
    cache=nothing,
    solveargs=(;)
)
    u₀ = [
        initial_guess_pos .|> _ustrip(m);
        initial_guess_rot .|> _ustrip(rad)
    ] |> Array

    ps = PoseOptimizationParams6DOF(point_features, line_features)
    cache = isnothing(cache) ? makecache(u₀, ps) : (reinit!(cache, u₀; p=ps); cache)
    solve!(cache; solveargs...)
    sol = (; u=cache.u, retcode=cache.retcode)

    !successful_retcode(sol.retcode) && throw(OptimizationFailedError(sol.retcode, sol))
    pos = WorldPoint(sol.u[1:3]m)
    rot = RotZYX(roll=sol.u[4]rad, pitch=sol.u[5]rad, yaw=sol.u[6]rad)
    return (; pos, rot)
end

# Convenience dispatch for points and projections
function estimatepose6dof(
    runway_corners::AbstractVector{<:WorldPoint},
    observed_corners::AbstractVector{<:ProjectionPoint{T,:offset}},
    camconfig::AbstractCameraConfig{:offset}=CAMERA_CONFIG_OFFSET,
    noise_model::N=_defaultnoisemodel(observed_corners);
    initial_guess_pos::AbstractVector{<:Length}=SA[-1000.0, 0.0, 100.0]m,
    initial_guess_rot::AbstractVector{<:DimensionlessQuantity}=SA[0.0, 0.0, 0.0]rad,
    cache=nothing,
    solveargs=(;)
) where {T,N}
    point_features = PointFeatures(
        runway_corners |> Vector, observed_corners |> Vector,
        camconfig, noise_model
    )
    return estimatepose6dof(point_features;
        initial_guess_pos, initial_guess_rot, cache)
end

# Dispatch that takes PointFeatures and LineFeatures directly
function estimatepose3dof(
    point_features::PointFeatures,
    line_features::LineFeatures,
    known_attitude::RotZYX;
    initial_guess_pos::AbstractVector{<:Length}=SA[-1000.0, 0.0, 100.0]m,
    cache=nothing
)
    u₀ = initial_guess_pos .|> _ustrip(m) |> Array

    ps = PoseOptimizationParams3DOF(point_features, line_features, known_attitude)
    cache = isnothing(cache) ? makecache(u₀, ps) : (reinit!(cache, u₀; p=ps):cache)
    solve!(cache; solveargs...)
    sol = (; u=cache.u, retcode=cache.retcode)

    !successful_retcode(sol.retcode) && throw(OptimizationFailedError(sol.retcode, sol))
    pos = WorldPoint(sol.u[1:3]m)
    rot = known_attitude
    return (; pos, rot)
end

# Convenience dispatch for points and projections
function estimatepose3dof(
    runway_corners::AbstractVector{<:WorldPoint},
    observed_corners::AbstractVector{<:ProjectionPoint{T,:offset}},
    known_attitude::RotZYX,
    camconfig::AbstractCameraConfig{:offset}=CAMERA_CONFIG_OFFSET,
    noise_model::N=_defaultnoisemodel(observed_corners);
    initial_guess_pos::AbstractVector{<:Length}=SA[-1000.0, 0.0, 100.0]m,
    cache=nothing
) where {T,N}
    point_features = PointFeatures(
        runway_corners |> Vector, observed_corners |> Vector,
        camconfig, noise_model
    )
    return estimatepose3dof(point_features, NO_LINES, known_attitude;
        initial_guess_pos, cache)
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
# const ALG = LevenbergMarquardt(; autodiff=AD, linsolve=CholeskyFactorization(),
#     disable_geodesic=Val(true))
const ALG = LevenbergMarquardt(; autodiff=AD, linsolve=CholeskyFactorization())

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
