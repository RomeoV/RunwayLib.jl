include("./pointfeatures.jl")
include("./linefeatures.jl")

abstract type AbstractPoseOptimizationParams end

"""
    PoseOptimizationParams6DOF{PF, LF}

Parameters for 6-DOF pose optimization (position + attitude).
"""
struct PoseOptimizationParams6DOF{
    PF<:PointFeatures,
    LF<:LineFeatures,
    NC
} <: AbstractPoseOptimizationParams
    point_features::PF
    line_features::LF
    constraints::NTuple{NC,Pair{Int,Float64}}
end
PoseOptimizationParams6DOF(point_features::PF, line_features::LF) where {PF,LF} =
    PoseOptimizationParams6DOF{PF,LF,0}(point_features, line_features, ())

"""
    PoseOptimizationParams3DOF{A, PF, LF}

Parameters for 3-DOF pose optimization (position only with known attitude).
"""
struct PoseOptimizationParams3DOF{
    A<:Rotation{3},
    PF<:PointFeatures,
    LF<:LineFeatures,
    NC
} <: AbstractPoseOptimizationParams
    point_features::PF
    line_features::LF
    known_attitude::A
    constraints::NTuple{NC,Pair{Int,Float64}}
end
PoseOptimizationParams3DOF(point_features::PF, line_features::LF, known_attitude::A) where {PF,LF,A} =
    PoseOptimizationParams3DOF{A,PF,LF,0}(point_features, line_features, known_attitude, ())

"From optimization space into regular space."
optvar2nominal(x::AT, ps::PoseOptimizationParams3DOF) where {AT} = SA[-exp(x[1]); x[2]; exp(x[3])] |> AT
# For some reason this solution introduces an allocation which costs 15% performance for one solve...
# We don't see this issue for [`nominal2optvar`](@ref) although it should be the same...
optvar2nominal(x::AT, ps::PoseOptimizationParams6DOF) where {AT} = reduce(
    vcat, (
        SA[-exp(x[1]); x[2]; exp(x[3])],
        # RotZYX(RodriguesParam(x[4], x[5], x[6])) |> Rotations.params,
        x[4:6],
    )
) |> AT
# So we instead implement it like this for StaticArrays
optvar2nominal(x::AT, ps::PoseOptimizationParams6DOF) where {AT<:StaticArray} = SA[
    -exp(x[1]);
    x[2];
    exp(x[3]);
    x[4];
    x[5];
    x[6]
]

"From regular space into optimization space."
nominal2optvar(x::AT, ps::PoseOptimizationParams3DOF) where {AT} = SA[log(-x[1]); x[2]; log(x[3])] |> AT
nominal2optvar(x::AT, ps::PoseOptimizationParams6DOF) where {AT} = reduce(
    vcat, (
        SA[log(-x[1]); x[2]; log(x[3])],
        # RodriguesParam(RotZYX(x[4], x[5], x[6])) |> Rotations.params,
        x[4:6],
    )
) |> AT

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
    cam_pos = WorldPoint(optvar[1:3] .* m)

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

    # Constraint residuals
    cstr_residuals = [1e1 * (optvar[i] - val) for (i, val) in SVector(ps.constraints)] |> SVector

    # Combine residuals
    return reduce(vcat, (point_residuals, line_residuals, cstr_residuals)) |> Array
end

"""
    makecache(u₀, ps::AbstractPoseOptimizationParams)

Create optimization cache.
"""
function makecache(u₀, ps::AbstractPoseOptimizationParams; kwargs...)
    T′ = Float64
    poseoptfn = NonlinearFunction{false,FullSpecialize}(pose_optimization_objective)
    prob = NonlinearLeastSquaresProblem{false}(poseoptfn, nominal2optvar(u₀, ps), ps)
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
    solveargs=(;),
    constraints::NTuple{NC,Pair{Int,Float64}}=(),
) where {NC}
    u₀ = [
        initial_guess_pos .|> _ustrip(m);
        initial_guess_rot .|> _ustrip(rad)
    ]

    ps = PoseOptimizationParams6DOF(point_features, line_features, constraints)
    cache = isnothing(cache) ? makecache(u₀, ps) : (reinit!(cache, nominal2optvar(u₀, ps); p=ps); cache)
    solve!(cache; solveargs...)
    sol = (; u=optvar2nominal(cache.u, ps), retcode=cache.retcode)

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
    solveargs=(;),
) where {T,N}
    point_features = PointFeatures(
        runway_corners |> Vector, observed_corners |> Vector,
        camconfig, noise_model
    )
    return estimatepose6dof(point_features;
        initial_guess_pos, initial_guess_rot, cache, solveargs)
end

# Dispatch that takes PointFeatures and LineFeatures directly
function estimatepose3dof(
    point_features::PointFeatures,
    line_features::LineFeatures,
    known_attitude::RotZYX;
    initial_guess_pos::AbstractVector{<:Length}=SA[-1000.0, 0.0, 100.0]m,
    cache=nothing,
    solveargs=(;),
    constraints::NTuple{NC,Pair{Int,Float64}}=(),
) where {NC}
    u₀ = initial_guess_pos .|> _ustrip(m)

    ps = PoseOptimizationParams3DOF(point_features, line_features, known_attitude, constraints)
    cache = isnothing(cache) ? makecache(u₀, ps) : (reinit!(cache, nominal2optvar(u₀, ps); p=ps):cache)
    solve!(cache; solveargs...)
    sol = (; u=optvar2nominal(cache.u, ps), retcode=cache.retcode)

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
    cache=nothing,
    solveargs=(;)
) where {T,N}
    point_features = PointFeatures(
        runway_corners |> Vector, observed_corners |> Vector,
        camconfig, noise_model
    )
    return estimatepose3dof(point_features, NO_LINES, known_attitude;
        initial_guess_pos, cache, solveargs)
end


function setup_for_precompile()
    runway_corners = SA[
        WorldPoint(1000.0m, -50.0m, 0.0m),
        WorldPoint(1000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, -50.0m, 0.0m),
    ]
    true_pos = WorldPoint(-1300.0m, 0.0m, 80.0m)
    true_rot = RotZYX(roll=0.03, pitch=0.04, yaw=0.05)
    projections = SVector([project(true_pos, true_rot, corner, CAMERA_CONFIG_OFFSET) for corner in runway_corners])
    return (; runway_corners, projections, true_pos, true_rot)
end


const _defaultnoisemodel_points(pts) =
    let
        distributions = [SA[Normal(0.0, 2.0), Normal(0.0, 2.0)] for _ in pts]
        UncorrGaussianNoiseModel(reduce(vcat, distributions))
    end
const _defaultnoisemodel(pts) = _defaultnoisemodel_points(pts)
const _defaultnoisemodel_lines(line_endpoints) =
    let
        distributions = [SA[Normal(0.0, 2.0), Normal(0.0, 2.0), Normal(0.0, 2.0)] for _ in line_endpoints]
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
    prob = NonlinearLeastSquaresProblem{false}(POSEOPTFN, @SVector(rand(6)), ps)
    T = Float64
    # sqrt of the default
    reltol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    abstol = real(oneunit(T)) * (eps(real(one(T))))^(2 // 5)
    init(prob, ALG; reltol, abstol)
end

const CACHE_6DOF_p = let
    (; runway_corners, projections, true_pos, true_rot) = setup_for_precompile()
    noise_model = _defaultnoisemodel(projections)
    point_features = PointFeatures(runway_corners, projections, CAMERA_CONFIG_OFFSET, noise_model)
    p = PoseOptimizationParams6DOF(point_features, NO_LINES)
end
function makecache(u₀::SVector{6,Float64}, ps::typeof(CACHE_6DOF_p))
    # @info "USING PREMADE CACHE"
    cache = CACHE_6DOF
    reinit!(cache, nominal2optvar(u₀, ps); p=ps)
    return cache
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
